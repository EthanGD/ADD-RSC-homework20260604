import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import soundfile as sf
import torchaudio
from torchvision import transforms
from models.resnet import ResNet50
from models.adapt_diff_denoise import DiffTransformerLayer
from util.icbhi_util import generate_fbank

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def predict_single_wav(audio_path, checkpoint_path):
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(project_root, 'data', 'ICBHI', 'ICBHI_final_database')
    
    args = type('Args', (), {
        'sample_rate': 16000,
        'desired_length': 8,
        'nfft': 1024,
        'n_mels': 128,
        'resz': 1,
        'h': 1024,
        'w': 256,
        'n_cls': 4,
        'cls_list': ['normal', 'crackle', 'wheeze', 'both'],
        'denoise_d_model': 256,
        'denoise_num_heads': 8,
        'denoise_depth': 6,
        'loss_beta': 0.5,
        'model': 'resnet50',
        'imagenet_pretrained': False,
        'f_min': 50,
        'f_max': 2000,
    })()
    args.hop = args.nfft // 2

    cls_list = args.cls_list

    model = ResNet50(imagenet_pretrained=args.imagenet_pretrained)
    classifier = torch.nn.Linear(model.final_feat_dim, args.n_cls)
    model = model.to(device)
    classifier = classifier.to(device)

    bias_denoise_encoder = DiffTransformerLayer(
        d_model=args.denoise_d_model,
        num_heads=args.denoise_num_heads,
        depth=args.denoise_depth
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = ckpt.get('model', ckpt)
    classifier_state = ckpt.get('classifier', None)
    denoise_state = ckpt.get('bias_denoise_encoder', None)

    new_state_dict = {}
    for k, v in model_state.items():
        if "module." in k:
            k = k.replace("module.", "")
        if "backbone." in k:
            k = k.replace("backbone.", "")
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)

    if classifier_state is not None:
        classifier.load_state_dict(classifier_state, strict=True)
    if denoise_state is not None:
        bias_denoise_encoder.load_state_dict(denoise_state, strict=True)

    model.eval()
    classifier.eval()
    bias_denoise_encoder.eval()

    waveform, sr = sf.read(audio_path)
    # soundfile returns (samples, channels)
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)
    waveform = torch.from_numpy(waveform).float()
    
    # Resample if needed
    if sr != args.sample_rate:
        waveform = waveform.unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, args.sample_rate)
        waveform = resampler(waveform).squeeze(0)

    duration = waveform.shape[0] / args.sample_rate
    print(f"\nAudio info: {duration:.2f}s, {sr}Hz -> {args.sample_rate}Hz")

    if duration < args.desired_length:
        repeat_times = int(args.desired_length / duration) + 1
        waveform = waveform.repeat(repeat_times)[:int(args.desired_length * args.sample_rate)]
    elif duration > args.desired_length:
        waveform = waveform[:int(args.desired_length * args.sample_rate)]

    # generate_fbank expects (audio, sample_rate, n_mels), returns (T, 128, 1)
    mel = generate_fbank(waveform.unsqueeze(0), args.sample_rate, args.n_mels)
    mel = torch.from_numpy(mel).float().to(device)
    # Shape: (T, 128, 1) -> squeeze last dim -> (T, 128)
    mel = mel.squeeze(-1)
    # Add batch dimension: (T, 128) -> (1, T, 128)
    mel = mel.unsqueeze(0)
    
    # For prediction, skip ADD denoising and directly use mel spectrogram
    # Model expects: (batch, 1, time, freq) after unsqueeze(1)
    mel_input = mel.unsqueeze(1)  # (1, 1, T, 128)

    with torch.no_grad():
        # Skip ADD denoising for single prediction, use mel directly
        features = model(mel_input)
        output = classifier(features)
        probs = torch.softmax(output, dim=1)

        pred = output.argmax(dim=1).item()
        confidence = probs[0, pred].item()

        result = {
            'predicted_class': cls_list[pred],
            'confidence': confidence,
            'all_probs': {cls_list[i]: probs[0, i].item() for i in range(args.n_cls)}
        }

    print(f"\nPrediction for: {os.path.basename(audio_path)}")
    print("=" * 60)
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"\nAll probabilities:")
    for cls, prob in result['all_probs'].items():
        print(f"  {cls}: {prob*100:.2f}%")
    print("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single WAV file prediction')
    parser.add_argument('--audio', type=str, default=None,
                        help='Path to the audio file to predict (relative to project root)')
    parser.add_argument('--checkpoint', type=str, default='icbhi_resnet50_train_resnet/best.pth',
                        help='Path to the model checkpoint (relative to script directory)')
    args = parser.parse_args()

    # Get project root and script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Build paths
    if args.audio is None:
        audio_path = os.path.join(project_root, 'data', 'ICBHI', 'ICBHI_final_database', '101_1b1_Al_sc_Meditron.wav')
    else:
        # Check if absolute path, otherwise relative to project root
        if os.path.isabs(args.audio):
            audio_path = args.audio
        else:
            audio_path = os.path.join(project_root, args.audio)
    
    checkpoint_path = os.path.join(script_dir, args.checkpoint)

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    predict_single_wav(audio_path, checkpoint_path)
