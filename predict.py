"""
使用训练好的模型进行预测
"""
import os
import torch
import torch.nn as nn
from torchvision import transforms
from models.resnet import ResNet50
from models.ast import ASTModel
from util.icbhi_dataset import ICBHIDataset
from util.icbhi_util import get_score

# ========== 配置 ==========
CHECKPOINT_PATH = "./save/icbhi_resnet50_train_resnet/best.pth"
DATA_FOLDER = "./data/ICBHI/ICBHI_final_database"
MODEL_TYPE = "resnet50"  # 或 "ast"
N_CLS = 4  # 4 类分类
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== 加载模型 ==========
def load_model(checkpoint_path, model_type, n_cls, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model']
    
    # 处理 DataParallel 保存的模型
    new_state_dict = {}
    for k, v in state_dict.items():
        if "module." in k:
            k = k.replace("module.", "")
        if "backbone." in k:
            k = k.replace("backbone.", "")
        new_state_dict[k] = v
    
    if model_type == "resnet50":
        model = ResNet50()
        classifier = nn.Linear(model.final_feat_dim, n_cls)
    elif model_type == "ast":
        model = ASTModel(
            input_fdim=1024,
            input_tdim=256,
            label_dim=n_cls,
            audioset_pretrained=False
        )
        classifier = nn.Linear(model.mlp_head.in_features, n_cls)
    else:
        raise ValueError(f"Unsupported model: {model_type}")
    
    model.load_state_dict(new_state_dict, strict=False)
    classifier.load_state_dict(checkpoint['classifier'])
    
    model.eval()
    classifier.eval()
    
    return model, classifier

# ========== 预测单个文件 ==========
def predict_file(model, classifier, audio_path, device=DEVICE):
    """预测单个音频文件"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(1024, 256))
    ])
    
    # 加载音频并转换（需要实现音频预处理）
    # 这里简化示例，实际需使用 ICBHIDataset 的预处理
    
    model.to(device)
    classifier.to(device)
    
    with torch.no_grad():
        inputs = transform(audio_path).unsqueeze(0).to(device)
        features = model(inputs)
        outputs = classifier(features)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    
    return pred, probs.cpu().numpy()[0]

# ========== 主函数 ==========
def main():
    print(f"设备：{DEVICE}")
    print(f"加载模型：{CHECKPOINT_PATH}")
    model, classifier = load_model(CHECKPOINT_PATH, MODEL_TYPE, N_CLS, DEVICE)
    
    model.to(DEVICE)
    classifier.to(DEVICE)
    
    print("\n模型加载成功!")
    print(f"模型类型：{MODEL_TYPE}")
    print(f"分类数：{N_CLS}")
    
    # 在验证集上评估
    print(f"\n数据路径：{DATA_FOLDER}")
    print("如需预测单个文件，请修改 predict_file() 函数")

if __name__ == "__main__":
    main()
