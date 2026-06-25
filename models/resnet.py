import torch
import torchvision
from torch import Tensor
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

class ResNet50(torchvision.models.resnet.ResNet):
    def __init__(self, track_bn=True, imagenet_pretrained=False):
        def norm_layer(*args, **kwargs):
            return nn.BatchNorm2d(*args, **kwargs, track_running_stats=track_bn)
        super().__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], norm_layer=norm_layer)
        del self.fc
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.final_feat_dim = 2048
        if imagenet_pretrained:
            state_dict = load_state_dict_from_url(
                "https://download.pytorch.org/models/resnet50-0676ba61.pth",
                progress=True,
            )
            if "conv1.weight" in state_dict:
                w = state_dict["conv1.weight"]
                if w.ndim == 4 and w.shape[1] == 3:
                    state_dict["conv1.weight"] = w.mean(dim=1, keepdim=True)
            state_dict.pop("fc.weight", None)
            state_dict.pop("fc.bias", None)
            self.load_state_dict(state_dict, strict=False)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x


