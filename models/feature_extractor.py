# =======================
# mandatory coding task 3
# =======================

import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        if pretrained:
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet18(weights=None)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.avgpool = backbone.avgpool

        self.out_features = backbone.fc.in_features

    def forward(self, x):
        # identical to ResNet _forward_impl() until avgpool
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

        return x