# =======================
# mandatory coding task 3
# =======================

import torch
import torch.nn as nn

from models.feature_extractor import ResNet18FeatureExtractor


class CustomClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.feature_net = ResNet18FeatureExtractor(pretrained=True)

        self.classifier = torch.nn.Linear(
            in_features=1024,
            out_features=num_classes
        )

    def forward(self, x):
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]

        f1 = self.feature_net(x1)
        f2 = self.feature_net(x2)

        fused = torch.cat([f1, f2], dim=1)

        logits = self.classifier(fused)

        return logits