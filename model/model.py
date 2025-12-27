# =======================
# mandatory coding task 2
# =======================

import torch.nn as nn

from torchvision import models


class Model():
    def __init__(self, num_classes: int):
        self.model = models.resnet18()

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def get_model(self):
        return self.model