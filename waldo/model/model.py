from __future__ import annotations

import torch

from torch import nn
from torchvision.models import resnet50


class Model(nn.Module):
    def __init__(self, device: str | torch.device = 'cpu'):
        super().__init__()

        self.device = device
        base = resnet50(weights='IMAGENET1K_V2')
        children = base.children()
        layers = list(children)[:-2]
        self.base = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()

        self.dense = nn.Sequential(
            nn.Linear(base.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.10)
        )

        # Classification
        self.classification = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.50),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            nn.Linear(128, 2),
            nn.Softmax(dim=1),
        )

        # Box
        self.box = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.50),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple(torch.Tensor, torch.Tensor):
        x = x.to(self.device)

        # Pass through the base model
        x = self.base(x)

        x = self.avgpool(x)
        x = self.flatten(x)

        # Dense feature extraction
        x = self.dense(x)

        return (
            self.classification(x),
            self.box(x)
        )
