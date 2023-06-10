from __future__ import annotations

import torch

from torch import nn
from torchvision.models import mobilenet_v2


class Model(nn.Module):
    def __init__(self, device: str | torch.device = 'cpu'):
        super().__init__()

        self.device = device

        # MobileNet
        base = mobilenet_v2(weights='DEFAULT')

        self.dense = nn.Sequential(
            nn.Linear(base.last_channel, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )

        children = base.children()
        children = list(children)

        base = nn.Sequential(*children[:-1])
        base.classifier = nn.Identity()

        self.base = base
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Classification
        self.classification = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.10),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.10),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.10),

            nn.Linear(128, 2),
            nn.Softmax(dim=1),
        )

        # Box
        self.box = nn.Sequential(
            nn.Linear(1024, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple(torch.Tensor, torch.Tensor):
        x = x.to(self.device)

        x = self.base(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense(x)

        return (
            self.classification(x),
            self.box(x)
        )
