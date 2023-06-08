from __future__ import annotations

import torch

from torch import nn


class Model(nn.Module):
    def __init__(self, device: str | torch.device = 'cpu'):
        super().__init__()
        self.device = device

        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(6),

            nn.Conv2d(6, 12, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(12),

            nn.Conv2d(12, 24, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(24),

            nn.Conv2d(24, 48, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48),

            nn.Conv2d(48, 192, 5),
            nn.ReLU(),
            nn.AvgPool2d(4, 2),
            nn.BatchNorm2d(192),
        )

        self.spatial = 2

        self.classification = nn.Sequential(
            nn.Linear(192 * self.spatial * self.spatial, 240),
            nn.ReLU(),

            nn.Linear(240, 120),
            nn.ReLU(),

            nn.Linear(120, 2),
            nn.Softmax(dim=1)
        )

        self.box = nn.Sequential(
            nn.Linear(192 * self.spatial * self.spatial, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple(torch.Tensor, torch.Tensor):
        x = x.to(self.device)

        x = self.convolutional(x)
        x = torch.flatten(x, 1)

        return (
            self.classification(x),
            self.box(x)
        )
