from __future__ import annotations

import torch

from torch import nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from model import Model
    from torch.optim.adam import Adam
    from torch.utils.data.dataloader import DataLoader


class Trainer():
    def __init__(
        self,
        device: str | torch.device = 'cpu',
        epoch: int = 0,
        model: Model = None,
        optimizer: Adam = None,
        testing: DataLoader = None,
        training: DataLoader = None,
        validating: DataLoader = None,

    ):
        self._model = model

        self.device = device
        self.epoch = epoch
        self.testing = testing
        self.training = training
        self.validating = validating
        self.optimizer = optimizer

    @property
    def model(self) -> Model:
        return self._model

    @model.setter
    def model(self, model: Model) -> None:
        self._model = model
        self.model.to(self.device)

    def _single_train(self) -> None:
        self.model.train()

        for x, y, z in self.training:
            x, y, z = (
                x.to(self.device),
                y.to(self.device),
                z.to(self.device),
            )

            self.optimizer.zero_grad()

            (yp, zp) = self.model(x)

            loss = nn.CrossEntropyLoss()
            classification_loss = loss(yp, y)

            loss = nn.MSELoss()
            bounding_box_loss = loss(zp, z)

            total = bounding_box_loss + classification_loss
            total.backward()

            self.optimizer.step()

        print(f"Loss: {total}")

    def _single_validation(self) -> None:
        self.model.eval()

        label = 0
        box = 0.0
        samples = 0

        for x, y, z in self.validating:
            x, y, z = (
                x.to(self.device),
                y.to(self.device),
                z.to(self.device),
            )

            (yp, zp) = self.model(x)

            prediction = yp.argmax(dim=1, keepdim=True)

            label = label + prediction.eq(
                y.view_as(prediction)
            ).sum().item()

            iou = self._iou(zp, z)
            box = box + iou.sum().item()

            samples = samples + x.size(0)

        label_accuracy = label / len(self.validating.dataset)
        print(f"Label accuracy: {label_accuracy}")

        box_accuracy = box / samples
        print(f"Bounding box accuracy: {box_accuracy}")

    def _iou(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x1 = torch.max(x[:, 0], y[:, 0])
        y1 = torch.max(x[:, 1], y[:, 1])
        x2 = torch.min(x[:, 2], y[:, 2])
        y2 = torch.min(x[:, 3], y[:, 3])

        intersection = (
            torch.clamp(x2 - x1 + 1, min=0) *
            torch.clamp(y2 - y1 + 1, min=0)
        )

        x_area = (
            (x[:, 2] - x[:, 0] + 1) *
            (x[:, 3] - x[:, 1] + 1)
        )

        y_area = (
            (y[:, 2] - y[:, 0] + 1) *
            (y[:, 3] - y[:, 1] + 1)
        )

        return (
            intersection /
            (x_area + y_area - intersection + 1e-6)
        )

    def predict(self, x: torch.Tensor, y: np.int64, mapping: list[str]) -> int:
        self.model.eval()

        with torch.no_grad():
            prediction = self.model(x)
            index = prediction[0].argmax(0)

            return (
                mapping[index],
                mapping[y]
            )

    def start(self) -> None:
        self.model.to(self.device)

        for i in range(self.epoch):
            print(f"Epoch {i + 1}")

            self._single_train()
            self._single_validation()

            print('\n')

        print('Training is complete')
