from __future__ import annotations

import torch

from tabulate import tabulate
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
        validating: DataLoader = None
    ):
        self.cse = CrossEntropyLoss()
        self.device = device
        self.epoch = epoch
        self.model = model
        self.sll = SmoothL1Loss()
        self.testing = testing
        self.training = training
        self.validating = validating
        self.optimizer = optimizer

    def _single_train(self) -> None:
        self.model.train()

        total_classification_loss = 0.0
        total_bounding_box_loss = 0.0
        total_classification = 0
        total_iou = 0.0

        total_classification_correct = 0

        total = len(self.training)
        progress = tqdm(self.training, total=total)

        for x, y, z in progress:
            x, y, z = (
                x.to(self.device),
                y.to(self.device),
                z.to(self.device),
            )

            xsize, ysize = x.size(0), y.size(0)

            self.optimizer.zero_grad()

            (yp, zp) = self.model(x)

            classification_loss = self.cse(yp, y)

            total_classification_loss = (
                total_classification_loss +
                (classification_loss.item() * xsize)
            )

            bounding_box_loss = self.sll(zp, z)

            total_bounding_box_loss = (
                total_bounding_box_loss +
                (bounding_box_loss.item() * xsize)
            )

            total = bounding_box_loss + classification_loss
            total.backward()

            self.optimizer.step()

            prediction = yp.argmax(dim=1)

            total_classification_correct = (
                total_classification_correct +
                (prediction == y).sum().item()
            )

            total_classification = total_classification + ysize

            total_iou = total_iou + self._iou(zp, z).sum().item()

        mean_classification_loss = (
            classification_loss /
            len(self.training.dataset)
        )

        mean_bounding_box_loss = (
            bounding_box_loss /
            len(self.training.dataset)
        )

        classification_accuracy = (
            total_classification_correct /
            total_classification
        )

        iou_score = total_iou / len(self.training.dataset)

        return (
            mean_classification_loss,
            mean_bounding_box_loss,
            classification_accuracy,
            iou_score
        )

    def _single_validation(self) -> None:
        self.model.eval()

        total_classification_loss = 0.0
        total_bounding_box_loss = 0.0
        total_classification = 0
        total_iou = 0.0

        total_classification_correct = 0

        total = len(self.validating)
        progress = tqdm(self.validating, total=total)

        for x, y, z in progress:
            x, y, z = (
                x.to(self.device),
                y.to(self.device),
                z.to(self.device),
            )

            xsize, ysize = x.size(0), y.size(0)

            (yp, zp) = self.model(x)

            classification_loss = self.cse(yp, y)

            total_classification_loss = (
                total_classification_loss +
                (classification_loss.item() * xsize)
            )

            bounding_box_loss = self.sll(zp, z)

            total_bounding_box_loss = (
                total_bounding_box_loss +
                (bounding_box_loss.item() * xsize)
            )

            prediction = yp.argmax(dim=1)

            total_classification_correct = (
                total_classification_correct +
                (prediction == y).sum().item()
            )

            total_classification = total_classification + ysize

            total_iou = total_iou + self._iou(zp, z).sum().item()

        mean_classification_loss = (
            total_classification_loss /
            len(self.validating.dataset)
        )

        mean_bounding_box_loss = (
            total_bounding_box_loss /
            len(self.validating.dataset)
        )

        classification_accuracy = (
            total_classification_correct /
            total_classification
        )

        iou_score = total_iou / len(self.validating.dataset)

        return (
            mean_classification_loss,
            mean_bounding_box_loss,
            classification_accuracy,
            iou_score
        )

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

    def start(self) -> None:
        self.model.to(self.device)

        for i in range(self.epoch):
            print(f"[Epoch {i + 1}]")

            c_loss, b_loss, accuracy, iou = self._single_train()

            table = {
                'Metric': [
                    'Training Classification Loss',
                    'Training Bounding Box Loss',
                    'Training Accuracy',
                    'Training IOU Score'
                ],
                'Score': [
                    f"{c_loss.cpu().item():.10f}",
                    f"{b_loss.cpu().item():.10f}",
                    f"{accuracy:.10f}",
                    f"{iou:.10f}"
                ]
            }

            print()

            table = tabulate(
                table,
                headers='keys',
                showindex=False,
                tablefmt='pretty'
            )

            print(table)

            c_loss, b_loss, accuracy, iou = self._single_validation()

            table = {
                'Metric': [
                    'Validation Classification Loss',
                    'Validation Bounding Box Loss',
                    'Validation Accuracy',
                    'Validation IOU Score'
                ],
                'Score': [
                    f"{c_loss:.10f}",
                    f"{b_loss:.10f}",
                    f"{accuracy:.10f}",
                    f"{iou:.10f}"
                ]
            }

            print()

            table = tabulate(
                table,
                headers='keys',
                showindex=False,
                tablefmt='pretty'
            )

            print(table)

            print()

        print('Training is complete')
