from __future__ import annotations

import torch

from tabulate import tabulate
from torch.nn import CrossEntropyLoss, SmoothL1Loss, parameter
from torch.optim.adam import Adam
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model import Model
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

        self.classification_weight = 0.001
        self.bounding_box_weight = 5

    def evaluate(self) -> float:
        _, _, _, accuracy = self._single_validation()
        return accuracy

    def _giou(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inter_x1 = torch.max(prediction[:, 0], target[:, 0])
        inter_y1 = torch.max(prediction[:, 1], target[:, 1])
        inter_x2 = torch.min(prediction[:, 2], target[:, 2])
        inter_y2 = torch.min(prediction[:, 3], target[:, 3])

        inter_area = (
            torch.clamp(inter_x2 - inter_x1 + 1, min=0) *
            torch.clamp(inter_y2 - inter_y1 + 1, min=0)
        )

        pred_area = (
            (prediction[:, 2] - prediction[:, 0] + 1) *
            (prediction[:, 3] - prediction[:, 1] + 1)
        )

        target_area = (
            (target[:, 2] - target[:, 0] + 1) *
            (target[:, 3] - target[:, 1] + 1)
        )

        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + 1e-6)

        enclose_x1 = torch.min(prediction[:, 0], target[:, 0])
        enclose_y1 = torch.min(prediction[:, 1], target[:, 1])
        enclose_x2 = torch.max(prediction[:, 2], target[:, 2])
        enclose_y2 = torch.max(prediction[:, 3], target[:, 3])

        enclose_area = (
            torch.clamp(enclose_x2 - enclose_x1 + 1, min=0) *
            torch.clamp(enclose_y2 - enclose_y1 + 1, min=0)
        )

        return iou - (enclose_area - union_area) / (enclose_area + 1e-6)

    def _giou_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._giou(prediction, target)

    def _single_train(self) -> tuple[float, ...]:
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

            bounding_box_loss = self._giou_loss(zp, z).mean()

            total_bounding_box_loss = (
                total_bounding_box_loss +
                (bounding_box_loss.item() * xsize)
            )

            total = (
                self.classification_weight * classification_loss +
                self.bounding_box_weight * bounding_box_loss
            )

            total.backward()

            self.optimizer.step()

            prediction = yp.argmax(dim=1)

            total_classification_correct = (
                total_classification_correct +
                (prediction == y).sum().item()
            )

            total_classification = total_classification + ysize

            total_iou = total_iou + self._giou(zp, z).sum().item()

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

    def _single_validation(self) -> tuple[float, ...]:
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

            bounding_box_loss = self._giou_loss(zp, z).mean()

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

            total_iou = total_iou + self._giou(zp, z).sum().item()

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
