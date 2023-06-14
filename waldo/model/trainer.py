from __future__ import annotations

import torch

from tabulate import tabulate
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model import Model
    from torch.utils.data.dataloader import DataLoader
    from torch.optim import AdamW


class Trainer():
    def __init__(
        self,
        device: str | torch.device = 'cpu',
        epoch: int = 0,
        model: Model = None,
        optimizer: AdamW = None,
        testing: DataLoader = None,
        training: DataLoader = None,
        validating: DataLoader = None
    ):
        self.cse = CrossEntropyLoss()
        self.device = device
        self.epoch = epoch
        self.model = model
        self.testing = testing
        self.training = training
        self.validating = validating
        self.optimizer = optimizer

        self.b_weight = 1.0
        self.c_weight = 0.0001

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

    def _giou_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
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

            ysize = y.size(0)

            self.optimizer.zero_grad()

            (yp, zp) = self.model(x)

            classification_loss = self.cse(yp, y)

            total_classification_loss = (
                total_classification_loss +
                (self.c_weight * classification_loss.item())
            )

            bounding_box_loss = self._giou_loss(zp, z).mean()

            total_bounding_box_loss = (
                total_bounding_box_loss +
                (self.b_weight * bounding_box_loss.item())
            )

            total = (
                (self.c_weight * classification_loss) +
                (self.b_weight * bounding_box_loss)
            )

            total.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            self.optimizer.step()

            prediction = yp.argmax(dim=1)

            total_classification_correct = (
                total_classification_correct +
                (prediction == y).sum().item()
            )

            total_classification = total_classification + ysize

            total_iou = total_iou + self._giou(zp, z).sum().item()

        mean_classification_loss = (
            total_classification_loss /
            len(self.training.dataset)
        )

        mean_bounding_box_loss = (
            total_bounding_box_loss /
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

        with torch.no_grad():
            for x, y, z in progress:
                x, y, z = (
                    x.to(self.device),
                    y.to(self.device),
                    z.to(self.device),
                )

                ysize = y.size(0)

                (yp, zp) = self.model(x)

                classification_loss = self.cse(yp, y)

                total_classification_loss = (
                    total_classification_loss +
                    (self.c_weight * classification_loss.item())
                )

                bounding_box_loss = self._giou_loss(zp, z).mean()

                total_bounding_box_loss = (
                    total_bounding_box_loss +
                    (self.b_weight * bounding_box_loss.item())
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

    def evaluate(self) -> float:
        self.model.eval()

        _, _, _, accuracy = self._single_validation()
        return accuracy

    def start(self) -> None:
        self.model.to(self.device)

        history = {
            'training': {
                'classification_accuracy': [],
                'classification_loss': [],
                'bounding_box_score': [],
                'bounding_box_loss': [],
            },
            'validation': {
                'classification_accuracy': [],
                'classification_loss': [],
                'bounding_box_score': [],
                'bounding_box_loss': [],
            }
        }

        for i in range(self.epoch):
            print(f"[Epoch {i + 1}]")

            c_loss, b_loss, accuracy, iou = self._single_train()

            c_loss = round(c_loss, 10)
            b_loss = round(b_loss, 10)
            accuracy = round(accuracy, 10)
            iou = round(iou, 10)

            history['training']['classification_accuracy'].append(accuracy)
            history['training']['classification_loss'].append(c_loss)
            history['training']['bounding_box_score'].append(iou)
            history['training']['bounding_box_loss'].append(b_loss)

            table = {
                'Metric': [
                    'Training Classification Loss',
                    'Training Bounding Box Loss',
                    'Training Accuracy',
                    'Training IOU Score'
                ],
                'Score': [
                    c_loss,
                    b_loss,
                    accuracy,
                    iou
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

            c_loss = round(c_loss, 10)
            b_loss = round(b_loss, 10)
            accuracy = round(accuracy, 10)
            iou = round(iou, 10)

            history['validation']['classification_accuracy'].append(accuracy)
            history['validation']['classification_loss'].append(c_loss)
            history['validation']['bounding_box_score'].append(iou)
            history['validation']['bounding_box_loss'].append(b_loss)

            table = {
                'Metric': [
                    'Validation Classification Loss',
                    'Validation Bounding Box Loss',
                    'Validation Accuracy',
                    'Validation IOU Score'
                ],
                'Score': [
                    c_loss,
                    b_loss,
                    accuracy,
                    iou
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

        return history
