from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from PIL import Image, ImageDraw
from typing import TYPE_CHECKING
from waldo.constant import CWD

if TYPE_CHECKING:
    from collections.abc import Callable
    from model import Model
    from pathlib import Path
    from torch.utils.data import DataLoader
    from typing_extensions import Any


class Predictor():
    def __init__(
        self,
        device: str | torch.device = 'cpu',
        examples: None = None,
        loader: DataLoader | None = None,
        mapping: dict[Any, Any] = None,
        model: Model = None,
        transformation: Callable | None = None
    ):
        self.device = device
        self.examples = examples
        self.loader = loader
        self.mapping = mapping
        self.model = model
        self.transformation = transformation

    @staticmethod
    def iou(x: tuple[float, ...], y: tuple[float, ...]) -> float:
        x1 = max(x[0], y[0])
        y1 = max(x[1], y[1])
        x2 = min(x[2], y[2])
        y2 = min(x[3], y[3])

        intersection = (
            max(0, x2 - x1 + 1) *
            max(0, y2 - y1 + 1)
        )

        x_area = (
            (x[2] - x[0] + 1) *
            (x[3] - x[1] + 1)
        )

        y_area = (
            (y[2] - y[0] + 1) *
            (y[3] - y[1] + 1)
        )

        return (
            intersection /
            (x_area + y_area - intersection + 1e-6)
        )

    def prediction(self, path: Path) -> dict[str, Any]:
        image = Image.open(path)
        image = self.transformation(image)
        image = image.unsqueeze(0)

        label, box = self.model(image)

        label = label.detach().flatten()
        label = torch.argmax(label, dim=0).item()

        x1, y1, x2, y2 = box.squeeze().tolist()

        x1 = min(1, x1)
        y1 = min(1, y1)
        x2 = min(1, x2)
        y2 = min(1, y2)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(0, x2)
        y2 = max(0, y2)

        w, h = (224, 224)
        x1 = int(w * x1)
        x2 = int(w * x2)
        y1 = int(h * y1)
        y2 = int(h * y2)

        if y2 < y1:
            y1, y2 = y2, y1

        if x2 < x1:
            x1, x2 = x2, x1

        p_box = (x1, y1, x2, y2)
        p_label = self.mapping[label]

        return {
            'prediction': {
                'box': p_box,
                'label': p_label,
            }
        }

    def truth(self, path: Path) -> dict[str, Any]:
        if path.name not in self.annotation.filename.tolist():
            return None

        x1 = (
            self.annotation
            .loc[self.annotation.filename == path.name, 'x1']
            .squeeze()
        )

        y1 = (
            self.annotation
            .loc[self.annotation.filename == path.name, 'y1']
            .squeeze()
        )

        x2 = (
            self.annotation
            .loc[self.annotation.filename == path.name, 'x2']
            .squeeze()
        )

        y2 = (
            self.annotation
            .loc[self.annotation.filename == path.name, 'y2']
            .squeeze()
        )

        label = (
            self.annotation
            .loc[self.annotation.filename == path.name, 'label']
            .squeeze()
        )

        t_box = (x1, y1, x2, y2)
        t_label = self.mapping[label]

        return {
            'truth': {
                'box': t_box,
                'label': t_label,
            }
        }

    def _build(self, path: Path) -> dict[str, Any]:
        p = self.prediction(path)
        t = self.truth(path)

        p_box = p.get('prediction').get('box')

        result = {**p} if t is None else {**p, **t}

        if t is not None:
            t_box = t.get('truth').get('box')
            jaccard = self.iou(t_box, p_box)
            result['jaccard'] = jaccard

        result['path'] = path

        return result

    def from_loader(self, loader: DataLoader) -> list[dict[Any, Any]]:
        truths = []
        predictions = []

        for x, y, z in loader:
            with torch.no_grad():
                w, h = 224, 224

                # Ground truth
                for label, box in zip(y, z, strict=True):
                    label = label.item()

                    x1, y1, x2, y2 = box.tolist()

                    x1 = round(float(w * x1), 1)
                    y1 = round(float(h * y1), 1)
                    x2 = round(float(w * x2), 1)
                    y2 = round(float(h * y2), 1)

                    box = (x1, y1, x2, y2)

                    truths.append(
                        (label, box)
                    )

                # Prediction
                labels, boxes = self.model(x)

                for label, box in zip(labels, boxes, strict=True):
                    label = torch.argmax(label, dim=0).item()

                    x1, y1, x2, y2 = box.tolist()

                    x1 = round(float(w * x1), 1)
                    y1 = round(float(h * y1), 1)
                    x2 = round(float(w * x2), 1)
                    y2 = round(float(h * y2), 1)

                    box = (x1, y1, x2, y2)

                    predictions.append(
                        (label, box)
                    )

        examples = []

        for truth, prediction in zip(truths, predictions, strict=True):
            t_label, t_box = truth

            if all(coordinate == 0 for coordinate in t_box):
                continue

            p_label, p_box = prediction
            x1, y1, x2, y2 = t_box

            example = {}

            condition = (
                (self.annotation.x1 == x1) &
                (self.annotation.y1 == y1) &
                (self.annotation.x2 == x2) &
                (self.annotation.y2 == y2)
            )

            filename = self.annotation.loc[condition, 'filename'].squeeze()
            path = self.annotation.loc[condition, 'path'].squeeze()

            t_label = self.mapping[t_label]
            p_label = self.mapping[p_label]

            jaccard = self.iou(t_box, p_box)

            example[filename] = {
                'truth': {
                    'box': t_box,
                    'label': t_label,
                },
                'prediction': {
                    'box': p_box,
                    'label': p_label,
                },
                'jaccard': jaccard,
                'path': path
            }

            examples.append(example)

        self.examples = examples

    def from_path(self, path: Path) -> dict[str, Any]:
        return self._build(path)

    def plot(self, prediction: dict[str, Any]) -> None:
        if 'truth' in prediction:
            t_box = prediction['truth']['box']
            t_label = prediction['truth']['label']

        p_box = prediction['prediction']['box']
        p_label = prediction['prediction']['label']

        if 'jaccard' in prediction:
            jaccard = round(prediction['jaccard'] * 100, 4)

        path = prediction['path']

        full = CWD.joinpath(path)

        image = Image.open(full)

        # Draw the ground truth bounding box
        if 'truth' in prediction and t_label == 'Waldo':
                draw = ImageDraw.Draw(image)
                draw.rectangle(t_box, outline='green', width=2)

        # Draw the prediction bounding box
        if any(coordinate > 0 for coordinate in p_box):
            draw = ImageDraw.Draw(image)
            draw.rectangle(p_box, outline='red', width=2)

        image = plt.imshow(image)

        plt.xticks([])
        plt.yticks([])

        if 'truth' in prediction:
            expected = f"Expected: {t_label}"
            index = f"Jaccard Index: {jaccard}"

            plt.figtext(
                0.50,
                0.99,
                index,
                fontsize='large',
                color='black',
                ha='center',
                va='top'
            )

            plt.figtext(
                0.50,
                0.93,
                expected,
                fontsize='large',
                color='green',
                ha='center',
                va='top'
            )

        predicted = f"Predicted: {p_label}"

        plt.figtext(
            0.50,
            0.05,
            predicted,
            fontsize='large',
            color='red',
            ha='center',
            va='bottom'
        )

        return image
