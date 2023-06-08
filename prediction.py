from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path
from PIL import Image, ImageDraw
from torchvision import transforms
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from model import Model
    from torch.utils.data import DataLoader
    from typing_extensions import Any


class Predictor():
    def __init__(
        self,
        device: str | torch.device = 'cpu',
        examples: None = None,
        loader: DataLoader | None = None,
        mapping: dict[Any, Any] = None,
        model: Model = None
    ):
        self._model = model

        self.device = device
        self.examples = examples
        self.loader = loader
        self.mapping = mapping

    @property
    def model(self) -> Model:
        return self._model

    @model.setter
    def model(self, model: Model) -> None:
        self._model = model
        self.model.to(self.device)

    def _transform(self, image: Image) -> npt.NDArray:
        size = (224, 224)
        image = image.resize(size)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = image / 255.0

        image = Image.fromarray(
            np.uint8(image * 255)
        )

        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ),
        ])

        image = transformation(image)
        return np.expand_dims(image, axis=0)

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

    def from_path(self, path: Path) -> list[dict[Any, Any]]:
        image = Image.open(path)
        image = self._transform(image)

        label, box = self.model(
            torch.from_numpy(image).float()
            .to('cpu')
        )

        label = label.detach().flatten()
        label = torch.argmax(label, dim=0).item()

        x1, y1, x2, y2 = box.squeeze().tolist()

        w, h = (224, 224)
        x1 = int(w * x1)
        x2 = int(w * x2)
        y1 = int(h * y1)
        y2 = int(h * y2)

        p_box = (x1, y1, x2, y2)
        p_label = self.mapping[label]

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

        jaccard = self.iou(t_box, p_box)

        example = {}

        example[path.name] = {
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

        self.examples = [example]

    def from_random(self, path: Path) -> list[dict[Any, Any]]:
        image = Image.open(path)
        image = self._transform(image)

        label, box = self.model(
            torch.from_numpy(image).float()
            .to('cpu')
        )

        label = label.detach().flatten()
        label = torch.argmax(label, dim=0).item()

        x1, y1, x2, y2 = box.squeeze().tolist()

        w, h = (224, 224)
        x1 = int(w * x1)
        x2 = int(w * x2)
        y1 = int(h * y1)
        y2 = int(h * y2)

        p_box = (x1, y1, x2, y2)
        p_label = self.mapping[label]

        example = {}

        example[path.name] = {
            'prediction': {
                'box': p_box,
                'label': p_label,
            },
            'path': path
        }

        self.examples = [example]

    def plot(self, save: bool = False, show: bool = True) -> None:
        current = Path.cwd()

        prediction = current.joinpath('prediction')
        preprocess = current.joinpath('preprocess')

        for example in self.examples:
            for file in example:
                filename = example[file]

                if 'truth' in filename:
                    t_box = filename['truth']['box']
                    t_label = filename['truth']['label']

                p_box = filename['prediction']['box']
                p_label = filename['prediction']['label']

                if 'jaccard' in filename:
                    jaccard = round(example[file]['jaccard'] * 100, 4)

                path = example[file]['path']

                full = preprocess.joinpath(path)

                image = Image.open(full)

                # Draw the ground truth bounding box
                if 'truth' in filename and t_label == 'Waldo':
                        draw = ImageDraw.Draw(image)
                        draw.rectangle(t_box, outline='green', width=2)

                # Draw the prediction bounding box
                if any(coordinate > 0 for coordinate in p_box):
                    draw = ImageDraw.Draw(image)
                    draw.rectangle(p_box, outline='red', width=2)

                image = plt.imshow(image)

                plt.xticks([])
                plt.yticks([])

                if 'truth' in filename:
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

                if show:
                    manager = plt.get_current_fig_manager()
                    manager.window.state('zoomed')

                    plt.show()

                if save:
                    fig = image.get_figure()

                    filename = Path(path).name
                    path = prediction.joinpath(filename)

                    fig.savefig(
                        path,
                        bbox_inches='tight',
                        dpi=300,
                        transparent=True
                    )

                plt.close()
