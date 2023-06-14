from __future__ import annotations

import matplotlib.pyplot as plt

from gui.canvas import Canvas
from PIL import Image, ImageDraw
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    # QScrollArea,
    QVBoxLayout,
    QWidget
)
from typing import TYPE_CHECKING
from waldo.constant import CWD

if TYPE_CHECKING:
    from typing_extensions import Any


class Plot(QWidget):
    def __init__(self):
        super().__init__()

        self.canvas = Canvas()

        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

        self.setFixedHeight(450)

    def display(self, prediction: dict[str, Any]) -> None:
        self.canvas.cleanup()

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

        image = self.canvas.ax.matshow(image)

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

        self.canvas.draw()
