from __future__ import annotations

import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class Canvas(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure()

        self.ax = self.figure.add_subplot(
            111,
            autoscale_on=True
        )

        super().__init__(self.figure)

        self.figure.patch.set_facecolor('#222222')

        self.ax.patch.set_facecolor('#222222')
        self.ax.set_axis_off()

        self.image = None

        self.draw()

    def cleanup(self) -> None:
        for text in self.ax.texts:
            text.remove()

        for patch in self.ax.patches:
            patch.remove()

        for line in self.ax.lines:
            line.remove()

        if self.image is not None:
            self.image.remove()

        plt.close(self.figure)
