from __future__ import annotations

import pandas as pd
import torch

from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QFileDialog,
    QWidget,
)
from gui.dropdown import Dropdown
from gui.explorer import FileExplorer
from gui.plot import Plot
from model.model import Model
from model.prediction import Predictor
from model.transformation import Transformation
from waldo.constant import (
    DATASET,
    MODEL,
    STATE,
    SUFFIX,
    WALDO
)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.resize(1200, 750)
        self.move(100, 100)

        self.setWindowTitle('Where\'s Waldo?')

        path = Path.cwd().joinpath('gui/asset/icon.png').as_posix()
        self.icon = QIcon(path)
        self.setWindowIcon(self.icon)

        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        self.layout = QVBoxLayout(self.widget)

        self.playout = QVBoxLayout()
        self.elayout = QVBoxLayout()

        self.current = None
        self.figure = None

        self.model = None
        self.prediction = None
        self.envelope = None
        self.spectrogram = None
        self.result = None

        self.dropdown = Dropdown()
        self.explorer = FileExplorer()
        self.plot = Plot()
        self.transformation = Transformation(device='cpu')

        width = int(self.width() / 1.2)

        self.plot.canvas.setFixedWidth(width)
        self.dropdown.box.setFixedWidth(width)
        self.explorer.list.setMinimumWidth(width)

        self.mapping = {
            0: 'Not Waldo',
            1: 'Waldo'
        }

        self.load()

        self.prediction = QLabel(
            'Prediction:',
            alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter
        )

        self.prediction.setObjectName('Prediction')

        self.elayout.addWidget(self.explorer)

        self.layout.addWidget(self.dropdown)
        self.layout.addWidget(self.plot)
        self.layout.addLayout(self.playout)
        self.layout.addWidget(self.prediction)
        self.layout.addLayout(self.elayout)
        self.layout.setStretch(2, 1)

        self.explorer.list.currentItemChanged.connect(self.on_selection_change)
        self.explorer.previous.clicked.connect(self.on_click_previous)
        self.explorer.predict.clicked.connect(self.on_click_predict)
        self.explorer.browse.clicked.connect(self.on_click_load)
        self.explorer.next.clicked.connect(self.on_click_next)

    def _predict(self) -> None:
        if self.explorer.list.count() == 0:
            QMessageBox.warning(
                self,
                'Warning',
                'Please select a folder to load.'
            )

            return

        current = self.explorer.list.currentItem().text()
        self.current = Path(current)

        result = self.predictor.from_path(self.current)

        label = (
            result
            .get('prediction')
            .get('label')
        )

        self.prediction.setText(f"Prediction: {label}")

        self.plot.display(result)

    def load(self) -> None:
        self.model = Model()
        self.model.device = 'cpu'

        path = MODEL.joinpath('state/model.pth')

        state = torch.load(path)
        self.model.load_state_dict(state)

        self.model.eval()

        csv = DATASET.joinpath('waldo.csv')
        self.annotation = pd.read_csv(csv)

        self.predictor = Predictor()
        self.predictor.annotation = self.annotation
        self.predictor.mapping = self.mapping
        self.predictor.model = self.model
        self.predictor.transformation = self.transformation

    def on_click_load(self) -> None:
        dialog = QFileDialog()

        index = self.dropdown.box.currentIndex()

        match index:
            case 0:
                # Load from a folder
                directory = DATASET.as_posix()

                dialog.setFileMode(QFileDialog.FileMode.Directory)
                dialog.setDirectory(directory)

                if dialog.exec() == QFileDialog.DialogCode.Accepted:
                    path, *_ = dialog.selectedFiles()

                    files = [
                        file.as_posix()
                        for file in Path(path).glob('*')
                        if file.suffix in SUFFIX and
                        file.exists() and file.is_file()
                    ]

                    self.explorer.list.clear()
                    self.explorer.add(files)

            case 1:
                # Load from a single file
                directory = WALDO.as_posix()

                dialog.setDirectory(directory)
                dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

                file, _ = dialog.getOpenFileName(self, 'Select File')

                if file:
                    self.explorer.insert(file)

            case 2:
                # Load from a loader
                directory = STATE.as_posix()

                dialog.setDirectory(directory)
                dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
                dialog.setNameFilter('Pickle (*.pkl)')

                path, _ = dialog.getOpenFileName(self, 'Select File')

                if path:
                    self.explorer.list.clear()

        self.explorer.list.setFocus()

    def on_click_next(self) -> None:
        index = self.explorer.list.currentIndex().row()
        index = 0 if index == self.explorer.list.count() - 1 else index + 1
        self.explorer.list.setCurrentRow(index)

    def on_click_predict(self) -> None:
        self._predict()

    def on_click_previous(self) -> None:
        index = self.explorer.list.currentIndex().row()
        index = self.explorer.list.count() - 1 if index == 0 else index - 1
        self.explorer.list.setCurrentRow(index)

    def on_selection_change(self) -> None:
        self._predict()
