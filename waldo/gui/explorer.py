from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QPushButton,
    QWidget
)


class FileExplorer(QWidget):
    def __init__(self):
        super().__init__()

        self.setFixedHeight(250)

        self.layout = QVBoxLayout(self)

        self.list = QListWidget()

        self.previous = QPushButton('Previous')
        self.predict = QPushButton('Where\'s Waldo?')
        self.browse = QPushButton('Browse')
        self.next = QPushButton('Next')

        self.predict.setObjectName('Predict')

        self.group = QHBoxLayout()
        self.group.addWidget(self.previous)
        self.group.addWidget(self.predict)
        self.group.addWidget(self.browse)
        self.group.addWidget(self.next)
        self.group.setContentsMargins(0, 0, 0, 0)

        self.container = QWidget()
        self.container.setLayout(self.group)

        self.layout.addWidget(self.list)
        self.layout.addWidget(self.container)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

    def add(self, path: list[str]) -> None:
        self.list.addItems(path)
        self.list.setCurrentRow(0)

    def insert(self, path: str) -> None:
        self.list.insertItem(0, path)
        self.list.setCurrentRow(0)
