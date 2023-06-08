from __future__ import annotations

import torch

from PIL import Image
from torch.utils.data import Dataset
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from collections.abc import Callable
    from pathlib import Path
    from typing_extensions import Any


class WaldoDataset(Dataset):
    def __init__(
        self,
        annotation: pd.DataFrame | None = None,
        current: Path | None = None,
        device: str | torch.device | None = None,
        settings: dict[str, Any] | None = None,
        transformation: Callable | None = None
    ):
        super().__init__()
        self.annotation = annotation
        self.current = current
        self.device = device
        self.transformation = transformation
        self.settings = settings

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        path = self.annotation.loc[index, 'path']
        label = self.annotation.loc[index, 'label'].astype('int')

        x1 = self.annotation.loc[index, 'x1'].astype('float')
        y1 = self.annotation.loc[index, 'y1'].astype('float')
        x2 = self.annotation.loc[index, 'x2'].astype('float')
        y2 = self.annotation.loc[index, 'y2'].astype('float')

        box = (x1, y1, x2, y2)
        box = [coordinate / 224 for coordinate in box]

        location = (
            self.current
            .joinpath('preprocess')
            .joinpath(path)
        )

        image = Image.open(location)
        image = image.convert('RGB')

        if self.transformation is not None:
            image = self.transformation(image)

        label = torch.tensor(label, dtype=torch.long)
        box = torch.tensor(box, dtype=torch.float)

        return (image, label, box)
