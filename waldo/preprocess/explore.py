from __future__ import annotations

import pandas as pd

from PIL import Image, ImageDraw
from waldo.constant import DATASET, SUFFIX, WALDO


def main() -> None:
    path = DATASET.joinpath('waldo.csv')
    dataframe = pd.read_csv(path)

    images = [
        file
        for file in WALDO.glob('*')
        if file.is_file() and file.suffix.lower() in SUFFIX
    ]

    for image in images:
        filename = image.name

        x1 = (
            dataframe
            .loc[dataframe.filename == filename, 'x1']
            .squeeze()
        )

        y1 = (
            dataframe
            .loc[dataframe.filename == filename, 'y1']
            .squeeze()
        )

        x2 = (
            dataframe
            .loc[dataframe.filename == filename, 'x2']
            .squeeze()
        )

        y2 = (
            dataframe
            .loc[dataframe.filename == filename, 'y2']
            .squeeze()
        )

        box = (x1, y1, x2, y2)

        image = Image.open(image)
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline='green', width=3)

        image.show()


if __name__ == '__main__':
    main()
