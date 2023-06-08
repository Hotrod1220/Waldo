import pandas as pd

from pathlib import Path
from PIL import Image, ImageDraw


def main() -> None:
    cwd = Path.cwd()

    # Create directories
    dataset = cwd.joinpath('dataset')

    waldo = dataset.joinpath('waldo')
    not_waldo = dataset.joinpath('not_waldo')

    waldo.mkdir(parents=True, exist_ok=True)
    not_waldo.mkdir(parents=True, exist_ok=True)

    suffix = [
        '.bmp',
        '.gif',
        '.jpg',
        '.jpeg',
        '.png',
        '.webp'
    ]

    path = dataset.joinpath('waldo.csv')
    dataframe = pd.read_csv(path)

    images = [
        file
        for file in waldo.glob('*')
        if file.is_file() and file.suffix.lower() in suffix
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
