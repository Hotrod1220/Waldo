from __future__ import annotations

from PIL import Image
from waldo.constant import NOT_WALDO, SUFFIX


def main() -> None:
    files = [
        file
        for file in NOT_WALDO.glob('*/*')
        if file.is_file() and file.suffix.lower() in SUFFIX
    ]

    size = (224, 224)
    width, height = size

    for file in files:
        print(f"Processing: {file}")

        image = Image.open(file)
        width, height = image.size

        condition = (
            width == width and
            height == height and
            image.mode == 'RGBA'
        )

        if condition:
            continue

        image = image.convert('RGBA')
        image = image.resize(size, Image.ANTIALIAS)

        path = file.parent.joinpath(file.stem + '.png')
        image.save(path)

        image.close()

        file.unlink()


if __name__ == '__main__':
    main()
