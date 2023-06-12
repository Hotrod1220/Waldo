from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw
from random import SystemRandom
from waldo.constant import (
    CHARACTER,
    CWD,
    SUFFIX,
    TRANSFORM
)


def main() -> None:
    character = CWD.joinpath('character')

    source = [
        file
        for file in TRANSFORM.glob('*')
        if file.is_file() and file.suffix.lower() in SUFFIX
    ]

    generator = SystemRandom()

    w = 2048
    h = 2048
    size = (512, 512)

    draw = True

    for path in source:
        with Image.open(path) as image:
            background = image.convert('RGBA')
            background = background.copy()

            for i in range(10):
                filename = f"{path.stem}_{i}{path.suffix}"
                destination = Path('images').joinpath(filename)

                width, height = background.size
                maximum_x = width - w
                maximum_y = height - h

                x = generator.randrange(0, maximum_x)
                y = generator.randrange(0, maximum_y)

                crop = background.crop(
                    (x, y, x + w, y + h)
                )

                width, height = crop.size
                maximum_x = width - 64
                maximum_y = height - 64

                x = generator.randrange(0, maximum_x)
                y = generator.randrange(0, maximum_y)

                character = generator.choice(CHARACTER)
                character.thumbnail(size)

                crop.paste(
                    character,
                    (x, y),
                    character
                )

                box = (x, y, x + character.width, y + character.height)
                box = tuple(float(coordinate) for coordinate in box)

                # Draw the bounding box
                if draw:
                    draw = ImageDraw.Draw(crop)
                    draw.rectangle(box, outline='green', width=6)

                crop = crop.convert('RGB')
                crop.save(destination)


if __name__ == '__main__':
    main()
