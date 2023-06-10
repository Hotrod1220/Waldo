from __future__ import annotations

import pickle

from PIL import Image, ImageDraw
from random import SystemRandom
from waldo.constant import (
    CHARACTER,
    CWD,
    DATASET,
    DYNAMIC,
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

    w = 224
    h = 224
    size = (64, 64)

    coordinates = {}

    draw = False

    for path in source:
        with Image.open(path) as image:
            image.thumbnail(
                (1920, 1080)
            )

            background = image.convert('RGBA')
            background = background.copy()

            for i in range(1000):
                filename = f"{path.stem}_{i}{path.suffix}"
                destination = DYNAMIC.joinpath(filename)

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

                coordinates[filename] = box

                crop = crop.convert('RGB')
                crop.save(destination)

                # Draw the bounding box
                if draw:
                    draw = ImageDraw.Draw(crop)
                    draw.rectangle(box, outline='green', width=2)
                    crop.show()

                # character.close()
                # crop.close()

    path = DATASET.joinpath('dynamic.pkl')

    with open(path, 'wb') as handle:
        pickle.dump(coordinates, handle)



if __name__ == '__main__':
    main()
