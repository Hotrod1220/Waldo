from __future__ import annotations

import pickle

from PIL import Image, ImageDraw
from random import SystemRandom
from waldo.constant import (
    CHARACTER,
    DATASET,
    STATIC,
    SUFFIX,
    TRANSFORM
)


def main() -> None:
    source = [
        file
        for file in TRANSFORM.glob('*')
        if file.is_file() and file.suffix.lower() in SUFFIX
    ]

    mapping = {
        '00': {
            'section': (960, 540),
            'size': (64, 64)
        },
        '01': {
            'section': (960, 540),
            'size': (72, 72)
        },
        '02': {
            'section': (600, 200),
            'size': (64, 64)
        },
        '03': {
            'section': (400, 400),
            'size': (80, 80)
        },
        '04': {
            'section': (700, 300),
            'size': (150, 150)
        },
        '05': {
            'section': (960, 540),
            'size': (80, 80)
        },
        '06': {
            'section': (960, 540),
            'size': (72, 72)
        },
        '07': {
            'section': (960, 540),
            'size': (72, 72)
        },
        '08': {
            'section': (400, 400),
            'size': (150, 150)
        },
    }

    generator = SystemRandom()

    w = 224
    h = 224

    coordinates = {}

    draw = False

    for i, path in enumerate(source, 0):
        i = str(i)
        i = i.zfill(2)

        section = mapping.get(i).get('section')
        size = mapping.get(i).get('size')

        scene = STATIC.joinpath(i)
        scene.mkdir(parents=True, exist_ok=True)

        with Image.open(path) as image:
            dimension = (1920, 1080)
            image.thumbnail(dimension)

            background = image.convert('RGBA')
            background = background.copy()

            for i in range(10000):
                filename = f"{i}_{path.name}"
                destination = scene.joinpath(filename)

                width, height = background.size
                maximum_x = width - w
                maximum_y = height - h

                x, y = section

                crop = background.crop(
                    (x, y, x + w, y + h)
                )

                padding, _ = size

                width, height = crop.size
                maximum_x = width - padding
                maximum_y = height - padding

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

    path = DATASET.joinpath('static.pkl')

    with open(path, 'wb') as handle:
        pickle.dump(coordinates, handle)


if __name__ == '__main__':
    main()
