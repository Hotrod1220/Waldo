import numpy as np
import pickle

from copy import deepcopy
from itertools import combinations
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from random import SystemRandom


def get_transparent_background() -> Image:
    width, height = 224, 224

    return Image.new(
        'RGBA',
        (width, height),
        (255, 0, 0, 0)
    )


def get_random_background() -> Image:
    generator = SystemRandom()

    width, height = 224, 224

    option = np.arange(0, 255, 1)

    red = generator.choice(option)
    green = generator.choice(option)
    blue = generator.choice(option)
    alpha = 255

    return Image.new(
        'RGBA',
        (width, height),
        (red, green, blue, alpha)
    )


def main() -> None:
    cwd = Path.cwd()

    # Create directories
    dataset = cwd.joinpath('dataset')

    character = dataset.joinpath('character')
    character.mkdir(parents=True, exist_ok=True)

    # Open the original Waldo image
    path = cwd.joinpath('character').joinpath('waldo.png')
    waldo = Image.open(path)
    waldo = waldo.convert('RGBA')

    generator = SystemRandom()

    coordinates = {}

    # Transform Waldo using each possible combination
    for i in range(5):
        filename = f"{i}_{path.name}"
        destination = character.joinpath(filename)

        background = get_random_background()
        # background = get_transparent_background()
        image = waldo.copy()

        # Resize Waldo to be at least 28x28px, but less than
        # the width and height of the background
        width, height = background.size
        x = generator.randrange(28, width)
        y = generator.randrange(28, height)

        image.thumbnail(
            (x, y)
        )

        box = image.getbbox()
        box = tuple(float(coordinate) for coordinate in box)

        coordinates[filename] = box

        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline='green', width=3)

        # Move Waldo to a random location
        width, height = background.size
        maximum_x = width - x
        maximum_y = height - y

        x = generator.randrange(0, maximum_x)
        y = generator.randrange(0, maximum_y)

        background.paste(
            image,
            (x, y),
            image
        )

        # background = background.convert('RGBA')
        # background.save(destination)

        # background.show()

        background.close()

    print(coordinates)

    # with open('coordinates.pickle', 'wb') as handle:
    #     pickle.dump(coordinates, handle)


if __name__ == '__main__':
    main()
