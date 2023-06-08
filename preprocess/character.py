import numpy as np
import pickle

from itertools import combinations
from pathlib import Path
from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageOps
)


def get_transparent_background() -> Image:
    width, height = 224, 224

    return Image.new(
        'RGBA',
        (width, height),
        (255, 0, 0, 0)
    )


def get_random_background(generator: np.random.RandomState) -> Image:
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

    waldo = dataset.joinpath('waldo')
    not_waldo = dataset.joinpath('not_waldo')

    # localization = waldo.joinpath('localization')

    waldo.mkdir(parents=True, exist_ok=True)
    not_waldo.mkdir(parents=True, exist_ok=True)

    # localization.mkdir(parents=True, exist_ok=True)

    # Open the original Waldo image
    path = cwd.joinpath('character').joinpath('waldo.png')
    character = Image.open(path)
    character = character.convert('RGBA')

    # generator = SystemRandom()
    generator = np.random.RandomState()

    callback = {
        'flip': ImageOps.flip,
        'rotate': lambda x, y: x.rotate(y, expand=True),
        'mirror': ImageOps.mirror,
        'brightness': ImageEnhance.Brightness,
        'contrast': ImageEnhance.Contrast,
        # 'grayscale': ImageOps.grayscale,
        'blur': lambda x: x.filter(ImageFilter.BLUR),
        # 'detail': lambda x: x.filter(ImageFilter.DETAIL),
        # 'edge_enhance': lambda x: x.filter(ImageFilter.EDGE_ENHANCE),
        # 'emboss': lambda x: x.filter(ImageFilter.EMBOSS),
        # 'find': lambda x: x.filter(ImageFilter.FIND_EDGES),
        # 'smooth': lambda x: x.filter(ImageFilter.SMOOTH),
        # 'sharpen': lambda x: x.filter(ImageFilter.SHARPEN)
    }

    additional = ['brightness', 'contrast']
    parameterize = ['rotate']

    k = list(callback)

    transformations = []

    for f in range(1, len(k) + 1):
        transformations.extend(
            combinations(k, f)
        )

    amount = len(transformations)
    iterate = 3

    print(f"There are {amount * 3} transformations")

    angles = [0, 45, 90, 135, 180, 225, 315, 360]

    choices = [True, False]
    factors = np.arange(0.2, 3, 0.05)

    coordinates = {}

    for i in range(iterate):
        # Transform Waldo using each possible combination
        for j, transformation in enumerate(transformations):
            filename = f"{i}_{j}_{path.name}"
            destination = waldo.joinpath(filename)

            print(f"Processing: {filename}")

            choice = generator.choice(choices)

            if choice:
                background = get_random_background(generator)
            else:
                background = get_transparent_background()

            # Copy the background and Waldo
            transparent = background.copy()
            transform = character.copy()
            temporary = get_transparent_background().copy()

            # Call each function on the image
            for function in transformation:
                if function in parameterize:
                    angle = generator.choice(angles)

                    transform = callback[function](transform, angle)
                elif function in additional:
                    factor = generator.choice(factors)

                    enhancement = callback[function](transform)
                    transform = enhancement.enhance(factor)
                else:
                    transform = callback[function](transform)

            # Resize Waldo to be at least 28x28px
            width, height = temporary.size
            x = generator.randint(28, width)
            y = generator.randint(28, height)

            transform.thumbnail(
                (x, y)
            )

            # Move Waldo to a random location
            maximum_x = width - x
            maximum_y = height - y

            x = generator.randint(0, maximum_x)
            y = generator.randint(0, maximum_y)

            temporary.paste(
                transform,
                (x, y),
                transform
            )

            box = temporary.getbbox()
            box = tuple(float(coordinate) for coordinate in box)

            coordinates[filename] = box

            transparent.paste(
                temporary,
                (0, 0),
                temporary
            )

            transparent.save(destination)

            # # Draw the bounding box
            # destination = localization.joinpath(filename)
            # draw = ImageDraw.Draw(transparent)
            # draw.rectangle(box, outline='green', width=3)
            # transparent.save(destination)

            # transparent.show()
            transform.close()
            temporary.close()
            transparent.close()

    path = dataset.joinpath('coordinates.pkl')

    with open(path, 'wb') as handle:
        pickle.dump(coordinates, handle)


if __name__ == '__main__':
    main()
