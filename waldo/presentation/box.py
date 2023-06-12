from __future__ import annotations

import numpy as np

from itertools import combinations
from pathlib import Path
from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageOps
)
from secrets import SystemRandom
from waldo.constant import CHARACTER


def get_transparent_background() -> Image:
    width, height = 2048, 2048

    return Image.new(
        'RGBA',
        (width, height),
        (255, 0, 0, 0)
    )


def main() -> None:
    # Define a list of transformation(s) to select from
    callback = {
        'contrast': ImageEnhance.Contrast,
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
    iterate = 400

    print(f"There are {amount * iterate} transformations")

    angles = [0, 45, 90, 135, 180, 225, 315, 360]

    factors = np.arange(0.2, 3, 0.05)

    coordinates = {}

    generator = SystemRandom()

    draw = True

    for i in range(iterate):
        # Transform Waldo using each possible combination
        for j, transformation in enumerate(transformations, 0):
            # Select a random image of Waldo
            character = generator.choice(CHARACTER)

            filename = f"{i}_{j}.png"
            destination = Path('images').joinpath(filename)

            print(f"Processing: {filename}")

            original = get_transparent_background()

            # Copy the background and Waldo
            background = original.copy()
            waldo = character.copy()

            # Call each selected transformation on the image
            for function in transformation:
                if function in parameterize:
                    angle = generator.choice(angles)

                    waldo = callback[function](waldo, angle)
                elif function in additional:
                    factor = generator.choice(factors)

                    enhancement = callback[function](waldo)
                    waldo = enhancement.enhance(factor)
                else:
                    waldo = callback[function](waldo)

            # Resize Waldo to be at least 28x28px
            width, height = background.size
            x = generator.randrange(28, width)
            y = generator.randrange(28, height)

            waldo.thumbnail(
                (x, y)
            )

            # Move Waldo to a random location
            maximum_x = width - x
            maximum_y = height - y

            x = generator.randrange(0, maximum_x)
            y = generator.randrange(0, maximum_y)

            background.paste(
                waldo,
                (x, y),
                waldo
            )

            box = (x, y, x + waldo.width, y + waldo.height)
            box = tuple(float(coordinate) for coordinate in box)

            coordinates[filename] = box

            # Draw the bounding box
            if draw:
                draw = ImageDraw.Draw(background)
                draw.rectangle(box, outline='green', width=12)

            background.save(destination)

            waldo.close()
            background.close()


if __name__ == '__main__':
    main()
