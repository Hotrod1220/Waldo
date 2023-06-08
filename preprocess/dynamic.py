from pathlib import Path
from PIL import Image
from random import SystemRandom


def main() -> None:
    cwd = Path.cwd()

    # Create directories
    character = cwd.joinpath('character')
    dataset = cwd.joinpath('dataset')
    wallpaper = cwd.joinpath('wallpaper')

    dynamic = dataset.joinpath('dynamic')
    original = wallpaper.joinpath('original')
    transform = wallpaper.joinpath('transform')
    upscale = wallpaper.joinpath('upscale')

    dynamic.mkdir(parents=True, exist_ok=True)
    original.mkdir(parents=True, exist_ok=True)
    transform.mkdir(parents=True, exist_ok=True)
    upscale.mkdir(parents=True, exist_ok=True)

    # Resize image
    suffix = [
        '.bmp',
        '.gif',
        '.jpg',
        '.jpeg',
        '.png',
        '.webp'
    ]

    source = [
        file
        for file in transform.glob('*')
        if file.is_file() and file.suffix.lower() in suffix
    ]

    waldo = character.joinpath('waldo.png')
    waldo = Image.open(waldo)
    waldo = waldo.convert('RGBA')

    waldo.thumbnail(
        (64, 64)
    )

    generator = SystemRandom()

    w = 224
    h = 224

    for path in source:
        with Image.open(path) as image:
            image.thumbnail(
                (1920, 1080)
            )

            background = image.convert('RGBA')
            background = background.copy()

            for i in range(1000):
                filename = f"{path.stem}_{i}{path.suffix}"
                destination = dynamic.joinpath(filename)

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

                crop.paste(
                    waldo,
                    (x, y),
                    waldo
                )

                # crop.show()

                crop = crop.convert('RGB')
                crop.save(destination)



if __name__ == '__main__':
    main()
