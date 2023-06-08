from pathlib import Path
from PIL import Image
from random import SystemRandom


def main() -> None:
    cwd = Path.cwd()

    # Create directories
    character = cwd.joinpath('character')
    dataset = cwd.joinpath('dataset')
    wallpaper = cwd.joinpath('wallpaper')

    static = dataset.joinpath('static')
    original = wallpaper.joinpath('original')
    transform = wallpaper.joinpath('transform')
    upscale = wallpaper.joinpath('upscale')

    static.mkdir(parents=True, exist_ok=True)
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

    for i, path in enumerate(source, 0):
        i = str(i)
        i = i.zfill(2)

        section = mapping.get(i).get('section')
        size = mapping.get(i).get('size')

        temporary = waldo.copy()
        temporary.thumbnail(size)

        scene = static.joinpath(i)
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

                crop.paste(
                    temporary,
                    (x, y),
                    temporary
                )

                # crop.show()

                crop = crop.convert('RGB')
                crop.save(destination)



if __name__ == '__main__':
    main()
