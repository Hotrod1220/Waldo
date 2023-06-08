from pathlib import Path
from PIL import Image


def main() -> None:
    cwd = Path.cwd()

    # Create directories
    dataset = cwd.joinpath('dataset')

    not_waldo = dataset.joinpath('not_waldo')
    not_waldo.mkdir(parents=True, exist_ok=True)

    suffix = [
        '.bmp',
        '.gif',
        '.jpg',
        '.jpeg',
        '.png',
        '.webp'
    ]

    files = [
        file
        for file in not_waldo.glob('*/*')
        if file.is_file() and file.suffix.lower() in suffix
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
