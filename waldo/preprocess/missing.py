import shutil

from pathlib import Path


def main() -> None:
    folder = Path.cwd().joinpath('missing')
    folder.mkdir(exist_ok=True, parents=True)

    missing = set()

    waldo = Path.cwd().joinpath('dataset/waldo')

    files = list(
        waldo.glob('*')
    )

    for file in files:
        name = file.stem

        name = name.replace('waldo_', '')
        name = int(name)

        if name in missing:
            path = folder.joinpath(file.name)
            shutil.copy(file, path)


if __name__ == '__main__':
    main()
