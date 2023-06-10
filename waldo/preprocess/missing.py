from __future__ import annotations

import shutil

from pathlib import Path
from waldo.constant import WALDO


def main() -> None:
    folder = Path.cwd().joinpath('missing')
    folder.mkdir(exist_ok=True, parents=True)

    missing = set()

    glob = WALDO.glob('*')
    files = list(glob)

    for file in files:
        name = file.stem

        name = name.replace('waldo_', '')
        name = int(name)

        if name in missing:
            path = folder.joinpath(file.name)
            shutil.copy(file, path)


if __name__ == '__main__':
    main()
