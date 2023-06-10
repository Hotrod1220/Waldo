from __future__ import annotations

from pathlib import Path
from PIL import Image


def walk(file: Path) -> Path | None:
    for path in file.parents:
        if path.is_dir():
            venv = list(
                path.glob('venv')
            )

            for environment in venv:
                return environment.parent

            walk(path.parent)

    return None


file = Path.cwd()
CWD = walk(file).joinpath('waldo')

MODEL = CWD.joinpath('model')
PREPROCESS = CWD.joinpath('preprocess')

DATASET = PREPROCESS.joinpath('dataset')
WALLPAPER = PREPROCESS.joinpath('wallpaper')

DYNAMIC = PREPROCESS.joinpath('dynamic')
STATIC = PREPROCESS.joinpath('static')

WALDO = DATASET.joinpath('waldo')
NOT_WALDO = DATASET.joinpath('not_waldo')

STATE = MODEL.joinpath('state')

ORIGINAL = WALLPAPER.joinpath('original')
TRANSFORM = WALLPAPER.joinpath('transform')
UPSCALE = WALLPAPER.joinpath('upscale')

# WALDO.mkdir(parents=True, exist_ok=True)
# NOT_WALDO.mkdir(parents=True, exist_ok=True)

# DYNAMIC.mkdir(parents=True, exist_ok=True)
# STATIC.mkdir(parents=True, exist_ok=True)

# ORIGINAL.mkdir(parents=True, exist_ok=True)
# TRANSFORM.mkdir(parents=True, exist_ok=True)
# UPSCALE.mkdir(parents=True, exist_ok=True)

SUFFIX = [
    '.bmp',
    '.gif',
    '.jpg',
    '.jpeg',
    '.png',
    '.webp'
]

ARTWORK = [
    path.as_posix()
    for path in DATASET.glob('artwork/*')
    if path.suffix in SUFFIX and
    path.exists() and path.is_file()
]

CHARACTER = [
    Image.open(path).convert('RGBA')
    for path in PREPROCESS.joinpath('character').glob('*.png')
]
