import pandas as pd
import pickle

from pathlib import Path, PurePath


def main() -> None:
    cwd = Path.cwd()

    # Create directories
    dataset = cwd.joinpath('dataset')

    waldo = dataset.joinpath('waldo')
    not_waldo = dataset.joinpath('not_waldo')

    waldo.mkdir(parents=True, exist_ok=True)
    not_waldo.mkdir(parents=True, exist_ok=True)

    suffix = [
        '.bmp',
        '.gif',
        '.jpg',
        '.jpeg',
        '.png',
        '.webp'
    ]

    label = 0
    x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0

    false = [
        {
            'filename': file.name,
            'path': PurePath(file).relative_to(cwd).as_posix().replace(
                'dataset',
                'dataset'
            ),
            'label': label,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        }
        for file in list(not_waldo.glob('*/*'))[:189]
        if file.is_file() and file.suffix.lower() in suffix
    ]

    label = 1

    true = [
        {
            'filename': file.name,
            'path': PurePath(file).relative_to(cwd).as_posix().replace(
                'dataset',
                'dataset'
            ),
            'label': label,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        }
        for file in waldo.glob('*')
        if file.is_file() and file.suffix.lower() in suffix
    ]

    true.extend(false)

    path = dataset.joinpath('coordinates.pkl')

    with open(path, 'rb') as handle:
        coordinates = pickle.load(handle)

    for row in true:
        file = row.get('filename')

        if file in coordinates:
            x1, y1, x2, y2 = coordinates[file]

            row['x1'] = float(x1)
            row['y1'] = float(y1)
            row['x2'] = float(x2)
            row['y2'] = float(y2)

    dataframe = pd.DataFrame(true)

    csv = dataset.joinpath('waldo.csv')
    dataframe.to_csv(csv, index=False)


if __name__ == '__main__':
    main()
