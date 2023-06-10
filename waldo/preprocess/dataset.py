from __future__ import annotations

import pandas as pd
import pickle

from pathlib import PurePath
from waldo.constant import (
    CWD,
    DATASET,
    NOT_WALDO,
    SUFFIX,
    WALDO
)


def main() -> None:
    label, x1, y1, x2, y2 = 0, 0.0, 0.0, 0.0, 0.0
    limit = 1000

    glob = NOT_WALDO.glob('*/*')

    false = [
        {
            'filename': file.name,
            'path': PurePath(file).relative_to(CWD).as_posix().replace(
                'dataset',
                'dataset'
            ),
            'label': label,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        }
        for file in list(glob)[:limit]
        if file.is_file() and file.suffix.lower() in SUFFIX
    ]

    label = 1

    glob = WALDO.glob('*')

    true = [
        {
            'filename': file.name,
            'path': PurePath(file).relative_to(CWD).as_posix().replace(
                'dataset',
                'dataset'
            ),
            'label': label,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        }
        for file in list(glob)[:limit]
        if file.is_file() and file.suffix.lower() in SUFFIX
    ]

    true.extend(false)

    path = DATASET.joinpath('coordinates.pkl')

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

    csv = DATASET.joinpath('waldo.csv')
    dataframe.to_csv(csv, index=False)


if __name__ == '__main__':
    main()
