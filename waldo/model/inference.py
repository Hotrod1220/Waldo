from __future__ import annotations

import pandas as pd
import pickle
import torch

from model import Model
from prediction import Predictor
from transformation import Transformation
from waldo.constant import (
    DATASET,
    DYNAMIC,
    STATIC,
    WALDO
)


def main() -> None:
    csv = DATASET.joinpath('waldo.csv')
    annotation = pd.read_csv(csv)

    # Load the model
    model = Model()
    model.device = 'cpu'

    state = torch.load('state/model.pth')
    model.load_state_dict(state)

    model.eval()

    # with open('state/trainer.pkl', 'rb') as handle:
    #     trainer = pickle.load(handle)

    # loader = trainer.testing

    mapping = {
        0: 'Not Waldo',
        1: 'Waldo'
    }

    transformation = Transformation(device='cpu')

    predictor = Predictor()
    predictor.annotation = annotation
    # predictor.loader = loader
    predictor.mapping = mapping
    predictor.model = model
    predictor.transformation = transformation

    # # From loader
    # predictor.from_loader(loader)
    # predictor.plot(show=True, save=False)


    # From image
    images = [
        file
        for file in WALDO.glob('*.png')
        if file.is_file()
    ]

    for image in images:
        predictor.from_path(image)
        predictor.plot(show=True, save=False)


    # # From random
    # images = DYNAMIC.glob('*.png')

    # # images = STATIC.glob('*/*.png')

    # for image in list(images)[5000:]:
    #     predictor.from_random(image)
    #     predictor.plot(show=True, save=False)


if __name__ == '__main__':
    main()
