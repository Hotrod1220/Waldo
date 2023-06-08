from __future__ import annotations

import pandas as pd
import pickle
import torch

from model import Model
from pathlib import Path
from prediction import Predictor


def main() -> None:
    current = Path.cwd()

    prediction = current.joinpath('prediction')
    prediction.mkdir(exist_ok=True, parents=True)

    csv = current.joinpath('preprocess/dataset/waldo.csv')
    annotation = pd.read_csv(csv)

    # Load the model
    model = Model()
    model.device = 'cpu'

    state = torch.load('model/model.pth')
    model.load_state_dict(state)

    model.eval()

    with open('model/trainer.pkl', 'rb') as handle:
        trainer = pickle.load(handle)

    loader = trainer.testing

    mapping = {
        0: 'Not Waldo',
        1: 'Waldo'
    }

    predictor = Predictor()
    predictor.annotation = annotation
    predictor.loader = loader
    predictor.mapping = mapping
    predictor.model = model

    # From loader
    predictor.from_loader(loader)
    predictor.plot(show=True, save=False)


    # # From image
    # waldo = current.joinpath('preprocess/dataset/waldo')
    # not_waldo = current.joinpath('preprocess/dataset/not_waldo')

    # file = waldo.joinpath('0_16_waldo.png')
    # file = not_waldo.joinpath('abstract/abstract_0112.png')

    # predictor.from_path(file)
    # predictor.plot(show=True, save=False)


    # # From random
    # images = (
    #     current
    #     .parent
    #     .joinpath('detection/dynamic')
    #     .glob('*.png')
    # )

    # images = (
    #     current
    #     .parent
    #     .joinpath('detection/static')
    #     .glob('*/*.png')
    # )

    # for image in images:
    #     predictor.from_random(image)
    #     predictor.plot(show=False, save=True)


if __name__ == '__main__':
    main()
