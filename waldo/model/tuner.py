from __future__ import annotations

import pandas as pd
import pickle
import torch

from dataset import WaldoDataset
from model import Model
from pathlib import Path
from torch.utils.data import DataLoader
from trainer import Trainer
from transformation import Transformation
from waldo.constant import DATASET
from torch.optim import AdamW, RMSprop


def main():
    current = Path.cwd()

    csv = DATASET.joinpath('waldo.csv')

    annotation = pd.read_csv(csv)

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    transformation = Transformation(device=device)

    dataset = WaldoDataset()
    dataset.annotation = annotation
    dataset.current = current
    dataset.device = device
    dataset.transformation = transformation

    length = len(dataset)

    trl = int(length * 0.80)
    tel = int(length * 0.10)
    val = length - (trl + tel)

    train, test, validation = torch.utils.data.random_split(
        dataset,
        [trl, tel, val]
    )

    param_grid = {
        'batch_size': [16, 32],
        'lr': [0.01, 0.001, 0.0001],
        'epochs': [10, 15],
        'optimizer': [AdamW, RMSprop]
    }

    best_score = 0.0
    best_params = {}

    for batch_size in param_grid['batch_size']:
        for lr in param_grid['lr']:
            for epochs in param_grid['epochs']:
                for optimizer in param_grid['optimizer']:

                    training = DataLoader(
                        dataset=train,
                        batch_size=batch_size,
                        shuffle=True
                    )

                    testing = DataLoader(
                        dataset=test,
                        batch_size=batch_size,
                        shuffle=False
                    )

                    validating = DataLoader(
                        dataset=validation,
                        batch_size=batch_size,
                        shuffle=False
                    )

                    model = Model()
                    model.device = device

                    torch.backends.cudnn.benchmark = True

                    optimizer = optimizer(model.parameters(), lr=lr)

                    trainer = Trainer()
                    trainer.device = device
                    trainer.epoch = epochs
                    trainer.model = model
                    trainer.optimizer = optimizer
                    trainer.testing = testing
                    trainer.training = training
                    trainer.validating = validating
                    trainer.start()

                    score = trainer.evaluate()

                    if score > best_score:
                        best_score = score
                        best_params = {
                            'batch_size': batch_size,
                            'lr': lr,
                            'epochs': epochs,
                            'optimizer': optimizer
                        }

    print(f'Best score: {best_score}')
    print(f'Best params: {best_params}')


if __name__ == '__main__':
    main()
