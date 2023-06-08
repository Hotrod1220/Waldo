from __future__ import annotations

import pandas as pd
import pickle
import torch

from dataset import WaldoDataset
from model import Model
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from trainer import Trainer


def main() -> None:
    current = Path.cwd()

    csv = current.joinpath('preprocess/dataset/waldo.csv')

    annotation = pd.read_csv(csv)

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])

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

    batch_size = 16

    training = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True
    )

    testing = DataLoader(
        dataset=test,
        batch_size=batch_size,
        shuffle=True
    )

    validating = DataLoader(
        dataset=validation,
        batch_size=batch_size,
        shuffle=True
    )

    model = Model()
    model.device = device

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    torch.backends.cudnn.benchmark = True

    trainer = Trainer()
    trainer.device = device
    trainer.epoch = 60
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.testing = testing
    trainer.training = training
    trainer.validating = validating
    trainer.start()

    torch.save(
        model.state_dict(),
        'model/model.pth'
    )

    with open('model/trainer.pkl', 'wb') as handle:
        pickle.dump(trainer, handle)


if __name__ == '__main__':
    main()
