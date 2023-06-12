from __future__ import annotations

import matplotlib.pyplot as plt
import pickle

from waldo.constant import MODEL


def main() -> None:
    path = MODEL.joinpath('state/history.pkl')

    with open(path, 'rb') as handle:
        history = pickle.load(handle)

    training = history.get('training')
    validation = history.get('validation')

    # Classification
    tc_loss = training.get('classification_loss')
    vc_loss = validation.get('classification_loss')

    figsize = (10, 5)
    plt.figure(figsize=figsize)

    plt.plot(
        tc_loss,
        label='Training Loss'
    )

    plt.plot(
        vc_loss,
        label='Validation Loss'
    )

    plt.title('Classification: Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(
        'classification_loss.png',
        bbox_inches='tight',
        dpi=300,
        format='png'
    )

    plt.show()

    # Bounding box
    tb_loss = training.get('bounding_box_loss')
    vb_loss = validation.get('bounding_box_loss')

    figsize = (10, 5)
    plt.figure(figsize=figsize)

    plt.plot(
        tb_loss,
        label='Training Loss'
    )

    plt.plot(
        vb_loss,
        label='Validation Loss'
    )

    plt.title('Bounding Box: Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(
        'bounding_box_loss.png',
        bbox_inches='tight',
        dpi=300,
        format='png'
    )

    plt.show()


if __name__ == '__main__':
    main()
