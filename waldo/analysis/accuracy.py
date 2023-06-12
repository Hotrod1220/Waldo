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
    tc_loss = training.get('classification_accuracy')
    vc_loss = validation.get('classification_accuracy')

    figsize = (10, 5)
    plt.figure(figsize=figsize)

    plt.plot(
        tc_loss,
        label='Training Accuracy'
    )

    plt.plot(
        vc_loss,
        label='Validation Accuracy'
    )

    plt.title('Classification: Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(
        'classification_accuracy.png',
        bbox_inches='tight',
        dpi=300,
        format='png'
    )

    plt.show()

    # Bounding box
    tb_loss = training.get('bounding_box_score')
    vb_loss = validation.get('bounding_box_score')

    figsize = (10, 5)
    plt.figure(figsize=figsize)

    plt.plot(
        tb_loss,
        label='Training GIoU Score'
    )

    plt.plot(
        vb_loss,
        label='Validation GIoU Score'
    )

    plt.title('Bounding Box: GIoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.savefig(
        'bounding_box_accuracy.png',
        bbox_inches='tight',
        dpi=300,
        format='png'
    )

    plt.show()


if __name__ == '__main__':
    main()
