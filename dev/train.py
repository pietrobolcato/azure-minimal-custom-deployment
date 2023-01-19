"""CLI interface to train the classifier on the MNIST dataset"""

import argparse
import mnist
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger


def parse_args():
    """Parse arguments from command line"""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        required=False,
        default=3,
        help="The number of epochs to train the model",
    )

    arguments = parser.parse_args()
    return arguments


def main(arguments):
    """Trains the classifier based on the parameters taken from the CLI"""

    model = mnist.Classifier()
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=arguments.epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        logger=CSVLogger(save_dir="logs/"),
    )

    trainer.fit(model)


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
