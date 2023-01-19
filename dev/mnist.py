"""This module defines a minimal neural network to train an image classifier
on the MNIST dataset"""

import torch
from pytorch_lightning import LightningModule
from torchvision import transforms
from torch import nn
from torchmetrics import Accuracy
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split


class Classifier(LightningModule):
    """Defines the neural network and it's main methods"""

    def __init__(self, data_dir=".", hidden_size=64, learning_rate=2e-4):
        super().__init__()

        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = 256 if torch.cuda.is_available() else 64

        # Dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = self.get_transform()

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    @staticmethod
    def get_transform():
        """Defines pre-processing input transformations"""

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        return transform

    def forward(self, x):
        """Defines forward pass"""

        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        """Defines the training step for the neural network to learn"""

        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        """Defines the validation step"""

        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # log to tensorboard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Defines the test step"""

        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # log to tensorboard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        """Defines optimizer"""

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def prepare_data(self):
        """Download MNIST dataset"""

        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Assign train/val/test datasets for use in dataloaders"""

        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            MNIST(self.data_dir, train=False, transform=self.transform)

        return self.mnist_train, self.mnist_val

    def train_dataloader(self):
        """Returns the train dataloader"""

        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        """Returns the validation dataloader"""

        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        """Returns the test dataloader"""

        return DataLoader(self.mnist_test, batch_size=self.batch_size)
