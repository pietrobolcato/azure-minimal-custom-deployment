"""Wrapper for mlflow model."""

import logging
import base64
import mnist
import numpy as np

import mlflow
import pandas as pd
import torch

from PIL import Image
from io import BytesIO


class ModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper for mlflow model.
    """

    def load_context(self, context):
        self.model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"])

    def predict(self, context, model_input):
        image_data = model_input.iloc[0][0].encode()
        image = Image.open(BytesIO(base64.decodebytes(image_data))).convert("L")

        transform = mnist.Classifier.get_transform()
        img_tensor = transform(image)

        logits = self.model(img_tensor)
        pred = np.argmax(logits.detach().numpy(), axis=1)

        return pred.tolist()
