"""This file implements the scoring methods for the online ML endpoint"""

import base64
import logging
import json
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from io import BytesIO


def init():
    """
    This method is called on initialization of the endpoint.
    It loads the model and the preprocessing function used in the inference method
    """

    global model, transform

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model/mnist_pt.pt")
    model = torch.jit.load(model_path)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    logging.info("Init complete")


def run(raw_data):
    """
    It accepts a json as input, where the image is encoded in base64.
    Performs inference and return the result
    """

    image_data = json.loads(raw_data)["image"].encode()
    image = Image.open(BytesIO(base64.decodebytes(image_data))).convert("L")

    img_tensor = transform(image)
    logits = model(img_tensor)
    pred = np.argmax(logits.detach().numpy(), axis=1)

    return pred
