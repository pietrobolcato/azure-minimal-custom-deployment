"""CLI interface to perform inference on the classifier"""

import argparse
import os
import mnist
import torch
import numpy as np
from PIL import Image
from io import BytesIO


def parse_args():
    """Parse arguments from command line"""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="The path to an input image to classify",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="The path to the trained model",
    )

    arguments = parser.parse_args()
    return arguments


def inference(arguments):
    """Performs inference with the classifier based on the parameters
    taken from the CLI"""

    model = torch.jit.load(arguments.model)
    image = Image.open(arguments.input).convert("L")
    transform = mnist.Classifier.get_transform()

    img_tensor = transform(image)
    logits = model(img_tensor)
    pred = np.argmax(logits.detach().numpy(), axis=1)

    return pred


if __name__ == "__main__":
    arguments = parse_args()
    result = inference(arguments)

    print(f"Inference result: {result}")
