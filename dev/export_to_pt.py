"""CLI interface to export .ckpt model to torchscript"""

import argparse
import mnist


def parse_args():
    """Parse arguments from command line"""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="The path to the input .ckpt to convert",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The path to output .pt file",
    )

    arguments = parser.parse_args()
    return arguments


def main(arguments):
    """Converts input .ckpt to torchscript"""

    model = mnist.Classifier().load_from_checkpoint(arguments.input)
    model.eval()

    compiled_model = model.to_torchscript()
    compiled_model.save(arguments.output)


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
