"""
CLI interface to export the trained model in the MLFlow format so that it
can be uploaded directly to Azure ML
"""

import shutil
import argparse
import mnist
import tempfile
import common
import mlflow
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema, TensorSpec, DataType
from model_wrapper import ModelWrapper


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="__doc__")

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="The path to the input .ckpt to convert",
    )

    parser.add_argument(
        "-o", "--output", type=str, help="The output directory where the model will be"
    )

    return parser.parse_args()


def export_base_model(model, out_path):
    """Export the base model without custom inference code"""

    code_paths = ["mnist.py"]

    input_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1, 28, 28, 1))])
    output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10))])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    shutil.rmtree(out_path, ignore_errors=True)
    mlflow.pytorch.save_model(
        pytorch_model=model,
        path=out_path,
        code_paths=code_paths,
        signature=signature,
    )


def export_pyfunc_model(model, base_model_path, out_path):
    """Exports model with custom inference code"""

    pyfunc_code_filenames = ["model_wrapper.py", "mnist.py"]

    pyfunc_input_schema = Schema([ColSpec(type="string", name="image")])
    pyfunc_output_schema = Schema([TensorSpec(np.dtype(np.int32), (-1, 1))])
    pyfunc_signature = ModelSignature(
        inputs=pyfunc_input_schema, outputs=pyfunc_output_schema
    )

    model = ModelWrapper()
    artifacts = {
        common.ARTIFACT_NAME: base_model_path,
    }

    shutil.rmtree(out_path, ignore_errors=True)
    mlflow.pyfunc.save_model(
        path=out_path,
        python_model=model,
        artifacts=artifacts,
        code_path=pyfunc_code_filenames,
        signature=pyfunc_signature,
    )


def main(arguments):
    """Export and test the model based on the command line arguments"""

    # export
    model = mnist.Classifier().load_from_checkpoint(arguments.input)

    with tempfile.TemporaryDirectory() as temp_dir:
        export_base_model(model, temp_dir)
        export_pyfunc_model(model, temp_dir, arguments.output)

    # test with sample image
    image_b64 = "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAA10lEQVR4nGNgGFhgy6xVdrCszBaLFN/mr28+/QOCr69DMCSnA8WvHti0acu/fx/10OS0X/975CDDw8DA1PDn/1pBVEmLf3+zocy2X/+8USXt/82Ds+/+m4sqeehfOpw97d9VFDmlO++t4JwQNMm6f6sZcEpee2+DR/I4A05J7tt4JJP+IUsu+ncRp6TxO9RAQJY0XvrvMAuypNNHuCTz8n+PzVEcy3DtqgiY1ptx6t8/ewY0yX9ntoDA63//Xs3hQpMMPPsPAv68qmDAAFKXwHIzMzCl6AoAxXp0QujtP+8AAAAASUVORK5CYII="

    loaded_model = mlflow.pyfunc.load_model(arguments.output)
    res = loaded_model.predict([{"image": image_b64}])

    assert res == [0], f"Prediction was wrong. Got {res}, but expected [0]"

    print(f"Predicted value: {res}. All done")


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
