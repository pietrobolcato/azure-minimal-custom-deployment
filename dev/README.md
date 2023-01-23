# Train a minimal NN classifier on the MNIST dataset

This part of the repo contains the code to train, test and export the classifier that will be deployed on the Azure online ML endpoint.

### How to run

1- Create conda environment from file `conda env create -f conda.yaml`

2- Start training `python train.py`

3- Convert generated .ckpt to mlflow model: `python mlfow_export.py -i $GENERATED_CKPT_PATH$ -o $OUT_PT_PATH$`

4- Test inference `python inference.py -m $OUT_PT_PATH$ -i $TEST_IMAGE_PATH$`

5- Done — follow the steps in the main `readme.md` file to deploy the model to Azure
