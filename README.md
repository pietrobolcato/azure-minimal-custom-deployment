# Introduction

This repo contains an example with code and instructions for the deployment of a minimal custom ML model on the Azure ML managed online endpoint, using MLFlow. This focuses on deployment only — training on the cloud is not covered here.

The `dev` folder contains all the code to create and train the most simple neural network to perform classification on a MNIST dataset. This very minimal, barebone example, shows the basics to easily extend the structure to much more complex models and projects.

# Train and export the model for deployment

Follow the instructions inside the `dev` folder

# Deployment

## Local deploy

Before deploying the model to the cloud, it's best practice to test it locally. Given that we're deploying a MLFlow model, testing is easier. Run the prediction on your model using MLFlow. If everything works as expected, so it will when deployed on the cloud.

`mlflow models predict --model-uri cloud/mnist-model-1-mlflow --input-path "cloud/sample-request.json" --content-type json`

## Cloud deploy

Now you're ready to deploy to the cloud. First, we have to register the model:

`az ml model create --path cloud/mnist-model-1-mlflow --name mnist-model-1-mlflow --version 1 --type mlflow_model`

Then, the following commands create and endpoint and its respective deployment:

`az ml online-endpoint create -f cloud/deployment/endpoint.yaml`

`az ml online-deployment create -f cloud/deployment/deployment.yaml --all-traffic`

Once these commands are succesfully executed, we can test our cloud deployment:

`az ml online-endpoint invoke --name mnist-endpoint-271934-mlflow --request-file cloud/sample-request.json`

# More

If you have any questions or doubts, feel free to open an issue. I will be happy to help!
