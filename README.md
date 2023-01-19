# Introduction

This repo contains an example with code and instructions for the deployment of a minimal custom ML model on the Azure ML managed online endpoint, without using MLFlow. This focuses on deployment only — training on the cloud is not covered here.

The `dev` folder contains all the code to create and train the most simple neural network to perform classification on a MNIST dataset. This very minimal, barebone example, shows the basics to easily extend the structure to much more complex models and projects.

# Train and export the model for deployment

Follow the instructions inside the `dev` folder

# Deployment

## Local deploy

Before deploying the model to the cloud, it's best practice to test it locally. To do that, we can use Azure ML’s local endpoint feature to test the actual endpoint and deployment on your local machine, using a Docker container.

Run the following commands to create the endpoint and the deployment for it (note the `--local` flag):

`az ml online-endpoint create -f cloud/deployment/endpoint.yaml --local`

`az ml online-deployment create -f cloud/deployment/deployment-local.yaml --local`

After the successful execution of the commands, you can test your local endpoint:

`az ml online-endpoint invoke --name mnist-endpoint-865021 --request-file cloud/mnist-model-1/sample-request.json --local`

You should get the response: `[0]`, which shows that the classifier correctly classified the image input as the number 0.

If something went wrong, you can easily debug what is going on using Azure's debug dev container. The commands remains the unaltered, and have only the `--vscode-debug` flag added to them:

`az ml online-endpoint delete -n mnist-endpoint-865021 -y --local`

`az ml online-endpoint create -f cloud/deployment/endpoint.yaml --local`

`az ml online-deployment create -f cloud/deployment/deployment-local.yaml --local --vscode-debug`

## Cloud deploy

Now you're ready to deploy to the cloud. First, we have to register the model:

`az ml model create --path cloud/mnist-model-1/model --name mnist-model --version 1`

Then, the following commands remain basically unaltered with respect to local development. We simply remove the `--local` flag:

`az ml online-endpoint create -f cloud/deployment/endpoint.yaml`

`az ml online-deployment create -f cloud/deployment/deployment.yaml --all-traffic`

Once these commands are succesfully executed, we can test our cloud deployment:

`az ml online-endpoint invoke --name mnist-endpoint-865021 --request-file cloud/mnist-model-1/sample-request.json`

# More

If you have any questions or doubts, feel free to open an issue. I will be happy to help!
