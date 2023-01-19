# `mnist-model-1`

This folder contains all the code and model needed for the inference, as well as a sample request to try it. The scoring file is located in the subfolder `onlinescoring`, and contains two methods that are consumed by the Azure infrastructure: `init` and `run`. The first, defines the code used to load the model, and the second performs inference.
