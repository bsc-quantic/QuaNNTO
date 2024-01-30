# QUANNTO USER GUIDE

This is a how-to-use guide for the continuous-variables quantum neural network simulator.

Before diving into the project usage, let's define some important observations or constraints that arise from the code implementation of this simulator version.

## OBSERVATIONS AND CONSTRAINTS

- QNN outputs are limited to one which is set as the sum of expectation values of the defined observables on each mode.

- Each mode observable must be a second-order operator, this is, an operator containing one or more combinations of two canonical or two ladder operators product. (TO-DO: Check explanation)

- The observables have to be defined in terms of creation and annihilation operators. 

- Non-linear part of each QNN layer is made of a single photon addition on mode 1.

- When there are more modes than inputs, the extra modes are not squeezed (i.e. squeezing factor equal to 1) while the input reuploading.

- Datasets should be normalized to ranges where the input encoding and the outcomes of the observables are defined. As the inputs are encoded as the factors of the squeezing operator matrix, these factors must always be positive. Moreover, if using the number operator on mode i $\hat{N}_i$ as the observable, the output range should always be positive because its possible outcomes are always equal or greater than 0.

## USER GUIDE

### Synthetic dataset training

In order to build and train a QNN to learn some continuous function, some parameters have to be defined.

First, the **quantum neural network hyperparameters**:

- **Number of modes $N$**. The number of modes of the quantum system serve as neurons of the NN, therefore, the number of modes must be always greater than the number of NN inputs.

- **Number of layers $L$**. The desired number of layers of the target QNN where, each layer, contains a Gaussian operator mimicking the linear transformation of the layer inputs and a non-Gaussian transformation (one photon addition on first mode) analogous to the non-linear activation function of the layer.

- **Observable**. For the output calculation, observables of the different modes have to be defined in ladder operators representation.
    - **Observable modes**. The modes that the observable's operators act onto. The range is $\{0,...,N-1\}$ where index $0$ is the first mode of the quantum system.
    - **Observable types**. The type of ladder operator, this is, $0$ for an annihilation operator and $1$ for a creation operator.


Now, let's define the (**synthetic**) dataset parameters to be specified:

- **Target function**. It stands for the name of the function which will be sampled to build the synthetic dataset.

- **Number of inputs**. The quantity of inputs to be generated for each dataset sample.

- **Dataset size**. The number of samples of the target function to be learned.

- **Input range**. Definition of the inputs' domain, this is, the minimum and maximum values the inputs can take.

- **Output range**. Definition of the outputs' domain. This part is automated for synthetic datasets by extracting approximate the minimum and maximum values of the target function.

- **Inputs normalization range**. Desired range for the dataset inputs while being fed to the QNN.

- **Outputs normalization range**. Desired range for the dataset outputs to be yielded by the QNN.

- **Model name**. A name for the model with which it will be saved to a file. It is typically set as the target function name for synthetic datasets plus the number of modes and layers.

- **Testing set size**. Number of samples to generate for the testing set of the QNN.

After defining all these parameters, one would execute the next command inside the project directory and the training will start showing the loss value per epoch and the final results at the end of the training.

```
python3 -u quannto/synthetic_build_and_train.py
```

