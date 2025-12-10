# QuaNNTO
**Qua**ntum **N**eural **N**etworks with **T**unable **O**ptics

_Article:_ [Hardware-inspired Continuous Variables Quantum Optical Neural Networks](https://arxiv.org/abs/2512.05204)

_QuaNNTO_ is an HPC library for exact simulation of continuous-variable quantum optical neural networks (QONN). The simulation technique is based on Wick-Isserlis expansions and Bogoliubov transformations, which allows an exact computation of expectation values of continuous observables (such as position) on quantum states built by a finite set of Gaussian and creation/annihilation operators, avoiding truncations in the infinite-dimensional Hilbert space that describe the quantum optical system.

The main QONN hyperparameters:

- ***modes*** is an integer denoting the number of optical modes in the QONN.
- ***is_addition*** is a boolean variable denoting which ladder operator (non-Gaussian resource), photon addition (`True`) or photon subtraction (`False`), is implemented in the QONN.
- ***layers*** is an integer denoting the amount of layers of the QONN.
- ***ladder_modes*** is a list of lists that defines the ladder operators' modes in each layer. E.g.: `[[0],[1]]` would mean a 2-layer QONN where the first layer has a ladder operator on mode $0$ and the second layer has another in mode $1$.
- ***n_inputs*** and ***n_outputs*** refer to the amount of inputs and outputs desired in the QONN model.
- ***observable*** is a string variable that allows to choose which operator's expectation value is to be measured, typically acting as the QONN output(s). In `expectation_value.py`, one can find the different observables that are available and, additionally, can implement different ones as long as the observable can be represented in Fock basis.
- ***in_norm_ranges*** and ***out_norm_ranges*** indicate the re-scaling ranges of the inputs (real coherent states) and outputs (position expectation values) respectively.

The different tasks train multiple QONN models with their hyperparameters defined at the beggining. They are divided in the following scripts:

- `multimode_curve_fitting.py` used for curve fitting of the target function indicated in the script (chosen from `synth_datasets.py`).

- `mnist_classification_example.py` is the MNIST classification task (discrete-variable inputs).

- `cv_classification.py` is the moons and circles classification task (continuous-variable inputs).

- `cubicphase_training.py` is the script for approximating the cubic phase gate's action of a specific strength over a set of states given their final state high-order statistical moments. In this example, the target states in which the action of the cubic gate (strength=$0.2$) is applied are real coherent states in the range $(-2,2)$.

The different tasks can be run directly from the `Quannto` directory with the command `python3 -m quannto.<file>` without the `.py` at the end. E.g.: `python3 -m quannto.multimode_curve_fitting` and the training will start for all the QONN setups defined in the hyperparameters of such a script.

In `qnn_trainers.py`, one could change parameter bounds such as the one for the squeezing, fixed by default to state-of-the-art limit $r \in (-1.7, 1.7)$.

<ins>**NOTE:**</ins> This library is still under development. 