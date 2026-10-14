# QONN examples

Run the cells in order. Each supervised example loads its numeric parameter
checkpoint if present; otherwise `load_or_train` trains with `train_scipy` and
saves the model and loss histories. Predictions and figures are computed in
the notebook. The saved outputs can also be viewed directly on GitHub.

Install from the repository root:

```bash
python -m pip install -e '.[sf]' 'setuptools<81' ipykernel pandas
python -m ipykernel install --user --name quannto --display-name QuaNNTO
```

Select the QuaNNTO kernel in Jupyter. Mode indices in the configurations start
at zero. Change the configuration and its model name together when trying a
new architecture.

- `binary_classification_QONN.ipynb`: moons and circles classification.
- `curve_fitting_QONN.ipynb`: Polynomial, cosh and oscillatory fits with nine saved models.
- `mnist_classification_QONN.ipynb`: Four six-mode classifiers, per-class scores and model averages.
- `nongauss_gate_synthesis_QONN.ipynb`: cubic-phase output moments.
- `catstate_engineering_QONN.ipynb`: cat-state fidelity and Wigner functions.
- `trainability_QONN.ipynb`: gradient variances from initialization checkpoints.
