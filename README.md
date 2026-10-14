# QuaNNTO

**Qua**ntum **N**eural **N**etworks with **T**unable **O**ptics.

**Related paper:** [Hardware-inspired Continuous Variables Quantum Optical Neural Networks](https://arxiv.org/abs/2512.05204v2)  
Todor Krasimirov-Ivanov, Alba Cervera-Lierta, Paolo Stornati and Federico Centrone.  
arXiv:2512.05204v2 [quant-ph].

QuaNNTO evaluates continuous-variable QONNs built from Gaussian transformations
and finite photon addition/subtraction sequences. Bogoliubov transformations and
Wick–Isserlis expansions provide analytic ladder-operator moments without a Fock
cutoff. JAX differentiates the moment evaluator. The optional independent Fock
backend validates conditional output states and fidelities numerically.

Examples cover curve fitting, binary and MNIST classification, cat-state
engineering, gate synthesis and finite-size trainability. The repository
includes saved models, datasets and executed notebooks. Scientific scope,
dependency details and archived-data provenance are documented in
[docs/reproducibility.md](docs/reproducibility.md).

## Install and verify

Python 3.12 on Linux is the tested reference environment. Python 3.10–3.12 is the
supported dependency range. To install the complete reference environment,
run these commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/reproducible-py312.txt
python -m pip install --no-build-isolation -c requirements/reproducible-py312.txt -e '.[dev,sf]'
python -m quannto.benchmarks.prr
python -m pytest
```

The verification checks archived SHA-256 hashes, all 13 quantum checkpoints,
the fixed trainability dataset, all 1,280 initialization vectors, 60 derivative
variances and 16 recorded finite-difference gradient checks. It does not retrain
models or claim a full independent rerun of every experiment.

For the analytic simulator alone, `python -m pip install -e .` is sufficient.
TensorFlow is optional (`.[mnist]`) and is needed only for MNIST preprocessing.
The wheel contains code and the checkpoint-selection manifest. Research data
remain in this source checkout; set `QUANNTO_REPO_ROOT` when using a wheel from
another directory.

## Repository layout

| Location | Contents |
|---|---|
| `quannto/core/` | QNN, analytic expectations, data processing and optimizers |
| `quannto/tasks/` | Training tasks and quantum-table exporter |
| `examples_ipynb/` | Executed notebooks that load saved models or train when needed |
| `quannto/tasks/trainability/` | Trainability protocol, isolated workers and campaign launchers |
| `quannto/tasks/diagnostics/` | Cat parity, moment hierarchy and parameter-space diagnostics |
| `quannto/benchmarks/` | Verification of the archived PRR artifacts |
| `data/datasets/` | Original numeric datasets |
| `results/models/` | Saved parameters, operators, losses and training records |
| `results/prr/` | Archived diagnostic campaigns and paired initial fidelity baselines |
| `figures/paper/` | Figure sources of the final manuscript |
| `figures/legacy/`, `archive/` | Historical exports and superseded code, retained for provenance |
| `test/` | Scientific and software regression tests |
| `requirements/`, `provenance/` | Tested dependency lock, source mapping and artifact hashes |

## Use the simulator

```python
import numpy as np
import jax.numpy as jnp
from quannto.core.qnn import QNN

qnn = QNN("example", "demo", N=1, layers=1, n_in=1, n_out=1,
          ladder_modes=[[0]], is_addition=False, observable="position")
qnn.build_QNN(np.zeros(5))
mean, branch_norm = qnn.eval_QNN(jnp.array([0.4, 0.0]))
# <x> = sqrt(2)*0.4; ideal subtraction-branch norm = 0.4**2.
```

Mode indices in code start at zero. Inputs use coherent amplitudes in `xxpp`
order: real parts, followed by imaginary parts. `[x,p]=i`. The returned branch
norm is not a physical detector-click probability unless a heralding model is
specified. A zero-norm branch has no normalized conditional state.

The default `loop_matchings` expectation backend is the one used in the
manuscript. `subset_dp` is an optional exact recurrence backend; it is not
silently substituted for the recorded training protocol.

## Reproduce the diagnostics

```bash
python -m quannto.tasks.quantum_tables --audit
python -m quannto.tasks.diagnostics.catstate_parity_diagnostics
python -m quannto.tasks.diagnostics.catstate_moment_hierarchy_sweep
```

Full fidelity exports require cutoff convergence and can use substantial
memory. See [docs/quantum_fidelity_baselines.md](docs/quantum_fidelity_baselines.md)
for the paired baseline and cutoff controls. A `not_converged` result requests
more numerical resolution; it is never replaced by an invented table value.

A small end-to-end trainability workflow is:

```bash
export QUANNTO_RESULTS_DIR="$PWD/results/local/trainability_smoke"
python -m quannto.tasks.trainability.trainability --quick --resume
```

Smoke results are separate from manuscript results. For new training, use a
separate `QUANNTO_RESULTS_DIR` so a reused model name cannot replace archived
checkpoints. `QUANNTO_CACHE_DIR` overrides the writable architecture cache;
`QUANNTO_DATASETS_DIR`/`DATASETS_DIR` and `QUANNTO_FIGURES_DIR`/`FIGURES_DIR` override
data and plot locations. See [docs/migration.md](docs/migration.md) when applying
this release to an existing Git checkout.

## Notebooks

Open `examples_ipynb/` for guided examples with saved outputs and figures.
The model-based examples load a checkpoint for the selected task and settings,
or train and save a model when no matching checkpoint exists. They evaluate
predictions or states and generate plots through the task utilities.
The trainability notebook analyzes stored initialization-gradient checkpoints.

See [examples_ipynb/README.md](examples_ipynb/README.md) for the notebook list.
Historical documentation is retained in `docs/archive/`;
[docs/reproducibility.md](docs/reproducibility.md) describes the current protocol,
and [docs/validation.md](docs/validation.md) records the release validation.
