"""Load numeric QNN checkpoints, or train and save a missing model."""
import inspect
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from quannto.core.qnn import QNN
from quannto.core.qnn_trainers import train_scipy, _compute_num_parameters
from quannto.utils.path_utils import (
    models_params_path, models_train_losses_path, models_valid_losses_path,
)


def _numeric(path):
    with Path(path).open("rb") as handle:
        magic = handle.read(6)
    data = np.load(path, allow_pickle=False) if magic == b"\x93NUMPY" else np.loadtxt(path)
    data = np.asarray(data)
    if data.dtype.kind not in "fciu" or not np.isfinite(data).all():
        raise ValueError(f"Invalid numeric checkpoint: {path}")
    return data


def _history(path_function, model_name):
    for extension in ("txt", "npy"):
        path = path_function(model_name, extension)
        if path.is_file():
            return _numeric(path).reshape(-1)
    return None


def load_or_train(*args, **kwargs):
    """Use the same arguments and return values as ``train_scipy``.

    A numeric parameter checkpoint is rebuilt with the requested architecture
    and preprocessing. Missing checkpoints are trained with ``train_scipy``
    and saved by default. Malformed checkpoints raise instead of being replaced.
    ``qnn.checkpoint_source`` identifies whether the model was loaded or trained.
    """
    bound = inspect.signature(train_scipy).bind(*args, **kwargs)
    bound.apply_defaults()
    config = bound.arguments
    name = config["model_name"]
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("model_name must be a nonempty filename stem")
    paths = [models_params_path(name, ext) for ext in ("npy", "txt")]
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        print(f"Training missing model: {name}")
        qnn, train_history, valid_history = train_scipy(*args, **kwargs)
        if config["save"]:
            for path_function, history in (
                (models_train_losses_path, train_history),
                (models_valid_losses_path, valid_history),
            ):
                if history is not None:
                    with path_function(name, "txt").open("wb") as handle:
                        np.save(handle, np.asarray(history, dtype=float))
        qnn.checkpoint_source = "trained"
        return qnn, train_history, valid_history

    parameters = _numeric(path).reshape(-1)
    expected = _compute_num_parameters(
        config["N"], config["layers"], config["include_initial_squeezing"],
        config["include_initial_mixing"], config["is_passive_gaussian"])
    if len(parameters) != expected or np.max(np.abs(parameters.imag), initial=0) > 1e-12:
        raise ValueError(f"Checkpoint parameters do not match this architecture: {path.name}")
    with redirect_stdout(io.StringIO()):
        qnn = QNN(
            name, config["task_name"], config["N"], config["layers"],
            config["n_inputs"], config["n_outputs"], config["ladder_modes"],
            config["is_addition"], config["observable"],
            config["include_initial_squeezing"], config["include_initial_mixing"],
            config["is_passive_gaussian"], parameters.real,
            config["in_preprocs"], config["out_prepocs"], config["postprocs"])
        qnn.build_QNN(parameters.real)
    qnn.checkpoint_source = str(path)
    print(f"Loaded model: {path.name}")
    return qnn, _history(models_train_losses_path, name), _history(models_valid_losses_path, name)
