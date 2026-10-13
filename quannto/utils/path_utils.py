"""Repository data and writable runtime locations.

Environment overrides remain compatible with earlier QuaNNTO releases. A wheel
contains code only; point it at a source/data checkout using QUANNTO_REPO_ROOT.
"""
from pathlib import Path
import os


def repository_root():
    override = os.environ.get("QUANNTO_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").is_file() else None


def _env_path(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser().resolve()
    return None


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def arch_cache_dir():
    path = _env_path("QUANNTO_CACHE_DIR")
    if path is None:
        cache = _env_path("XDG_CACHE_HOME") or Path.home() / ".cache"
        path = cache / "quannto" / "architectures"
    return ensure_dir(path)


def quannto_results_dir():
    path = _env_path("QUANNTO_RESULTS_DIR")
    if path is not None:
        return path
    root = repository_root()
    return root / "results" if root is not None else Path.home() / ".quannto" / "results"


def datasets_dir():
    path = _env_path("QUANNTO_DATASETS_DIR", "DATASETS_DIR")
    if path is not None:
        return path
    root = repository_root()
    return root / "data" / "datasets" if root is not None else Path.home() / ".quannto" / "datasets"


def figures_dir():
    path = _env_path("QUANNTO_FIGURES_DIR", "FIGURES_DIR")
    return ensure_dir(path if path is not None else quannto_results_dir() / "generated_figures")


def models_json_path(model_name, ext):
    return ensure_dir(quannto_results_dir() / "models" / "pickle_json") / f"{model_name}.{ext}"


def models_params_path(model_name, ext):
    return ensure_dir(quannto_results_dir() / "models" / "params") / f"{model_name}.{ext}"


def models_operators_path(model_name, operator_name, ext):
    return ensure_dir(quannto_results_dir() / "models" / "trained_operators") / f"{model_name}_{operator_name}.{ext}"


def models_train_losses_path(model_name, ext):
    return ensure_dir(quannto_results_dir() / "models" / "train_losses") / f"{model_name}.{ext}"


def models_valid_losses_path(model_name, ext):
    return ensure_dir(quannto_results_dir() / "models" / "valid_losses") / f"{model_name}.{ext}"


def models_testing_results_path(model_name, ext):
    return ensure_dir(quannto_results_dir() / "models" / "testing_results") / f"{model_name}.{ext}"
