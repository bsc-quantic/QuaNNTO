from __future__ import annotations
from pathlib import Path
import os

def quannto_results_dir() -> Path:
    """
    Base directory where QuaNNTO writes outputs (params, losses, results, figures, datasets).
    Priority:
      1) env var QUANNTO_RESULTS_DIR
      2) <repo_root>/quannto/tasks   (when running from source tree)
      3) ~/.quannto/tasks            (fallback)
    """
    env = os.getenv("QUANNTO_RESULTS_DIR")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # repo_root/
    candidate = repo_root / "quannto" / "tasks"
    if candidate.exists():
        return candidate

    return (Path.home() / ".quannto" / "tasks").resolve()

def datasets_dir() -> Path:
    """
    Base directory where QuaNNTO writes datasets.
    Priority:
      1) env var DATASETS_DIR
      2) <repo_root>/datasets   (when running from source tree)
      3) ~/datasets            (fallback)
    """
    env = os.getenv("DATASETS_DIR")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # repo_root/
    candidate = repo_root / "datasets"
    if candidate.exists():
        return candidate

    return (Path.home() / "datasets").resolve()

def figures_dir() -> Path:
    """
    Base directory where QuaNNTO writes figures.
    Priority:
      1) env var FIGURES_DIR
      2) <repo_root>/figures   (when running from source tree)
      3) ~/figures            (fallback)
    """
    env = os.getenv("FIGURES_DIR")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # repo_root/
    candidate = repo_root / "figures"
    if candidate.exists():
        return candidate

    return (Path.home() / "figures").resolve()

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def models_json_path(model_name: str, ext: str) -> Path:
    return ensure_dir(quannto_results_dir() / "models" / "pickle_json") / f"{model_name}.{ext}"

def models_params_path(model_name: str, ext: str) -> Path:
    return ensure_dir(quannto_results_dir() / "models" / "params") / f"{model_name}.{ext}"

def models_operators_path(model_name: str, operator_name: str, ext: str) -> Path:
    return ensure_dir(quannto_results_dir() / "models" / "trained_operators") / f"{model_name}_{operator_name}.{ext}"

def models_train_losses_path(model_name: str, ext: str) -> Path:
    return ensure_dir(quannto_results_dir() / "models" / "train_losses") / f"{model_name}.{ext}"

def models_valid_losses_path(model_name: str, ext: str) -> Path:
    return ensure_dir(quannto_results_dir() / "models" / "valid_losses") / f"{model_name}.{ext}"

def models_testing_results_path(model_name: str, ext: str) -> Path:
    return ensure_dir(quannto_results_dir() / "models" / "testing_results") / f"{model_name}.{ext}"
