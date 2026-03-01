import sys
from pathlib import Path
import pytest

# === Ensure repo root is importable (so "import quannto" works) ===
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

@pytest.fixture(autouse=True)
def _sandbox_quannto_paths(tmp_path, monkeypatch):
    """
    Redirect QuaNNTO outputs/caches to a temp dir for clean test runs.
    """
    monkeypatch.setenv("QUANNTO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("QUANNTO_RESULTS_DIR", str(tmp_path / "tasks"))
    monkeypatch.setenv("DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("FIGURES_DIR", str(tmp_path / "figures"))