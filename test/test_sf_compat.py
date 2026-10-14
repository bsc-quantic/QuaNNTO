"""Compatibility with setuptools versions that no longer ship pkg_resources."""
from importlib.util import find_spec
from pathlib import Path
import sys

import pytest

from quannto.utils.strawberryfields import compat


def test_sf_dataset_paths_remain_available_without_global_shim():
    if find_spec("strawberryfields") is None:
        pytest.skip("Strawberry Fields is not installed")
    previous = sys.modules.get("pkg_resources")
    compat.import_strawberryfields()
    from strawberryfields.apps.data import sample, feature
    assert Path(sample.DATA_PATH).is_dir()
    assert Path(feature.DATA_PATH).is_dir()
    assert sys.modules.get("pkg_resources") is previous


def test_failed_import_removes_temporary_resource_module(monkeypatch):
    monkeypatch.delitem(sys.modules, "strawberryfields", raising=False)
    monkeypatch.delitem(sys.modules, "pkg_resources", raising=False)
    monkeypatch.setattr(compat, "find_spec", lambda name: None)

    def fail(name):
        assert "pkg_resources" in sys.modules
        raise ImportError("dependency import failed")

    monkeypatch.setattr(compat, "import_module", fail)
    with pytest.raises(ImportError, match="dependency import failed"):
        compat.import_strawberryfields()
    assert "pkg_resources" not in sys.modules
