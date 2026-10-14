"""Load Strawberry Fields 0.23 with modern setuptools.

Its two bundled dataset modules only use pkg_resources.resource_filename.
Supply that operation with importlib.resources during import, without installing
an obsolete setuptools or leaving a replacement pkg_resources module globally.
"""
from importlib import import_module
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType

_IMPORT_LOCK = RLock()


def _resource_filename(package, resource):
    if package != "strawberryfields" or resource not in (
        "apps/data/sample_data", "apps/data/feature_data",
    ):
        raise ValueError("Unsupported Strawberry Fields resource request")
    path = files(package).joinpath(*resource.split("/"))
    if not isinstance(path, Path):
        raise RuntimeError("Strawberry Fields must be installed as an unpacked package")
    return str(path)


def import_strawberryfields():
    """Return the optional Strawberry Fields module, preserving import errors."""
    with _IMPORT_LOCK:
        if "strawberryfields" in sys.modules:
            return import_module("strawberryfields")
        if "pkg_resources" in sys.modules or find_spec("pkg_resources") is not None:
            return import_module("strawberryfields")
        resources = ModuleType("pkg_resources")
        resources.resource_filename = _resource_filename
        sys.modules["pkg_resources"] = resources
        try:
            return import_module("strawberryfields")
        finally:
            if sys.modules.get("pkg_resources") is resources:
                del sys.modules["pkg_resources"]
