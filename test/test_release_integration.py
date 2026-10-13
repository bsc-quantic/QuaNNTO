"""Integration checks for packaged data and independent QNN instances."""
from pathlib import Path

import pytest

from quannto.benchmarks.prr import verify_hashes, verify_quantum_checkpoints, verify_trainability


def test_archived_paper_artifacts_are_complete_and_consistent():
    root = Path(__file__).resolve().parents[1]
    assert verify_hashes(root) > 1000
    assert verify_quantum_checkpoints(root) == {"cubic": 5, "cat": 8}
    assert verify_trainability(root)["configurations"] == 20


def test_modified_artifact_is_rejected(tmp_path):
    import hashlib
    (tmp_path / "provenance").mkdir()
    artifact = tmp_path / "saved.csv"
    artifact.write_text("original numerical result\n")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "provenance/artifacts.sha256").write_text(digest + "  saved.csv\n")
    assert verify_hashes(tmp_path) == 1
    artifact.write_text("different result\n")
    with pytest.raises(ValueError, match="Modified archived artifact"):
        verify_hashes(tmp_path)


def test_qnn_instances_do_not_mutate_shared_preprocessors():
    from quannto.core.qnn import QNN
    preprocessors = []
    first = QNN("first", "integration", 1, 1, 1, 1, ladder_modes=[[]],
                in_preprocessors=preprocessors)
    second = QNN("second", "integration", 1, 1, 1, 1, ladder_modes=[[]],
                 in_preprocessors=preprocessors)
    assert preprocessors == []
    assert len(first.in_preprocessors) == len(second.in_preprocessors) == 1
    first.in_preprocessors.append(lambda x: x)
    assert len(second.in_preprocessors) == 1
