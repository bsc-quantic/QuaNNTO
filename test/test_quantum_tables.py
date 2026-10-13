"""Scientific and provenance regression tests; no optimizer is invoked."""
import json
from pathlib import Path

import numpy as np
import pytest

from quannto.tasks.quantum_tables import (load_numeric, load_checkpoint, load_initial_operators,
                                           load_task_data, audit_model, main)
from quannto.utils.strawberryfields.general_tools import (
    coherent_ket, fidelity_two_pure, fidelity_pure_vs_mixed, mode_moments_from_ket,
)
from quannto.utils.strawberryfields.quantum_state_evaluation import (
    projected_qonn_ket, reduced_output, output_fidelity, output_moments,
    reference_ket, operators_from_parameters, moment_operators, validate_layers,
)


def test_squared_fidelity_and_global_phase():
    a = np.array([1, 0], complex)
    b = np.array([0.8, 0.6], complex)
    for phase in (1, -1, 1j, np.exp(0.71j)):
        assert fidelity_two_pure(a, phase*b) == pytest.approx(0.64)
        assert fidelity_two_pure(a, phase*a) == pytest.approx(1)
    assert fidelity_pure_vs_mixed(a, 3*np.outer(b, b.conj())) == pytest.approx(0.64)
    with pytest.raises(ValueError):
        fidelity_two_pure(a, [0, 0])
    with pytest.raises(ValueError):
        fidelity_pure_vs_mixed(a, np.diag([2, -1]))


def test_coherent_ket_high_cutoff_and_truncated_norm():
    from scipy.stats import poisson
    for alpha, cutoff in ((0, 40), (1+2j, 80), (15, 300)):
        ket = coherent_ket(alpha, cutoff)
        assert np.vdot(ket, ket).real == pytest.approx(poisson.cdf(cutoff-1, abs(alpha)**2), abs=2e-13)
        assert np.allclose(ket[1:]*np.sqrt(np.arange(1, cutoff)), alpha*ket[:-1], atol=1e-13)
    assert np.linalg.norm(coherent_ket(3, 3)) < 0.1


def test_reduced_fidelity_includes_mixedness_and_target_tail():
    entangled = np.zeros((2, 2), complex)
    entangled[0, 0] = entangled[1, 1] = 1/np.sqrt(2)
    rho = reduced_output(entangled)
    assert np.allclose(rho, np.eye(2)/2)
    assert output_fidelity(rho, [1, 0]) == pytest.approx(0.5)
    # A large target must not be renormalized after slicing to the output cutoff.
    target = np.sqrt([0.25, 0.25, 0.5])
    assert output_fidelity(np.diag([1, 0]), target) == pytest.approx(0.25)


def test_moments_use_unbounded_ordering_and_complex_components():
    D = 12
    rho = np.zeros((D, D), complex)
    rho[-1, -1] = 1
    # a^3 a^dagger has the correct top-boundary matrix element; padding is essential.
    op = moment_operators("cat", D)[7]
    assert op[D-3, D-1] == pytest.approx(D*np.sqrt((D-1)*(D-2)))
    assert output_moments(rho, "cat")[11] == pytest.approx((D-1)**3)
    psi = coherent_ket(0.2+0.3j, D)
    pure = np.outer(psi, psi.conj())/np.vdot(psi, psi)
    assert output_moments(pure, "cubic")[0] == pytest.approx(0.2+0.3j, abs=1e-10)
    ket = np.outer(psi, coherent_ket(-0.4, D))
    a = np.diag(np.sqrt(np.arange(1, D)), 1)
    assert mode_moments_from_ket(ket, {"a": a}, D)["a"] == pytest.approx(0.2+0.3j, abs=1e-10)


def test_numeric_file_detection_and_pickle_rejection(tmp_path):
    binary = tmp_path / "history.txt"
    with binary.open("wb") as f:
        np.save(f, np.array([1.0, 0.5]))
    text = tmp_path / "parameters.npy"
    np.savetxt(text, [0.1, 0.2])
    assert np.array_equal(load_numeric(binary), [1, 0.5])
    assert np.array_equal(load_numeric(text), [0.1, 0.2])
    np.save(tmp_path / "object.npy", np.array([{}], dtype=object))
    with pytest.raises(ValueError):
        load_numeric(tmp_path / "object.npy")


def test_invalid_circuits_and_memory_guard():
    with pytest.raises(ValueError):
        validate_layers(np.ones((1, 2, 2)), np.zeros((1, 2)), [[]])
    with pytest.raises(ValueError):
        validate_layers(np.eye(2)[None], np.zeros((1, 2)), [[1]])
    with pytest.raises(MemoryError):
        projected_qonn_ket(np.eye(8)[None], np.zeros((1, 8)), [[]], 0, 100, max_tensor_bytes=1000)


@pytest.mark.parametrize("L", [1, 2, 3])
def test_affine_displacements_and_subtraction_norms(L):
    pytest.importorskip("thewalrus")
    ds = np.array([[0.1, 0.08], [-0.02, 0.07], [0.03, -0.05]])[:L]
    alpha = 0.4-0.1j
    S = np.repeat(np.eye(2)[None], L, axis=0)
    ket, weight = projected_qonn_ket(S, ds, [[0]]*L, alpha, 20)
    cumulative = alpha + np.cumsum(ds[:, 0] + 1j*ds[:, 1])
    assert weight == pytest.approx(np.prod(abs(cumulative)**2), rel=1e-11)
    assert fidelity_two_pure(ket, coherent_ket(cumulative[-1], 20)) == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("N,L,subs", [(1, 1, [[0, 0]]), (2, 1, [[0, 1]]),
                                     (2, 2, [[1], [0]]), (2, 3, [[1], [0], [1]])])
def test_gaussian_polynomial_matches_literal_strawberry_fields(N, L, subs, monkeypatch):
    sf = pytest.importorskip("strawberryfields")
    monkeypatch.setattr(sf, "hbar", 2)
    from quannto.utils.strawberryfields.qonn_builder import run_qonn
    # Low energy makes the independent, literal Fock simulation converge through sixth moments at D=30.
    rng = np.random.default_rng(218+L)
    parameters = rng.normal(0, 0.06, L*(2*N*N+3*N))
    S, d = operators_from_parameters(parameters, N, L)
    layers = [{"S": s, "d": dd} for s, dd in zip(S, d)]
    alpha, D = 0.3+0.1j, 30
    actual, weight = projected_qonn_ket(S, d, subs, alpha, D)
    expected, factors = run_qonn(N, D, layers, subs, input_alpha=alpha)
    assert fidelity_two_pure(actual, expected) == pytest.approx(1, abs=5e-10)
    assert weight == pytest.approx(np.prod(factors), rel=1e-8, abs=1e-13)
    assert np.allclose(output_moments(reduced_output(actual), "cat"),
                       output_moments(reduced_output(expected), "cat"), atol=1e-8, rtol=1e-8)


def test_projection_consistency_with_larger_cutoff():
    pytest.importorskip("thewalrus")
    S = np.array([[[1.3, 0], [0, 1/1.3]], [[1.1, 0], [0, 1/1.1]]])
    d = np.array([[0.1, 0.2], [-0.05, 0.13]])
    small, wsmall = projected_qonn_ket(S, d, [[0], [0]], 0.1, 12)
    large, wlarge = projected_qonn_ket(S, d, [[0], [0]], 0.1, 30)
    assert wlarge >= wsmall
    # Exact projected coefficients, not merely similar normalized output moments.
    assert np.allclose(small, large[:12], atol=1e-13, rtol=1e-12)


@pytest.mark.parametrize("hbar", [1.0, 2.0])
def test_cubic_reference_matches_public_vgate(hbar, monkeypatch):
    sf = pytest.importorskip("strawberryfields")
    from strawberryfields.ops import Coherent, Vgate
    monkeypatch.setattr(sf, "hbar", hbar)
    D, alpha, gamma = 40, 0.4, 0.2
    prog = sf.Program(1)
    with prog.context as q:
        Coherent(alpha) | q[0]
        Vgate(gamma) | q[0]
    state = sf.Engine("fock", backend_options={"cutoff_dim": D}).run(prog).state
    target = reference_ket("cubic", alpha, D, gamma=gamma, hbar=hbar)
    assert fidelity_two_pure(target, state.ket()) == pytest.approx(1, abs=1e-13)
    assert np.allclose(target, reference_ket("cubic", alpha, D, gamma=gamma*np.sqrt(hbar), hbar=1), atol=1e-12)
    identity = reference_ket("cubic", alpha, D, gamma=0, hbar=hbar)
    assert fidelity_two_pure(identity, coherent_ket(alpha, D)) == pytest.approx(1, abs=1e-13)


def test_checkpoint_source_is_explicit_and_conflicts_fail(tmp_path):
    (tmp_path / "params").mkdir()
    (tmp_path / "trained_operators").mkdir()
    spec = {"N": 1, "subtractions": [[]], "checkpoint_stem": "saved"}
    np.save(tmp_path / "params/saved.npy", np.zeros(5))
    np.save(tmp_path / "trained_operators/saved_S_l.npy", np.eye(2)[None])
    np.save(tmp_path / "trained_operators/saved_D_l.npy", np.ones((1, 2)))
    with pytest.raises(ValueError, match="different checkpoints"):
        load_checkpoint(spec, tmp_path)
    S, d, _, source = load_checkpoint(dict(spec, checkpoint_source="operators"), tmp_path)
    assert source == "operators"
    assert np.array_equal(d, np.ones((1, 2)))
    S, d, _, source = load_checkpoint(dict(spec, checkpoint_source="parameters"), tmp_path)
    assert source == "parameters_decoded_without_training"
    assert np.array_equal(d, np.zeros((1, 2)))


def test_saved_quantum_table_mapping_matches_revised_manuscript():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "quannto/tasks/quantum_tables_manifest.json").read_text())
    models_root = root / "results/models"
    if not models_root.exists():
        pytest.skip("Model artifacts not included in this checkout")
    cubic = [s for s in manifest["models"] if s["task"] == "cubic"]
    assert [s["id"] for s in cubic] == [
        "cubic_n2_l1", "cubic_n2_l2", "cubic_n3_l1", "cubic_n4_l1", "cubic_n2_double"
    ]
    assert [s["subtractions"] for s in cubic] == [
        [[1]], [[1], [1]], [[1, 2]], [[1, 2, 3]], [[1, 1]]
    ]
    for spec in manifest["models"]:
        definition = manifest["tasks"][spec["task"]]
        data = load_task_data(spec["task"], definition, root / "data/datasets")
        row, _ = audit_model(spec, models_root, data, definition)
        assert row["status"] == "ready", row.get("error")
        initial, metadata = load_initial_operators(spec, models_root)
        assert initial is not None and metadata is not None
        assert metadata["seed"] == 42
        assert metadata["parameters_sha256"] == row["initial_parameters_sha256"]
        if spec["task"] == "cubic":
            assert len(data["evaluation"]) == 49
            assert not np.intersect1d(data["inputs"], data["evaluation"]).size



def test_cli_missing_checkpoint_exports_explicit_failure(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "evaluation"
    code = main(["--audit", "--task", "cubic", "--models-root", str(tmp_path / "empty"),
                 "--datasets-root", str(root / "data/datasets"), "--output", str(output)])
    assert code == 2
    report = json.loads((output / "audit.json").read_text())
    assert report["complete"] is False
    assert all(r["status"] == "missing_checkpoint" for r in report["models"])
    assert all(r["fidelity_mean"] is None for r in report["models"])


def test_cubic_cli_end_to_end_against_position_quadrature(tmp_path):
    pytest.importorskip("strawberryfields")
    from scipy.special import roots_hermite
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "quannto/tasks/quantum_tables_manifest.json").read_text())
    # Exactly solvable reference circuit: subtract from a coherent ancilla.
    # The observed output remains |alpha>. This is a test fixture, not training.
    model = {"id": "coherent_reference", "figure": 3, "task": "cubic", "N": 2,
             "subtractions": [[1]], "checkpoint_stem": "coherent_reference"}
    manifest["models"] = [model]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    operators = tmp_path / "models/trained_operators"
    operators.mkdir(parents=True)
    np.save(operators / "coherent_reference_S_l.npy", np.eye(4)[None])
    np.save(operators / "coherent_reference_D_l.npy", np.array([[0, 1, 0, 0]]))
    output = tmp_path / "results"
    args = ["--manifest", str(manifest_path), "--models-root", str(operators.parent),
            "--datasets-root", str(root / "data/datasets"), "--output", str(output)]
    assert main(args) == 0
    report = json.loads((output / "quantum_tables.json").read_text())
    row = report["models"][0]
    assert report["complete"] and row["fidelity_sample_count"] == 49
    assert len(row["points"]) == 99
    # Independent wavefunction integral, <alpha|V|alpha> in SF's hbar=2 units.
    nodes, weights = roots_hermite(256)
    for point in row["points"]:
        x = 2*point["alpha"] + np.sqrt(2)*nodes
        overlap = np.sum(weights*np.exp(1j*0.2*x**3/6))/np.sqrt(np.pi)
        assert point["fidelity"] == pytest.approx(abs(overlap)**2, abs=2e-10)
    table = (output / "cubic_table.tex").read_text()
    assert "F_{\\mathrm{test,min}}" in table
    assert "\\overline{F}_{\\mathrm{test}}^{\\mathrm{init}}" in table
    # A convention change must be rejected, not silently applied to saved runs.
    manifest["tasks"]["cubic"]["gamma_sf"] = 0.21
    manifest_path.write_text(json.dumps(manifest))
    assert main(args) == 2
    failed = json.loads((output / "quantum_tables.json").read_text())["models"][0]
    assert "Target mismatch" in failed["error"]


@pytest.mark.parametrize("id", ["cat_n1_double", "cat_n2_aux_double", "cat_n2_both", "cat_n3_aux_pair", "cat_n2_l2", "cat_n1_single", "cat_n1_triple", "cat_n1_l2"])
def test_saved_cat_operators_against_independent_fock(id, monkeypatch):
    sf = pytest.importorskip("strawberryfields")
    monkeypatch.setattr(sf, "hbar", 2)
    from quannto.utils.strawberryfields.qonn_builder import run_qonn
    root = Path(__file__).resolve().parents[1]
    if not (root / "results/models").exists():
        pytest.skip("Model artifacts not included in this checkout")
    manifest = json.loads((root / "quannto/tasks/quantum_tables_manifest.json").read_text())
    spec = next(s for s in manifest["models"] if s["id"] == id)
    S, d, _, _ = load_checkpoint(spec, root / "results/models")
    # The one-mode run has an intermediate displaced squeezed state requiring
    # more levels in the literal gate simulation than in the final projection.
    D = 48 if spec["N"] == 1 else 32
    if id == "cat_n2_l2":
        # The updated model has substantial intermediate Fock tails. D=32
        # fails even though final-state fidelity may appear fairly stable.
        # Generator exponentials avoid SF's dense four-index beam-splitter
        # tensors at the independently verified intermediate cutoff D=240.
        D = 240
    projected, _ = projected_qonn_ket(S, d, spec["subtractions"], -1, D)
    if id == "cat_n2_l2":
        from quannto.utils.strawberryfields.parameter_fock_reference import parameter_fock_ket
        params = load_numeric(root / "results/models/params" / (spec["checkpoint_stem"] + ".npy"))
        literal = parameter_fock_ket(params, spec["N"], spec["subtractions"], -1, D)
        assert np.vdot(literal, literal).real == pytest.approx(np.vdot(projected, projected).real, rel=1e-8)
    else:
        literal, _ = run_qonn(spec["N"], D, [{"S": s, "d": dd} for s, dd in zip(S, d)],
                              spec["subtractions"], input_alpha=-1)
    assert fidelity_two_pure(projected, literal) == pytest.approx(1, abs=1e-9)
    target = reference_ket("cat", -1, max(300, D))
    assert output_fidelity(reduced_output(projected), target) == pytest.approx(
        output_fidelity(reduced_output(literal), target), abs=1e-8)


@pytest.mark.parametrize("N,L,subs", [(1, 2, [[0], [0]]), (2, 2, [[1], [0]]), (3, 1, [[1, 2]])])
def test_parameter_generator_reference(N, L, subs):
    pytest.importorskip("thewalrus")
    from quannto.utils.strawberryfields.parameter_fock_reference import parameter_fock_ket
    params = np.random.default_rng(821+N).normal(0, .05, L*(2*N*N+3*N))
    S, d = operators_from_parameters(params, N, L)
    literal = parameter_fock_ket(params, N, subs, -1, 24)
    projected, weight = projected_qonn_ket(S, d, subs, -1, 24)
    assert fidelity_two_pure(projected, literal) == pytest.approx(1, abs=1e-10)
    assert np.vdot(literal, literal).real == pytest.approx(weight, rel=1e-10)


def test_parameter_decoder_matches_qnn_builder():
    pytest.importorskip("jax")
    from quannto.core.qnn import QNN
    from quannto.utils.cvquantum_utils import CanonicalLadderTransformations
    for N, L in ((1, 1), (2, 3), (3, 2)):
        rng = np.random.default_rng(481+N)
        parameters = rng.normal(0, 0.2, L*(2*N*N+3*N))
        # Only the deterministic Gaussian builder is needed; no training or
        # symbolic compilation/cache generation is involved.
        qnn = object.__new__(QNN)
        qnn.N, qnn.layers = N, L
        qnn.include_initial_squeezing = qnn.include_initial_mixing = False
        qnn.is_passive_gaussian = False
        qnn.u_bar = CanonicalLadderTransformations(N)
        qnn.build_QNN(parameters)
        S, d = operators_from_parameters(parameters, N, L)
        assert np.allclose(S, np.asarray(qnn.S_l), atol=1e-12, rtol=1e-12)
        assert np.allclose(d, np.asarray(qnn.D_l), atol=1e-12, rtol=1e-12)
