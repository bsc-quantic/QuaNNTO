# === test_qnn_2mode_double_sub_vs_sf.py ===
import numpy as np
import pytest
import scipy.integrate as si
# Temporary fix for older scipy versions where simps was named simpson
if not hasattr(si, "simps") and hasattr(si, "simpson"):
    si.simps = si.simpson
sf = pytest.importorskip("strawberryfields")
from strawberryfields import Engine, Program
from strawberryfields.ops import Dgate, GaussianTransform

from quannto.core.qnn import QNN
from quannto.utils.strawberryfields.general_tools import (
    apply_annihilation_to_ket,
    ladder_ops,
    mode_moments_from_ket,
)

np.random.seed(42)

def test_gaussian_outputs_match_sf_means():
    # === Match QuaNNTO convention: vacuum cov = 0.5 I ===
    sf.hbar = 1  # SF default is 2; it is configurable

    N, layers = 2, 1
    n_pars = 2*(N**2) + 3*N   # active 1-layer

    qnn = QNN(
        model_name="sf_equiv",
        task_name="unit",
        N=N, layers=layers,
        n_in=1, n_out=1,
        ladder_modes=[[]],          # no non-G
        is_addition=False,
        observable="position",
        include_initial_squeezing=False,
        include_initial_mixing=False,
        is_passive_gaussian=False,
        parameters=np.zeros(n_pars),
        in_preprocessors=[],
        out_preprocessors=[],
        postprocessors=[],
    )

    # Choose random params for the single-layer Gaussian QONN
    params = np.random.rand(n_pars)
    qnn.build_QNN(params)

    # Input alpha on mode 0 only (QuaNNTO uses (Re,Im) per mode)
    inputs_disp = np.array([0.20, 0.00, 0.00, 0.00])  # (Re0,Re1,Im0,Im1)?? -> careful: QuaNNTO uses 2N vector [x1..xN,p1..pN] == [Re.., Im..]
    # For N=2: [Re0, Re1, Im0, Im1]
    qnn_out, _ = qnn.eval_QNN(inputs_disp)
    qnn_x0 = np.real_if_close(np.array(qnn_out[0]))

    # === Build equivalent SF program ===
    eng = Engine("gaussian")
    prog = Program(N)
    with prog.context as q:
        # input displacements
        alpha0 = inputs_disp[0] + 1j*inputs_disp[N]
        alpha1 = inputs_disp[1] + 1j*inputs_disp[N+1]
        Dgate(np.abs(alpha0), np.angle(alpha0)) | q[0]  # Dgate supports complex alpha
        Dgate(np.abs(alpha1), np.angle(alpha1)) | q[1]

        # layer symplectic
        S = np.real_if_close(np.array(qnn.S_l[0]))
        GaussianTransform(S) | (q[0], q[1])  # takes symplectic S

        # layer displacement (same mapping: d = Re + i Im)
        d0 = np.real_if_close(qnn.D_l[0, 0]) + 1j*np.real_if_close(qnn.D_l[0, N])
        d1 = np.real_if_close(qnn.D_l[0, 1]) + 1j*np.real_if_close(qnn.D_l[0, N+1])
        Dgate(np.abs(d0), np.angle(d0)) | q[0]
        Dgate(np.abs(d1), np.angle(d1)) | q[1]

    state = eng.run(prog).state
    mu = state.means()  # (x1..xN, p1..pN)
    sf_x0 = np.real_if_close(mu[0])

    assert np.allclose(qnn_x0, sf_x0, rtol=0, atol=1e-9)

def test_2mode_double_subtraction_mode0_matches_sf():
    # === Match QuaNNTO convention: vacuum cov = 0.5 I ===
    sf.hbar = 1

    # === Setup ===
    N, layers = 2, 1
    # active 1-layer parameter count: 2*N^2 + 3*N = 14
    n_pars = 2 * (N**2) + 3 * N
    params = np.random.rand(n_pars)  # random Gaussian layer params

    # === QNN with two photon subtractions on mode 0 in the single layer ===
    qnn = QNN(
        model_name="sf_equiv_double_sub_mode0",
        task_name="unit",
        N=N,
        layers=layers,
        n_in=1,
        n_out=1,
        ladder_modes=[[0, 0]],  # two subtractions in mode 0
        is_addition=False,
        observable="position",
        include_initial_squeezing=False,
        include_initial_mixing=False,
        is_passive_gaussian=False,
        parameters=params,
        in_preprocessors=[],
        out_preprocessors=[],
        postprocessors=[],
    )

    # === Use small random params to keep the Fock cutoff stable ===
    # (Large squeezing would require higher cutoff for accurate SF results.)
    rng = np.random.default_rng(42)
    params = 0.05 * rng.standard_normal(n_pars)
    qnn.build_QNN(params)

    # === Input coherent displacement in QuaNNTO convention: [Re0, Re1, Im0, Im1] ===
    inputs_disp = np.array([0.20, 0.05, -0.10, 0.00], dtype=float)

    # === QuaNNTO prediction ===
    qnn_out, qnn_norm = qnn.eval_QNN(inputs_disp)
    qnn_x0 = np.real_if_close(np.array(qnn_out[0]))
    qnn_p_succ = float(np.real_if_close(np.array(qnn_norm)))

    # === Build equivalent SF program (Fock backend so we can apply annihilation on ket) ===
    cutoff = 20
    eng = Engine("fock", backend_options={"cutoff_dim": cutoff})
    prog = Program(N)

    with prog.context as q:
        # input displacements
        alpha0 = inputs_disp[0] + 1j * inputs_disp[N]
        alpha1 = inputs_disp[1] + 1j * inputs_disp[N + 1]
        Dgate(np.abs(alpha0), np.angle(alpha0)) | q[0]
        Dgate(np.abs(alpha1), np.angle(alpha1)) | q[1]

        # layer symplectic
        S = np.real_if_close(np.array(qnn.S_l[0]))
        GaussianTransform(S) | (q[0], q[1])

        # layer displacement (mapping: d = Re + i Im)
        d0 = np.real_if_close(qnn.D_l[0, 0]) + 1j * np.real_if_close(qnn.D_l[0, N])
        d1 = np.real_if_close(qnn.D_l[0, 1]) + 1j * np.real_if_close(qnn.D_l[0, N + 1])
        Dgate(np.abs(d0), np.angle(d0)) | q[0]
        Dgate(np.abs(d1), np.angle(d1)) | q[1]

    state = eng.run(prog).state
    ket = np.array(state.ket())  # shape (cutoff, cutoff)

    # === Apply a^2 on mode 0 using QuaNNTO helper ===
    ket1, p1 = apply_annihilation_to_ket(ket, cutoff=cutoff, mode=0)   # normalized ket1, conditional prob p1
    ket2, p2 = apply_annihilation_to_ket(ket1, cutoff=cutoff, mode=0)  # normalized ket2, conditional prob p2
    sf_p_succ = float(np.real_if_close(p1 * p2))  # total success probability

    # === Compute <x0> on the final (normalized) state ===
    a, adag = ladder_ops(cutoff)
    x_op = (a + adag) / np.sqrt(2.0)

    obs = mode_moments_from_ket(ket2, moments={"x": x_op}, cutoff=cutoff, mode=0)
    sf_x0 = float(np.real_if_close(obs["x"]))

    # === Compare ===
    # Norm/probabilities are more sensitive to cutoff; relax slightly vs Gaussian-only tests.
    assert np.allclose(qnn_x0, sf_x0, rtol=0, atol=1e-6)
    assert np.allclose(qnn_p_succ, sf_p_succ, rtol=0, atol=5e-5)
