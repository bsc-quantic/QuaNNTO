import numpy as np
import jax.numpy as jnp
from quannto.core.qnn import QNN

def test_photon_subtraction_keeps_coherent_quadrature_mean():
    # === QNN with single subtraction (a) at last layer ===
    qnn_sub = QNN(
        model_name="test_sub_coh",
        task_name="unit",
        N=1, layers=1,
        n_in=1, n_out=1,
        ladder_modes=[[0]],        # subtract on mode 0
        is_addition=False,
        observable="position",
        include_initial_squeezing=False,
        include_initial_mixing=False,
        is_passive_gaussian=False,
        parameters=np.zeros(5),
        in_preprocessors=[],
        out_preprocessors=[],
        postprocessors=[],
    )
    params = np.zeros(qnn_sub.N**2 * 2 + qnn_sub.N * 3)  # all params = 0 gives identity Gaussian
    qnn_sub.build_QNN(params)

    # === Gaussian baseline ===
    qnn_gauss = QNN(
        model_name="test_gauss_baseline",
        task_name="unit",
        N=1, layers=1,
        n_in=1, n_out=1,
        ladder_modes=[[]],
        is_addition=False,
        observable="position",
        include_initial_squeezing=False,
        include_initial_mixing=False,
        is_passive_gaussian=False,
        parameters=np.zeros(5),
        in_preprocessors=[],
        out_preprocessors=[],
        postprocessors=[],
    )
    qnn_gauss.build_QNN(params)

    # avoid alpha=0 (subtraction would give ~zero state)
    x, p = 0.4, 0.0
    inputs_disp = jnp.array([x, p], dtype=jnp.float64)

    out_sub, norm_sub = qnn_sub.eval_QNN(inputs_disp)
    out_g, norm_g = qnn_gauss.eval_QNN(inputs_disp)

    # === Position mean must match (coherent is invariant under a, up to normalization) ===
    assert np.allclose(np.array(out_sub[0]), np.array(out_g[0]), rtol=0, atol=1e-10)

    # === Norm should be ~ <a† a> = |alpha|^2 = x^2 + p^2 for identity Gaussian ===
    expected_norm = x**2 + p**2
    assert np.allclose(np.array(norm_sub), expected_norm, rtol=0, atol=1e-10)

def test_2mode_double_subtraction_mode0_matches_analytic_position_and_norm():
    # === QNN definition: N=2, L=1, two subtractions on mode 0 ===
    qnn = QNN(
        model_name="unit_2mode_double_sub_mode0",
        task_name="unit",
        N=2,
        layers=1,
        n_in=2,
        n_out=1,
        ladder_modes=[[0, 0]],     # two subtractions in mode 0
        is_addition=False,
        observable="position",
        include_initial_squeezing=False,
        include_initial_mixing=False,
        is_passive_gaussian=False,
        parameters=np.zeros(14),
        in_preprocessors=[],
        out_preprocessors=[],
        postprocessors=[],
    )

    params = np.zeros(2 * (qnn.N**2) + 3 * qnn.N)  # all params = 0 gives identity Gaussian
    # === Identity Gaussian (all params = 0) ===
    qnn.build_QNN(params)

    # === Input coherent amplitudes in QuaNNTO convention: [Re0, Re1, Im0, Im1] ===
    re0, im0 = 0.40, -0.10
    re1, im1 = 0.20, 0.30
    inputs_disp = jnp.array([re0, re1, im0, im1], dtype=jnp.float64)

    # === Evaluate ===
    out, norm = qnn.eval_QNN(inputs_disp)

    # === Analytic expectations ===
    expected_x0 = np.sqrt(2.0) * re0
    expected_norm = (re0**2 + im0**2) ** 2  # |alpha0|^4

    assert np.allclose(np.array(out[0]), expected_x0, rtol=0, atol=1e-10)
    assert np.allclose(np.array(norm), expected_norm, rtol=0, atol=1e-10)