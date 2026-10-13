import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
try:
    import jax.numpy as jnp
    from quannto.core.qnn import QNN
except ModuleNotFoundError:
    jnp = None
    QNN = None


def _build_identity_qnn(ladder_modes):
    n_modes = 2
    layers = 2
    num_parameters = layers * (2 * n_modes**2 + 3 * n_modes)
    parameters = np.zeros(num_parameters, dtype=np.float64)
    qnn = QNN(
        model_name=f"test_multilayer_{ladder_modes}",
        task_name="unit",
        N=n_modes,
        layers=layers,
        n_in=n_modes,
        n_out=1,
        ladder_modes=ladder_modes,
        is_addition=False,
        observable="position",
        include_initial_squeezing=False,
        include_initial_mixing=False,
        is_passive_gaussian=False,
        parameters=parameters,
        in_preprocessors=[],
        out_preprocessors=[],
        postprocessors=[],
    )
    qnn.build_QNN(parameters)
    return qnn


@unittest.skipIf(QNN is None, "JAX/QONN integration dependencies are unavailable")
class MultilayerGaussianControlTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self._env_patch = patch.dict(
            os.environ,
            {"QUANNTO_CACHE_DIR": self._temp_dir.name},
        )
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._temp_dir.cleanup()

    def test_multilayer_gaussian_control_uses_unit_trace_coefficients(self):
        qnn = _build_identity_qnn([[], []])
        self.assertTrue(hasattr(qnn, "jax_ones_coefs"))
        self.assertFalse(hasattr(qnn, "jax_traceterms_coefs_inds"))

        re0, re1, im0, im1 = 0.4, 0.2, -0.1, 0.3
        input_displacement = jnp.asarray(
            [re0, re1, im0, im1], dtype=jnp.float64
        )
        output, norm = qnn.eval_QNN(input_displacement)

        np.testing.assert_allclose(
            np.asarray(output[0]), np.sqrt(2.0) * re0, atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(np.asarray(norm), 1.0, atol=1e-10, rtol=0)

        # Reproduce the array layout stored by older Gaussian L>1 caches.
        base_coefficients = qnn.jax_ones_coefs
        qnn.jax_traceterms_coefs = base_coefficients
        qnn.jax_traceterms_coefs_inds = jnp.empty(
            (*base_coefficients.shape, 0), dtype=jnp.int32
        )
        del qnn.jax_ones_coefs

        cached_output, cached_norm = qnn.eval_QNN(input_displacement)
        np.testing.assert_allclose(cached_output, output, atol=1e-10, rtol=0)
        np.testing.assert_allclose(cached_norm, norm, atol=1e-10, rtol=0)

    def test_multilayer_non_gaussian_coefficients_are_unchanged(self):
        qnn = _build_identity_qnn([[0], [0]])
        self.assertTrue(hasattr(qnn, "jax_traceterms_coefs_inds"))

        re0, re1, im0, im1 = 0.4, 0.2, -0.1, 0.3
        output, norm = qnn.eval_QNN(
            jnp.asarray([re0, re1, im0, im1], dtype=jnp.float64)
        )

        expected_norm = (re0**2 + im0**2) ** 2
        np.testing.assert_allclose(
            np.asarray(output[0]), np.sqrt(2.0) * re0, atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(
            np.asarray(norm), expected_norm, atol=1e-10, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
