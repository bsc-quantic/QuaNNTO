import unittest

import jax
import jax.numpy as jnp
import numpy as np

from quannto.core.wick_subset_dp import wick_moment_subset_dp

jax.config.update("jax_enable_x64", True)


def _loop_matchings(indices):
    if not indices:
        yield ()
        return
    first, *remaining = indices
    yield from (
        ((first, first),) + suffix for suffix in _loop_matchings(remaining)
    )
    for position, partner in enumerate(remaining):
        rest = remaining[:position] + remaining[position + 1:]
        yield from (
            ((first, partner),) + suffix for suffix in _loop_matchings(rest)
        )


class WickSubsetDPTests(unittest.TestCase):
    def test_matches_explicit_loop_matching_sum(self):
        rng = np.random.default_rng(7)
        n_modes = 3
        means = rng.normal(size=2 * n_modes)
        quadratic = (
            rng.normal(size=(4, n_modes, n_modes))
            + 1j * rng.normal(size=(4, n_modes, n_modes))
        )

        for term_length in range(2, 8):
            modes = rng.integers(0, n_modes, size=term_length, dtype=np.int32)
            types = rng.integers(0, 2, size=term_length, dtype=np.int32)
            loop_weights = (means[modes] + 1j * (-2 * types + 1) * means[n_modes + modes]) / np.sqrt(2.0)
            pair_weights = quadratic[
                types[:, None] + 2 * types[None, :],
                modes[:, None],
                modes[None, :],
            ]
            expected = 0.0 + 0.0j
            for matching in _loop_matchings(list(range(term_length))):
                product = 1.0 + 0.0j
                for first, second in matching:
                    product *= loop_weights[first] if first == second else pair_weights[first, second]
                expected += product

            obtained = wick_moment_subset_dp(
                jnp.asarray(modes), jnp.asarray(types), n_modes,
                jnp.asarray(quadratic), jnp.asarray(means),
            )
            np.testing.assert_allclose(np.asarray(obtained), expected, rtol=1e-11, atol=1e-11)

    def test_is_differentiable(self):
        modes = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
        types = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
        means = jnp.asarray([0.2, -0.1, 0.3, 0.4])
        quadratic = jnp.arange(16, dtype=jnp.float64).reshape(4, 2, 2) / 10.0
        derivative = jax.grad(
            lambda scale: jnp.real(
                wick_moment_subset_dp(modes, types, 2, scale * quadratic, scale * means)
            )
        )(jnp.asarray(1.0))
        self.assertTrue(np.isfinite(float(derivative)))


if __name__ == "__main__":
    unittest.main()
