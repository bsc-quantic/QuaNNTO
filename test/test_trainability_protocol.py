import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from quannto.tasks.trainability import trainability
from quannto.tasks.trainability.trainability_protocol import (
    bootstrap_scalar_variance_ci, build_scan_points, dataset_sha256,
    make_fixed_scalar_dataset,
    matched_probe_indices, parameter_seed, parameter_types, sample_initial_parameters,
    parameter_diagnostic_rows, semilog_fit, loglog_fit, summarize_gradients,
    unique_architectures,
)
from quannto.tasks.trainability.trainability_supervisor import (
    classify_worker_outcome,
)


class TrainabilityProtocolTests(unittest.TestCase):
    def test_width_depth_and_ladder_resources_are_not_confounded(self):
        points = build_scan_points(
            widths=(2, 3, 4, 5), depths=(1, 2, 3),
            ladder_orders=(0, 1, 2, 3, 4),
            energy_scales=(0.5, 1.0, 2.0),
            ensemble="energy_controlled",
        )
        width = [point for point in points if point.sweep == "width"]
        for family, expected_k in (("gaussian", 0), ("subtracted", 1)):
            selected = [point for point in width if point.family == family]
            self.assertEqual([point.n_modes for point in selected], [2, 3, 4, 5])
            self.assertTrue(all(point.layers == 1 for point in selected))
            self.assertTrue(all(point.k_subtractions == expected_k for point in selected))

        depth = [point for point in points if point.sweep == "depth"]
        for family, expected_k in (("gaussian", 0), ("subtracted", 1)):
            selected = [point for point in depth if point.family == family]
            self.assertEqual([point.layers for point in selected], [1, 2, 3])
            self.assertTrue(all(point.n_modes == 2 for point in selected))
            self.assertTrue(all(point.k_subtractions == expected_k for point in selected))

        ladder = [point for point in points if point.sweep == "ladder"]
        self.assertEqual([point.k_subtractions for point in ladder], [0, 1, 2, 3, 4])
        self.assertTrue(all(point.n_modes == 4 and point.layers == 1 for point in ladder))

    def test_recorded_profiles_include_manuscript_and_extended_resources(self):
        manuscript = trainability.SCAN_PROFILES["manuscript"]
        self.assertEqual(manuscript["widths"], (2, 3, 4, 5))
        self.assertEqual(manuscript["depths"], (1, 2, 3))
        self.assertEqual(manuscript["ladder_orders"], (0, 1, 2, 3, 4))
        self.assertEqual(
            manuscript["energy_scales"], (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
        )
        extended = trainability.SCAN_PROFILES["extended_mn5"]
        self.assertIn(6, extended["widths"])
        self.assertIn(4, extended["depths"])
        self.assertIn(5, extended["ladder_orders"])
        points = build_scan_points(
            widths=extended["widths"],
            depths=extended["depths"],
            ladder_orders=extended["ladder_orders"],
            energy_scales=extended["energy_scales"],
            fixed_modes_for_depth=extended["fixed_modes_for_depth"],
            fixed_modes_for_ladder=extended["fixed_modes_for_ladder"],
            fixed_modes_for_energy=extended["fixed_modes_for_energy"],
            ensemble="energy_controlled",
        )
        self.assertTrue(any(p.n_modes == 2 and p.layers == 4 for p in points))
        self.assertTrue(any(p.n_modes == 5 and p.k_subtractions == 5 for p in points))

    def test_legacy_ensemble_does_not_silently_define_an_energy_sweep(self):
        points = build_scan_points(
            sweeps=("width", "energy"), ensemble="legacy_uniform",
        )
        self.assertFalse(any(point.sweep == "energy" for point in points))

    def test_duplicate_scan_memberships_share_one_raw_architecture(self):
        points = build_scan_points(
            sweeps=("width", "depth", "ladder", "energy"),
            widths=(2, 3, 4), depths=(1, 2), ladder_orders=(0, 1, 2),
            energy_scales=(1.0, 2.0), ensemble="energy_controlled",
        )
        unique = unique_architectures(points)
        self.assertLess(len(unique), len(points))
        keys = [point.architecture_key for point in unique]
        self.assertEqual(len(keys), len(set(keys)))

    def test_fixed_scalar_dataset_is_reproducible_and_anchored(self):
        first_x, first_y = make_fixed_scalar_dataset(seed=1234)
        second_x, second_y = make_fixed_scalar_dataset(seed=1234)
        np.testing.assert_array_equal(first_x, second_x)
        np.testing.assert_array_equal(first_y, second_y)
        np.testing.assert_array_equal(first_x, first_y)
        self.assertEqual(first_x.shape, (32, 1))
        self.assertEqual(float(first_x.min()), -0.75)
        self.assertEqual(float(first_x.max()), 0.75)
        self.assertEqual(
            dataset_sha256(first_x, first_y),
            dataset_sha256(second_x, second_y),
        )

    def test_parameter_seed_pairs_k_and_energy_scale_comparisons(self):
        sequence_a = parameter_seed(
            base_seed=2026, ensemble="energy_controlled",
            n_modes=4, layers=1, trial=7,
        )
        sequence_b = parameter_seed(
            base_seed=2026, ensemble="energy_controlled",
            n_modes=4, layers=1, trial=7,
        )
        self.assertEqual(sequence_a.entropy, sequence_b.entropy)
        rng_a = np.random.default_rng(sequence_a)
        rng_b = np.random.default_rng(sequence_b)
        types = parameter_types(4, 1)
        params_a = sample_initial_parameters(
            rng=rng_a, types=types, layers=1,
            ensemble="energy_controlled", energy_scale=0.5,
        )
        params_b = sample_initial_parameters(
            rng=rng_b, types=types, layers=1,
            ensemble="energy_controlled", energy_scale=2.0,
        )
        type_array = np.asarray(types)
        np.testing.assert_allclose(
            params_a[type_array == "passive"],
            params_b[type_array == "passive"],
        )
        np.testing.assert_allclose(
            4.0 * params_a[type_array != "passive"],
            params_b[type_array != "passive"],
        )

    def test_depth_scaling_does_not_rescale_passive_parameters(self):
        rng_l1 = np.random.default_rng(11)
        rng_l2 = np.random.default_rng(11)
        types_l1 = parameter_types(2, 1)
        types_l2 = parameter_types(2, 2)
        params_l1 = sample_initial_parameters(
            rng=rng_l1, types=types_l1, layers=1,
            ensemble="energy_controlled",
        )
        params_l2 = sample_initial_parameters(
            rng=rng_l2, types=types_l2, layers=2,
            ensemble="energy_controlled",
        )[:len(types_l1)]
        types = np.asarray(types_l1)
        np.testing.assert_allclose(
            params_l1[types == "passive"], params_l2[types == "passive"],
        )
        np.testing.assert_allclose(
            params_l1[types != "passive"] / np.sqrt(2.0),
            params_l2[types != "passive"],
        )

    def test_legacy_uniform_matches_the_recorded_distribution(self):
        params = sample_initial_parameters(
            rng=np.random.default_rng(42), types=parameter_types(3, 1),
            layers=1, ensemble="legacy_uniform",
        )
        self.assertTrue(np.all((0.0 <= params) & (params < 1.0)))
        with self.assertRaisesRegex(ValueError, "does not define an energy-scale"):
            sample_initial_parameters(
                rng=np.random.default_rng(42), types=parameter_types(3, 1),
                layers=1, ensemble="legacy_uniform", energy_scale=2.0,
            )

    def test_gradient_summary_is_size_normalized_and_bootstrap_reproducible(self):
        gradients = np.asarray([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
        ])
        types = ("passive", "passive", "squeezing", "displacement", "displacement")
        first = summarize_gradients(
            gradients, types=types, bootstrap_repetitions=100,
            bootstrap_seed=91,
            probe_indices={"passive": 0, "squeezing": 2, "displacement": 3},
        )
        second = summarize_gradients(
            gradients, types=types, bootstrap_repetitions=100,
            bootstrap_seed=91,
            probe_indices={"passive": 0, "squeezing": 2, "displacement": 3},
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["median_per_param_var"], 5.0 / 3.0)
        self.assertLess(first["moment_identity_max_abs_error"], 1e-12)
        self.assertEqual(first["passive__num_parameters"], 2)
        self.assertEqual(first["squeezing__num_parameters"], 1)
        self.assertEqual(first["displacement__num_parameters"], 2)
        self.assertAlmostEqual(
            first["probe_squeezing__gradient_variance"], 5.0 / 3.0,
        )
        self.assertEqual(
            first["probe_squeezing__gradient_variance_ci95_low"],
            second["probe_squeezing__gradient_variance_ci95_low"],
        )

    def test_primary_probes_are_not_conflated_with_inactive_coordinates(self):
        base = np.arange(1.0, 6.0)
        gradients = np.column_stack((
            base,
            2.0 * base,
            -3.0 * base,
            np.zeros_like(base),
            np.zeros_like(base),
            np.zeros_like(base),
            np.zeros_like(base),
        ))
        types = (
            "passive", "squeezing", "displacement", "passive",
            "passive", "squeezing", "displacement",
        )
        summary = summarize_gradients(
            gradients,
            types=types,
            probe_indices={"passive": 0, "squeezing": 1, "displacement": 2},
            bootstrap_repetitions=50,
            bootstrap_seed=7,
        )
        self.assertEqual(summary["median_per_param_var"], 0.0)
        self.assertEqual(summary["empirically_inactive_parameter_count"], 4)
        self.assertAlmostEqual(
            summary["empirically_inactive_parameter_fraction"], 4.0 / 7.0,
        )
        self.assertGreater(summary["probe_passive__gradient_variance"], 0.0)
        self.assertGreater(summary["probe_squeezing__gradient_variance"], 0.0)
        self.assertGreater(summary["probe_displacement__gradient_variance"], 0.0)

    def test_scalar_variance_bootstrap_and_parameter_table_are_reproducible(self):
        values = np.asarray((1.0, 2.0, 4.0, 8.0, 16.0))
        first = bootstrap_scalar_variance_ci(values, repetitions=100, seed=13)
        second = bootstrap_scalar_variance_ci(values, repetitions=100, seed=13)
        self.assertEqual(first, second)
        gradients = np.column_stack((values, np.zeros_like(values)))
        rows = parameter_diagnostic_rows(
            gradients, types=("passive", "squeezing"),
        )
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0]["empirically_inactive"])
        self.assertTrue(rows[1]["empirically_inactive"])

    def test_partial_checkpoint_can_resume_or_finalize_every_trial_prefix(self):
        point = build_scan_points(
            sweeps=("width",), widths=(2,), ensemble="energy_controlled",
        )[0]
        trials = 3
        n_parameters = len(parameter_types(point.n_modes, point.layers))
        signature = trainability._configuration_signature(
            point,
            ensemble="energy_controlled",
            trials=trials,
            dataset_hash="abc",
            parameter_seed_value=2026,
            expectation_backend="loop_matchings",
        )
        arrays = {
            "parameters": np.ones((trials, n_parameters)),
            "gradients": np.ones((trials, n_parameters)),
            "losses": np.ones(trials),
            "gradient_norms": np.ones(trials),
            "mean_norms": np.ones(trials),
            "median_norms": np.ones(trials),
            "tiny_norm_fractions": np.zeros(trials),
            "nonpositive_norm_fractions": np.zeros(trials),
            "mean_core_energy_per_mode": np.ones(trials),
            "mean_core_total_energy": np.ones(trials),
            "mean_abs_outputs": np.ones(trials),
            "trial_small_gradient_fractions": np.zeros(trials),
        }
        gradcheck = {
            "gradcheck_indices": np.empty(0, dtype=np.int64),
            "gradcheck_autodiff": np.empty(0),
            "gradcheck_finite_difference": np.empty(0),
            "gradcheck_absolute_error": np.empty(0),
            "gradcheck_relative_error": np.empty(0),
            "gradcheck_passed": np.empty(0, dtype=np.int8),
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "partial.npz"
            for completed in (1, trials):
                payload = trainability._trial_payload(
                    signature=signature,
                    completed_trials=completed,
                    gradcheck=gradcheck,
                    include_completed_counter=True,
                    **arrays,
                )
                trainability._write_npz_atomic(path, payload)
                loaded = trainability._load_partial_checkpoint(
                    path,
                    expected_signature=signature,
                    expected_parameters=n_parameters,
                )
                self.assertEqual(int(loaded["completed_trials"]), completed)
                self.assertEqual(loaded["gradients"].shape[0], completed)

    def test_worker_exit_classification_is_explicit(self):
        self.assertEqual(classify_worker_outcome(0)["worker_status"], "completed")
        killed = classify_worker_outcome(-9)
        self.assertEqual(killed["signal"], "SIGKILL")
        self.assertTrue(killed["probable_oom"])
        ordinary = classify_worker_outcome(1)
        self.assertFalse(ordinary["probable_oom"])
        self.assertEqual(ordinary["failure_kind"], "worker_error")
        timeout = classify_worker_outcome(-15, timed_out=True)
        self.assertEqual(timeout["failure_kind"], "timeout")

    def test_matched_probe_indices_preserve_bloch_messiah_locations(self):
        probes = matched_probe_indices(4)
        types = parameter_types(4, 1)
        self.assertEqual(types[probes["passive"]], "passive")
        self.assertEqual(types[probes["squeezing"]], "squeezing")
        self.assertEqual(types[probes["displacement"]], "displacement")

    def test_exploratory_fit_helpers_recover_known_scalings(self):
        exponential = semilog_fit((1, 2, 3, 4), np.exp(-2.0 * np.arange(1, 5)))
        self.assertAlmostEqual(exponential["slope"], -2.0)
        self.assertAlmostEqual(exponential["r2_log_fit"], 1.0)
        power = loglog_fit((1, 2, 4, 8), np.asarray((1, 2, 4, 8)) ** -3.0)
        self.assertAlmostEqual(power["power_exponent"], -3.0)
        self.assertAlmostEqual(power["r2_loglog_fit"], 1.0)

    def test_reported_fits_use_probes_and_omit_ladder_order(self):
        rows = []
        for sweep in ("width", "ladder"):
            for size in (1.0, 2.0, 3.0):
                rows.append({
                    "sweep": sweep,
                    "family": "subtracted" if sweep == "width" else "ladder",
                    "size": size,
                    "mean_gaussian_core_photons_per_mode": size,
                    "probe_passive__gradient_variance": np.exp(-size),
                    "probe_squeezing__gradient_variance": np.exp(-2.0 * size),
                    "probe_displacement__gradient_variance": np.exp(-3.0 * size),
                    "median_per_param_var": 0.0,
                })
        fits = trainability._fit_sweeps(rows)
        self.assertIn("width:subtracted", fits["fits"])
        self.assertNotIn("ladder:ladder", fits["fits"])
        self.assertNotIn(
            "median_per_param_var", fits["fits"]["width:subtracted"],
        )


if __name__ == "__main__":
    unittest.main()
