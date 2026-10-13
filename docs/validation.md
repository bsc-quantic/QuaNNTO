# Release validation — 2026-10-13

Validation ran on CPU/Linux with Python 3.12.14 and the included dependency lock.

- 72 tests passed, zero failures and zero skips, from a fresh ZIP extraction and
  freshly installed environment (58.24 seconds for the final test suite).
- All 1,892 archived artifact hashes verified. All five cubic and eight cat
  checkpoints passed the dataset/trajectory/operator/parameter audit, and their
  saved prediction arrays reproduced the recorded training losses.
- Twenty production trainability configurations were complete, with all 1,280
  parameter vectors reproducible from the recorded seeds. Sixty derivative
  variances matched the raw gradients; all 16 recorded finite-difference checks
  passed (maximum absolute error 1.9254e-9).
- Fresh analytic reevaluation of four Gaussian/photon-subtracted one/two-layer
  configurations matched their stored first-trial gradients within 6.67e-16.
- The isolated trainability smoke completed all 11 reduced-grid configurations
  without failures. A separate short optimizer/save smoke completed successfully.
- Fresh parity reconstruction of all eight cat checkpoints matched the archived
  fidelities within 3.34e-16 and losses within 1.43e-13. The degree-2–12 hierarchy
  diagnostic reproduced its archived losses and odd-sector behavior.
- Paired trained/random-initial fidelity exports passed for the single-mode
  single-subtraction cat and two-mode two-layer cat. The latter required a trained
  cutoff of 160 to satisfy the strict consecutive norm/moment/fidelity checks.
- Source/wheel builds, wheel manifest inclusion, external data lookup, dependency
  consistency, active-module syntax and ZIP CRC checks passed.

These checks preserve the original training results. The full basinhopping
campaign, every cubic fidelity point at converged Fock cutoffs, TensorFlow data
preprocessing, GPU execution and cluster launchers were not rerun. The unchanged
numeric artifacts and scientific regression tests are not presented as those
additional computations. Third-party deprecation warnings occurred without
failed checks.

Machine-readable details are in `provenance/validation.json`.
