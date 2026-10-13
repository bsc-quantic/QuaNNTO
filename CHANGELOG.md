# Integration release 0.1.1 — 2026-10-13

- Integrate the October QuaNNTO models, production trainability data, cat parity
  and hierarchy sweep, full-parameter cat audit, and paired random fidelity
  baseline patch in one source tree.
- Restore the multilayer Gaussian-control coefficient fix and Gaussian-core
  helper used by trainability. Retain the October training API and original
  loop-perfect-matching backend. Add the separately selectable exact subset-DP
  backend with its independent Wick-sum regression checks.
- Remove shared mutable preprocessor defaults. Invalidate architecture caches
  from the previous compiler version without distributing generated caches.
- Move numerical artifacts out of package code, update active data paths, make
  diagnostic modules import safe, and preserve historical code/figures separately.
- Correct the quantum manifest to include all five cubic and eight cat rows,
  including the repeated auxiliary-mode subtraction and paired initial baselines.
- Add an archived-artifact verifier, dependency lock, package-data declaration,
  migration notes and automated tests. Clear notebook execution output.

Saved model parameters, operators, datasets, trajectories and original
trainability arrays are preserved byte for byte. No model is retrained, no
checkpoint is selected by fidelity, and diagnostic refinements do not overwrite
manuscript checkpoints. The scientific manuscript and its bibliography are not
edited in this software release.
