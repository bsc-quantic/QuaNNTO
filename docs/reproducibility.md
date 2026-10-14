# PRR reproducibility and provenance

## Included sources

This integration uses the October 1 QuaNNTO snapshot as its primary source and
saved-model collection. It adds the production trainability campaign from
`trainability(1).zip`, the associated September 2 protocol/code patch, and the
cat parameter audit of October 1. The separate September 25 fidelity-baseline
patch supplies the exporter and original paired-initialization records that
were absent from the October snapshot.

`provenance/source_file_map.json` records the original hashes and destination of
the initial merged files; `provenance/integration.json` records subsequent
merges, intentional edits and source ZIP hashes. `provenance/artifacts.sha256`
checks the archived numerical artifacts and paper figure sources. Code changes
are listed in `CHANGELOG.md`.

## Quantum benchmarks

`quannto/tasks/quantum_tables_manifest.json` selects the five cubic architectures
of revised Table II and eight cat architectures of Table III. Indices in the
manifest start at zero; manuscript indices start at one. The excluded old
three-layer cubic architecture has no supplied checkpoint and is documented in
`provenance/excluded_quantum_architectures.json`. It is not a missing result of
the revised paper and is not replaced by a fabricated model.

A training record, its dataset fingerprints, selected parameters, Gaussian
operators and loss trajectory belong to the same checkpoint. Historical `.npy`
and newer binary-NPY files named `.txt` can coexist; the manifest explicitly
selects the intended trajectory. Numeric arrays are read without pickle. Old
`pickle_json` exports are retained as historical artifacts; they are not needed
to evaluate the quantum tables. Selection is by training loss, not fidelity.

Fidelity is the squared overlap with the normalized reduced output of mode 0.
For the cubic benchmark, train/test means use 50 real coherent amplitudes and
49 intervening midpoints. These are conditional output-state fidelities, not
process or Haar-average gate fidelities. Strawberry Fields `Vgate(0.2)` at
`hbar=2` corresponds to `gamma=0.2*sqrt(2)` in `[x,p]=i` coordinates.

The paired baseline uses the exact `initialization.parameters` vector from the
same training record (seed 42). It is not a new random initialization. Original
baseline values and their convergence settings are retained in
`results/prr/quantum_baselines/`.

The analytic training objective is cutoff free. Its independent Fock validation
has explicit norm, moment and fidelity convergence checks. The default cutoff
ladder is a starting point, not a guarantee: highly squeezed or multilayer
checkpoints can require substantially larger cutoffs and more memory. A
non-converged export returns exit code 2. Do not remove that criterion to make a
run appear successful. For example:

```bash
python -m quannto.tasks.quantum_tables --audit
python -m quannto.tasks.quantum_tables --model cat_n1_single \
  --cutoffs 20 30 40 60 --output results/local/cat_single
python -m quannto.tasks.quantum_tables --model cubic_n2_l2 \
  --cutoffs 40 60 80 100 120 160 --initial-cutoffs 40 60 80 100 120 160 \
  --output results/local/cubic_two_layers
```

Re-evaluation must pass convergence and reproduce the selected loss before
being treated as a replacement for the archived results. Extending the cutoff
ladder does not retrain or change the checkpoint. Use `--max-memory-mib` to set
the explicit Fock workspace bound; it is an estimate, not a hard process limit.

## Trainability

The canonical paper campaign is:

`results/prr/trainability/barren_plateaus/bp_scalar_identity_energy_controlled_T64_v4_manuscript_c0739531ce/`

It has 20 unique configurations, 64 independent initializations per
configuration, one fixed 32-input dataset, and 25 plotted scan points because
some configurations participate in more than one sweep. Dataset seed is 1234;
parameter seed is 2026. The original parameters and full gradients are in the
NPZ checkpoints; the CSV files contain trial, parameter and selected-probe
statistics. Confidence intervals use 1,000 trial-bootstrap replicates.

The historical directory name `energy_controlled` denotes the initialization
ensemble; it does not assert a fixed energy per mode. The active-coordinate
widths scale as `1/sqrt(L)` while passive-angle widths remain fixed. Statements
are specific to the cost, sizes and ensemble; they do not exclude an asymptotic
barren plateau. The ideal subtraction branch denominator is not an experimental
click probability.

The legacy-uniform campaign and incomplete historical v3 request are retained
alongside the production run with their original completion/status records.
They do not substitute for the canonical campaign. New smoke runs are written
to `results/local/`, not into this archive.

To recompute from random initializations, use a fresh results directory:

```bash
export QUANNTO_RESULTS_DIR="$PWD/results/local/trainability_full"
python -m quannto.tasks.trainability.trainability --resume
```

This is the full numerical campaign and can be expensive. The manuscript uses
`loop_matchings`; the optional `subset_dp` backend computes the same ordered
Wick sum, but choosing it constitutes a separately recorded run.

## Cat diagnostics

Archived parity and degree-2–12 hierarchy results are in `results/prr/`.
The hierarchy sweep examines a two-coordinate slice; the later parameter audit
examines the full trained coordinates of three near-odd checkpoints. Nearly
flat Hessian directions do not certify a strict/global minimum. Sharing 9 of
15 moments does not make ideal even and odd cats indistinguishable under the
loss, and convergence to a nonzero loss does not provide a fidelity certificate.
The diagnostic refinements remain separate from the trained checkpoints.

## Validation boundaries

The release validation report lists the checks actually executed. Archived
results are preserved rather than replaced by smoke tests. The packaging check
starts from a fresh extraction and checks that no original machine path is
required to load the data. TensorFlow preprocessing, GPU execution, cluster
launches and full basinhopping retraining are outside the release verification.
