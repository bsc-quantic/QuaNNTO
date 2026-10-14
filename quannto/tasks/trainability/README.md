# Finite-size trainability campaign

The three Python modules in this package preserve the production protocol
version 4. Commands, seeds, stored campaigns and their scientific scope are
in [docs/reproducibility.md](../../../docs/reproducibility.md).

From the repository root:

```bash
export QUANNTO_RESULTS_DIR="$PWD/results/local/trainability_smoke"
python -m quannto.tasks.trainability.trainability --quick --resume
```

For a complete manuscript-profile campaign, omit `--quick`. Workers execute in
separate processes and checkpoint each completed trial. `--resume` checks the
recorded signature before reuse. `--aggregate-only` rebuilds statistics from
matching checkpoints; it does not recompute gradients. `--allow-partial` is an
explicit choice and does not certify a complete campaign.

`run_trainability_campaign.sh` runs the near-identity and legacy-uniform
ensembles. `run_trainability_mn5_extended.sh` is an optional larger profile and
is not the profile used for the paper. It has no site-specific scheduler
configuration. The manuscript backend is `loop_matchings`.
