# Paired random-initialization fidelity baselines

The quantum-application tables can be regenerated with the trained-checkpoint
fidelities and a paired untrained baseline from the **exact random parameter
vector that initialized the same optimization run**.

The baseline is not a fresh draw: it is read from each run's
`training_records/<model>.json` under `initialization.parameters`. This makes the
comparison strictly paired:

- cat-state engineering: `F_init` vs. trained `F` for the single target state;
- cubic-phase synthesis: mean `F_test_init` on the 49 unseen midpoint coherent
  inputs vs. trained `F_train`, `F_test`, and `F_test,min`.

The state fidelities are reconstructed independently in Fock space using the
same evaluator as the trained fidelities. The trained checkpoint keeps the
strict moment/norm/fidelity convergence checks used for the manuscript. The
random baseline is certified directly for normalized-state fidelity and raw
projection norm; high-order moment convergence is not required because the
baseline column reports fidelity only.

From the repository root:

```bash
python -m quannto.tasks.quantum_tables --audit --output results/local/quantum_evaluation
python -m quannto.tasks.quantum_tables --task cat --output results/local/quantum_evaluation
python -m quannto.tasks.quantum_tables --task cubic --output results/local/quantum_evaluation
```

The generated files include:

- `quantum_tables.json`: full-precision results and cutoff diagnostics;
- `quantum_tables.csv`: compact numerical summary;
- `cat_table.tex`: cat-state table including `F_init`;
- `cubic_table.tex`: cubic-phase table including mean `F_test_init`;
- after the cat run, `python -m quannto.tasks.catstate_report --evaluation results/local/quantum_evaluation`
  exports the manuscript-ready cat table and checkpoint-integrity diagnostics.

If a random initialization does not satisfy two successive cutoff comparisons,
provide a longer common ladder with `--initial-cutoffs ...`. The default
initial-cutoff ladders are mode-count dependent to avoid impractical tensors
for the four-mode architecture.

Default cutoffs may be insufficient for a highly squeezed checkpoint. Extend
the trained and initial cutoff ladders as needed and retain the convergence
checks; see [reproducibility.md](reproducibility.md).
