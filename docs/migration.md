# Applying the integrated release

The ZIP has one `QuaNNTO/` directory and contains no `.git` history, virtual
environment or compiled cache. Use its contents as the working tree on a new
branch of the existing GitHub checkout, preserving that checkout's `.git/`.
Commit and push from the checkout so the existing history and remote are kept.

Before copying, preserve any local modifications or results not contained in
this release. Review `git diff --stat` and `git status` before committing.
This archive does not claim to contain experiments that were never uploaded.

Paths migrated by this release:

| Old path | New path |
|---|---|
| `datasets/` | `data/datasets/` |
| `quannto/tasks/models/` | `results/models/` |
| Root cat diagnostic output directories | `results/prr/` |
| Root cat diagnostic implementations | `quannto/tasks/diagnostics/` |
| Cached architecture arrays under `quannto/cache/` | Runtime cache outside the tracked source tree |
| Dated Python backups | `archive/legacy_code/` |
| Historical plot exports | `figures/legacy/` |
| Final manuscript figure sources | `figures/paper/` |

The two original root cat diagnostic commands remain as compatibility wrappers.
All active helpers use the new paths. An old checkout overlay can leave obsolete
copies at the old paths; remove only the tracked obsolete copies after reviewing
the migration, not new local results. The GitHub branch should contain one
canonical working copy of the package and data.

The package version is 0.1.1. Architecture cache version 2 prevents reuse of stale
arrays after the multilayer Gaussian-control fix. The default backend remains
`loop_matchings`. Training protocols and model-selection conventions are not
changed by this reorganization.

For fresh training, set `QUANNTO_RESULTS_DIR` to `results/local/<campaign>`.
For a wheel installed away from the source tree, set `QUANNTO_REPO_ROOT` to the
extracted repository containing `data/` and `results/`.
