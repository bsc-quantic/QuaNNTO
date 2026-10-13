#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"
python_bin="${QUANNTO_PYTHON:-python3}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
"${python_bin}" -m pip check
"${python_bin}" -m quannto.benchmarks.prr
"${python_bin}" -m pytest -ra --tb=short
