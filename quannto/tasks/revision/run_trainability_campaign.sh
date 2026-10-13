#!/usr/bin/env bash
set -uo pipefail

# Run the two pre-registered initialization ensembles in separate processes so
# JAX/XLA returns all compiled memory between them.  The Python supervisor also
# isolates every architecture, checkpoints each trial, and continues after an
# OOM-killed or otherwise failed worker.

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
python_bin="${QUANNTO_PYTHON:-python3}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd "${repo_root}"

# Avoid the accidental duplicate execution that would otherwise compile the
# same JAX configurations twice and contend for the same checkpoints.
results_dir="${QUANNTO_RESULTS_DIR:-${repo_root}/revision_trainability_results}"
export QUANNTO_RESULTS_DIR="${results_dir}"
mkdir -p "${results_dir}/campaign_logs"
if command -v flock >/dev/null 2>&1; then
    lock_file="${results_dir}/campaign_logs/trainability_v4.lock"
    exec 9>"${lock_file}"
    if ! flock -n 9; then
        printf 'Another trainability campaign is already using %s\n' "${results_dir}" >&2
        exit 2
    fi
fi

profile="${QUANNTO_TRAINABILITY_PROFILE:-manuscript}"
failures=()

run_ensemble() {
    local ensemble="$1"
    local sweeps="$2"
    local log_file="${results_dir}/campaign_logs/trainability_v4_${profile}_${ensemble}.log"
    printf '[start] trainability %s (%s profile)\n' "${ensemble}" "${profile}"
    "${python_bin}" -m quannto.tasks.revision.trainability \
        --profile "${profile}" \
        --ensemble "${ensemble}" \
        --sweeps "${sweeps}" \
        --resume 2>&1 | tee "${log_file}"
    local exit_code=${PIPESTATUS[0]}
    if [[ ${exit_code} -ne 0 ]]; then
        failures+=("${ensemble}:${exit_code}")
        printf '[failed:%s] %s finished all possible configurations; continuing\n' \
            "${exit_code}" "${ensemble}"
    else
        printf '[done] trainability %s\n' "${ensemble}"
    fi
}

run_ensemble energy_controlled width,depth,ladder,energy
run_ensemble legacy_uniform width,depth,ladder

if [[ ${#failures[@]} -gt 0 ]]; then
    printf 'Trainability campaign completed with partial failures: %s\n' "${failures[*]}"
    exit 1
fi

printf 'Trainability campaign completed successfully. Results: %s\n' "${results_dir}"
