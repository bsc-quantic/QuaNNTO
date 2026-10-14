#!/usr/bin/env bash
set -uo pipefail

# Extended finite-size grid intended for a memory-rich MareNostrum allocation.
# Scheduler/resource directives remain site- and partition-specific; invoke
# this launcher from inside the allocation.  It reuses the same failure-
# tolerant supervisor and checkpoint format as the manuscript profile.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export QUANNTO_TRAINABILITY_PROFILE=extended_mn5
export QUANNTO_RESULTS_DIR="${QUANNTO_RESULTS_DIR:-${PWD}/revision_trainability_mn5}"

bash "${script_dir}/run_trainability_campaign.sh"
