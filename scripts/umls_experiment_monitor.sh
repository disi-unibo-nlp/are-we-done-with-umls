#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${LOAD_DOTENV:-1}" != "0" && -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

export HF_HOME="${HF_HOME:-/mnt/nvme/llms/}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/nvme/llms/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/nvme/llms/datasets}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/nvme/llms/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/mnt/nvme/llms/venvs/are-we-done-with-umls}"
UV_BIN="${UV_BIN:-/home/molfetta/.local/bin/uv}"

MODE="${MODE:-audit}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/operational_umls}"
STATUS_JSON="${STATUS_JSON:-${OUTPUT_ROOT}/status.json}"
STATUS_MD="${STATUS_MD:-${OUTPUT_ROOT}/missing_runs.md}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"

mkdir -p "${OUTPUT_ROOT}"

run_status() {
  "${UV_BIN}" run python3 src/experiments/umls_experiment_status.py \
    --output-json "${STATUS_JSON}" \
    --output-md "${STATUS_MD}" \
    "$@"
}

case "${MODE}" in
  audit)
    run_status
    ;;
  json)
    run_status --print-json
    ;;
  commands)
    run_status --commands
    ;;
  analyze)
    run_status
    if [[ -f "${OUTPUT_ROOT}/missing_runs.md" ]]; then
      sed -n '1,220p' "${OUTPUT_ROOT}/missing_runs.md"
    else
      echo "No operational status markdown yet: ${OUTPUT_ROOT}/missing_runs.md"
    fi
    ;;
  run-missing)
    mapfile -t commands < <(run_status --commands)
    if (( ${#commands[@]} == 0 )); then
      echo "No missing-run recovery command needed."
      exit 0
    fi
    mkdir -p "${LOG_DIR}"
    timestamp="$(date +%Y%m%d_%H%M%S)"
    unit="umls-operational-${timestamp}"
    log_file="${LOG_DIR}/${unit}.log"
    command_to_run="${commands[0]}"
    printf -v quoted_log '%q' "${log_file}"
    scheduled_command="${command_to_run} > ${quoted_log} 2>&1"
    systemd-run --user \
      --unit "${unit}" \
      --property "WorkingDirectory=${REPO_ROOT}" \
      bash -lc "${scheduled_command}"
    echo "Scheduled next recovery command as ${unit}.service"
    echo "Command: ${command_to_run}"
    echo "Log: ${log_file}"
    echo "Check: journalctl --user -u ${unit}.service -n 80 --no-pager"
    ;;
  *)
    echo "Unknown MODE=${MODE}; use audit, json, commands, analyze, or run-missing." >&2
    exit 2
    ;;
esac
