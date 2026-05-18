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

DATASET="${DATASET:-medQA}"
SPLIT="${SPLIT:-test}"
LINKER="${LINKER:-metamap}"
LIMIT="${LIMIT:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rq3_failure_attribution}"
PREDICTIONS="${PREDICTIONS:-outputs/model_sweeps/full_medqa_test_vllm_optimized/Qwen__Qwen3_5-4B_medQA_test_non_thinking.jsonl}"
CONDITIONS="${CONDITIONS:-question_grounded correct_answer_grounded all_options_grounded any_option_grounded gold_path_available gold_has_name gold_has_definition gold_has_semantic_type}"

mkdir -p "${OUTPUT_DIR}/diagnostics"

if [[ ! -s "${PREDICTIONS}" ]]; then
  echo "Prediction file not found: ${PREDICTIONS}" >&2
  echo "Set PREDICTIONS=/path/to/model_predictions.jsonl or run RQ2 first." >&2
  exit 2
fi

limit_args=()
if [[ "${LIMIT}" != "0" ]]; then
  limit_args+=(--limit "${LIMIT}")
fi

run_diag() {
  local experiment="$1"
  local script="$2"
  local stem="${experiment}_${DATASET}_${SPLIT}_${LINKER}"
  echo "+ Running ${experiment}"
  uv run python3 "${script}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --linker "${LINKER}" \
    --output-jsonl "${OUTPUT_DIR}/diagnostics/${stem}.jsonl" \
    --summary-csv "${OUTPUT_DIR}/diagnostics/${stem}.summary.csv" \
    "${limit_args[@]}"
}

run_diag "exp1_evidence_availability" "src/experiments/exp1_evidence_availability.py"
run_diag "exp2_answer_grounding" "src/experiments/exp2_answer_grounding.py"
run_diag "exp3_retrieval_faithfulness" "src/experiments/exp3_retrieval_faithfulness.py"

condition_args=()
read -r -a condition_list <<< "${CONDITIONS}"
for condition in "${condition_list[@]}"; do
  condition_args+=(--condition "${condition}")
done

mapfile -t diagnostic_files < <(find "${OUTPUT_DIR}/diagnostics" -name '*.jsonl' -type f | sort)

echo "+ Running conditioned model metrics"
uv run python3 src/model_sweeps/conditioned_metrics.py \
  --predictions "${PREDICTIONS}" \
  --diagnostics "${diagnostic_files[@]}" \
  "${condition_args[@]}" \
  --output-csv "${OUTPUT_DIR}/conditioned_metrics.csv"

echo "RQ3 outputs written under ${OUTPUT_DIR}"
