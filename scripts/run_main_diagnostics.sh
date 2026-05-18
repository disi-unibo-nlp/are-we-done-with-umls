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

DATASETS="${DATASETS:-medQA medMCQA}"
SECONDARY_DATASETS="${SECONDARY_DATASETS:-pubmedQA}"
SPLITS="${SPLITS:-test}"
LINKERS="${LINKERS:-metamap spacy}"
LIMIT="${LIMIT:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/main}"

mkdir -p "${OUTPUT_DIR}"

limit_args=()
if [[ "${LIMIT}" != "0" ]]; then
  limit_args+=(--limit "${LIMIT}")
fi

run_py() {
  echo "+ uv run python3 $*"
  uv run python3 "$@"
}

run_case() {
  local experiment="$1"
  local script="$2"
  local dataset="$3"
  local split="$4"
  local linker="$5"
  local stem="${experiment}_${dataset}_${split}_${linker}"
  run_py "${script}" \
    --dataset "${dataset}" \
    --split "${split}" \
    --linker "${linker}" \
    --output-jsonl "${OUTPUT_DIR}/${stem}.jsonl" \
    --summary-csv "${OUTPUT_DIR}/${stem}.summary.csv" \
    "${limit_args[@]}"
}

read -r -a dataset_list <<< "${DATASETS}"
read -r -a secondary_dataset_list <<< "${SECONDARY_DATASETS}"
read -r -a split_list <<< "${SPLITS}"
read -r -a linker_list <<< "${LINKERS}"

for dataset in "${dataset_list[@]}"; do
  for split in "${split_list[@]}"; do
    for linker in "${linker_list[@]}"; do
      run_case "exp1_evidence_availability" "src/experiments/exp1_evidence_availability.py" "${dataset}" "${split}" "${linker}"
      run_case "exp2_answer_grounding" "src/experiments/exp2_answer_grounding.py" "${dataset}" "${split}" "${linker}"
      run_case "exp3_retrieval_faithfulness" "src/experiments/exp3_retrieval_faithfulness.py" "${dataset}" "${split}" "${linker}"
    done
  done
done

for dataset in "${secondary_dataset_list[@]}"; do
  for split in "${split_list[@]}"; do
    for linker in "${linker_list[@]}"; do
      run_case "exp1_evidence_availability" "src/experiments/exp1_evidence_availability.py" "${dataset}" "${split}" "${linker}"
      run_case "exp2_answer_grounding" "src/experiments/exp2_answer_grounding.py" "${dataset}" "${split}" "${linker}"
    done
  done
done

echo "Main diagnostic outputs written under ${OUTPUT_DIR}"
