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
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

DATASET="${DATASET:-medQA}"
SPLIT="${SPLIT:-test}"
LINKER="${LINKER:-metamap}"
LIMIT="${LIMIT:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rq2_llm_evidence_use}"
RUN_MEDGEMMA="${RUN_MEDGEMMA:-1}"
RUN_QWEN="${RUN_QWEN:-0}"
PROMPT_CONDITION="${PROMPT_CONDITION:-no_umls}"
MOCK_MODEL="${MOCK_MODEL:-0}"

MEDGEMMA_MODEL_KEY="${MEDGEMMA_MODEL_KEY:-MedGemma-1.5-4B-it}"
MEDGEMMA_MODEL_ID="${MEDGEMMA_MODEL_ID:-google/medgemma-1.5-4b-it}"
MEDGEMMA_DTYPE="${MEDGEMMA_DTYPE:-bfloat16}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
NUM_SHUFFLES="${NUM_SHUFFLES:-1}"
SEED="${SEED:-42}"

QWEN_MODEL_LIST="${QWEN_MODEL_LIST:-Qwen/Qwen3.5-4B}"
QWEN_MODES="${QWEN_MODES:-non_thinking}"
QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-256}"
QWEN_CONCURRENCY="${QWEN_CONCURRENCY:-1}"
QWEN_TIMEOUT_S="${QWEN_TIMEOUT_S:-300}"

mkdir -p "${OUTPUT_DIR}/medgemma" "${OUTPUT_DIR}/qwen"

slugify_model() {
  local model="$1"
  echo "${model//\//__}" | tr '.' '_'
}

if [[ "${PROMPT_CONDITION}" != "no_umls" ]]; then
  echo "PROMPT_CONDITION=${PROMPT_CONDITION}: delegating to the downstream UMLS probe wrapper."
  CONDITIONS="${PROMPT_CONDITION}" \
  RUN_MEDGEMMA="${RUN_MEDGEMMA}" \
  RUN_QWEN="${RUN_QWEN}" \
  MOCK_MODEL="${MOCK_MODEL}" \
  DATASET="${DATASET}" \
  SPLIT="${SPLIT}" \
  LINKER="${LINKER}" \
  LIMIT="${LIMIT}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  bash scripts/run_downstream_umls_probe.sh
  exit 0
fi

if [[ "${RUN_MEDGEMMA}" == "1" ]]; then
  echo "+ Running full-model MedGemma local logits for RQ2"
  model_args=(
    --model-key "${MEDGEMMA_MODEL_KEY}"
    --model-id "${MEDGEMMA_MODEL_ID}"
    --dtype "${MEDGEMMA_DTYPE}"
  )
  if [[ "${MOCK_MODEL}" == "1" ]]; then
    echo "+ MOCK_MODEL=1: using deterministic mock logits instead of loading MedGemma"
    model_args=(
      --model-key "MockLogits"
      --model-id "mock-local-logits"
      --mock-model
    )
  fi
  uv run --extra models python3 src/model_sweeps/local_logits_sweep.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --linker "${LINKER}" \
    "${model_args[@]}" \
    --limit "${LIMIT}" \
    --num-samples "${NUM_SAMPLES}" \
    --num-shuffles "${NUM_SHUFFLES}" \
    --seed "${SEED}" \
    --prompt-condition "${PROMPT_CONDITION}" \
    --output-dir "${OUTPUT_DIR}/medgemma"
fi

if [[ "${RUN_QWEN}" == "1" ]]; then
  read -r -a qwen_models <<< "${QWEN_MODEL_LIST}"
  read -r -a qwen_modes <<< "${QWEN_MODES}"
  for model in "${qwen_models[@]}"; do
    model_slug="$(slugify_model "${model}")"
    for mode in "${qwen_modes[@]}"; do
      stem="${model_slug}_${DATASET}_${SPLIT}_${mode}"
      echo "+ Running Qwen sweep for ${model} ${mode}; requires vLLM at ${OPENAI_BASE_URL}"
      uv run python3 src/model_sweeps/qwen_vllm_sweep.py \
        --dataset "${DATASET}" \
        --split "${SPLIT}" \
        --linker "${LINKER}" \
        --model "${model}" \
        --mode "${mode}" \
        --limit "${LIMIT}" \
        --max-tokens "${QWEN_MAX_TOKENS}" \
        --timeout-s "${QWEN_TIMEOUT_S}" \
        --concurrency "${QWEN_CONCURRENCY}" \
        --output-jsonl "${OUTPUT_DIR}/qwen/${stem}.jsonl" \
        --summary-csv "${OUTPUT_DIR}/qwen/${stem}.summary.csv" \
        --resume
    done
  done
fi

echo "RQ2 outputs written under ${OUTPUT_DIR}"
