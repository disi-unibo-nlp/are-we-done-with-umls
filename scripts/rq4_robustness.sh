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
OUTPUT_DIR="${OUTPUT_DIR:-outputs/appendix/model_robustness}"
RUN_MEDGEMMA="${RUN_MEDGEMMA:-1}"
RUN_QWEN="${RUN_QWEN:-0}"
MOCK_MODEL="${MOCK_MODEL:-0}"

MEDGEMMA_MODEL_KEY="${MEDGEMMA_MODEL_KEY:-MedGemma-1.5-4B-it}"
MEDGEMMA_MODEL_ID="${MEDGEMMA_MODEL_ID:-google/medgemma-1.5-4b-it}"
MEDGEMMA_DTYPE="${MEDGEMMA_DTYPE:-bfloat16}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
NUM_SHUFFLES="${NUM_SHUFFLES:-4}"
SEED="${SEED:-42}"

QWEN_MODEL_LIST="${QWEN_MODEL_LIST:-Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B}"
QWEN_MODES="${QWEN_MODES:-non_thinking thinking}"
QWEN_NON_THINKING_MAX_TOKENS="${QWEN_NON_THINKING_MAX_TOKENS:-256}"
QWEN_THINKING_MAX_TOKENS="${QWEN_THINKING_MAX_TOKENS:-768}"
QWEN_CONCURRENCY="${QWEN_CONCURRENCY:-1}"
QWEN_TIMEOUT_S="${QWEN_TIMEOUT_S:-300}"

mkdir -p "${OUTPUT_DIR}/medgemma_option_shuffle" "${OUTPUT_DIR}/qwen_modes"

slugify_model() {
  local model="$1"
  echo "${model//\//__}" | tr '.' '_'
}

if [[ "${RUN_MEDGEMMA}" == "1" ]]; then
  echo "+ Running MedGemma option-order robustness sweep"
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
    --prompt-condition "no_umls_option_shuffle" \
    --output-dir "${OUTPUT_DIR}/medgemma_option_shuffle"
fi

if [[ "${RUN_QWEN}" == "1" ]]; then
  read -r -a qwen_models <<< "${QWEN_MODEL_LIST}"
  read -r -a qwen_modes <<< "${QWEN_MODES}"
  for model in "${qwen_models[@]}"; do
    model_slug="$(slugify_model "${model}")"
    for mode in "${qwen_modes[@]}"; do
      max_tokens="${QWEN_NON_THINKING_MAX_TOKENS}"
      if [[ "${mode}" == "thinking" ]]; then
        max_tokens="${QWEN_THINKING_MAX_TOKENS}"
      fi
      stem="${model_slug}_${DATASET}_${SPLIT}_${mode}"
      echo "+ Running Qwen robustness sweep for ${model} ${mode}; requires vLLM at ${OPENAI_BASE_URL}"
      uv run python3 src/model_sweeps/qwen_vllm_sweep.py \
        --dataset "${DATASET}" \
        --split "${SPLIT}" \
        --linker "${LINKER}" \
        --model "${model}" \
        --mode "${mode}" \
        --limit "${LIMIT}" \
        --max-tokens "${max_tokens}" \
        --timeout-s "${QWEN_TIMEOUT_S}" \
        --concurrency "${QWEN_CONCURRENCY}" \
        --output-jsonl "${OUTPUT_DIR}/qwen_modes/${stem}.jsonl" \
        --summary-csv "${OUTPUT_DIR}/qwen_modes/${stem}.summary.csv" \
        --resume
    done
  done
else
  echo "Skipping live Qwen runs. Set RUN_QWEN=1 after starting a vLLM OpenAI-compatible endpoint at OPENAI_BASE_URL."
  echo "Existing Qwen summaries remain under outputs/model_sweeps/full_medqa_test_vllm_optimized/."
fi

echo "RQ4 outputs written under ${OUTPUT_DIR}"
