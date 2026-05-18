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
OUTPUT_DIR="${OUTPUT_DIR:-outputs/model_sweeps/downstream_umls_probe}"
RUN_MEDGEMMA="${RUN_MEDGEMMA:-1}"
RUN_QWEN="${RUN_QWEN:-0}"
MOCK_MODEL="${MOCK_MODEL:-0}"
CONDITIONS="${CONDITIONS:-no_umls relation_only_evidence metadata_rich_evidence q_to_a_path_evidence random_same_size_evidence shuffled_answer_evidence gold_only_evidence distractor_contrast_evidence oracle_gold_evidence}"

MEDGEMMA_MODEL_KEY="${MEDGEMMA_MODEL_KEY:-MedGemma-1.5-4B-it}"
MEDGEMMA_MODEL_ID="${MEDGEMMA_MODEL_ID:-google/medgemma-1.5-4b-it}"
MEDGEMMA_DTYPE="${MEDGEMMA_DTYPE:-bfloat16}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
NUM_SHUFFLES="${NUM_SHUFFLES:-1}"
SEED="${SEED:-42}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.5-4B}"
QWEN_MODE="${QWEN_MODE:-non_thinking}"
QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-256}"
QWEN_TIMEOUT_S="${QWEN_TIMEOUT_S:-300}"

mkdir -p "${OUTPUT_DIR}/medgemma" "${OUTPUT_DIR}/qwen"

condition_args=()
read -r -a condition_list <<< "${CONDITIONS}"
for condition in "${condition_list[@]}"; do
  condition_args+=(--condition "${condition}")
done

if [[ "${RUN_MEDGEMMA}" == "1" ]]; then
  model_args=(
    --model-key "${MEDGEMMA_MODEL_KEY}"
    --model-id "${MEDGEMMA_MODEL_ID}"
    --dtype "${MEDGEMMA_DTYPE}"
  )
  if [[ "${MOCK_MODEL}" == "1" ]]; then
    model_args=(--mock-model --model-key MockLogits --model-id mock-local-logits)
  fi
  echo "+ Running MedGemma downstream UMLS probe"
  uv run --extra models python3 src/model_sweeps/downstream_umls_probe.py \
    --backend medgemma \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --linker "${LINKER}" \
    --limit "${LIMIT}" \
    "${model_args[@]}" \
    --num-samples "${NUM_SAMPLES}" \
    --num-shuffles "${NUM_SHUFFLES}" \
    --seed "${SEED}" \
    "${condition_args[@]}" \
    --output-jsonl "${OUTPUT_DIR}/medgemma/${DATASET}_${SPLIT}_${LINKER}_medgemma_probe.jsonl" \
    --summary-csv "${OUTPUT_DIR}/medgemma/${DATASET}_${SPLIT}_${LINKER}_medgemma_probe.summary.csv"
fi

if [[ "${RUN_QWEN}" == "1" ]]; then
  echo "+ Running Qwen downstream UMLS probe; requires vLLM at ${OPENAI_BASE_URL}"
  uv run python3 src/model_sweeps/downstream_umls_probe.py \
    --backend qwen \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --linker "${LINKER}" \
    --limit "${LIMIT}" \
    --qwen-model "${QWEN_MODEL}" \
    --qwen-mode "${QWEN_MODE}" \
    --base-url "${OPENAI_BASE_URL}" \
    --api-key "${OPENAI_API_KEY}" \
    --timeout-s "${QWEN_TIMEOUT_S}" \
    --max-tokens "${QWEN_MAX_TOKENS}" \
    "${condition_args[@]}" \
    --output-jsonl "${OUTPUT_DIR}/qwen/${DATASET}_${SPLIT}_${LINKER}_qwen_probe.jsonl" \
    --summary-csv "${OUTPUT_DIR}/qwen/${DATASET}_${SPLIT}_${LINKER}_qwen_probe.summary.csv"
else
  echo "Skipping Qwen probe. Set RUN_QWEN=1 after starting a vLLM endpoint."
fi

echo "Downstream probe outputs written under ${OUTPUT_DIR}"
