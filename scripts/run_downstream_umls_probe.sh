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
export OPENAI_API_KEY="${VLLM_API_KEY:-EMPTY}"
UV_BIN="${UV_BIN:-/home/molfetta/.local/bin/uv}"

DATASET="${DATASET:-medQA}"
SPLIT="${SPLIT:-test}"
LINKER="${LINKER:-metamap}"
LIMIT="${LIMIT:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/model_sweeps/downstream_umls_probe}"
ITEM_ID_FILE="${ITEM_ID_FILE:-}"
TRIPLETS_ZIP="${TRIPLETS_ZIP:-data/UMLS/triplets.csv.zip}"
RELATION_SCOPE="${RELATION_SCOPE:-item_cuis}"
RUN_MEDGEMMA="${RUN_MEDGEMMA:-1}"
RUN_QWEN="${RUN_QWEN:-0}"
MOCK_MODEL="${MOCK_MODEL:-0}"
DRY_RUN="${DRY_RUN:-0}"
CONDITIONS="${CONDITIONS:-no_umls relation_only_evidence q_to_a_path_evidence metadata_rich_evidence}"
MAX_ITEMS_PER_CONDITION="${MAX_ITEMS_PER_CONDITION:-${MAX_EXPERIMENT_INSTANCES:-0}}"
REQUIRE_Q_TO_A_PATH="${REQUIRE_Q_TO_A_PATH:-0}"
EVIDENCE_CACHE_JSONL="${EVIDENCE_CACHE_JSONL:-}"
REQUIRE_EVIDENCE_CACHE="${REQUIRE_EVIDENCE_CACHE:-0}"

MEDGEMMA_MODEL_KEY="${MEDGEMMA_MODEL_KEY:-MedGemma-1.5-4B-it}"
MEDGEMMA_MODEL_ID="${MEDGEMMA_MODEL_ID:-google/medgemma-1.5-4b-it}"
MEDGEMMA_DTYPE="${MEDGEMMA_DTYPE:-bfloat16}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
NUM_SHUFFLES="${NUM_SHUFFLES:-1}"
SEED="${SEED:-42}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.5-4B}"
QWEN_MODE="${QWEN_MODE:-non_thinking}"
QWEN_TIMEOUT_S="${QWEN_TIMEOUT_S:-300}"
QWEN_TEMPERATURE="${QWEN_TEMPERATURE:-0.0}"
QWEN_TOP_P="${QWEN_TOP_P:-1.0}"
QWEN_TOP_K="${QWEN_TOP_K:-20}"
QWEN_PRESENCE_PENALTY="${QWEN_PRESENCE_PENALTY:-0.0}"
QWEN_SEED="${QWEN_SEED:-42}"
QWEN_CONCURRENCY="${QWEN_CONCURRENCY:-1}"
QWEN_THINKING_TOKEN_BUDGET="${QWEN_THINKING_TOKEN_BUDGET:-}"
if [[ "${QWEN_MODE}" == "thinking" ]]; then
  QWEN_THINKING_TOKEN_BUDGET="${QWEN_THINKING_TOKEN_BUDGET:-4096}"
fi
QWEN_ANSWER_TOKEN_BUDGET="${QWEN_ANSWER_TOKEN_BUDGET:-64}"
_default_max_tokens=32
if [[ "${QWEN_MODE}" == "thinking" ]]; then
  _default_max_tokens=$((QWEN_THINKING_TOKEN_BUDGET + QWEN_ANSWER_TOKEN_BUDGET))
fi
QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-${_default_max_tokens}}"
RESUME="${RESUME:-1}"

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUTPUT_DIR}/medgemma" "${OUTPUT_DIR}/qwen"
fi

dry_run_args=()
if [[ "${DRY_RUN}" == "1" ]]; then
  dry_run_args+=(--dry-run)
fi

condition_args=()
read -r -a condition_list <<< "${CONDITIONS}"
for condition in "${condition_list[@]}"; do
  condition_args+=(--condition "${condition}")
done

subset_args=()
if [[ -n "${ITEM_ID_FILE}" ]]; then
  subset_args+=(--item-id-file "${ITEM_ID_FILE}")
fi

filter_args=()
if [[ "${REQUIRE_Q_TO_A_PATH}" == "1" ]]; then
  filter_args+=(--require-q-to-a-path)
fi
if [[ -n "${MAX_ITEMS_PER_CONDITION}" && "${MAX_ITEMS_PER_CONDITION}" != "0" ]]; then
  filter_args+=(--max-items-per-condition "${MAX_ITEMS_PER_CONDITION}")
fi
if [[ -n "${EVIDENCE_CACHE_JSONL}" ]]; then
  filter_args+=(--evidence-cache-jsonl "${EVIDENCE_CACHE_JSONL}")
fi
if [[ "${REQUIRE_EVIDENCE_CACHE}" == "1" ]]; then
  filter_args+=(--require-evidence-cache)
fi

relation_args=(--triplets-zip "${TRIPLETS_ZIP}" --relation-scope "${RELATION_SCOPE}")

thinking_budget_args=()
if [[ -n "${QWEN_THINKING_TOKEN_BUDGET}" ]]; then
  thinking_budget_args+=(--thinking-token-budget "${QWEN_THINKING_TOKEN_BUDGET}")
fi

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
  "${UV_BIN}" run --extra models python3 src/model_sweeps/downstream_umls_probe.py \
    --backend medgemma \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --linker "${LINKER}" \
    --limit "${LIMIT}" \
    "${subset_args[@]}" \
    "${filter_args[@]}" \
    "${relation_args[@]}" \
    "${model_args[@]}" \
    --num-samples "${NUM_SAMPLES}" \
    --num-shuffles "${NUM_SHUFFLES}" \
    --seed "${SEED}" \
    "${dry_run_args[@]}" \
    "${condition_args[@]}" \
    --output-jsonl "${OUTPUT_DIR}/medgemma/${DATASET}_${SPLIT}_${LINKER}_medgemma_probe.jsonl" \
    --summary-csv "${OUTPUT_DIR}/medgemma/${DATASET}_${SPLIT}_${LINKER}_medgemma_probe.summary.csv"
fi

if [[ "${RUN_QWEN}" == "1" ]]; then
  qwen_model_slug="${QWEN_MODEL//\//__}"
  qwen_model_slug="${qwen_model_slug//./_}"
  resume_args=()
  if [[ "${RESUME}" == "1" ]]; then
    resume_args+=(--resume)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ Dry-run Qwen downstream UMLS probe; no vLLM request will be sent"
  else
    echo "+ Running Qwen downstream UMLS probe; requires vLLM at ${OPENAI_BASE_URL}"
  fi
  "${UV_BIN}" run python3 src/model_sweeps/downstream_umls_probe.py \
    --backend qwen \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --linker "${LINKER}" \
    --limit "${LIMIT}" \
    "${subset_args[@]}" \
    "${filter_args[@]}" \
    "${relation_args[@]}" \
    --qwen-model "${QWEN_MODEL}" \
    --qwen-mode "${QWEN_MODE}" \
    --base-url "${OPENAI_BASE_URL}" \
    --api-key "${OPENAI_API_KEY}" \
    --timeout-s "${QWEN_TIMEOUT_S}" \
    --max-tokens "${QWEN_MAX_TOKENS}" \
    --temperature "${QWEN_TEMPERATURE}" \
    --top-p "${QWEN_TOP_P}" \
    --top-k "${QWEN_TOP_K}" \
    --presence-penalty "${QWEN_PRESENCE_PENALTY}" \
    "${thinking_budget_args[@]}" \
    --seed "${QWEN_SEED}" \
    --concurrency "${QWEN_CONCURRENCY}" \
    "${resume_args[@]}" \
    "${dry_run_args[@]}" \
    "${condition_args[@]}" \
    --output-jsonl "${OUTPUT_DIR}/qwen/${DATASET}_${SPLIT}_${LINKER}_${qwen_model_slug}_${QWEN_MODE}_qwen_probe.jsonl" \
    --summary-csv "${OUTPUT_DIR}/qwen/${DATASET}_${SPLIT}_${LINKER}_${qwen_model_slug}_${QWEN_MODE}_qwen_probe.summary.csv"
else
  echo "Skipping Qwen probe. Set RUN_QWEN=1 after starting a vLLM endpoint."
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Downstream probe dry-run complete; planned outputs are under ${OUTPUT_DIR}"
else
  echo "Downstream probe outputs written under ${OUTPUT_DIR}"
fi
