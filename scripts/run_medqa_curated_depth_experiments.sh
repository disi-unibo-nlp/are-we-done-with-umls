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
LINKER="${LINKER:-spacy}"
LIMIT="${LIMIT:-0}"
ANNOTATIONS_JSONL="${ANNOTATIONS_JSONL:-outputs/operational_umls/curated_depth/medqa_spacy_curated_annotations_silver_coverage_wide.jsonl}"
CANDIDATE_JSONL="${CANDIDATE_JSONL:-outputs/operational_umls/curated_depth/medqa_spacy_curated_candidates_wide.jsonl}"
ITEM_ID_FILE="${ITEM_ID_FILE:-outputs/operational_umls/step1_subsets/medQA_test_spacy_Qwen__Qwen3_5-9B_non_thinking_baseline_wrong.jsonl}"
CURATED_EVIDENCE_CACHE_JSONL="${CURATED_EVIDENCE_CACHE_JSONL:-outputs/operational_umls/curated_depth/medqa_spacy_curated_evidence_cache_coverage_wide.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/operational_umls/step3_curated_predictions}"
QWEN_MODEL_LIST="${QWEN_MODEL_LIST:-Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B}"
QWEN_MODE="${QWEN_MODE:-non_thinking}"
QWEN_CONCURRENCY="${QWEN_CONCURRENCY:-1}"
QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-32}"
QWEN_TEMPERATURE="${QWEN_TEMPERATURE:-0.0}"
QWEN_TOP_P="${QWEN_TOP_P:-1.0}"
QWEN_TOP_K="${QWEN_TOP_K:-20}"
QWEN_PRESENCE_PENALTY="${QWEN_PRESENCE_PENALTY:-0.0}"
QWEN_SEED="${QWEN_SEED:-42}"
MAX_ITEMS_PER_CONDITION="${MAX_ITEMS_PER_CONDITION:-100}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"

DEPTHS="${DEPTHS:-2 3 4}"
CONDITIONS="${CONDITIONS:-}"
if [[ -z "${CONDITIONS}" ]]; then
  for depth in ${DEPTHS}; do
    CONDITIONS+="${CONDITIONS:+ }q_to_a_curated_depth_${depth}_evidence"
  done
fi

depth_args=()
for depth in ${DEPTHS}; do
  depth_args+=(--depth "${depth}")
done

candidate_args=()
if [[ -n "${CANDIDATE_JSONL}" ]]; then
  candidate_args+=(--candidate-jsonl "${CANDIDATE_JSONL}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[curated-depth] DRY_RUN=1: validating annotations and planning outputs"
fi

"${UV_BIN}" run python3 src/experiments/curated_evidence.py build-cache \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --linker "${LINKER}" \
  --limit "${LIMIT}" \
  --item-id-file "${ITEM_ID_FILE}" \
  "${depth_args[@]}" \
  --annotations-jsonl "${ANNOTATIONS_JSONL}" \
  "${candidate_args[@]}" \
  --output-jsonl "${CURATED_EVIDENCE_CACHE_JSONL}"

for model in ${QWEN_MODEL_LIST}; do
  echo "[curated-depth] >>> ${model}"
  RUN_MEDGEMMA=0 \
  RUN_QWEN=1 \
  DATASET="${DATASET}" \
  SPLIT="${SPLIT}" \
  LINKER="${LINKER}" \
  LIMIT="${LIMIT}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  ITEM_ID_FILE="${ITEM_ID_FILE}" \
  CONDITIONS="${CONDITIONS}" \
  RELATION_SCOPE="item_cuis" \
  REQUIRE_Q_TO_A_PATH=1 \
  EVIDENCE_CACHE_JSONL="${CURATED_EVIDENCE_CACHE_JSONL}" \
  REQUIRE_EVIDENCE_CACHE=1 \
  MAX_ITEMS_PER_CONDITION="${MAX_ITEMS_PER_CONDITION}" \
  QWEN_MODEL="${model}" \
  QWEN_MODE="${QWEN_MODE}" \
  QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS}" \
  QWEN_TEMPERATURE="${QWEN_TEMPERATURE}" \
  QWEN_TOP_P="${QWEN_TOP_P}" \
  QWEN_TOP_K="${QWEN_TOP_K}" \
  QWEN_PRESENCE_PENALTY="${QWEN_PRESENCE_PENALTY}" \
  QWEN_SEED="${QWEN_SEED}" \
  QWEN_CONCURRENCY="${QWEN_CONCURRENCY}" \
  RESUME="${RESUME}" \
  DRY_RUN="${DRY_RUN}" \
  bash scripts/run_downstream_umls_probe.sh
  echo "[curated-depth] <<< ${model}"
done
