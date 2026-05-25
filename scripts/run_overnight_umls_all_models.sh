#!/usr/bin/env bash
# Run STEPS 1-3 for all Qwen3.5 models sequentially, starting and stopping
# a fresh vLLM instance for each model checkpoint.
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

VLLM_BIN="${VLLM_BIN:-/mnt/nvme/llms/venvs/spiral-benchmark-env/bin/vllm}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
VLLM_REASONING_CONFIG="${VLLM_REASONING_CONFIG:-{\"reasoning_start_str\":\"<think>\",\"reasoning_end_str\":\"</think>\"}}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---enable-prefix-caching --enable-chunked-prefill --kv-cache-dtype fp8 --no-enable-log-requests --disable-log-stats --performance-mode throughput}"

QWEN_MODEL_LIST="${QWEN_MODEL_LIST:-Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B Qwen/Qwen3.5-35B}"
QWEN_CONCURRENCY="${QWEN_CONCURRENCY:-32}"
LIMIT="${LIMIT:-0}"
LINKERS="${LINKERS:-metamap spacy}"
STEP23_LINKERS="${STEP23_LINKERS:-spacy}"
STEP1_SUBSET_FALLBACK_MODEL="${STEP1_SUBSET_FALLBACK_MODEL:-Qwen/Qwen3.5-9B}"
MAX_ITEMS_PER_CONDITION="${MAX_ITEMS_PER_CONDITION:-${MAX_EXPERIMENT_INSTANCES:-100}}"
STEP3_REQUIRE_Q_TO_A_PATH="${STEP3_REQUIRE_Q_TO_A_PATH:-1}"
VLLM_READY_TIMEOUT_S="${VLLM_READY_TIMEOUT_S:-300}"
RUN_STEPS="${RUN_STEPS:-2 3}"

LOG_DIR="${LOG_DIR:-outputs/operational_umls/logs}"
mkdir -p "${LOG_DIR}"

log() { printf '[overnight-umls] %s\n' "$*" | tee -a "${LOG_DIR}/overnight_run.log"; }

VLLM_PID=""

stop_vllm() {
  if [[ -n "${VLLM_PID}" ]]; then
    log "Stopping vLLM PID=${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
    VLLM_PID=""
    sleep 8
  fi
}

trap 'stop_vllm' EXIT INT TERM

start_vllm() {
  local model="$1"
  local slug="${model//\//__}"
  slug="${slug//./_}"
  mkdir -p "${LOG_DIR}"
  local vllm_log="${LOG_DIR}/vllm_${slug}_$(date +%Y%m%dT%H%M%SZ).log"
  local reasoning_args=()
  if [[ -n "${VLLM_REASONING_PARSER}" ]]; then
    reasoning_args+=(--reasoning-parser "${VLLM_REASONING_PARSER}")
  fi
  if [[ -n "${VLLM_REASONING_CONFIG}" ]]; then
    reasoning_args+=(--reasoning-config "${VLLM_REASONING_CONFIG}")
  fi

  log "Starting vLLM for ${model} (log: ${vllm_log})"

  # shellcheck disable=SC2086
  HF_HOME="${HF_HOME}" \
  "${VLLM_BIN}" serve "${model}" \
    --port "${VLLM_PORT}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
    "${reasoning_args[@]}" \
    ${VLLM_EXTRA_ARGS} \
    > "${vllm_log}" 2>&1 &
  VLLM_PID=$!

  local waited=0
  while ! curl -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    if (( waited >= VLLM_READY_TIMEOUT_S )); then
      log "ERROR: vLLM did not become ready within ${VLLM_READY_TIMEOUT_S}s. Check ${vllm_log}"
      stop_vllm
      exit 1
    fi
    sleep 5
    waited=$(( waited + 5 ))
  done
  log "vLLM ready after ${waited}s (PID=${VLLM_PID})"
}

run_steps() {
  local model="$1"
  local steps="${2:-1 2 3}"
  for step in ${steps}; do
    log "  STEP ${step} for ${model}"
    QWEN_MODEL_LIST="${model}" \
    STEP="${step}" \
    LIMIT="${LIMIT}" \
    LINKERS="${LINKERS}" \
    STEP23_LINKERS="${STEP23_LINKERS}" \
    STEP1_SUBSET_FALLBACK_MODEL="${STEP1_SUBSET_FALLBACK_MODEL}" \
    QWEN_CONCURRENCY="${QWEN_CONCURRENCY}" \
    MAX_ITEMS_PER_CONDITION="${MAX_ITEMS_PER_CONDITION}" \
    STEP3_REQUIRE_Q_TO_A_PATH="${STEP3_REQUIRE_Q_TO_A_PATH}" \
      bash scripts/run_operational_umls_experiments.sh
  done
}

log "=== Overnight UMLS run start: $(date) ==="
log "Models: ${QWEN_MODEL_LIST}"
log "LIMIT=${LIMIT}  CONCURRENCY=${QWEN_CONCURRENCY}"
log "STEP1_LINKERS=${LINKERS}"
log "STEP23_LINKERS=${STEP23_LINKERS}"
log "STEP1_SUBSET_FALLBACK_MODEL=${STEP1_SUBSET_FALLBACK_MODEL}"
log "MAX_ITEMS_PER_CONDITION=${MAX_ITEMS_PER_CONDITION}"
log "STEP3_REQUIRE_Q_TO_A_PATH=${STEP3_REQUIRE_Q_TO_A_PATH}"
log "STEPS=${RUN_STEPS}"

for model in ${QWEN_MODEL_LIST}; do
  log ""
  log ">>> Model: ${model}"
  start_vllm "${model}"
  run_steps "${model}" "${RUN_STEPS}"
  stop_vllm
  log "<<< Done: ${model}"
done

log ""
log "Building final tables across all models..."
LINKERS="${LINKERS}" STEP23_LINKERS="${STEP23_LINKERS}" STEP=tables bash scripts/run_operational_umls_experiments.sh

log "=== Overnight UMLS run complete: $(date) ==="
log "Outputs: outputs/operational_umls/"
