#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-outputs/operational_umls/step1_subsets/medQA_test_spacy_Qwen__Qwen3_5-9B_non_thinking_baseline_wrong.jsonl}"
GRAPH_DISTANCE_MANIFEST="${GRAPH_DISTANCE_MANIFEST:-/tmp/medqa_graph_distance_manifest_${RUN_ID}.jsonl}"
GRAPH_DISTANCE_AUDIT_JSONL="${GRAPH_DISTANCE_AUDIT_JSONL:-/tmp/medqa_graph_distance_audit_${RUN_ID}.jsonl}"
EVIDENCE_CACHE_JSONL="${EVIDENCE_CACHE_JSONL:-/tmp/medqa_graph_distance_evidence_${RUN_ID}.jsonl}"
LOG_DIR="${LOG_DIR:-outputs/operational_umls/logs/medqa_graph_distance_${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/operational_umls/graph_distance_predictions}"
TABLES_DIR="${TABLES_DIR:-outputs/operational_umls/tables}"
mkdir -p "${LOG_DIR}"

log() { printf '[graph-distance] %s\n' "$*" | tee -a "${LOG_DIR}/run.log"; }

export HF_HOME="${HF_HOME:-/mnt/nvme/llms/}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/nvme/llms/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/nvme/llms/datasets}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/nvme/llms/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/mnt/nvme/llms/venvs/are-we-done-with-umls}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
export OPENAI_API_KEY="${VLLM_API_KEY:-EMPTY}"

VLLM_BIN="${VLLM_BIN:-/mnt/nvme/llms/venvs/spiral-benchmark-env/bin/vllm}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_READY_TIMEOUT_S="${VLLM_READY_TIMEOUT_S:-7200}"
VLLM_STOP_TIMEOUT_S="${VLLM_STOP_TIMEOUT_S:-180}"
QWEN_MODEL_LIST="${QWEN_MODEL_LIST:-Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B Qwen/Qwen3.5-35B-A3B}"
CONDITIONS="${CONDITIONS:-q_to_a_raw_shortest_path_evidence q_to_a_semantic_shortest_path_evidence q_to_a_topology_only_path_evidence}"
QWEN_CONCURRENCY="${QWEN_CONCURRENCY:-32}"
QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-32}"
MAX_PER_STRATUM="${MAX_PER_STRATUM:-100}"
MIN_STRATUM_SIZE="${MIN_STRATUM_SIZE:-30}"
GRAPH_DISTANCE_RELATION_SCOPE="${GRAPH_DISTANCE_RELATION_SCOPE:-full}"
BUILD_ONLY="${BUILD_ONLY:-0}"
REUSE_CACHE="${REUSE_CACHE:-0}"
WAIT_FOR_USER_UNITS="${WAIT_FOR_USER_UNITS:-}"
VLLM_PID=""

build_graph_distance_cache() {
  log "Building graph-distance cache from ${SOURCE_MANIFEST}"
  condition_args=()
  for condition in ${CONDITIONS}; do
    condition_args+=(--condition "${condition}")
  done
  /home/molfetta/are-we-done-with-umls/.venv/bin/python -m src.experiments.graph_distance_mechanism build-cache \
    --dataset medQA \
    --split test \
    --linker spacy \
    --item-id-file "${SOURCE_MANIFEST}" \
    --max-per-stratum "${MAX_PER_STRATUM}" \
    --min-stratum-size "${MIN_STRATUM_SIZE}" \
    --relation-scope "${GRAPH_DISTANCE_RELATION_SCOPE}" \
    --seed 42 \
    "${condition_args[@]}" \
    --audit-jsonl "${GRAPH_DISTANCE_AUDIT_JSONL}" \
    --manifest-jsonl "${GRAPH_DISTANCE_MANIFEST}" \
    --evidence-cache-jsonl "${EVIDENCE_CACHE_JSONL}" \
    | tee -a "${LOG_DIR}/cache_build.log"
  log "Graph-distance cache ready: cache=${EVIDENCE_CACHE_JSONL} manifest=${GRAPH_DISTANCE_MANIFEST}"
}

stop_vllm() {
  if [[ -z "${VLLM_PID}" ]]; then
    return
  fi
  log "Stopping vLLM PID=${VLLM_PID}"
  kill "${VLLM_PID}" 2>/dev/null || true
  local waited=0
  while kill -0 "${VLLM_PID}" 2>/dev/null && (( waited < VLLM_STOP_TIMEOUT_S )); do
    sleep 5
    waited=$((waited + 5))
  done
  if kill -0 "${VLLM_PID}" 2>/dev/null; then
    log "Force-stopping vLLM PID=${VLLM_PID} after ${waited}s"
    kill -KILL "${VLLM_PID}" 2>/dev/null || true
  fi
  wait "${VLLM_PID}" 2>/dev/null || true
  VLLM_PID=""
  sleep 8
}

trap 'stop_vllm' EXIT INT TERM

wait_for_user_units() {
  for unit in ${WAIT_FOR_USER_UNITS}; do
    local waited=0
    while systemctl --user is-active --quiet "${unit}" 2>/dev/null; do
      if (( waited == 0 )); then
        log "Waiting for active unit before starting vLLM: ${unit}"
      fi
      sleep 60
      waited=$((waited + 60))
      if (( waited % 600 == 0 )); then
        log "Still waiting for ${unit}: ${waited}s"
      fi
    done
  done
}

wait_for_port_clear() {
  local waited=0
  while curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
    if (( waited >= 7200 )); then
      log "ERROR: port ${VLLM_PORT} is still serving after ${waited}s"
      return 1
    fi
    sleep 10
    waited=$((waited + 10))
  done
}

start_vllm() {
  local model="$1"
  local slug="${model//\//__}"
  slug="${slug//./_}"
  local vllm_log="${LOG_DIR}/vllm_${slug}_$(date -u +%Y%m%dT%H%M%SZ).log"
  log "Starting vLLM for ${model}"
  HF_HOME="${HF_HOME}" "${VLLM_BIN}" serve "${model}" \
    --port "${VLLM_PORT}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs 64 \
    --max-num-batched-tokens 65536 \
    --reasoning-parser qwen3 \
    --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --kv-cache-dtype fp8 \
    --no-enable-log-requests \
    --disable-log-stats \
    --performance-mode throughput \
    > "${vllm_log}" 2>&1 &
  VLLM_PID=$!

  local waited=0
  while ! curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      wait "${VLLM_PID}" 2>/dev/null || true
      log "ERROR: vLLM exited before becoming ready for ${model}; see ${vllm_log}"
      exit 1
    fi
    if (( waited >= VLLM_READY_TIMEOUT_S )); then
      log "ERROR: vLLM did not become ready for ${model} after ${waited}s; see ${vllm_log}"
      exit 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  log "vLLM ready after ${waited}s for ${model}"
}

run_graph_distance_model() {
  local model="$1"
  local n
  n="$(wc -l < "${GRAPH_DISTANCE_MANIFEST}")"
  log "Graph-distance run: model=${model} items=${n}"
  RUN_MEDGEMMA=0 \
  RUN_QWEN=1 \
  DATASET=medQA \
  SPLIT=test \
  LINKER=spacy \
  LIMIT=0 \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  CONDITIONS="${CONDITIONS}" \
  ITEM_ID_FILE="${GRAPH_DISTANCE_MANIFEST}" \
  TRIPLETS_ZIP=data/UMLS/triplets.csv.zip \
  RELATION_SCOPE=item_cuis \
  QWEN_MODEL="${model}" \
  QWEN_MODE=non_thinking \
  QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS}" \
  QWEN_TEMPERATURE=0.0 \
  QWEN_TOP_P=1.0 \
  QWEN_TOP_K=20 \
  QWEN_PRESENCE_PENALTY=0.0 \
  QWEN_SEED=42 \
  QWEN_CONCURRENCY="${QWEN_CONCURRENCY}" \
  REQUIRE_Q_TO_A_PATH=0 \
  MAX_ITEMS_PER_CONDITION=0 \
  EVIDENCE_CACHE_JSONL="${EVIDENCE_CACHE_JSONL}" \
  REQUIRE_EVIDENCE_CACHE=1 \
  RESUME=1 \
  bash scripts/run_downstream_umls_probe.sh
}

run_model() {
  local model="$1"
  log ">>> ${model}"
  wait_for_user_units
  wait_for_port_clear
  start_vllm "${model}"
  run_graph_distance_model "${model}"
  stop_vllm
  log "<<< ${model}"
}

summarize_graph_distance() {
  mapfile -t pred_files < <(find "${OUTPUT_DIR}/qwen" -name 'medQA_test_spacy_*_qwen_probe.jsonl' -type f | sort)
  if (( ${#pred_files[@]} == 0 )); then
    log "No prediction files found to summarize."
    return
  fi
  /home/molfetta/are-we-done-with-umls/.venv/bin/python -m src.experiments.graph_distance_mechanism summarize \
    --pred-jsonl "${pred_files[@]}" \
    --evidence-cache-jsonl "${EVIDENCE_CACHE_JSONL}" \
    --audit-jsonl "${GRAPH_DISTANCE_AUDIT_JSONL}" \
    --summary-csv "${TABLES_DIR}/graph_distance_mechanism_summary.csv" \
    --summary-jsonl "${TABLES_DIR}/graph_distance_mechanism_summary.jsonl" \
    --output-md "${TABLES_DIR}/graph_distance_mechanism_report.md"
  log "Graph-distance report written under ${TABLES_DIR}"
}

log "Run start"
if [[ "${REUSE_CACHE}" == "1" && -s "${EVIDENCE_CACHE_JSONL}" && -s "${GRAPH_DISTANCE_MANIFEST}" && -s "${GRAPH_DISTANCE_AUDIT_JSONL}" ]]; then
  log "Reusing graph-distance cache: cache=${EVIDENCE_CACHE_JSONL} manifest=${GRAPH_DISTANCE_MANIFEST}"
else
  build_graph_distance_cache
fi
if [[ "${BUILD_ONLY}" == "1" ]]; then
  log "BUILD_ONLY=1; skipping model generation"
  exit 0
fi
for model in ${QWEN_MODEL_LIST}; do
  run_model "${model}"
done
summarize_graph_distance
log "Run complete"
