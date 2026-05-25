#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/tmp/medqa_step3_shared08_manifests_20260525T165438Z/medQA_spacy_Qwen3_5-0_8B_depth4_shared08.jsonl}"
SHORTCUT_MANIFEST="${SHORTCUT_MANIFEST:-/tmp/medqa_d4_shortcut_manifest_${RUN_ID}.jsonl}"
EVIDENCE_CACHE_JSONL="${EVIDENCE_CACHE_JSONL:-/tmp/medqa_d4_shortcuts_evidence_${RUN_ID}.jsonl}"
LOG_DIR="${LOG_DIR:-outputs/operational_umls/logs/medqa_d4_shortcuts_${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/operational_umls/step3_shortcut_predictions}"
mkdir -p "${LOG_DIR}"

log() { printf '[d4-shortcuts] %s\n' "$*" | tee -a "${LOG_DIR}/run.log"; }

export HF_HOME="${HF_HOME:-/mnt/nvme/llms/}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/nvme/llms/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/nvme/llms/datasets}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/nvme/llms/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/mnt/nvme/llms/venvs/are-we-done-with-umls}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
export OPENAI_API_KEY="${VLLM_API_KEY:-EMPTY}"
export SOURCE_MANIFEST SHORTCUT_MANIFEST EVIDENCE_CACHE_JSONL LOG_DIR

VLLM_BIN="${VLLM_BIN:-/mnt/nvme/llms/venvs/spiral-benchmark-env/bin/vllm}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_READY_TIMEOUT_S="${VLLM_READY_TIMEOUT_S:-7200}"
VLLM_STOP_TIMEOUT_S="${VLLM_STOP_TIMEOUT_S:-180}"
QWEN_MODEL_LIST="${QWEN_MODEL_LIST:-Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B Qwen/Qwen3.5-35B-A3B}"
CONDITIONS="${CONDITIONS:-q_to_a_d4_shortcut_1_evidence q_to_a_d4_shortcut_2_evidence q_to_a_d4_shortcut_3_evidence q_to_a_d4_shortcut_4_evidence}"
QWEN_CONCURRENCY="${QWEN_CONCURRENCY:-32}"
QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-32}"
BUILD_ONLY="${BUILD_ONLY:-0}"
REUSE_CACHE="${REUSE_CACHE:-0}"
WAIT_FOR_USER_UNITS="${WAIT_FOR_USER_UNITS:-umls-final-medqa-35b-a3b-step23-20260525T1939Z.service}"
VLLM_PID=""

build_shortcut_cache_and_manifest() {
  log "Building synthetic D4 shortcut cache from ${SOURCE_MANIFEST}"
  python3 - <<'PY' | tee -a "${LOG_DIR}/cache_build.log"
import json
import os
import signal
from collections import OrderedDict
from pathlib import Path

from src.experiments.cli_utils import default_input_path, default_reference_path, load_items
from src.experiments.evidence_cards import build_evidence_card_details
from src.experiments.umls_index import CuiMetadataIndex, RelationIndex
from src.model_sweeps.downstream_umls_probe import evidence_card_to_cache_row

SOURCE_MANIFEST = Path(os.environ["SOURCE_MANIFEST"])
SHORTCUT_MANIFEST = Path(os.environ["SHORTCUT_MANIFEST"])
EVIDENCE_CACHE_JSONL = Path(os.environ["EVIDENCE_CACHE_JSONL"])
TIMEOUT_S = int(os.environ.get("EVIDENCE_CARD_TIMEOUT_S", "120"))
CONDITIONS = (
    "q_to_a_d4_shortcut_1_evidence",
    "q_to_a_d4_shortcut_2_evidence",
    "q_to_a_d4_shortcut_3_evidence",
    "q_to_a_d4_shortcut_4_evidence",
)


class Timeout(Exception):
    pass


def timeout_handler(signum, frame):
    raise Timeout()


signal.signal(signal.SIGALRM, timeout_handler)

if not SOURCE_MANIFEST.exists():
    raise SystemExit(f"Missing source manifest: {SOURCE_MANIFEST}")

selected = OrderedDict()
with SOURCE_MANIFEST.open("r", encoding="utf-8") as fh:
    for raw in fh:
        if not raw.strip():
            continue
        row = json.loads(raw)
        selected[str(row["item_id"])] = None

items = load_items(
    "medQA",
    "test",
    "spacy",
    input_path=default_input_path("medQA", "test", "spacy"),
    reference_path=default_reference_path("medQA", "test"),
)
item_map = {item.item_id: item for item in items}
metadata = CuiMetadataIndex.from_zip(Path("data/UMLS/cui_detail.tsv.zip"))
relations = RelationIndex.from_zip(Path("data/UMLS/triplets.csv.zip"), keep_cuis=None)

EVIDENCE_CACHE_JSONL.parent.mkdir(parents=True, exist_ok=True)
SHORTCUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
cache_tmp = EVIDENCE_CACHE_JSONL.with_suffix(EVIDENCE_CACHE_JSONL.suffix + ".tmp")
manifest_tmp = SHORTCUT_MANIFEST.with_suffix(SHORTCUT_MANIFEST.suffix + ".tmp")
counts = {"source_items": len(selected), "kept_items": 0, "cache_rows": 0, "missing": 0, "timeout": 0, "error": 0, "path_missing": 0}

with cache_tmp.open("w", encoding="utf-8") as cache_fh, manifest_tmp.open("w", encoding="utf-8") as manifest_fh:
    for item_id in selected:
        item = item_map.get(item_id)
        if item is None:
            counts["missing"] += 1
            continue
        rows = []
        failed = False
        for condition in CONDITIONS:
            try:
                signal.alarm(TIMEOUT_S)
                card = build_evidence_card_details(item, condition, metadata, relations)
                signal.alarm(0)
            except Timeout:
                signal.alarm(0)
                counts["timeout"] += 1
                failed = True
                print(json.dumps({"event": "timeout", "item_id": item_id, "condition": condition}), flush=True)
                break
            except Exception as exc:
                signal.alarm(0)
                counts["error"] += 1
                failed = True
                print(json.dumps({"event": "error", "item_id": item_id, "condition": condition, "error": str(exc)}), flush=True)
                break
            if not card.q_to_a_path_found:
                counts["path_missing"] += 1
                failed = True
                print(json.dumps({"event": "path_missing", "item_id": item_id, "condition": condition}), flush=True)
                break
            rows.append(evidence_card_to_cache_row(item, condition, card))
        if failed:
            continue
        manifest_fh.write(json.dumps({"item_id": item_id}) + "\n")
        counts["kept_items"] += 1
        for row in rows:
            cache_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["cache_rows"] += 1
        if counts["kept_items"] % 10 == 0:
            print(json.dumps({"event": "cache_progress", **counts}), flush=True)

cache_tmp.replace(EVIDENCE_CACHE_JSONL)
manifest_tmp.replace(SHORTCUT_MANIFEST)
print(json.dumps({"event": "cache_done", "cache": str(EVIDENCE_CACHE_JSONL), "manifest": str(SHORTCUT_MANIFEST), **counts}), flush=True)
if counts["kept_items"] == 0:
    raise SystemExit("No D4 shortcut items remained after cache construction")
PY
  log "Shortcut cache ready: cache=${EVIDENCE_CACHE_JSONL} manifest=${SHORTCUT_MANIFEST}"
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
    --max-model-len 65536 \
    --gpu-memory-utilization 0.85 \
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
    if (( waited >= VLLM_READY_TIMEOUT_S )); then
      log "ERROR: vLLM did not become ready for ${model} after ${waited}s; see ${vllm_log}"
      exit 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  log "vLLM ready after ${waited}s for ${model}"
}

run_shortcuts() {
  local model="$1"
  local n
  n="$(wc -l < "${SHORTCUT_MANIFEST}")"
  log "STEP3 synthetic D4 shortcuts: model=${model} items=${n}"
  RUN_MEDGEMMA=0 \
  RUN_QWEN=1 \
  DATASET=medQA \
  SPLIT=test \
  LINKER=spacy \
  LIMIT=0 \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  CONDITIONS="${CONDITIONS}" \
  ITEM_ID_FILE="${SHORTCUT_MANIFEST}" \
  TRIPLETS_ZIP=data/UMLS/triplets.csv.zip \
  RELATION_SCOPE=full \
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
  run_shortcuts "${model}"
  stop_vllm
  log "<<< ${model}"
}

log "Run start"
if [[ "${REUSE_CACHE}" == "1" && -s "${EVIDENCE_CACHE_JSONL}" && -s "${SHORTCUT_MANIFEST}" ]]; then
  log "Reusing shortcut cache: cache=${EVIDENCE_CACHE_JSONL} manifest=${SHORTCUT_MANIFEST}"
else
  build_shortcut_cache_and_manifest
fi
if [[ "${BUILD_ONLY}" == "1" ]]; then
  log "BUILD_ONLY=1; skipping model generation"
  exit 0
fi
for model in ${QWEN_MODEL_LIST}; do
  run_model "${model}"
done
log "Run complete"
