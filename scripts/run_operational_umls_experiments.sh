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

STEP="${STEP:-all}"
LIMIT="${LIMIT:-10}"
SPLIT="${SPLIT:-test}"
DATASETS="${DATASETS:-medQA medMCQA}"
LINKERS="${LINKERS:-metamap spacy}"
QWEN_MODEL_LIST="${QWEN_MODEL_LIST:-Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B}"
QWEN_MODE="${QWEN_MODE:-non_thinking}"
QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-32}"
QWEN_TEMPERATURE="${QWEN_TEMPERATURE:-0.0}"
QWEN_TOP_P="${QWEN_TOP_P:-1.0}"
QWEN_TOP_K="${QWEN_TOP_K:-20}"
QWEN_PRESENCE_PENALTY="${QWEN_PRESENCE_PENALTY:-0.0}"
QWEN_SEED="${QWEN_SEED:-42}"
QWEN_CONCURRENCY="${QWEN_CONCURRENCY:-1}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/operational_umls}"
TRIPLETS_ZIP="${TRIPLETS_ZIP:-data/UMLS/triplets.csv.zip}"
EXTENDED_TRIPLETS_ZIP="${EXTENDED_TRIPLETS_ZIP:-data/UMLS/triplets_extended.csv.zip}"

STEP1_CONDITIONS="${STEP1_CONDITIONS:-no_umls metadata_rich_evidence}"
STEP2_CONDITIONS="${STEP2_CONDITIONS:-relation_only_evidence metadata_rich_evidence gold_only_evidence shuffled_answer_evidence q_to_a_path_evidence}"

CHAIN_CONDITIONS="${CHAIN_CONDITIONS:-}"
if [[ -z "${CHAIN_CONDITIONS}" ]]; then
  for depth in 1 2 3 4; do
    for width in tight medium wide full; do
      CHAIN_CONDITIONS+="${CHAIN_CONDITIONS:+ }chain_depth_${depth}_width_${width}_evidence"
    done
  done
fi

log() {
  printf '[operational-umls] %s\n' "$*"
}

qwen_slug() {
  local value="$1"
  value="${value//\//__}"
  value="${value//./_}"
  printf '%s' "${value}"
}

run_probe() {
  local dataset="$1"
  local linker="$2"
  local model="$3"
  local conditions="$4"
  local output_dir="$5"
  local item_id_file="${6:-}"
  local relation_scope="${7:-item_cuis}"
  local triplets_zip="${8:-${TRIPLETS_ZIP}}"

  log "Qwen probe: dataset=${dataset} linker=${linker} model=${model} step_dir=${output_dir}"
  RUN_MEDGEMMA=0 \
  RUN_QWEN=1 \
  DATASET="${dataset}" \
  SPLIT="${SPLIT}" \
  LINKER="${linker}" \
  LIMIT="${LIMIT}" \
  OUTPUT_DIR="${output_dir}" \
  CONDITIONS="${conditions}" \
  ITEM_ID_FILE="${item_id_file}" \
  TRIPLETS_ZIP="${triplets_zip}" \
  RELATION_SCOPE="${relation_scope}" \
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
}

prediction_file() {
  local output_dir="$1"
  local dataset="$2"
  local linker="$3"
  local model="$4"
  local slug
  slug="$(qwen_slug "${model}")"
  printf '%s/qwen/%s_%s_%s_%s_%s_qwen_probe.jsonl' "${output_dir}" "${dataset}" "${SPLIT}" "${linker}" "${slug}" "${QWEN_MODE}"
}

manifest_file() {
  local dataset="$1"
  local linker="$2"
  local model="$3"
  local slug
  slug="$(qwen_slug "${model}")"
  printf '%s/step1_subsets/%s_%s_%s_%s_%s_baseline_wrong.jsonl' "${OUTPUT_ROOT}" "${dataset}" "${SPLIT}" "${linker}" "${slug}" "${QWEN_MODE}"
}

run_step1() {
  local step_dir="${OUTPUT_ROOT}/step1_predictions"
  for dataset in ${DATASETS}; do
    for linker in ${LINKERS}; do
      for model in ${QWEN_MODEL_LIST}; do
        run_probe "${dataset}" "${linker}" "${model}" "${STEP1_CONDITIONS}" "${step_dir}"
      done
    done
  done

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN=1: skipping STEP 1 subset manifest build."
    return
  fi

  mapfile -t prediction_files < <(find "${step_dir}/qwen" -name '*_qwen_probe.jsonl' -type f | sort)
  if (( ${#prediction_files[@]} == 0 )); then
    echo "No STEP 1 prediction files found under ${step_dir}/qwen" >&2
    exit 2
  fi
  uv run python3 src/experiments/operational_subset_analysis.py \
    --predictions "${prediction_files[@]}" \
    --output-dir "${OUTPUT_ROOT}/step1_subsets" \
    --standard-condition metadata_rich_evidence
}

run_filtered_step() {
  local step_name="$1"
  local conditions="$2"
  local relation_scope="$3"
  local triplets_zip="$4"
  local step_dir="${OUTPUT_ROOT}/${step_name}_predictions"

  for dataset in ${DATASETS}; do
    for linker in ${LINKERS}; do
      for model in ${QWEN_MODEL_LIST}; do
        local manifest
        manifest="$(manifest_file "${dataset}" "${linker}" "${model}")"
        if [[ ! -s "${manifest}" ]]; then
          if [[ "${DRY_RUN}" == "1" ]]; then
            log "DRY_RUN=1: would use manifest ${manifest}"
            continue
          fi
          echo "Missing STEP 1 manifest: ${manifest}" >&2
          echo "Run STEP=1 first." >&2
          exit 2
        fi
        run_probe "${dataset}" "${linker}" "${model}" "${conditions}" "${step_dir}" "${manifest}" "${relation_scope}" "${triplets_zip}"
      done
    done
  done
}

run_step2() {
  run_filtered_step "step2" "${STEP2_CONDITIONS}" "item_cuis" "${TRIPLETS_ZIP}"
}

run_step3() {
  local graph_zip="${TRIPLETS_ZIP}"
  if [[ -s "${EXTENDED_TRIPLETS_ZIP}" ]]; then
    graph_zip="${EXTENDED_TRIPLETS_ZIP}"
  fi
  run_filtered_step "step3" "${CHAIN_CONDITIONS}" "full" "${graph_zip}"
}

run_tables() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN=1: skipping table/status build."
    return
  fi
  MODE=audit OUTPUT_ROOT="${OUTPUT_ROOT}" bash scripts/umls_experiment_monitor.sh

  local diagnostics_dir="${DIAGNOSTICS_DIR:-outputs/diagnostics/main}"
  if [[ ! -d "${diagnostics_dir}" ]]; then
    diagnostics_dir="outputs/diagnostics"
  fi
  if [[ ! -d "${diagnostics_dir}" ]]; then
    log "Skipping operational tables: no diagnostics directory found."
    return
  fi

  mapfile -t diagnostic_files < <(find "${diagnostics_dir}" -name '*.jsonl' -type f | sort)
  if (( ${#diagnostic_files[@]} == 0 )); then
    log "Skipping operational tables: no diagnostic JSONL files found under ${diagnostics_dir}."
    return
  fi

  local tables_dir="${OUTPUT_ROOT}/tables"
  mapfile -t step2_predictions < <(find "${OUTPUT_ROOT}/step2_predictions/qwen" -name '*_qwen_probe.jsonl' -type f 2>/dev/null | sort)
  if (( ${#step2_predictions[@]} > 0 )); then
    log "Building STEP 2 conditioned tables under ${tables_dir}"
    DIAGNOSTICS_DIR="${diagnostics_dir}" \
    OUTPUT_DIR="${tables_dir}" \
    PREDICTIONS="${step2_predictions[*]}" \
    bash scripts/run_paper_tables.sh
  else
    log "Skipping STEP 2 conditioned tables: no STEP 2 prediction files found."
  fi

  mapfile -t step3_predictions < <(find "${OUTPUT_ROOT}/step3_predictions/qwen" -name '*_qwen_probe.jsonl' -type f 2>/dev/null | sort)
  mapfile -t exp3_files < <(find "${diagnostics_dir}" -name 'exp3_retrieval_faithfulness_*.jsonl' -type f | sort)
  if (( ${#step3_predictions[@]} > 0 && ${#exp3_files[@]} > 0 )); then
    log "Building STEP 3 depth-width tables under ${tables_dir}"
    uv run python3 src/experiments/depth_width_analysis.py \
      --exp3-jsonl "${exp3_files[@]}" \
      --pred-jsonl "${step3_predictions[@]}" \
      --output-dir "${tables_dir}"
  else
    log "Skipping STEP 3 depth-width tables: need STEP 3 predictions and Exp3 diagnostics."
  fi
}

case "${STEP}" in
  1|step1)
    run_step1
    ;;
  2|step2)
    run_step2
    ;;
  3|step3)
    run_step3
    ;;
  tables)
    run_tables
    ;;
  all)
    run_step1
    run_step2
    run_step3
    run_tables
    ;;
  *)
    echo "Unsupported STEP=${STEP}; use 1, 2, 3, tables, or all." >&2
    exit 2
    ;;
esac

log "Finished STEP=${STEP}. Outputs: ${OUTPUT_ROOT}"
