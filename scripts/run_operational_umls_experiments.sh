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

STEP="${STEP:-all}"
LIMIT="${LIMIT:-10}"
SPLIT="${SPLIT:-test}"
DATASETS="${DATASETS:-medQA medMCQA}"
LINKERS="${LINKERS:-metamap spacy}"
STEP23_LINKERS="${STEP23_LINKERS:-spacy}"
QWEN_MODEL_LIST="${QWEN_MODEL_LIST:-Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B Qwen/Qwen3.5-35B}"
QWEN_MODE="${QWEN_MODE:-non_thinking}"
QWEN_TEMPERATURE="${QWEN_TEMPERATURE:-0.0}"
QWEN_TOP_P="${QWEN_TOP_P:-1.0}"
QWEN_TOP_K="${QWEN_TOP_K:-20}"
QWEN_PRESENCE_PENALTY="${QWEN_PRESENCE_PENALTY:-0.0}"
QWEN_SEED="${QWEN_SEED:-42}"
QWEN_CONCURRENCY="${QWEN_CONCURRENCY:-1}"
MAX_ITEMS_PER_CONDITION="${MAX_ITEMS_PER_CONDITION:-${MAX_EXPERIMENT_INSTANCES:-100}}"
STEP3_REQUIRE_Q_TO_A_PATH="${STEP3_REQUIRE_Q_TO_A_PATH:-1}"
QWEN_THINKING_TOKEN_BUDGET="${QWEN_THINKING_TOKEN_BUDGET:-}"
if [[ "${QWEN_MODE}" == "thinking" ]]; then
  QWEN_THINKING_TOKEN_BUDGET="${QWEN_THINKING_TOKEN_BUDGET:-4096}"
fi
QWEN_ANSWER_TOKEN_BUDGET="${QWEN_ANSWER_TOKEN_BUDGET:-64}"
# thinking mode needs the reasoning budget plus a small final-answer window
_default_max_tokens=32
if [[ "${QWEN_MODE}" == "thinking" ]]; then
  _default_max_tokens=$((QWEN_THINKING_TOKEN_BUDGET + QWEN_ANSWER_TOKEN_BUDGET))
fi
QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-${_default_max_tokens}}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/operational_umls}"
STEP1_SUBSET_ROOT="${STEP1_SUBSET_ROOT:-${OUTPUT_ROOT}/step1_subsets}"
STEP1_SUBSET_MODE="${STEP1_SUBSET_MODE:-${QWEN_MODE}}"
STEP1_SUBSET_FALLBACK_MODEL="${STEP1_SUBSET_FALLBACK_MODEL:-Qwen/Qwen3.5-9B}"
TRIPLETS_ZIP="${TRIPLETS_ZIP:-data/UMLS/triplets.csv.zip}"
EXTENDED_TRIPLETS_ZIP="${EXTENDED_TRIPLETS_ZIP:-data/UMLS/triplets_extended.csv.zip}"

STEP1_CONDITIONS="${STEP1_CONDITIONS:-no_umls metadata_rich_evidence}"
STEP2_CONDITIONS="${STEP2_CONDITIONS:-relation_only_evidence q_to_a_path_evidence metadata_rich_evidence}"

Q2A_DEPTH_CONDITIONS="${Q2A_DEPTH_CONDITIONS:-}"
if [[ -z "${Q2A_DEPTH_CONDITIONS}" ]]; then
  for depth in 1 2 3 4; do
    Q2A_DEPTH_CONDITIONS+="${Q2A_DEPTH_CONDITIONS:+ }q_to_a_exact_depth_${depth}_evidence"
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
  local require_q_to_a_path="${9:-0}"

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
  QWEN_THINKING_TOKEN_BUDGET="${QWEN_THINKING_TOKEN_BUDGET}" \
  MAX_ITEMS_PER_CONDITION="${MAX_ITEMS_PER_CONDITION}" \
  REQUIRE_Q_TO_A_PATH="${require_q_to_a_path}" \
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
  printf '%s/%s_%s_%s_%s_%s_baseline_wrong.jsonl' "${STEP1_SUBSET_ROOT}" "${dataset}" "${SPLIT}" "${linker}" "${slug}" "${STEP1_SUBSET_MODE}"
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
  "${UV_BIN}" run python3 src/experiments/operational_subset_analysis.py \
    --predictions "${prediction_files[@]}" \
    --output-dir "${OUTPUT_ROOT}/step1_subsets" \
    --standard-condition metadata_rich_evidence
}

run_filtered_step() {
  local step_name="$1"
  local conditions="$2"
  local relation_scope="$3"
  local triplets_zip="$4"
  local linkers="$5"
  local require_q_to_a_path="${6:-0}"
  local step_dir="${OUTPUT_ROOT}/${step_name}_predictions"

  for dataset in ${DATASETS}; do
    for linker in ${linkers}; do
      for model in ${QWEN_MODEL_LIST}; do
        local manifest
        manifest="$(manifest_file "${dataset}" "${linker}" "${model}")"
        if [[ ! -s "${manifest}" && -n "${STEP1_SUBSET_FALLBACK_MODEL}" && "${model}" != "${STEP1_SUBSET_FALLBACK_MODEL}" ]]; then
          local fallback_manifest
          fallback_manifest="$(manifest_file "${dataset}" "${linker}" "${STEP1_SUBSET_FALLBACK_MODEL}")"
          if [[ -s "${fallback_manifest}" ]]; then
            log "Using STEP 1 fallback manifest for model=${model}: fallback_model=${STEP1_SUBSET_FALLBACK_MODEL} manifest=${fallback_manifest}"
            manifest="${fallback_manifest}"
          fi
        fi
        if [[ ! -s "${manifest}" ]]; then
          if [[ "${DRY_RUN}" == "1" ]]; then
            log "DRY_RUN=1: would use manifest ${manifest}"
            continue
          fi
          echo "Missing STEP 1 manifest: ${manifest}" >&2
          echo "Run STEP=1 first." >&2
          exit 2
        fi
        run_probe "${dataset}" "${linker}" "${model}" "${conditions}" "${step_dir}" "${manifest}" "${relation_scope}" "${triplets_zip}" "${require_q_to_a_path}"
      done
    done
  done
}

run_step2() {
  run_filtered_step "step2" "${STEP2_CONDITIONS}" "item_cuis" "${TRIPLETS_ZIP}" "${STEP23_LINKERS}" "0"
}

run_step3() {
  local graph_zip="${TRIPLETS_ZIP}"
  if [[ -s "${EXTENDED_TRIPLETS_ZIP}" ]]; then
    graph_zip="${EXTENDED_TRIPLETS_ZIP}"
  fi
  run_filtered_step "step3" "${Q2A_DEPTH_CONDITIONS}" "full" "${graph_zip}" "${STEP23_LINKERS}" "${STEP3_REQUIRE_Q_TO_A_PATH}"
}

collect_prediction_files_for_step23_linkers() {
  local root="$1"
  local -n output_ref="$2"
  output_ref=()
  for linker in ${STEP23_LINKERS}; do
    while IFS= read -r file; do
      output_ref+=("${file}")
    done < <(find "${root}" -name "*_${linker}_*_qwen_probe.jsonl" -type f 2>/dev/null | sort)
  done
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
  local step2_predictions=()
  collect_prediction_files_for_step23_linkers "${OUTPUT_ROOT}/step2_predictions/qwen" step2_predictions
  if (( ${#step2_predictions[@]} > 0 )); then
    log "Building STEP 2 conditioned tables under ${tables_dir}"
    DIAGNOSTICS_DIR="${diagnostics_dir}" \
    OUTPUT_DIR="${tables_dir}" \
    PREDICTIONS="${step2_predictions[*]}" \
    bash scripts/run_paper_tables.sh
  else
    log "Skipping STEP 2 conditioned tables: no STEP 2 prediction files found."
  fi

  local step3_predictions=()
  collect_prediction_files_for_step23_linkers "${OUTPUT_ROOT}/step3_predictions/qwen" step3_predictions
  if (( ${#step3_predictions[@]} > 0 )); then
    log "Building STEP 3 Q-to-answer depth tables under ${tables_dir}"
    "${UV_BIN}" run python3 src/experiments/q_to_a_depth_analysis.py \
      --pred-jsonl "${step3_predictions[@]}" \
      --output-dir "${tables_dir}"
  else
    log "Skipping STEP 3 Q-to-answer depth tables: no STEP 3 prediction files found."
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
