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

DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-outputs/diagnostics/main}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/paper_tables}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-1000}"
PREDICTIONS="${PREDICTIONS:-}"
CONDITIONS="${CONDITIONS:-correct_answer_grounded all_options_grounded gold_path_available gold_answer_evidence_available distractor_contrast_available}"
UMLS_CONDITION="${UMLS_CONDITION:-q_to_a_path_evidence}"
EVIDENCE_CONDITIONS="${EVIDENCE_CONDITIONS:-relation_only_evidence metadata_rich_evidence q_to_a_path_evidence gold_only_evidence distractor_contrast_evidence oracle_gold_evidence random_same_size_evidence shuffled_answer_evidence}"

if [[ ! -d "${DIAGNOSTICS_DIR}" ]]; then
  DIAGNOSTICS_DIR="outputs/diagnostics"
fi

mapfile -t diagnostic_files < <(find "${DIAGNOSTICS_DIR}" -name '*.jsonl' -type f | sort)
if (( ${#diagnostic_files[@]} == 0 )); then
  echo "No diagnostic JSONL files found under ${DIAGNOSTICS_DIR}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

echo "+ Building diagnostic paper tables"
uv run python3 src/experiments/make_paper_diagnostic_tables.py \
  "${diagnostic_files[@]}" \
  --bootstrap-iterations "${BOOTSTRAP_ITERATIONS}" \
  --output-dir "${OUTPUT_DIR}"

if [[ -z "${PREDICTIONS}" ]]; then
  candidate="${OUTPUT_DIR}/../model_sweeps/downstream_umls_probe/medgemma/medQA_test_metamap_medgemma_probe.jsonl"
  if [[ -s "${candidate}" ]]; then
    PREDICTIONS="${candidate}"
  fi
fi

if [[ -n "${PREDICTIONS}" && -s "${PREDICTIONS}" ]]; then
  condition_args=()
  read -r -a condition_list <<< "${CONDITIONS}"
  for condition in "${condition_list[@]}"; do
    condition_args+=(--condition "${condition}")
  done
  evidence_condition_args=()
  read -r -a evidence_condition_list <<< "${EVIDENCE_CONDITIONS}"
  for condition in "${evidence_condition_list[@]}"; do
    evidence_condition_args+=(--evidence-condition "${condition}")
  done
  echo "+ Building downstream conditioned paper tables from ${PREDICTIONS}"
  uv run python3 src/model_sweeps/conditioned_metrics.py \
    --predictions "${PREDICTIONS}" \
    --diagnostics "${diagnostic_files[@]}" \
    "${condition_args[@]}" \
    --delta-table \
    --umls-condition "${UMLS_CONDITION}" \
    --output-csv "${OUTPUT_DIR}/table_downstream_conditioned.csv" \
    --output-md "${OUTPUT_DIR}/table_downstream_conditioned.md"
  uv run python3 src/model_sweeps/conditioned_metrics.py \
    --predictions "${PREDICTIONS}" \
    --diagnostics "${diagnostic_files[@]}" \
    "${condition_args[@]}" \
    "${evidence_condition_args[@]}" \
    --intervention-table \
    --output-csv "${OUTPUT_DIR}/table_intervention_by_diagnostic.csv" \
    --output-md "${OUTPUT_DIR}/table_intervention_by_diagnostic.md"
  uv run python3 src/model_sweeps/conditioned_metrics.py \
    --predictions "${PREDICTIONS}" \
    --diagnostics "${diagnostic_files[@]}" \
    "${condition_args[@]}" \
    "${evidence_condition_args[@]}" \
    --paired-table \
    --output-csv "${OUTPUT_DIR}/table_paired_intervention_effects.csv" \
    --output-md "${OUTPUT_DIR}/table_paired_intervention_effects.md"
  uv run python3 src/model_sweeps/conditioned_metrics.py \
    --predictions "${PREDICTIONS}" \
    --diagnostics "${diagnostic_files[@]}" \
    --downstream-table-kind realistic \
    --output-csv "${OUTPUT_DIR}/table_downstream_realistic.csv" \
    --output-md "${OUTPUT_DIR}/table_downstream_realistic.md"
  uv run python3 src/model_sweeps/conditioned_metrics.py \
    --predictions "${PREDICTIONS}" \
    --diagnostics "${diagnostic_files[@]}" \
    --downstream-table-kind controls \
    --output-csv "${OUTPUT_DIR}/table_downstream_controls.csv" \
    --output-md "${OUTPUT_DIR}/table_downstream_controls.md"
  uv run python3 src/model_sweeps/conditioned_metrics.py \
    --predictions "${PREDICTIONS}" \
    --diagnostics "${diagnostic_files[@]}" \
    --downstream-table-kind diagnostic_oracle \
    --output-csv "${OUTPUT_DIR}/table_downstream_diagnostic_oracle.csv" \
    --output-md "${OUTPUT_DIR}/table_downstream_diagnostic_oracle.md"
  uv run python3 src/model_sweeps/conditioned_metrics.py \
    --predictions "${PREDICTIONS}" \
    --diagnostics "${diagnostic_files[@]}" \
    --main-claim-summary \
    --output-csv "${OUTPUT_DIR}/table_main_claim_summary.csv" \
    --output-md "${OUTPUT_DIR}/table_main_claim_summary.md"
else
  echo "Skipping downstream conditioned table; set PREDICTIONS=/path/to/probe.jsonl after running scripts/run_downstream_umls_probe.sh."
fi

echo "Paper tables written under ${OUTPUT_DIR}"
