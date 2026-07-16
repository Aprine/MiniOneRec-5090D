#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B}"
SAMPLE="${SAMPLE:-1024}"
EPOCHS="${EPOCHS:-1}"
RUN_ID="${RUN_ID:-sainit_freeze_sid_qwen25_3b_industrial}"
OUTPUT_DIR="${OUTPUT_DIR:-output_dir/qwen25_3b_${CATEGORY}_single5090d_sainit_sft}"
RESULT_DIR="${RESULT_DIR:-results/sainit_single5090d}"
LOG_DIR="${LOG_DIR:-repro/logs}"

mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SFT_LOG="${LOG_DIR}/${RUN_ID}_sft_${STAMP}.log"
EVAL_LOG="${LOG_DIR}/${RUN_ID}_eval_${STAMP}.log"

export CATEGORY MODEL SAMPLE EPOCHS OUTPUT_DIR RESULT_DIR

echo "[A1] SA-Init freeze-SID SFT -> ${OUTPUT_DIR}"
bash repro/run_sft_sainit_single_5090d.sh 2>&1 | tee "${SFT_LOG}"

echo "[A1] Evaluation -> ${RESULT_DIR}"
MODEL_PATH="${OUTPUT_DIR}/final_checkpoint" \
RESULT_DIR="${RESULT_DIR}" \
CATEGORY="${CATEGORY}" \
bash repro/run_eval_single_5090d.sh 2>&1 | tee "${EVAL_LOG}"

RESULT_FILE="${RESULT_DIR}/final_result_${CATEGORY}.json"
INFO_FILE="data/Amazon/info/${CATEGORY}_5_2016-10-2018-11.txt"
TRAINER_STATE="$(find "${OUTPUT_DIR}" -path "*/trainer_state.json" | sort -V | tail -1)"

ARCHIVE_ARGS=(
  python repro/archive_run.py
  --result "${RESULT_FILE}"
  --item-info "${INFO_FILE}"
  --checkpoint "${OUTPUT_DIR}/final_checkpoint"
  --experiment-name "${RUN_ID}"
  --model "${MODEL}"
  --dataset "${CATEGORY}"
  --train-log "${SFT_LOG}"
  --trainable-setting "SA-Init item_text_mean; freeze_LLM=True; train new SID input/output rows only"
)

if [[ -n "${TRAINER_STATE}" ]]; then
  ARCHIVE_ARGS+=(--trainer-state "${TRAINER_STATE}")
fi

echo "[A1] Archive"
"${ARCHIVE_ARGS[@]}"

echo "[A1] Done"
