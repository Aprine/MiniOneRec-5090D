#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
VARIANT="${VARIANT:-Industrial_and_Scientific_b1recent5}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B}"
SAMPLE="${SAMPLE:-1024}"
EPOCHS="${EPOCHS:-1}"
MAX_HISTORY="${MAX_HISTORY:-5}"
RUN_ID="${RUN_ID:-history_recent5_freeze_qwen25_3b_industrial}"
OUTPUT_DIR="${OUTPUT_DIR:-output_dir/qwen25_3b_${VARIANT}_single5090d_finalonly_sft}"
RESULT_DIR="${RESULT_DIR:-results/b1_recent5_single5090d}"
LOG_DIR="${LOG_DIR:-repro/logs}"

mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SFT_LOG="${LOG_DIR}/${RUN_ID}_sft_${STAMP}.log"
EVAL_LOG="${LOG_DIR}/${RUN_ID}_eval_${STAMP}.log"

echo "[B1] Generate recent-history variant -> ${VARIANT}, max_history=${MAX_HISTORY}"
python repro/make_history_recent_variant.py \
  --category "${CATEGORY}" \
  --variant "${VARIANT}" \
  --max-history "${MAX_HISTORY}"

export CATEGORY VARIANT MODEL SAMPLE EPOCHS OUTPUT_DIR RESULT_DIR

echo "[B1] Freeze-SID SFT -> ${OUTPUT_DIR}"
bash repro/run_b1_history_recent_sft_single_5090d.sh 2>&1 | tee "${SFT_LOG}"

echo "[B1] Evaluation -> ${RESULT_DIR}"
MODEL_PATH="${OUTPUT_DIR}/final_checkpoint" \
RESULT_DIR="${RESULT_DIR}" \
CATEGORY="${CATEGORY}" \
TEST_FILE="./data/Amazon/test/${VARIANT}_5_2016-10-2018-11.csv" \
INFO_FILE="./data/Amazon/info/${CATEGORY}_5_2016-10-2018-11.txt" \
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
  --dataset "${VARIANT}"
  --train-log "${SFT_LOG}"
  --trainable-setting "B1 recent-history max_history=${MAX_HISTORY}; freeze_LLM=True; train new SID token embeddings only"
)

if [[ -n "${TRAINER_STATE}" ]]; then
  ARCHIVE_ARGS+=(--trainer-state "${TRAINER_STATE}")
fi

echo "[B1] Archive"
"${ARCHIVE_ARGS[@]}"

echo "[B1] Done"
