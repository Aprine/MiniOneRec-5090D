#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
VARIANT="${VARIANT:-Industrial_and_Scientific_c1collision}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B}"
SAMPLE="${SAMPLE:-1024}"
EPOCHS="${EPOCHS:-1}"
RUN_ID="${RUN_ID:-collision_sid_freeze_qwen25_3b_industrial}"
OUTPUT_DIR="${OUTPUT_DIR:-output_dir/qwen25_3b_${VARIANT}_single5090d_sft}"
RESULT_DIR="${RESULT_DIR:-results/c1_collision_single5090d}"
LOG_DIR="${LOG_DIR:-repro/logs}"

mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SFT_LOG="${LOG_DIR}/${RUN_ID}_sft_${STAMP}.log"
EVAL_LOG="${LOG_DIR}/${RUN_ID}_eval_${STAMP}.log"

echo "[C1] Generate collision-aware SID variant -> ${VARIANT}"
python repro/make_collision_sid_variant.py \
  --category "${CATEGORY}" \
  --variant "${VARIANT}"

export CATEGORY VARIANT MODEL SAMPLE EPOCHS OUTPUT_DIR RESULT_DIR

echo "[C1] Freeze-SID SFT -> ${OUTPUT_DIR}"
bash repro/run_c1_collision_sft_single_5090d.sh 2>&1 | tee "${SFT_LOG}"

echo "[C1] Evaluation -> ${RESULT_DIR}"
MODEL_PATH="${OUTPUT_DIR}/final_checkpoint" \
RESULT_DIR="${RESULT_DIR}" \
CATEGORY="${CATEGORY}" \
TEST_FILE="./data/Amazon/test/${VARIANT}_5_2016-10-2018-11.csv" \
INFO_FILE="./data/Amazon/info/${VARIANT}_5_2016-10-2018-11.txt" \
bash repro/run_eval_single_5090d.sh 2>&1 | tee "${EVAL_LOG}"

RESULT_FILE="${RESULT_DIR}/final_result_${CATEGORY}.json"
INFO_FILE="data/Amazon/info/${VARIANT}_5_2016-10-2018-11.txt"
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
  --trainable-setting "C1 collision-aware SID; freeze_LLM=True; train new SID token embeddings only"
)

if [[ -n "${TRAINER_STATE}" ]]; then
  ARCHIVE_ARGS+=(--trainer-state "${TRAINER_STATE}")
fi

echo "[C1] Archive"
"${ARCHIVE_ARGS[@]}"

echo "[C1] Done"
