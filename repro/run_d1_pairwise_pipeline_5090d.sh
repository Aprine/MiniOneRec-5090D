#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
SAMPLE="${SAMPLE:-1024}"
EPOCHS="${EPOCHS:-1}"
RUN_ID="${RUN_ID:-d1_pairwise_sidpref_qwen25_3b_industrial}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-output_dir/qwen25_3b_${CATEGORY}_single5090d_finalonly_sft/final_checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-output_dir/qwen25_3b_${CATEGORY}_d1_pairwise_single5090d}"
RESULT_DIR="${RESULT_DIR:-results/d1_pairwise_single5090d}"
LOG_DIR="${LOG_DIR:-repro/logs}"

TRAIN_FILE="./data/Amazon/train/${CATEGORY}_5_2016-10-2018-11.csv"
EVAL_FILE="./data/Amazon/valid/${CATEGORY}_5_2016-10-2018-11.csv"
TEST_FILE="./data/Amazon/test/${CATEGORY}_5_2016-10-2018-11.csv"
INFO_FILE="./data/Amazon/info/${CATEGORY}_5_2016-10-2018-11.txt"

if [[ ! -d "${BASE_CHECKPOINT}" ]]; then
  echo "Base checkpoint not found: ${BASE_CHECKPOINT}"
  echo "Run R0 first: bash repro/run_a0_finalonly_pipeline_5090d.sh"
  exit 1
fi

mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="${LOG_DIR}/${RUN_ID}_train_${STAMP}.log"
EVAL_LOG="${LOG_DIR}/${RUN_ID}_eval_${STAMP}.log"

if [[ -d "${OUTPUT_DIR}/final_checkpoint" && "${ALLOW_OVERWRITE:-false}" != "true" ]]; then
  echo "Output checkpoint already exists: ${OUTPUT_DIR}/final_checkpoint"
  echo "Set a unique OUTPUT_DIR/RESULT_DIR for a new run, or set ALLOW_OVERWRITE=true to overwrite intentionally."
  exit 1
fi

echo "[D1] Pairwise SID preference tuning -> ${OUTPUT_DIR}"
python -u repro/pairwise_preference_train.py \
  --model_path "${BASE_CHECKPOINT}" \
  --train_file "${TRAIN_FILE}" \
  --eval_file "${EVAL_FILE}" \
  --info_file "${INFO_FILE}" \
  --category "${CATEGORY}" \
  --output_dir "${OUTPUT_DIR}" \
  --sample "${SAMPLE}" \
  --eval_sample "${EVAL_SAMPLE:-256}" \
  --num_epochs "${EPOCHS}" \
  --micro_batch_size "${MICRO_BATCH_SIZE:-1}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-16}" \
  --learning_rate "${LR:-2e-5}" \
  --beta "${BETA:-0.1}" \
  --sft_weight "${SFT_WEIGHT:-0.1}" \
  --max_len "${MAX_LEN:-512}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --eval_steps "${PAIRWISE_EVAL_STEPS:-256}" \
  --seed "${SEED:-42}" \
  --length_normalize "${LENGTH_NORMALIZE:-True}" \
  2>&1 | tee "${TRAIN_LOG}"

echo "[D1] Constrained decoding evaluation -> ${RESULT_DIR}"
MODEL_PATH="${OUTPUT_DIR}/final_checkpoint" \
RESULT_DIR="${RESULT_DIR}" \
CATEGORY="${CATEGORY}" \
TEST_FILE="${TEST_FILE}" \
INFO_FILE="${INFO_FILE}" \
bash repro/run_eval_single_5090d.sh 2>&1 | tee "${EVAL_LOG}"

RESULT_FILE="${RESULT_DIR}/final_result_${CATEGORY}.json"
TRAINER_STATE="${OUTPUT_DIR}/trainer_state.json"

ARCHIVE_ARGS=(
  python repro/archive_run.py
  --result "${RESULT_FILE}"
  --item-info "${INFO_FILE}"
  --checkpoint "${OUTPUT_DIR}/final_checkpoint"
  --experiment-name "${RUN_ID}"
  --model "${BASE_CHECKPOINT}"
  --dataset "${CATEGORY}"
  --train-log "${TRAIN_LOG}"
  --trainable-setting "D1 reference-free pairwise SID preference; base=R0 final-only checkpoint; freeze LLM; train gradient-masked SID token rows; hard negatives by SID prefix"
)

if [[ -f "${TRAINER_STATE}" ]]; then
  ARCHIVE_ARGS+=(--trainer-state "${TRAINER_STATE}")
fi

echo "[D1] Archive"
"${ARCHIVE_ARGS[@]}"

echo "[D1] Done"
