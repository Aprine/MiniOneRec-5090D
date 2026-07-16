#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
MODEL_PATH="${MODEL_PATH:-output_dir/qwen25_3b_${CATEGORY}_single5090d_sft/final_checkpoint}"
RESULT_DIR="${RESULT_DIR:-results/single5090d}"
mkdir -p "${RESULT_DIR}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model checkpoint not found: ${MODEL_PATH}"
  echo "Run SFT successfully first, or set MODEL_PATH to an existing local checkpoint/Hugging Face model id."
  exit 1
fi

TEST_FILE="${TEST_FILE:-./data/Amazon/test/${CATEGORY}_5_2016-10-2018-11.csv}"
INFO_FILE="${INFO_FILE:-./data/Amazon/info/${CATEGORY}_5_2016-10-2018-11.txt}"
RESULT_JSON="${RESULT_DIR}/final_result_${CATEGORY}.json"

python -u ./evaluate.py \
  --base_model "${MODEL_PATH}" \
  --info_file "${INFO_FILE}" \
  --category "${CATEGORY}" \
  --test_data_path "${TEST_FILE}" \
  --result_json_data "${RESULT_JSON}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --num_beams "${NUM_BEAMS:-20}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
  --length_penalty 0.0

python ./calc.py \
  --path "${RESULT_JSON}" \
  --item_path "${INFO_FILE}"
