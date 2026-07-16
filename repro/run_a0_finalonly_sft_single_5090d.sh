#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_IB_DISABLE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B}"
SAMPLE="${SAMPLE:-1024}"
EPOCHS="${EPOCHS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-output_dir/qwen25_3b_${CATEGORY}_single5090d_finalonly_sft}"

TRAIN_FILE="./data/Amazon/train/${CATEGORY}_5_2016-10-2018-11.csv"
EVAL_FILE="./data/Amazon/valid/${CATEGORY}_5_2016-10-2018-11.csv"
SID_INDEX="./data/Amazon/index/${CATEGORY}.index.json"
ITEM_META="./data/Amazon/index/${CATEGORY}.item.json"

python -u sft.py \
  --base_model "${MODEL}" \
  --batch_size 16 \
  --micro_batch_size 1 \
  --num_epochs "${EPOCHS}" \
  --learning_rate 1e-4 \
  --cutoff_len 512 \
  --sample "${SAMPLE}" \
  --train_file "${TRAIN_FILE}" \
  --eval_file "${EVAL_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --wandb_project "" \
  --wandb_run_name "qwen25-3b-single5090d-a0-finalonly-sft" \
  --category "${CATEGORY}" \
  --train_from_scratch False \
  --seed 42 \
  --sid_index_path "${SID_INDEX}" \
  --item_meta_path "${ITEM_META}" \
  --freeze_LLM True \
  --save_intermediate_checkpoints False \
  --load_best_model_at_end False

echo "A0 final-only SFT checkpoint: ${OUTPUT_DIR}/final_checkpoint"
