# MiniOneRec on a Local 5090D with Qwen 3B

This is the single-GPU reproduction path for the local `MiniOneRec` checkout.
It is not the paper's full 4-8x A100/H100 setting. The goal is to make the
pipeline reproducible on one 5090D first, then scale the training knobs.

## Model Choice

Use `Qwen/Qwen2.5-3B` for the Qwen 3B target. The Qwen3 family does not have a
standard 3B checkpoint; the closest Qwen3 dense checkpoint is usually 4B. The
MiniOneRec README also recommends switching from Instruct to a base model when
constrained decoding produces invalid items, so the default here is the base
`Qwen/Qwen2.5-3B`.

DeepSeek is possible as an ablation, but not as the first reproduction target.
Use Qwen first to validate SID construction, SFT, constrained decoding, and
metrics. After that, try a DeepSeek distilled Qwen checkpoint as a model swap.

## Why Not the Upstream requirements.txt

The upstream `requirements.txt` pins `torch==2.6.0` and also includes
`torchrec==0.6.0+cu118` / `fbgemm_gpu==0.8.0+cu118`. That stack targets older
CUDA wheels and is a bad default for RTX 50-series Blackwell GPUs. Install
PyTorch separately with CUDA 12.8 wheels, then install the trimmed dependency
file in `repro/requirements-5090d-cu128.txt`.

The single-GPU smoke path intentionally does not install DeepSpeed. If DeepSpeed
is installed without a full CUDA Toolkit and `CUDA_HOME`, `transformers.Trainer`
can import it through Accelerate and fail before training starts. For the local
5090D path, uninstall it unless you are explicitly setting up multi-GPU or
ZeRO-based training:

```bash
python -m pip uninstall -y deepspeed
```

## Recommended Platform

Use Ubuntu or WSL2 Ubuntu. Native Windows is not recommended for this project
because DeepSpeed, Triton, bitsandbytes, and NCCL-adjacent tooling are much more
predictable on Linux.

If WSL does not have `conda`, `pip`, or `python` yet, run the bootstrap script:

```bash
cd /mnt/d/Document/OneminiRec/MiniOneRec
bash repro/bootstrap_wsl_miniforge_5090d.sh
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate minionerec-5090d
```

Install or verify the NVIDIA driver first:

```bash
nvidia-smi
```

Then create the Python environment:

```bash
conda create -n minionerec-5090d python=3.11 -y
conda activate minionerec-5090d
python -m pip install -U pip setuptools wheel

# Pick the CUDA 12.8 PyTorch wheel from the official PyTorch index.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r repro/requirements-5090d-cu128.txt
python -m pip uninstall -y deepspeed
```

If you do not have conda yet, install Miniforge or Miniconda first. Do not
install the full CUDA Toolkit unless another package explicitly requires local
`nvcc`; PyTorch wheels bundle the CUDA runtime they need.

## Environment Check

From the repo root:

```bash
python repro/check_5090d_env.py --skip-model-load
python repro/check_5090d_env.py --model Qwen/Qwen2.5-3B
```

Expected signals:

- `cuda available: True`
- GPU name contains `5090`
- `bf16 supported: True`
- The Qwen model loads onto CUDA.

## Smoke SFT

The upstream SFT script is full-parameter training by default. On a single
32GB-class 5090D, start with frozen LLM weights and train the added SID token
embeddings only. This validates the whole data/model/tokenizer path.

```bash
bash repro/run_sft_single_5090d.sh
```

The script defaults to:

- category: `Industrial_and_Scientific`
- model: `Qwen/Qwen2.5-3B`
- sample: `1024`
- epochs: `1`
- output: `output_dir/qwen25_3b_Industrial_and_Scientific_single5090d_sft/final_checkpoint`

For a full local run after the smoke test:

```bash
SAMPLE=-1 EPOCHS=3 bash repro/run_sft_single_5090d.sh
```

## Evaluation

```bash
bash repro/run_eval_single_5090d.sh
```

The single-GPU evaluation script uses conservative defaults:

- batch size: `1`
- beams: `20`
- max new tokens: `128`

For closer paper-style evaluation, increase beams toward 50 if VRAM allows:

```bash
NUM_BEAMS=50 BATCH_SIZE=1 bash repro/run_eval_single_5090d.sh
```

## RL Stage

Treat RL as phase 2. The upstream `rl.sh` assumes 8 processes and generates many
candidates per prompt. On one 5090D, start with smaller values:

```bash
accelerate launch --num_processes 1 rl.py \
  --model_path output_dir/qwen25_3b_Industrial_and_Scientific_single5090d_sft/final_checkpoint \
  --train_batch_size 1 \
  --eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --num_generations 4 \
  --learning_rate 1e-6 \
  --beta 1e-3 \
  --reward_type ranking \
  --beam_search True \
  --test_during_training False \
  --train_file ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --eval_file ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --info_file ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
  --category Industrial_and_Scientific \
  --output_dir output_dir/qwen25_3b_Industrial_and_Scientific_single5090d_rl \
  --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
  --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json
```

If this OOMs, lower `num_generations` first, then lower sequence/eval frequency.

## Reproduction Ladder

1. Environment check.
2. Smoke SFT with `SAMPLE=1024`.
3. Smoke evaluation with `NUM_BEAMS=20`.
4. Full SFT with `SAMPLE=-1`.
5. Evaluation with `NUM_BEAMS=50`.
6. Reduced RL with `num_generations=4`.
7. Full-ish single-GPU RL only if memory and runtime are acceptable.

For exact paper metrics, expect to need the upstream multi-GPU setting or a
larger GPU. The single 5090D path is a faithful engineering reproduction of the
pipeline, not a guarantee of the published leaderboard numbers.
