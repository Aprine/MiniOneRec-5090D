# RTX 5090D Research Code

This directory contains the local research implementation added on top of the
upstream MiniOneRec framework. It is the primary code entry point for reproducing
the single-GPU benchmark and its ablations.

## Main Method

[`pairwise_preference_train.py`](./pairwise_preference_train.py) implements D1,
a reference-free pairwise SID preference stage. Given a recommendation prompt,
it scores the target SID against a prefix-matched hard negative and optimizes:

```text
pairwise logistic loss + 0.1 * chosen-SID SFT loss
```

Qwen2.5-3B remains frozen except for gradient-masked SID token rows. This keeps
the additional stage short and memory-efficient on one RTX 5090D.

Run the complete D1 stage with:

```bash
bash repro/run_d1_pairwise_pipeline_5090d.sh
```

## Code Index

| File | Function |
|---|---|
| [`pairwise_preference_train.py`](./pairwise_preference_train.py) | D1 dataset, hard-negative sampler, pairwise objective, training, validation, and checkpoint export |
| [`sid_diagnostics.py`](./sid_diagnostics.py) | SID collision, prefix, codebook-usage, and interaction-frequency diagnostics |
| [`make_collision_sid_variant.py`](./make_collision_sid_variant.py) | Collision-aware fourth-level SID construction |
| [`make_history_recent_variant.py`](./make_history_recent_variant.py) | Controlled recent-history pruning variant |
| [`archive_run.py`](./archive_run.py) | Metric recomputation and compact experiment archiving |
| [`check_5090d_env.py`](./check_5090d_env.py) | CUDA, PyTorch, GPU, and Qwen loading checks |
| [`bootstrap_wsl_miniforge_5090d.sh`](./bootstrap_wsl_miniforge_5090d.sh) | WSL and conda bootstrap |
| [`requirements-5090d-cu128.txt`](./requirements-5090d-cu128.txt) | Reproduction dependency lock |

## Experiment Runners

| ID | Command | Question |
|---|---|---|
| A0 | `bash repro/run_sft_single_5090d.sh` | Can SID-only SFT run correctly on one 5090D? |
| R0 | `bash repro/run_a0_finalonly_pipeline_5090d.sh` | What is the fair final-only baseline? |
| A1 | `bash repro/run_a1_sainit_pipeline_5090d.sh` | Does item-text semantic SID initialization help under frozen-LLM training? |
| A2 | `bash repro/run_a2_sainit_lora_pipeline_5090d.sh` | Can small LoRA capacity rescue semantic initialization? |
| C1 | `bash repro/run_c1_collision_pipeline_5090d.sh` | Are full SID collisions the main bottleneck? |
| B1 | `bash repro/run_b1_history_recent_pipeline_5090d.sh` | Does uniform recent-history pruning improve speed or quality? |
| D1 | `bash repro/run_d1_pairwise_pipeline_5090d.sh` | Can lightweight pairwise tuning improve R0 without GRPO? |

## Evidence

- Full benchmark narrative: [`BENCHMARK_5090D.md`](./BENCHMARK_5090D.md)
- Current restart command: [`RESUME_5090D.md`](./RESUME_5090D.md)
- Compact machine-readable results: [`archive/`](./archive/)
- Upstream-versus-local boundary: [`../RESEARCH_CONTRIBUTIONS.md`](../RESEARCH_CONTRIBUTIONS.md)

Large checkpoints, raw datasets, logs, and per-example prediction files are
excluded from Git. The compact archive retains the configuration, runtime,
metrics, and artifact paths needed to audit each run.
