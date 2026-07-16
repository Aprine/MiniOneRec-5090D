# Research Contributions and Code Provenance

This document separates the upstream MiniOneRec foundation from the local
research implementation in `MiniOneRec-5090D`. The distinction matters for
reproducibility, academic attribution, and code review.

The comparison below was checked against the default branch of
[`AkaliKong/MiniOneRec`](https://github.com/AkaliKong/MiniOneRec) on
2026-07-16. The preserved upstream documentation is available in
[`README_UPSTREAM.md`](./README_UPSTREAM.md).

## Contribution Boundary

### Upstream Foundation

The original MiniOneRec project provides the core generative recommendation
framework, including data classes, RQ-based SID construction, SFT and RL entry
points, constrained decoding, and evaluation utilities. Representative upstream
files include:

- `data.py`, `evaluate.py`, and `LogitProcessor.py`;
- `minionerec_trainer.py`, `rl.py`, and `rl_gpr.py`;
- `rq/` and the original data-processing scripts;
- the base structure of `sft.py`.

These files are retained because the local experiments execute on top of that
framework. They are not presented as original work from this project.

### Local Research Implementation

The complete `repro/` directory is the local single-GPU benchmark and research
layer. It is absent from the upstream repository.

| Local component | Implementation | Research role |
|---|---|---|
| Pairwise preference objective | [`repro/pairwise_preference_train.py`](./repro/pairwise_preference_train.py) | Reference-free logistic preference loss plus chosen-SID SFT regularization |
| Prefix-aware hard negatives | [`HardNegativeSampler`](./repro/pairwise_preference_train.py#L86) | Samples difficult negatives from two-token or one-token SID prefix groups |
| SID-row-only adaptation | [`enable_sid_token_row_training`](./repro/pairwise_preference_train.py#L223) | Freezes Qwen2.5-3B while updating gradient-masked SID rows |
| SID diagnostics | [`repro/sid_diagnostics.py`](./repro/sid_diagnostics.py) | Measures full collisions, layer usage, prefix skew, and Gini statistics |
| Collision-aware variant | [`repro/make_collision_sid_variant.py`](./repro/make_collision_sid_variant.py) | Adds fourth-level codes and rewrites all affected splits consistently |
| Recent-history variant | [`repro/make_history_recent_variant.py`](./repro/make_history_recent_variant.py) | Generates a controlled max-history-5 ablation |
| Result archiving | [`repro/archive_run.py`](./repro/archive_run.py) | Recomputes HR/NDCG, validates predictions, and stores compact manifests |
| 5090D environment check | [`repro/check_5090d_env.py`](./repro/check_5090d_env.py) | Verifies PyTorch, CUDA, GPU visibility, and model loading |
| Reproducible runners | [`repro/run_*_5090d.sh`](./repro/) | Encodes matched hyperparameters, paths, logging, evaluation, and archiving |

## Modified Upstream File

The local [`sft.py`](./sft.py) starts from the upstream training entry point but
contains substantive research and engineering changes:

| Local addition | Code location | Why it was needed |
|---|---|---|
| Semantic SID initialization | [`semantic_aware_sid_init`](./sft.py#L74) | Initializes a SID token from the mean Qwen embeddings of associated item text |
| Input and output SID-row training | [`enable_new_token_embedding_training`](./sft.py#L153) | Updates both input embeddings and an untied output head while preserving old vocabulary rows |
| Configurable LoRA | [`normalize_lora_target_modules`](./sft.py#L192) and `train` arguments | Tests whether limited attention adaptation can rescue semantic initialization |
| Final-only checkpoint mode | `save_intermediate_checkpoints` in [`train`](./sft.py#L237) | Avoids large optimizer checkpoints and reduces I/O on the local D drive |
| LoRA merge for evaluation | final checkpoint block in [`sft.py`](./sft.py#L470) | Produces a standalone checkpoint compatible with the existing evaluator |
| Vocabulary-boundary fix | `original_vocab_size` before resize | Makes gradient masking well-defined for newly introduced SID tokens |

The upstream version trains only the input embedding rows under `freeze_LLM`
and does not provide the local semantic initialization, LoRA, output-head row
masking, or final-only saving controls.

## Experiment-to-Code Traceability

| Experiment | Main command | Method implementation | Archived evidence |
|---|---|---|---|
| R0 final-only baseline | [`run_a0_finalonly_pipeline_5090d.sh`](./repro/run_a0_finalonly_pipeline_5090d.sh) | local checkpoint-control path in `sft.py` | [`baseline_finalonly...json`](./repro/archive/baseline_finalonly_freeze_qwen25_3b_industrial.json) |
| A1 semantic initialization | [`run_a1_sainit_pipeline_5090d.sh`](./repro/run_a1_sainit_pipeline_5090d.sh) | `semantic_aware_sid_init` | [`sainit_freeze...json`](./repro/archive/sainit_freeze_sid_qwen25_3b_industrial.json) |
| A2 semantic init + LoRA | [`run_a2_sainit_lora_pipeline_5090d.sh`](./repro/run_a2_sainit_lora_pipeline_5090d.sh) | local LoRA path in `sft.py` | [`sainit_lora...json`](./repro/archive/sainit_lora_qwen25_3b_industrial.json) |
| C1 collision-aware SID | [`run_c1_collision_pipeline_5090d.sh`](./repro/run_c1_collision_pipeline_5090d.sh) | `make_collision_sid_variant.py` | [`collision_sid...json`](./repro/archive/collision_sid_freeze_qwen25_3b_industrial.json) |
| B1 recent-history pruning | [`run_b1_history_recent_pipeline_5090d.sh`](./repro/run_b1_history_recent_pipeline_5090d.sh) | `make_history_recent_variant.py` | [`history_recent5...json`](./repro/archive/history_recent5_freeze_qwen25_3b_industrial.json) |
| D1 pairwise preference | [`run_d1_pairwise_pipeline_5090d.sh`](./repro/run_d1_pairwise_pipeline_5090d.sh) | `pairwise_preference_train.py` | [`d1_pairwise...json`](./repro/archive/d1_pairwise_sidpref_qwen25_3b_industrial.json) |

## Current Evidence Boundary

D1 is the first local variant to improve both HR@20 and NDCG@20 over the matched
R0 baseline. This is a promising single-seed sampled result, not a claim that the
method outperforms the published full-scale MiniOneRec system. The next required
evidence is seed replication, followed by larger-sample and cross-category
evaluation.

## Reviewing the Difference

After adding the official repository as `upstream`, reviewers can inspect the
modified shared file directly:

```bash
git remote add upstream https://github.com/AkaliKong/MiniOneRec.git
git fetch upstream
git diff upstream/main -- sft.py
```

New local files can be reviewed from [`repro/`](./repro/). The benchmark design,
negative results, and current limitations are documented in
[`repro/BENCHMARK_5090D.md`](./repro/BENCHMARK_5090D.md).
