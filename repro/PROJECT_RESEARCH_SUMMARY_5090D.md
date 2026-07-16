# MiniOneRec-5090D Research Summary

## Project Positioning

This is a long-term independent research project on compute-efficient
generative recommendation. It reproduces and extends a MiniOneRec-style
semantic-ID pipeline on one NVIDIA RTX 5090D with `Qwen/Qwen2.5-3B`.

The present stage is a controlled sampled benchmark and method exploration
platform. It is not yet a full-scale reproduction of every result reported by
MiniOneRec.

## One-Sentence Research Argument

Under a single-consumer-GPU constraint, a reproducible Qwen2.5-3B semantic-ID
recommendation pipeline can be built with frozen-LLM adaptation, and a short
reference-free pairwise SID preference stage can improve the current matched
baseline at K=20, although cross-seed and cross-dataset validation remain open.

## Long-Term Research Direction

The long-term goal is to identify which components of LLM-based generative
recommendation provide the greatest accuracy-per-compute benefit:

1. semantic-ID construction and collision control;
2. semantic initialization of newly added SID tokens;
3. parameter-efficient adaptation through frozen embeddings or LoRA;
4. sequence and history pruning;
5. lightweight preference optimization as an alternative to expensive GRPO;
6. stability across seeds, data scales, item categories, and model backbones.

## Experimental Setting

| Component | Setting |
|---|---|
| GPU | NVIDIA RTX 5090D |
| Environment | WSL, conda env `minionerec-5090d` |
| Backbone | `Qwen/Qwen2.5-3B` |
| Dataset | Amazon `Industrial_and_Scientific` |
| Test records | 4,533 |
| Evaluation | Constrained decoding with HR@K, NDCG@K, and invalid count |
| Fair baseline | R0 final-only freeze-SID SFT |
| Best current local variant | D1 pairwise SID preference, LR `2e-5` |

## Completed Ablations

| ID | Variant | Runtime | HR@20 | NDCG@20 | Invalid | Conclusion |
|---|---|---:|---:|---:|---:|---|
| A0 | Initial freeze-SID smoke run | 3129.5 s | 0.01544231 | 0.00536662 | 0 | Intermediate checkpoint I/O dominated runtime |
| R0 | Final-only freeze-SID baseline | 772.6 s | 0.01500110 | 0.00476696 | 0 | Main fair baseline |
| A1 | Semantic initialization + frozen LLM | 3283.7 s | 0.00132363 | 0.00051600 | 0 | Semantic initialization alone was unstable |
| A2 | Semantic initialization + LoRA | 1662.8 s | 0.00750055 | 0.00229111 | 0 | LoRA recovered over A1 but remained below R0 |
| C1 | Collision-aware SID extension | 3323.1 s | 0.01478050 | 0.00421362 | 0 | Removing collisions alone was insufficient |
| B1 | Keep the five most recent history items | 735.1 s | 0.01411869 | 0.00455581 | 0 | Small speed gain with lower quality |
| D1 | Pairwise SID preference, LR `2e-5` | +119.7 s | **0.01544231** | **0.00517191** | 0 | Best current local result at K=20 |
| D1-LR | Pairwise SID preference, LR `1e-5` | +123.5 s | 0.01500110 | 0.00464173 | 0 | Lower LR removed the gain |

Relative to R0, D1 currently changes:

- HR@20: `0.01500110 -> 0.01544231`, approximately `+2.9%`;
- NDCG@20: `0.00476696 -> 0.00517191`, approximately `+8.5%`;
- invalid predictions: `0 -> 0`.

These results are promising but not yet conclusive. D1 reduced HR@10 and its
pairwise validation accuracy remained near random, so seed replication is
required before making a robust improvement claim.

## SID Diagnostics

| Diagnostic | Value |
|---|---:|
| Items | 3,686 |
| Unique full SID sequences | 3,670 |
| Full SID collision groups | 15 |
| Collision-affected items | 31 |
| SID token-usage Gini | 0.38698 |
| Interaction-frequency Gini | 0.46471 |
| Largest observed two-token prefix | `<a_223><b_198>`, 47 items |

The C1 experiment removed all observed full-SID collisions without improving
R0, indicating that SID quality cannot be reduced to collision count alone.

## Immediate Next Experiment

Run D1 with seed 43 while holding all default hyperparameters fixed:

```bash
cd /mnt/d/Document/OneminiRec/MiniOneRec
source ~/miniforge3/etc/profile.d/conda.sh
conda activate minionerec-5090d
RUN_ID=d1_pairwise_sidpref_seed43_qwen25_3b_industrial \
SEED=43 \
OUTPUT_DIR=output_dir/qwen25_3b_Industrial_and_Scientific_d1_pairwise_seed43_single5090d \
RESULT_DIR=results/d1_pairwise_seed43_single5090d \
bash repro/run_d1_pairwise_pipeline_5090d.sh
```

If the gain persists, the next controlled factor should be negative-sampling
strategy, followed by a larger training sample and a second Amazon category.

## CV Description: English

**Efficient Generative Recommendation with Qwen2.5-3B on RTX 5090D**  
Independent Research Project

- Reproduced and extended a MiniOneRec-style semantic-ID generative
  recommendation pipeline on a single RTX 5090D using Qwen2.5-3B.
- Built reproducible workflows for SFT, constrained decoding, experiment
  archiving, and SID collision and distribution diagnostics.
- Conducted controlled ablations on SID semantic initialization, LoRA,
  collision-aware SID construction, history pruning, and lightweight preference
  optimization.
- Designed a reference-free pairwise SID preference objective with prefix-aware
  hard negatives, improving HR@20 by `2.9%` and NDCG@20 by `8.5%` over the
  matched final-only baseline in the current sampled benchmark, with zero
  invalid predictions.
- Current work evaluates seed stability, stronger negative mining, and transfer
  to larger samples and additional item domains.

## 简历描述：中文

**基于 Qwen2.5-3B 与 RTX 5090D 的轻量化生成式推荐复现与改进**  
独立研究项目

- 在单张 RTX 5090D 上复现并扩展 MiniOneRec 风格的语义 ID 生成式推荐流程，
  使用 Qwen2.5-3B 完成监督微调与受约束推荐解码。
- 搭建可复现的环境检查、训练、评估、实验归档和 SID 碰撞与分布诊断流程。
- 围绕 SID 语义初始化、LoRA、碰撞感知 SID 构造、历史序列裁剪和轻量偏好优化
  开展控制变量消融实验。
- 设计基于前缀 hard negative 的无参考模型 pairwise SID preference objective；
  在当前 sampled benchmark 中，相对匹配的 final-only 基线将 HR@20 提升约
  `2.9%`、NDCG@20 提升约 `8.5%`，并保持 invalid prediction 为 `0`。
- 后续研究聚焦随机种子稳定性、更强 hard-negative 构造、更大训练规模和跨品类
  泛化验证。

