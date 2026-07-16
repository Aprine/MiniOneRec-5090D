# 5090D MiniOneRec Benchmark Plan

## Current Archived Baseline

Experiment `baseline_freeze_sid_qwen25_3b_industrial` is the first completed
single-GPU smoke baseline.

- GPU: RTX 5090D
- Model: `Qwen/Qwen2.5-3B`
- Dataset: `Industrial_and_Scientific`
- SFT setting: `freeze_LLM=True`, train added SID token embeddings only
- Train records after task mixing: 3072
- Eval records during SFT: 1024
- Test records: 4533
- SFT runtime: 3129.5365 seconds
- SFT steps: 192
- Train loss: 4.915262444565694
- Final eval loss: 8.162327766418457
- HR@20: 0.01544231
- NDCG@20: 0.00536662

## A1 Result: SA-Init + Freeze SID

Experiment `sainit_freeze_sid_qwen25_3b_industrial` completed with the same
single-GPU smoke setting.

- SFT runtime: 3283.7433 seconds
- SFT steps: 190
- Train loss: 5.018499409159024
- Final eval loss: 9.654293060302734
- HR@20: 0.00132363
- NDCG@20: 0.00051600

Compared with A0, A1 is clearly worse, so do not spend a full-sample run on
freeze-only SA-Init. The next benchmark should test whether small LoRA capacity
can use the semantic initialization without collapsing retrieval quality.

## A2 Result: SA-Init + LoRA

Experiment `sainit_lora_qwen25_3b_industrial` completed with the same
single-GPU smoke setting.

- SFT runtime: 1662.785 seconds
- SFT steps: 190
- Train loss: 3.67922892421484
- Final eval loss: 4.772631645202637
- HR@20: 0.00750055
- NDCG@20: 0.00229111

A2 is much better than A1, but still below A0. Do not run full-sample A2 yet.
Move to C1 collision-aware SID, because it changes only the SID construction
surface and keeps the freeze-only training path comparable to A0.

## C1 Result: Collision-Aware SID

Experiment `collision_sid_freeze_qwen25_3b_industrial` completed with the same
single-GPU smoke setting.

- SFT runtime: 3323.147 seconds
- SFT steps: 190
- Train loss: 5.002043507993221
- Final eval loss: 8.301302909851074
- HR@20: 0.01478050
- NDCG@20: 0.00421362

C1 is close to A0 on HR@20 but still lower on both HR@20 and NDCG@20. Do not run
full-sample C1 yet. Move to B1 history pruning/dropout, because the SID-only
changes tested so far do not beat the freeze-only baseline.

## B1 Setup: Recent-History Pruning

B1 variant files have been generated as `Industrial_and_Scientific_b1recent5`.
This keeps only the most recent 5 history items and leaves target SID, item info,
and SID codebook unchanged.

- Train rows changed: 7280 / 36259
- Valid rows changed: 1345 / 4532
- Test rows changed: 1472 / 4533
- Train mean history length: 3.769988 -> 3.160953
- Valid mean history length: 4.642542 -> 3.755958
- Test mean history length: 4.956541 -> 3.980807
- Max history length: 10 -> 5
- B1 SFT uses final-only checkpoint saving to avoid large intermediate optimizer checkpoints on `/mnt/d`.

## B1 Result: Recent-History Pruning

Experiment `history_recent5_freeze_qwen25_3b_industrial` completed with the
same single-GPU smoke setting.

- SFT runtime: 735.0695 seconds
- SFT steps: 192
- Train loss: 4.93153091520071
- Final eval loss: 8.298267364501953
- HR@20: 0.01411869
- NDCG@20: 0.00455581

B1 is faster and reasonably close to A0, but still lower on HR@20 and NDCG@20.
The runtime is not directly comparable with A0 because B1 uses final-only saving
while A0 saved intermediate checkpoints. Run A0-finalonly before drawing a speed
claim.

## R0 Result: A0 Final-Only Runtime Control

Experiment `baseline_finalonly_freeze_qwen25_3b_industrial` completed with the
same A0 data/model/training setting, but disables intermediate checkpoint saves.

- SFT runtime: 772.5501 seconds
- SFT steps: 192
- Train loss: 4.926515669872363
- Final eval loss: 8.310744285583496
- HR@20: 0.01500110
- NDCG@20: 0.00476696
- Invalid predictions: 0

R0 is the fair runtime control for B1. Compared with R0, B1 is only about 4.9%
faster, while HR@20 and NDCG@20 are both lower. Therefore the main practical
speed gain came from final-only checkpoint saving, not from recent-history
pruning. Do not pursue B1 as a quality-improvement direction unless a later
variant adds adaptive pruning or dropout.

## D1 Setup: Pairwise SID Preference Tuning

D1 is implemented as a reference-free pairwise preference stage on top of the R0
final checkpoint. For each sequence recommendation prompt, it compares the gold
target SID against a hard negative SID sampled from the same SID prefix when
possible. The model is frozen except for gradient-masked SID token rows.

- Base checkpoint: `output_dir/qwen25_3b_Industrial_and_Scientific_single5090d_finalonly_sft/final_checkpoint`
- Output checkpoint: `output_dir/qwen25_3b_Industrial_and_Scientific_d1_pairwise_single5090d/final_checkpoint`
- Result directory: `results/d1_pairwise_single5090d`
- Default sample: 1024 train rows, 256 pairwise eval rows
- Default loss: reference-free logistic preference loss plus 0.1 chosen-SID SFT regularization

D1 should be compared primarily against R0, because R0 is the fair final-only
baseline. A useful D1 outcome is any HR@20/NDCG@20 improvement over R0 without a
large invalid-prediction increase.

## D1 Result: Pairwise SID Preference Tuning

Experiment `d1_pairwise_sidpref_qwen25_3b_industrial` completed on top of the
R0 final checkpoint.

- Pairwise tuning runtime: 119.68395686149597 seconds
- Pairwise global steps: 1024
- Optimizer steps: 64
- Train loss: 1.3964805391151458
- Train pair accuracy: 0.5869140625
- Eval pair loss: 0.6966824349947274
- Eval pair accuracy: 0.49609375
- Eval margin: -0.055023374035954475
- HR@20: 0.01544231
- NDCG@20: 0.00517191
- Invalid predictions: 0

Compared with R0, D1 improves HR@20 from 0.01500110 to 0.01544231 and NDCG@20
from 0.00476696 to 0.00517191, while keeping invalid predictions at 0. This is
the first local variant that beats the fair final-only baseline on both @20
metrics. The result is still mixed because HR@10 drops from 0.00661813 to
0.00529451 and pairwise eval accuracy stays near random. Treat D1 as a promising
but not yet stable direction; validate it with a small hyperparameter/seed sweep
before any full-sample run.

## D1 Sweep Result: LR=1e-5

Experiment `d1_pairwise_sidpref_lr1e5_qwen25_3b_industrial` completed on top of
the R0 final checkpoint.

- Pairwise tuning runtime: 123.45749974250793 seconds
- Pairwise global steps: 1024
- Optimizer steps: 64
- Train loss: 1.459086395218037
- Train pair accuracy: 0.5810546875
- Eval pair loss: 0.696898553520441
- Eval pair accuracy: 0.51171875
- Eval margin: -0.06055472046136856
- HR@20: 0.01500110
- NDCG@20: 0.00464173
- Invalid predictions: 0

Lowering the learning rate from 2e-5 to 1e-5 removes the default D1 @20 gain.
HR@20 falls back to the R0 value and NDCG@20 drops below R0. Do not pursue
LR=1e-5 as the main D1 setting. The next stability check should keep the
default D1 hyperparameters and change only the random seed.

## SID Diagnostics

Diagnostics are archived in
`repro/archive/sid_diagnostics_industrial.json`.

- Items: 3686
- Unique SID sequences: 3670
- Full SID collision groups: 15
- Items affected by full SID collision: 31
- SID token usage Gini: 0.386981952303439
- Target interaction-frequency Gini: 0.46470944541863535
- Top skewed prefix: `<a_223><b_198>` with 47 assigned items

These numbers make C1 worth testing after A1/A2: add a fourth code only for
collided SID groups and keep the same constrained decoding/evaluation path.

C1 variant files have been generated as `Industrial_and_Scientific_c1collision`.
Diagnostics are archived in
`repro/archive/sid_diagnostics_industrial_c1collision.json`.

- C1 full SID collision groups: 0
- C1 items affected by full SID collision: 0
- Added suffix tokens: `<d_0>`, `<d_1>`, `<d_2>`
- Changed train targets: 1455
- Changed valid targets: 238
- Changed test targets: 317

## Literature-To-Pipeline Map

| Paper | Relevant claim | Local benchmark translation |
|---|---|---|
| MiniOneRec | End-to-end SID construction, SFT, RL pipeline over Qwen backbones | Keep the same data, Qwen2.5-3B, constrained decoding, and HR/NDCG metrics |
| TS-Rec | SID tokens suffer meaningless initialization; SA-Init uses semantic text pooling | `sid_init_strategy=item_text_mean` before freeze-only SFT |
| FORGE | SID construction quality and collision handling strongly affect GR | Run SID diagnostics before training; track collisions, code usage, prefix skew |
| RASTP | SID token pruning can reduce training time while keeping quality | Start with data-level history pruning before model-internal pruning |
| Variable-Length SID | Fixed SID length is a poor fit for heterogeneous item popularity | Use diagnostics to decide whether high-frequency items deserve shorter/easier SIDs |

## Benchmark Variants

| ID | Name | Implementation status | Target question |
|---|---|---|---|
| A0 | Freeze SID baseline | done | What is the 5090D smoke baseline? |
| A1 | SA-Init + freeze SID | done, worse than A0 | Does semantic SID init improve HR/NDCG without extra trainable LLM weights? |
| A2 | SA-Init + LoRA | done, better than A1 but worse than A0 | Does small LoRA capacity beat freeze-only? |
| B1 | History pruning/dropout | done, faster but lower than A0 | Can we reduce runtime or noise with lighter SID histories? |
| C1 | SID collision-aware extension | done, close to A0 but lower | Are collisions/prefix skew limiting this category? |
| R0 | A0 final-only runtime control | done, fair runtime control for B1 | How much speedup comes from disabling intermediate checkpoint saves? |
| D1 | Pairwise RL-light | done, default improves @20 vs R0; LR=1e-5 fails | Can preference training improve R0 without costly GRPO on one GPU? |

For A1/A2, `sft.py` now keeps old vocabulary rows frozen but makes the newly
added SID token rows trainable in both input embeddings and the output lm_head
when the model does not tie those weights.

## Primary Metrics

- HR@1/3/5/10/20
- NDCG@1/3/5/10/20
- invalid prediction count
- SFT runtime
- train loss and eval loss trajectory
- SID collision count and codebook usage Gini

## Run Order

1. Archive A0 and run SID diagnostics.
2. Run A1 with the same `SAMPLE=1024`, `EPOCHS=1`. Completed; worse than A0.
3. Run A2 with the same `SAMPLE=1024`, `EPOCHS=1`. Completed; better than A1 but worse than A0.
4. Run C1 collision-aware SID with the same `SAMPLE=1024`, `EPOCHS=1`. Completed; close to A0 but lower.
5. Run B1 recent-history pruning with the same `SAMPLE=1024`, `EPOCHS=1`. Completed; faster but lower than A0.
6. Run R0 A0-finalonly runtime control before making speed claims. Completed; B1 has only a small runtime edge and lower quality.
7. Run D1 pairwise preference tuning on top of the R0 final checkpoint. Completed; improves HR@20/NDCG@20 vs R0 but is mixed at @10.
8. Run D1 LR=1e-5 sweep. Completed; worse than default D1 and below R0 on NDCG@20.
9. Next: run D1 seed=43 with default hyperparameters before full-sample training, because pairwise eval accuracy is still near random.

## Commands

Archive current A0:

```bash
python repro/archive_run.py \
  --result results/single5090d/final_result_Industrial_and_Scientific.json \
  --item-info data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
  --checkpoint output_dir/qwen25_3b_Industrial_and_Scientific_single5090d_sft/final_checkpoint \
  --runtime-seconds 3129.5365 \
  --steps 192 \
  --train-loss 4.915262444565694 \
  --final-eval-loss 8.162327766418457
```

SID diagnostics:

```bash
python repro/sid_diagnostics.py \
  --index data/Amazon/index/Industrial_and_Scientific.index.json \
  --train data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --valid data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --test data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv
```

Run A1:

```bash
bash repro/run_a1_sainit_pipeline_5090d.sh
```

Run A2:

```bash
python -m pip install peft
bash repro/run_a2_sainit_lora_pipeline_5090d.sh
```

Run C1:

```bash
bash repro/run_c1_collision_pipeline_5090d.sh
```

Run B1:

```bash
bash repro/run_b1_history_recent_pipeline_5090d.sh
```

Run R0:

```bash
bash repro/run_a0_finalonly_pipeline_5090d.sh
```

Run D1:

```bash
bash repro/run_d1_pairwise_pipeline_5090d.sh
```

D1 stability sweep candidates:

```bash
RUN_ID=d1_pairwise_sidpref_seed43_qwen25_3b_industrial SEED=43 OUTPUT_DIR=output_dir/qwen25_3b_Industrial_and_Scientific_d1_pairwise_seed43_single5090d RESULT_DIR=results/d1_pairwise_seed43_single5090d bash repro/run_d1_pairwise_pipeline_5090d.sh
RUN_ID=d1_pairwise_sidpref_beta03_qwen25_3b_industrial BETA=0.3 OUTPUT_DIR=output_dir/qwen25_3b_Industrial_and_Scientific_d1_pairwise_beta03_single5090d RESULT_DIR=results/d1_pairwise_beta03_single5090d bash repro/run_d1_pairwise_pipeline_5090d.sh
```
