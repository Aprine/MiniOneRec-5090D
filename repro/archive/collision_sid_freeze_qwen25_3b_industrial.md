# MiniOneRec 5090D Experiment Archive

- Created at: 2026-07-09T17:32:56.920255+00:00
- Experiment: `collision_sid_freeze_qwen25_3b_industrial`
- Model: `Qwen/Qwen2.5-3B`
- Dataset: `Industrial_and_Scientific_c1collision`
- Checkpoint: `output_dir/qwen25_3b_Industrial_and_Scientific_c1collision_single5090d_sft/final_checkpoint`
- Result file: `results/c1_collision_single5090d/final_result_Industrial_and_Scientific.json`

## Training Summary

- SFT runtime: 3323.147 seconds
- SFT steps: 190
- Train loss: 5.002043507993221
- Final eval loss: 8.301302909851074
- Trainable setting: C1 collision-aware SID; freeze_LLM=True; train new SID token embeddings only

## Evaluation Metrics

- Records: 4533
- Beam count: 20
- Invalid prediction count: 0

| K | HR@K | NDCG@K |
|---:|---:|---:|
| 1 | 0.00000000 | 0.00000000 |
| 3 | 0.00044121 | 0.00022060 |
| 5 | 0.00154423 | 0.00067631 |
| 10 | 0.00463269 | 0.00166030 |
| 20 | 0.01478050 | 0.00421362 |

## Notes

- This is a single-RTX-5090D smoke baseline, not the full multi-GPU MiniOneRec setting.
- SFT used frozen LLM weights plus trainable SID token embeddings.
- Evaluation used the single-GPU script with constrained decoding.
