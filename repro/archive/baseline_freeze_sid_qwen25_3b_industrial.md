# MiniOneRec 5090D Experiment Archive

- Created at: 2026-07-09T12:47:41.949103+00:00
- Experiment: `baseline_freeze_sid_qwen25_3b_industrial`
- Model: `Qwen/Qwen2.5-3B`
- Dataset: `Industrial_and_Scientific`
- Checkpoint: `output_dir/qwen25_3b_Industrial_and_Scientific_single5090d_sft/final_checkpoint`
- Result file: `results\single5090d\final_result_Industrial_and_Scientific.json`

## Training Summary

- SFT runtime: 3129.5365 seconds
- SFT steps: 192
- Train loss: 4.915262444565694
- Final eval loss: 8.162327766418457
- Trainable setting: freeze_LLM=True; train added SID token embeddings only

## Evaluation Metrics

- Records: 4533
- Beam count: 20
- Invalid prediction count: 0

| K | HR@K | NDCG@K |
|---:|---:|---:|
| 1 | 0.00000000 | 0.00000000 |
| 3 | 0.00264725 | 0.00149693 |
| 5 | 0.00441209 | 0.00223767 |
| 10 | 0.00970660 | 0.00390613 |
| 20 | 0.01544231 | 0.00536662 |

## Notes

- This is a single-RTX-5090D smoke baseline, not the full multi-GPU MiniOneRec setting.
- SFT used frozen LLM weights plus trainable SID token embeddings.
- Evaluation used the single-GPU script with constrained decoding.
