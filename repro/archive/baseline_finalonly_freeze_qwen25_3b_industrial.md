# MiniOneRec 5090D Experiment Archive

- Created at: 2026-07-10T14:50:44.143938+00:00
- Experiment: `baseline_finalonly_freeze_qwen25_3b_industrial`
- Model: `Qwen/Qwen2.5-3B`
- Dataset: `Industrial_and_Scientific`
- Checkpoint: `output_dir/qwen25_3b_Industrial_and_Scientific_single5090d_finalonly_sft/final_checkpoint`
- Result file: `results/a0_finalonly_single5090d/final_result_Industrial_and_Scientific.json`

## Training Summary

- SFT runtime: 772.5501 seconds
- SFT steps: 192
- Train loss: 4.926515669872363
- Final eval loss: 8.310744285583496
- Trainable setting: A0 final-only; freeze_LLM=True; train new SID token embeddings only

## Evaluation Metrics

- Records: 4533
- Beam count: 20
- Invalid prediction count: 0

| K | HR@K | NDCG@K |
|---:|---:|---:|
| 1 | 0.00000000 | 0.00000000 |
| 3 | 0.00220604 | 0.00118967 |
| 5 | 0.00330907 | 0.00164538 |
| 10 | 0.00661813 | 0.00267119 |
| 20 | 0.01500110 | 0.00476696 |

## Notes

- This is a single-RTX-5090D smoke baseline, not the full multi-GPU MiniOneRec setting.
- SFT used frozen LLM weights plus trainable SID token embeddings.
- Evaluation used the single-GPU script with constrained decoding.
