# MiniOneRec 5090D Experiment Archive

- Created at: 2026-07-10T13:17:01.581922+00:00
- Experiment: `history_recent5_freeze_qwen25_3b_industrial`
- Model: `Qwen/Qwen2.5-3B`
- Dataset: `Industrial_and_Scientific_b1recent5`
- Checkpoint: `output_dir/qwen25_3b_Industrial_and_Scientific_b1recent5_single5090d_finalonly_sft/final_checkpoint`
- Result file: `results/b1_recent5_single5090d/final_result_Industrial_and_Scientific.json`

## Training Summary

- SFT runtime: 735.0695 seconds
- SFT steps: 192
- Train loss: 4.93153091520071
- Final eval loss: 8.298267364501953
- Trainable setting: B1 recent-history max_history=5; freeze_LLM=True; train new SID token embeddings only

## Evaluation Metrics

- Records: 4533
- Beam count: 20
- Invalid prediction count: 0

| K | HR@K | NDCG@K |
|---:|---:|---:|
| 1 | 0.00000000 | 0.00000000 |
| 3 | 0.00176484 | 0.00099795 |
| 5 | 0.00375028 | 0.00181436 |
| 10 | 0.00683874 | 0.00277005 |
| 20 | 0.01411869 | 0.00455581 |

## Notes

- This is a single-RTX-5090D smoke baseline, not the full multi-GPU MiniOneRec setting.
- SFT used frozen LLM weights plus trainable SID token embeddings.
- Evaluation used the single-GPU script with constrained decoding.
