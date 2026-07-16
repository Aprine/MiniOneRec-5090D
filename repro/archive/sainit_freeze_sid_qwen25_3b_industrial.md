# MiniOneRec 5090D Experiment Archive

- Created at: 2026-07-09T14:06:54.197863+00:00
- Experiment: `sainit_freeze_sid_qwen25_3b_industrial`
- Model: `Qwen/Qwen2.5-3B`
- Dataset: `Industrial_and_Scientific`
- Checkpoint: `output_dir/qwen25_3b_Industrial_and_Scientific_single5090d_sainit_sft/final_checkpoint`
- Result file: `results/sainit_single5090d/final_result_Industrial_and_Scientific.json`

## Training Summary

- SFT runtime: 3283.7433 seconds
- SFT steps: 190
- Train loss: 5.018499409159024
- Final eval loss: 9.654293060302734
- Trainable setting: SA-Init item_text_mean; freeze_LLM=True; train new SID input/output rows only

## Evaluation Metrics

- Records: 4533
- Beam count: 20
- Invalid prediction count: 0

| K | HR@K | NDCG@K |
|---:|---:|---:|
| 1 | 0.00000000 | 0.00000000 |
| 3 | 0.00044121 | 0.00024949 |
| 5 | 0.00044121 | 0.00024949 |
| 10 | 0.00110302 | 0.00046407 |
| 20 | 0.00132363 | 0.00051600 |

## Notes

- This is a single-RTX-5090D smoke baseline, not the full multi-GPU MiniOneRec setting.
- SFT used frozen LLM weights plus trainable SID token embeddings.
- Evaluation used the single-GPU script with constrained decoding.
