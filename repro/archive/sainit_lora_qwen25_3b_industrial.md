# MiniOneRec 5090D Experiment Archive

- Created at: 2026-07-09T15:29:55.424160+00:00
- Experiment: `sainit_lora_qwen25_3b_industrial`
- Model: `Qwen/Qwen2.5-3B`
- Dataset: `Industrial_and_Scientific`
- Checkpoint: `output_dir/qwen25_3b_Industrial_and_Scientific_single5090d_sainit_lora_sft/final_checkpoint`
- Result file: `results/sainit_lora_single5090d/final_result_Industrial_and_Scientific.json`

## Training Summary

- SFT runtime: 1662.785 seconds
- SFT steps: 190
- Train loss: 3.67922892421484
- Final eval loss: 4.772631645202637
- Trainable setting: SA-Init item_text_mean; freeze_LLM=True; train new SID input/output rows plus LoRA q_proj/v_proj/o_proj

## Evaluation Metrics

- Records: 4533
- Beam count: 20
- Invalid prediction count: 0

| K | HR@K | NDCG@K |
|---:|---:|---:|
| 1 | 0.00022060 | 0.00022060 |
| 3 | 0.00044121 | 0.00033091 |
| 5 | 0.00110302 | 0.00058693 |
| 10 | 0.00308846 | 0.00122453 |
| 20 | 0.00750055 | 0.00229111 |

## Notes

- This is a single-RTX-5090D smoke baseline, not the full multi-GPU MiniOneRec setting.
- SFT used frozen LLM weights plus trainable SID token embeddings.
- Evaluation used the single-GPU script with constrained decoding.
