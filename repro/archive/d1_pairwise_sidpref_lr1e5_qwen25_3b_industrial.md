# MiniOneRec 5090D Experiment Archive

- Created at: 2026-07-11T10:29:58.223065+00:00
- Experiment: `d1_pairwise_sidpref_lr1e5_qwen25_3b_industrial`
- Model: `output_dir/qwen25_3b_Industrial_and_Scientific_single5090d_finalonly_sft/final_checkpoint`
- Dataset: `Industrial_and_Scientific`
- Checkpoint: `output_dir/qwen25_3b_Industrial_and_Scientific_d1_pairwise_lr1e5_single5090d/final_checkpoint`
- Result file: `results/d1_pairwise_lr1e5_single5090d/final_result_Industrial_and_Scientific.json`

## Training Summary

- SFT runtime: 123.45749974250793 seconds
- SFT steps: 1024
- Train loss: 1.459086395218037
- Final eval loss: n/a
- Trainable setting: D1 reference-free pairwise SID preference; base=R0 final-only checkpoint; freeze LLM; train gradient-masked SID token rows; hard negatives by SID prefix

## Evaluation Metrics

- Records: 4533
- Beam count: 20
- Invalid prediction count: 0

| K | HR@K | NDCG@K |
|---:|---:|---:|
| 1 | 0.00000000 | 0.00000000 |
| 3 | 0.00176484 | 0.00094019 |
| 5 | 0.00264725 | 0.00131055 |
| 10 | 0.00617692 | 0.00242944 |
| 20 | 0.01500110 | 0.00464173 |

## Notes

- This is a single-RTX-5090D smoke baseline, not the full multi-GPU MiniOneRec setting.
- SFT used frozen LLM weights plus trainable SID token embeddings.
- Evaluation used the single-GPU script with constrained decoding.
