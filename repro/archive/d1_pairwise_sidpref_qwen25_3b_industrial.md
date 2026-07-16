# MiniOneRec 5090D Experiment Archive

- Created at: 2026-07-11T10:07:53.156241+00:00
- Experiment: `d1_pairwise_sidpref_qwen25_3b_industrial`
- Model: `output_dir/qwen25_3b_Industrial_and_Scientific_single5090d_finalonly_sft/final_checkpoint`
- Dataset: `Industrial_and_Scientific`
- Checkpoint: `output_dir/qwen25_3b_Industrial_and_Scientific_d1_pairwise_single5090d/final_checkpoint`
- Result file: `results/d1_pairwise_single5090d/final_result_Industrial_and_Scientific.json`

## Training Summary

- SFT runtime: 119.68395686149597 seconds
- SFT steps: 1024
- Train loss: 1.3964805391151458
- Final eval loss: n/a
- Trainable setting: D1 reference-free pairwise SID preference; base=R0 final-only checkpoint; freeze LLM; train gradient-masked SID token rows; hard negatives by SID prefix

## Evaluation Metrics

- Records: 4533
- Beam count: 20
- Invalid prediction count: 0

| K | HR@K | NDCG@K |
|---:|---:|---:|
| 1 | 0.00066181 | 0.00066181 |
| 3 | 0.00286786 | 0.00179372 |
| 5 | 0.00375028 | 0.00217376 |
| 10 | 0.00529451 | 0.00264813 |
| 20 | 0.01544231 | 0.00517191 |

## Notes

- This is a single-RTX-5090D smoke baseline, not the full multi-GPU MiniOneRec setting.
- SFT used frozen LLM weights plus trainable SID token embeddings.
- Evaluation used the single-GPU script with constrained decoding.
