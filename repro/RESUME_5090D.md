# Resume: MiniOneRec 5090D Reproduction

## Current State

- Project path in WSL: `/mnt/d/Document/OneminiRec/MiniOneRec`
- Conda env: `minionerec-5090d`
- Model: `Qwen/Qwen2.5-3B`
- Dataset: `Industrial_and_Scientific`
- Completed fair baseline: R0 A0 final-only
- Completed local improvement: D1 pairwise SID preference tuning
- R0 checkpoint: `output_dir/qwen25_3b_Industrial_and_Scientific_single5090d_finalonly_sft/final_checkpoint`
- R0 result: `results/a0_finalonly_single5090d/final_result_Industrial_and_Scientific.json`
- R0 metrics: HR@20 `0.01500110`, NDCG@20 `0.00476696`
- D1 checkpoint: `output_dir/qwen25_3b_Industrial_and_Scientific_d1_pairwise_single5090d/final_checkpoint`
- D1 result: `results/d1_pairwise_single5090d/final_result_Industrial_and_Scientific.json`
- D1 metrics: HR@20 `0.01544231`, NDCG@20 `0.00517191`
- D1 LR=1e-5 result: HR@20 `0.01500110`, NDCG@20 `0.00464173`

## Next Step

Run D1 with seed 43 and default hyperparameters before any full-sample training.
The LR=1e-5 sweep did not keep the default D1 gain, so the key question is
whether the default D1 setting is stable across seeds.

```bash
cd /mnt/d/Document/OneminiRec/MiniOneRec
source ~/miniforge3/etc/profile.d/conda.sh
conda activate minionerec-5090d
RUN_ID=d1_pairwise_sidpref_seed43_qwen25_3b_industrial SEED=43 OUTPUT_DIR=output_dir/qwen25_3b_Industrial_and_Scientific_d1_pairwise_seed43_single5090d RESULT_DIR=results/d1_pairwise_seed43_single5090d bash repro/run_d1_pairwise_pipeline_5090d.sh
```

## Before Shutdown

If no training/evaluation process is running, it is safe to close Codex, WSL,
and shut down the PC. If training is running, wait for it to finish or stop it
knowing that D1 only writes the final checkpoint at the end.
