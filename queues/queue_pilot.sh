#!/bin/bash
set -eo pipefail
export HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
R=/root/autodl-tmp/s-flm; cd $R; O=$R/eval_runs/rft_pilot; mkdir -p $O
for t in 0.1 0.4 0.7 1.0; do
  echo "=== pilot t$t $(date +%T)"
  python -u -m main mode=gsm8k_eval seed=1 eval.checkpoint_path=$R/checkpoints/tinygsm/mdlm.ckpt \
    data=gsm8k-test data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir=$R/data_cache_rftpilot data.data_path=$R/data/rft_pilot.json \
    model=small model.length=512 algo=mdlm sampler=ancestral sampler.steps=128 sampler.temperature=$t \
    sampler.use_float64=false loader.eval_batch_size=32 loader.num_workers=4 trainer.num_nodes=1 trainer.devices=1 \
    gsm8k.output_dir=$O/s128_t$t +wandb.offline=True hydra.run.dir=$O/s128_t$t \
    2>&1 | tee $O/s128_t$t.log | grep --line-buffered -E "accuracy|Error"
  python -I /root/autodl-tmp/analysis/grade_v2.py $O/s128_t$t | tail -1
done
python -I /root/autodl-tmp/analysis/rft_pilot_analyze.py $O/s128_t0.1 $O/s128_t0.4 $O/s128_t0.7 $O/s128_t1.0 | tee $O/summary.txt
echo "ALL DONE $(date +%T)"
#read -p .git .gitignore README.md __pycache__ algo.py callbacks candi_utils.py checkpoints configs data data_cache data_cache_dev data_cache_sft data_cache_test300 dataclass_patch.py dataloader.py eval_runs eval_runs_mdlm_baseline.log eval_runs_mdlm_baseline_rerun.log eval_runs_mdlm_bs128.log eval_runs_mdlm_fp32_timing.log flm_utils.py main.py metrics.py models noise_schedules.py outputs requirements.txt samplers.py sandbox_gsm8k.py scripts sft_sanity_train.log sudoku_generator.py trainer_base.py utils.py 
"
