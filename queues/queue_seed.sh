#!/bin/bash
set -eo pipefail
export HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
R=/root/autodl-tmp/s-flm; cd $R
for t in 0.1 1.0; do
  echo "=== dev s128 t$t seed2 $(date +%T)"
  python -u -m main mode=gsm8k_eval seed=2 eval.checkpoint_path=$R/checkpoints/tinygsm/mdlm.ckpt \
    data=gsm8k-test data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir=$R/data_cache_dev data.data_path=$R/data/gsm8k_dev.json \
    model=small model.length=512 algo=mdlm sampler=ancestral sampler.steps=128 sampler.temperature=$t \
    loader.eval_batch_size=32 loader.num_workers=4 trainer.num_nodes=1 trainer.devices=1 \
    gsm8k.output_dir=$R/eval_runs/dev/s128_t${t}_seed2 +wandb.offline=True \
    hydra.run.dir=$R/eval_runs/dev/s128_t${t}_seed2 2>&1 | tee $R/eval_runs/seed2_t${t}.log | grep --line-buffered -E "accuracy|Error"
done
echo "ALL DONE $(date +%T)"
read -p "按回车关闭窗口"
