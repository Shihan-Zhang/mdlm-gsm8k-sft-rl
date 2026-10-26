#!/bin/bash
# Usage: bash eval_seeds.sh CKPT NAME "2 3 4 5" [TEMP]   -> dev, 128 steps, float64, raw weights
set -eo pipefail
CK=$1; NAME=$2; SEEDS=$3; T=${4:-0.1}
export HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
R=/root/autodl-tmp/s-flm; cd $R; mkdir -p $R/eval_runs/final_dev/$NAME
for s in $SEEDS; do
  OUT=$R/eval_runs/final_dev/$NAME/t${T}_seed$s
  echo "=== $NAME t$T seed$s $(date +%T)"
  python -u -m main mode=gsm8k_eval seed=$s eval.checkpoint_path=$CK training.ema=0 \
    data=gsm8k-test data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir=$R/data_cache_dev data.data_path=$R/data/gsm8k_dev.json \
    model=small model.length=512 algo=mdlm sampler=ancestral sampler.steps=128 sampler.temperature=$T \
    loader.eval_batch_size=32 loader.num_workers=4 trainer.num_nodes=1 trainer.devices=1 \
    gsm8k.output_dir=$OUT +wandb.offline=True hydra.run.dir=$OUT 2>&1 | tee $OUT.log | grep --line-buffered -E "accuracy|Error"
  python -I /root/autodl-tmp/analysis/grade_v2.py $OUT | tail -1
done
