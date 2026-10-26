#!/bin/bash
set -eo pipefail
export HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
R=/root/autodl-tmp/s-flm; cd $R; O=$R/eval_runs/rft_full; mkdir -p $O
for c in 0 1 2 3; do
  OUT=$O/chunk$c
  if [ -f $OUT/rank_0000.json ]; then echo "chunk$c already done, skip"; continue; fi
  echo "=== chunk$c start $(date +%T)"
  python -u -m main mode=gsm8k_eval seed=$((1000+c)) eval.checkpoint_path=$R/checkpoints/tinygsm/mdlm.ckpt \
    data=gsm8k-test data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir=$R/data_cache_rft_c$c data.data_path=$R/data/rft_full/chunk$c.json \
    model=small model.length=512 algo=mdlm sampler=ancestral sampler.steps=128 sampler.temperature=0.4 \
    sampler.use_float64=false loader.eval_batch_size=32 loader.num_workers=4 trainer.num_nodes=1 trainer.devices=1 \
    gsm8k.output_dir=$OUT +wandb.offline=True hydra.run.dir=$OUT \
    2>&1 | tee $O/chunk$c.log | grep --line-buffered -E "accuracy|Error"
  python -I /root/autodl-tmp/analysis/grade_v2.py $OUT | tail -1
  echo "=== chunk$c end $(date +%T)"
done
echo "ALL DONE $(date +%T)"
read -p "Press Enter to close"
