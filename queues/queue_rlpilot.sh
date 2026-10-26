#!/bin/bash
set -eo pipefail
export HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
O=/root/autodl-tmp/s-flm/outputs; A=/root/autodl-tmp/analysis
I=$O/sft_b_ans_lr3e-5_e10/checkpoints/10-1650.ckpt
for lr in 1e-5 2e-6; do
  echo "=== pilot lr=$lr $(date +%T)"
  python -u $A/rl_grpo.py --variant v1 --init_ckpt $I --out_dir $O/rl_pilot_lr$lr \
    --steps 20 --lr $lr --kl_every 5 --save_every 1000 2>&1 | tee $O/rl_pilot_lr$lr.log \
    | grep --line-buffered -E '"step"|prompt pool|Error'
done
python -I $A/rl_table.py $O/rl_pilot_lr1e-5 $O/rl_pilot_lr2e-6
echo "ALL DONE $(date +%T)"
read -p "Press Enter to close"
