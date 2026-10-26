#!/bin/bash
# Waits until the GPU has been idle for 10 consecutive minutes (queue day4a finished), then trains and evaluates
# the B-q arm as bq_pure (same evaluation as A / C / B-ans: dev seed 1 at epochs 2-10, then epoch 10 on seeds 2-5).
set -o pipefail
R=/root/autodl-tmp/s-flm; O=$R/outputs; A=/root/autodl-tmp/analysis; D=$R/data/sft
idle=0
echo "waiting for the GPU to be idle for 10 minutes ... $(date +%T)"
while [ $idle -lt 10 ]; do
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then idle=$((idle+1)); else idle=0; fi
  sleep 60
done
echo "##### queue day4b start $(date +%T)"
test -f $D/bq_pure.jsonl || { echo "!! $D/bq_pure.jsonl missing"; read -p "Press Enter to close"; exit 1; }
bash $A/sft_run.sh bq_pure_lr3e-5 $D/bq_pure.jsonl 3e-5 10 0 330,660,990,1320,1650 || echo "!! FAILED sft_run"
bash $A/eval_seeds.sh $O/sft_bq_pure_lr3e-5/checkpoints/10-1650.ckpt bq_pure_lr3e-5_e10 "2 3 4 5" || echo "!! FAILED seeds"
echo "##### queue day4b ALL DONE $(date +%T)"
read -p "Press Enter to close"
