#!/bin/bash
set -uo pipefail
cd /root/autodl-tmp/s-flm
CKPT=/root/autodl-tmp/s-flm/checkpoints/tinygsm/mdlm.ckpt
SUMMARY=/root/autodl-tmp/s-flm/eval_runs/sweep_summary.txt

run () {
  local name=$1 data=$2 cache=$3 steps=$4 temp=$5
  local out=/root/autodl-tmp/s-flm/eval_runs/$name
  echo "=== $name (steps=$steps temp=$temp) started $(date +%H:%M:%S)" | tee -a $SUMMARY
  python -u -m main mode=gsm8k_eval \
    eval.checkpoint_path=$CKPT \
    data=gsm8k-test data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir=$cache data.data_path=$data \
    model=small model.length=512 algo=mdlm \
    sampler=ancestral sampler.steps=$steps sampler.temperature=$temp \
    loader.eval_batch_size=32 loader.num_workers=4 \
    trainer.num_nodes=1 trainer.devices=1 \
    gsm8k.output_dir=$out +wandb.offline=True hydra.run.dir=$out > $out.log 2>&1
  grep "GSM8K accuracy" $out.log | tee -a $SUMMARY
  grep -o "Rank 0 sampling: 100%.*" $out.log | tail -1 | tee -a $SUMMARY
  python -I /root/autodl-tmp/analysis/error_breakdown.py $out/rank_0000.json | tee -a $SUMMARY
}

run repro/test300_s1024_t1.0 /root/autodl-tmp/s-flm/data/gsm8k_test_sub300.json /root/autodl-tmp/s-flm/data_cache_test300 1024 1.0
for s in 32 128 256; do
  for t in 1.0 0.1; do
    run dev/s${s}_t${t} /root/autodl-tmp/s-flm/data/gsm8k_dev.json /root/autodl-tmp/s-flm/data_cache_dev $s $t
  done
done
echo "ALL DONE $(date +%H:%M:%S)" | tee -a $SUMMARY
