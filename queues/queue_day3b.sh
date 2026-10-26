#!/bin/bash
# Day 3 afternoon + night: RL (V1, V2, R0) -> RL dev evals -> B-q sampling -> base seeds 3-5 -> pass@k of the SFT init.
# Stages are independent: a failed stage is reported and the queue moves on. Finished stages are skipped on rerun.
set -o pipefail
export HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
R=/root/autodl-tmp/s-flm; O=$R/outputs; A=/root/autodl-tmp/analysis; cd $R
I=$O/sft_b_ans_lr3e-5_e10/checkpoints/10-1650.ckpt
BASE=$R/checkpoints/tinygsm/mdlm.ckpt
COMMON="data=gsm8k-test data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M model=small model.length=512 algo=mdlm sampler=ancestral sampler.steps=128 loader.eval_batch_size=32 loader.num_workers=4 trainer.num_nodes=1 trainer.devices=1 +wandb.offline=True"
stage() { "$@" || echo "!! FAILED: $* ($(date +%T))"; }

run_rl() {  # NAME VARIANT STEPS
  local D=$O/rl_$1
  if [ -f $D/step$3.ckpt ]; then echo "rl_$1 already done, skip"; return 0; fi
  if [ -e $D/log.jsonl ]; then echo "rl_$1 has a partial run - move $D away to rerun"; return 1; fi
  echo "=== RL $1 start $(date +%T)"
  python -u $A/rl_grpo.py --variant $2 --init_ckpt $I --out_dir $D --steps $3 --lr 1e-5 2>&1 | tee $D.log \
    | grep --line-buffered -E 'prompt pool|Error|"step": [0-9]*0,'
  echo "=== RL $1 end $(date +%T)"
}

base_eval() {  # SEED TEMP  (base checkpoint keeps EMA weights: NO training.ema=0 here)
  local OUT=$R/eval_runs/dev/s128_t$2_seed$1
  if [ -f $OUT/graded.jsonl ]; then echo "base seed$1 t$2 done, skip"; return 0; fi
  echo "=== base dev t$2 seed$1 $(date +%T)"
  python -u -m main mode=gsm8k_eval seed=$1 eval.checkpoint_path=$BASE $COMMON sampler.temperature=$2 \
    data.cache_dir=$R/data_cache_dev data.data_path=$R/data/gsm8k_dev.json \
    gsm8k.output_dir=$OUT hydra.run.dir=$OUT 2>&1 | tee $OUT.log | grep --line-buffered -E "accuracy|Error"
  python -I $A/grade_v2.py $OUT | tail -1
}

bq_chunk() {  # C
  local OUT=$R/eval_runs/bq/chunk$1; mkdir -p $R/eval_runs/bq
  if [ -f $OUT/rank_0000.json ]; then echo "bq chunk$1 done, skip"; return 0; fi
  echo "=== bq chunk$1 start $(date +%T)"
  python -u -m main mode=gsm8k_eval seed=$((3000+$1)) eval.checkpoint_path=$BASE $COMMON sampler.temperature=0.4 \
    sampler.use_float64=false data.cache_dir=$R/data_cache_bq_c$1 data.data_path=$R/data/bq/chunk$1.json \
    gsm8k.output_dir=$OUT hydra.run.dir=$OUT 2>&1 | tee $R/eval_runs/bq/chunk$1.log | grep --line-buffered -E "accuracy|Error"
  echo "=== bq chunk$1 end $(date +%T)"
}

passk_eval() {  # NAME CKPT   (raw weights; temperature 1, float64)
  local OUT=$R/eval_runs/passk/$1; mkdir -p $R/eval_runs/passk
  if [ -f $OUT/rank_0000.json ]; then echo "passk $1 done, skip"; return 0; fi
  echo "=== passk $1 start $(date +%T)"
  python -u -m main mode=gsm8k_eval seed=11 eval.checkpoint_path=$2 training.ema=0 $COMMON sampler.temperature=1.0 \
    data.cache_dir=$R/data_cache_passk data.data_path=$R/data/passk/test300x32.json \
    gsm8k.output_dir=$OUT hydra.run.dir=$OUT 2>&1 | tee $OUT.log | grep --line-buffered -E "accuracy|Error"
  echo "=== passk $1 end $(date +%T)"
}

ev() { [ -f $1 ] && bash $A/eval_seeds.sh "$@" || echo "!! missing checkpoint $1"; }   # CKPT NAME SEEDS [TEMP]

echo "##### queue day3b start $(date +%T)"
stage run_rl v1 v1 200
stage run_rl v2 v2 200
stage run_rl r0 r0 100
for v in v1 v2; do for s in 50 100 150 200; do stage ev $O/rl_$v/step$s.ckpt rl_${v}_s$s "2"; done; done
for s in 50 100; do stage ev $O/rl_r0/step$s.ckpt rl_r0_s$s "2"; done
stage ev $I b_ans_lr3e-5_e10 "2" 1.0
stage ev $O/rl_v1/step200.ckpt rl_v1_s200 "2" 1.0
stage ev $O/rl_v2/step200.ckpt rl_v2_s200 "2" 1.0
stage ev $O/rl_r0/step100.ckpt rl_r0_s100 "2" 1.0
for c in 0 1 2 3; do stage bq_chunk $c; done
for s in 3 4 5; do stage base_eval $s 0.1; done
stage passk_eval sft_init $I
echo "##### queue day3b ALL DONE $(date +%T)"
read -p "Press Enter to close"
