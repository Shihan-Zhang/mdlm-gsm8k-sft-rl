#!/bin/bash
# Day 4: drift -> test (fresh seeds 101-103, T=0.1; seed 101 at T=1) -> pass@k for V2 and V1 -> B-q SFT -> extra test runs.
# Stages are independent; finished stages are skipped on rerun.
set -o pipefail
export HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
R=/root/autodl-tmp/s-flm; O=$R/outputs; A=/root/autodl-tmp/analysis; D=$R/data/sft; cd $R
SFT=$O/sft_b_ans_lr3e-5_e10/checkpoints/10-1650.ckpt
BASE=$R/checkpoints/tinygsm/mdlm.ckpt
V1=$O/rl_v1/step200.ckpt; V2=$O/rl_v2/step200.ckpt; R0=$O/rl_r0/step100.ckpt
AE8=$O/sft_a_tinygsm_lr3e-5/checkpoints/8-1320.ckpt; CE4=$O/sft_c_omi_lr3e-5/checkpoints/4-660.ckpt
COMMON="data=gsm8k-test data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M model=small model.length=512 algo=mdlm sampler=ancestral sampler.steps=128 loader.eval_batch_size=32 loader.num_workers=4 trainer.num_nodes=1 trainer.devices=1 +wandb.offline=True"
stage() { "$@" || echo "!! FAILED: $* ($(date +%T))"; }

test_eval() {  # NAME CKPT SEED TEMP   (base keeps EMA weights; everything else is evaluated with training.ema=0)
  local OUT=$R/eval_runs/test/$1/t$4_seed$3; mkdir -p $R/eval_runs/test/$1
  if [ -f $OUT/graded.jsonl ]; then echo "test $1 t$4 seed$3 done, skip"; return 0; fi
  [ -f $2 ] || { echo "!! missing checkpoint $2"; return 1; }
  local EMA="training.ema=0"; [ "$1" = base ] && EMA=""
  echo "=== test $1 t$4 seed$3 $(date +%T)"
  python -u -m main mode=gsm8k_eval seed=$3 eval.checkpoint_path=$2 $EMA $COMMON sampler.temperature=$4 \
    data.cache_dir=$R/data_cache_testfull data.data_path=$R/data/gsm8k_test.json \
    gsm8k.output_dir=$OUT hydra.run.dir=$OUT 2>&1 | tee $OUT.log | grep --line-buffered -E "accuracy|Error"
  python -I $A/grade_v2.py $OUT | tail -1
}

passk_eval() {  # NAME CKPT   (raw weights; temperature 1, float64; same seed 11 as the SFT run)
  local OUT=$R/eval_runs/passk/$1; mkdir -p $R/eval_runs/passk
  if [ -f $OUT/rank_0000.json ]; then echo "passk $1 done, skip"; return 0; fi
  echo "=== passk $1 start $(date +%T)"
  python -u -m main mode=gsm8k_eval seed=11 eval.checkpoint_path=$2 training.ema=0 $COMMON sampler.temperature=1.0 \
    data.cache_dir=$R/data_cache_passk data.data_path=$R/data/passk/test300x32.json \
    gsm8k.output_dir=$OUT hydra.run.dir=$OUT 2>&1 | tee $OUT.log | grep --line-buffered -E "accuracy|Error"
  echo "=== passk $1 end $(date +%T)"
}

bq_sft() {
  [ -f $D/bq_mix.jsonl ] || { echo "!! $D/bq_mix.jsonl missing (bq.py build not finished or infeasible)"; return 1; }
  if [ ! -f $O/sft_bq_mix_lr3e-5/checkpoints/10-1650.ckpt ]; then
    bash $A/sft_run.sh bq_mix_lr3e-5 $D/bq_mix.jsonl 3e-5 10 0 1650 || return 1
  fi
  bash $A/eval_seeds.sh $O/sft_bq_mix_lr3e-5/checkpoints/10-1650.ckpt bq_mix_lr3e-5_e10 "2 3 4 5"
}

echo "##### queue day4a start $(date +%T)"
stage python -u $A/drift.py --ref $SFT v1_s50=$O/rl_v1/step50.ckpt v1_s100=$O/rl_v1/step100.ckpt v1_s150=$O/rl_v1/step150.ckpt v1_s200=$V1 \
  v2_s50=$O/rl_v2/step50.ckpt v2_s100=$O/rl_v2/step100.ckpt v2_s150=$O/rl_v2/step150.ckpt v2_s200=$V2 r0_s50=$O/rl_r0/step50.ckpt r0_s100=$R0
for s in 101 102 103; do for m in base:$BASE sft:$SFT v1:$V1 v2:$V2; do stage test_eval ${m%%:*} ${m#*:} $s 0.1; done; done
for m in base:$BASE sft:$SFT v1:$V1 v2:$V2; do stage test_eval ${m%%:*} ${m#*:} 101 1.0; done
stage passk_eval v2_s200 $V2
stage passk_eval v1_s200 $V1
stage bq_sft
stage test_eval r0 $R0 101 0.1
stage test_eval r0 $R0 101 1.0
stage test_eval a_e8 $AE8 101 0.1
stage test_eval c_e4 $CE4 101 0.1
echo "##### queue day4a ALL DONE $(date +%T)"
read -p "Press Enter to close"
