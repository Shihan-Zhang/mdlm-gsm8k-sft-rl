#!/bin/bash
# Waits for queue day4b (bq_pure) to finish, then evaluates base / SFT / V1 / V2 on the GSM-Plus and GSM-Symbolic subsets
# (T=0.1, seed 101, float64 - the same seed as their first test run, so per-question drops are paired).
set -o pipefail
export HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
R=/root/autodl-tmp/s-flm; O=$R/outputs; A=/root/autodl-tmp/analysis; cd $R
DONE=$R/eval_runs/final_dev/bq_pure_lr3e-5_e10/t0.1_seed5/graded.jsonl
idle=0
echo "waiting for day4b to finish ... $(date +%T)"
while true; do
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then idle=$((idle+1)); else idle=0; fi
  if [ -f $DONE ] && [ $idle -ge 2 ]; then break; fi
  if [ $idle -ge 30 ]; then echo "GPU idle 30 min without day4b output - starting anyway"; break; fi
  sleep 60
done
echo "##### queue day4c start $(date +%T)"
COMMON="data=gsm8k-test data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M model=small model.length=512 algo=mdlm sampler=ancestral sampler.steps=128 sampler.temperature=0.1 loader.eval_batch_size=32 loader.num_workers=4 trainer.num_nodes=1 trainer.devices=1 +wandb.offline=True"
gen_eval() {  # NAME CKPT SET
  local OUT=$R/eval_runs/gen/$1/$3; mkdir -p $R/eval_runs/gen/$1
  if [ -f $OUT/graded.jsonl ]; then echo "gen $1 $3 done, skip"; return 0; fi
  [ -f $2 ] || { echo "!! missing checkpoint $2"; return 1; }
  local EMA="training.ema=0"; [ "$1" = base ] && EMA=""
  echo "=== gen $1 $3 $(date +%T)"
  python -u -m main mode=gsm8k_eval seed=101 eval.checkpoint_path=$2 $EMA $COMMON \
    data.cache_dir=$R/data_cache_$3 data.data_path=$R/data/gen/$3.json \
    gsm8k.output_dir=$OUT hydra.run.dir=$OUT 2>&1 | tee $OUT.log | grep --line-buffered -E "accuracy|Error"
  python -I $A/grade_v2.py $OUT | tail -1
}
for set in gsmplus gsmsym; do
  for m in base:$R/checkpoints/tinygsm/mdlm.ckpt sft:$O/sft_b_ans_lr3e-5_e10/checkpoints/10-1650.ckpt v1:$O/rl_v1/step200.ckpt v2:$O/rl_v2/step200.ckpt; do
    gen_eval ${m%%:*} ${m#*:} $set || echo "!! FAILED gen ${m%%:*} $set"
  done
done
python -I $A/prep_gen.py analyze --models base=eval_runs/gen/base:eval_runs/test/base/t0.1_seed101 \
  sft=eval_runs/gen/sft:eval_runs/test/sft/t0.1_seed101 v1=eval_runs/gen/v1:eval_runs/test/v1/t0.1_seed101 \
  v2=eval_runs/gen/v2:eval_runs/test/v2/t0.1_seed101 | tee eval_runs/gen/summary.txt
echo "##### queue day4c ALL DONE $(date +%T)"
read -p "Press Enter to close"
