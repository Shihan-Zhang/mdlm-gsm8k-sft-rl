#!/bin/bash
# Usage: bash sft_run.sh NAME DATA_JSONL LR [EPOCHS]
# Trains one SFT run (checkpoint once per epoch) and evaluates every epoch checkpoint on dev
# (ancestral, 128 steps, temperature 0.1, seed 1, float64, raw weights).
set -eo pipefail
NAME=$1; DATA=$2; LR=$3; EPOCHS=${4:-5}; SKIP_STEPS=${5:-0}; EVAL_STEPS=${6:-}
export HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
R=/root/autodl-tmp/s-flm; cd $R
RUN=$R/outputs/sft_$NAME; EV=$R/eval_runs/sft/$NAME; mkdir -p $EV
trap 'echo "FAILED at line $LINENO"; tail -n 30 $RUN.log 2>/dev/null' ERR
test ! -e $RUN || { echo "$RUN exists - would auto-resume, stop"; exit 1; }
N=$(wc -l < $DATA); NVAL=$(( (N + 99) / 100 )); NTR=$((N - NVAL))
MB=$(( (NTR + 15) / 16 ))            # micro-batches (16 rows) per epoch = validation interval
SPE=$(( (MB + 3) / 4 ))              # optimizer steps per epoch (global batch 64 = 4 x 16)
MAX=$((EPOCHS * SPE))
echo "[$NAME] rows=$N train=$NTR microbatches/epoch=$MB steps/epoch=$SPE max_steps=$MAX lr=$LR  start $(date +%T)"
python -u -m main data=tiny-gsm data.cache_dir=$R/data_cache_sft \
  data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
  data.wrap=False data.train_on_prompt=False data.train_on_pad=True data.filter_too_long=True \
  +data.sft_json_path=$DATA \
  model=small model.length=512 algo=mdlm \
  training.finetune_path=$R/checkpoints/tinygsm/mdlm.ckpt +training.finetune_init_from_ema=true \
  optim.lr=$LR lr_scheduler.num_warmup_steps=20 \
  loader.global_batch_size=64 loader.batch_size=16 loader.eval_batch_size=16 loader.num_workers=4 \
  eval.generate_samples=False trainer.num_nodes=1 trainer.devices=1 \
  trainer.max_steps=$MAX trainer.val_check_interval=$MB trainer.limit_val_batches=1.0 trainer.log_every_n_steps=10 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=null +callbacks.checkpoint_every_n_steps.every_n_epochs=1 +callbacks.checkpoint_every_n_steps.save_weights_only=true "~callbacks.checkpoint_monitor" \
  +wandb.offline=True hydra.run.dir=$RUN > $RUN.log 2>&1
grep -q "\[SFT patch\] initialized" $RUN.log || { echo "SFT patch line missing - stop"; exit 1; }
echo "[$NAME] train done $(date +%T)"; ls $RUN/checkpoints
CKS=$(ls $RUN/checkpoints/*-*.ckpt)
ls $RUN/checkpoints/*-$MAX.ckpt >/dev/null 2>&1 || CKS="$CKS $RUN/checkpoints/last.ckpt"
for ck in $CKS; do
  name=$(basename $ck .ckpt); OUT=$EV/$name
  step=${name##*-}; if [[ "$step" =~ ^[0-9]+$ ]] && [ "$step" -le "$SKIP_STEPS" ]; then echo "skip $name"; continue; fi
  if [ -n "$EVAL_STEPS" ] && [[ ",$EVAL_STEPS," != *",$step,"* ]]; then echo "skip $name"; continue; fi
  echo "=== [$NAME] eval $name $(date +%T)"
  python -u -m main mode=gsm8k_eval seed=1 eval.checkpoint_path=$ck training.ema=0 \
    data=gsm8k-test data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir=$R/data_cache_dev data.data_path=$R/data/gsm8k_dev.json \
    model=small model.length=512 algo=mdlm sampler=ancestral sampler.steps=128 sampler.temperature=0.1 \
    loader.eval_batch_size=32 loader.num_workers=4 trainer.num_nodes=1 trainer.devices=1 \
    gsm8k.output_dir=$OUT +wandb.offline=True hydra.run.dir=$OUT 2>&1 | tee $OUT.log | grep --line-buffered -E "accuracy|Error"
  python -I /root/autodl-tmp/analysis/grade_v2.py $OUT | tail -1
done
echo "[$NAME] done $(date +%T)"
