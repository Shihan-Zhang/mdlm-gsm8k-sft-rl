#!/bin/bash
set -eo pipefail
trap 'echo "FAILED at line $LINENO"; tail -n 30 /root/autodl-tmp/s-flm/outputs/sft_b_ans_lr3e-5.log 2>/dev/null; read -p "Press Enter to close"' ERR
export HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
R=/root/autodl-tmp/s-flm; cd $R
DATA=$R/data/sft/b_ans.jsonl
RUN=$R/outputs/sft_b_ans_lr3e-5
test ! -e $RUN || { echo "$RUN exists - would auto-resume, stop"; exit 1; }
N=$(wc -l < $DATA); NVAL=$(( (N + 99) / 100 )); NTR=$((N - NVAL)); SPE=$(( (NTR + 63) / 64 )); MAX=$((5 * SPE))
echo "rows=$N train=$NTR steps/epoch=$SPE max_steps=$MAX   start $(date +%T)"

python -u -m main data=tiny-gsm data.cache_dir=$R/data_cache_sft \
  data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
  data.wrap=False data.train_on_prompt=False data.train_on_pad=True data.filter_too_long=True \
  +data.sft_json_path=$DATA \
  model=small model.length=512 algo=mdlm \
  training.finetune_path=$R/checkpoints/tinygsm/mdlm.ckpt +training.finetune_init_from_ema=true \
  optim.lr=3e-5 lr_scheduler.num_warmup_steps=20 \
  loader.global_batch_size=64 loader.batch_size=16 loader.eval_batch_size=16 loader.num_workers=4 \
  eval.generate_samples=False trainer.num_nodes=1 trainer.devices=1 \
  trainer.max_steps=$MAX trainer.val_check_interval=$SPE trainer.limit_val_batches=1.0 trainer.log_every_n_steps=10 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=null +callbacks.checkpoint_every_n_steps.every_n_epochs=1 \
  +wandb.offline=True hydra.run.dir=$RUN > $RUN.log 2>&1
grep -q "\[SFT patch\] initialized" $RUN.log || { echo "SFT patch line missing - stop"; exit 1; }
echo "train done $(date +%T)"; ls $RUN/checkpoints

for ck in $RUN/checkpoints/*-*.ckpt; do
  name=$(basename $ck .ckpt); OUT=$R/eval_runs/sft/b_ans_lr3e-5/$name
  echo "=== eval $name $(date +%T)"
  python -u -m main mode=gsm8k_eval seed=1 eval.checkpoint_path=$ck training.ema=0 \
    data=gsm8k-test data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir=$R/data_cache_dev data.data_path=$R/data/gsm8k_dev.json \
    model=small model.length=512 algo=mdlm sampler=ancestral sampler.steps=128 sampler.temperature=0.1 \
    loader.eval_batch_size=32 loader.num_workers=4 trainer.num_nodes=1 trainer.devices=1 \
    gsm8k.output_dir=$OUT +wandb.offline=True hydra.run.dir=$OUT 2>&1 | tee $OUT.log | grep --line-buffered -E "accuracy|Error"
  python -I /root/autodl-tmp/analysis/grade_v2.py $OUT | tail -1
done

for t in 0.1 1.0; do
  OUT=$R/eval_runs/dev_conf/s128_t$t
  echo "=== confidence sampler base model t$t $(date +%T)"
  python -u -m main mode=gsm8k_eval seed=1 eval.checkpoint_path=$R/checkpoints/tinygsm/mdlm.ckpt \
    data=gsm8k-test data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir=$R/data_cache_dev data.data_path=$R/data/gsm8k_dev.json \
    model=small model.length=512 algo=mdlm sampler=confidence sampler.steps=128 sampler.temperature=$t \
    loader.eval_batch_size=32 loader.num_workers=4 trainer.num_nodes=1 trainer.devices=1 \
    gsm8k.output_dir=$OUT +wandb.offline=True hydra.run.dir=$OUT 2>&1 | tee $OUT.log | grep --line-buffered -E "accuracy|Error"
  python -I /root/autodl-tmp/analysis/grade_v2.py $OUT
done
echo "ALL DONE $(date +%T)"
read -p "Press Enter to close"
