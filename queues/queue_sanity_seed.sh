#!/bin/bash
set -eo pipefail
export HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
R=/root/autodl-tmp/s-flm; cd $R
test ! -e $R/outputs/sft_sanity || { echo "outputs/sft_sanity 已存在，会触发续训，先删掉"; exit 1; }

echo "=== sanity train $(date +%T)"
python -u -m main data=tiny-gsm data.cache_dir=$R/data_cache_sft \
  data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
  data.wrap=False data.train_on_prompt=False data.train_on_pad=True data.filter_too_long=True \
  +data.sft_json_path=$R/data/sft_sanity64.jsonl \
  model=small model.length=512 algo=mdlm \
  training.finetune_path=$R/checkpoints/tinygsm/mdlm.ckpt +training.finetune_init_from_ema=true \
  optim.lr=0 loader.global_batch_size=8 loader.batch_size=8 loader.eval_batch_size=8 loader.num_workers=4 \
  eval.generate_samples=False trainer.num_nodes=1 trainer.devices=1 \
  trainer.max_steps=1 trainer.val_check_interval=1 trainer.limit_val_batches=1 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1 +wandb.offline=True \
  hydra.run.dir=$R/outputs/sft_sanity 2>&1 | tee $R/sft_sanity_train.log
grep -q "\[SFT patch\] initialized" $R/sft_sanity_train.log || { echo "没有出现补丁打印，停止"; exit 1; }
ls -la $R/outputs/sft_sanity/checkpoints

echo "=== sanity eval $(date +%T)"
python -u -m main mode=gsm8k_eval eval.checkpoint_path=$R/outputs/sft_sanity/checkpoints/last.ckpt training.ema=0 \
  data=gsm8k-test data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
  data.cache_dir=$R/data_cache data.data_path=$R/data/gsm8k_test.json \
  model=small model.length=512 algo=mdlm sampler=ancestral sampler.steps=32 \
  loader.eval_batch_size=32 loader.num_workers=4 trainer.num_nodes=1 trainer.devices=1 \
  gsm8k.output_dir=$R/eval_runs/sanity_lr0 +wandb.offline=True hydra.run.dir=$R/eval_runs/sanity_lr0

for t in 0.1 1.0; do
  echo "=== dev s128 t$t seed2 $(date +%T)"
  python -u -m main mode=gsm8k_eval seed=2 eval.checkpoint_path=$R/checkpoints/tinygsm/mdlm.ckpt \
    data=gsm8k-test data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir=$R/data_cache_dev data.data_path=$R/data/gsm8k_dev.json \
    model=small model.length=512 algo=mdlm sampler=ancestral sampler.steps=128 sampler.temperature=$t \
    loader.eval_batch_size=32 loader.num_workers=4 trainer.num_nodes=1 trainer.devices=1 \
    gsm8k.output_dir=$R/eval_runs/dev/s128_t${t}_seed2 +wandb.offline=True hydra.run.dir=$R/eval_runs/dev/s128_t${t}_seed2 \
    2>&1 | grep -E "accuracy|Error|error" || true
done
echo "ALL DONE $(date +%T)"
