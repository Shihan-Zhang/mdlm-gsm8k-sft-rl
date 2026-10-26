#!/bin/bash
set -eo pipefail
export HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
R=/root/autodl-tmp/s-flm; cd $R
trap 'echo "FAILED at line $LINENO"; read -p "Press Enter to close"' ERR
OUT=$R/eval_runs/sft/b_ans_lr3e-5/last-825
echo "=== eval b_ans_lr3e-5 last.ckpt $(date +%T)"
python -u -m main mode=gsm8k_eval seed=1 eval.checkpoint_path=$R/outputs/sft_b_ans_lr3e-5/checkpoints/last.ckpt training.ema=0 \
  data=gsm8k-test data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
  data.cache_dir=$R/data_cache_dev data.data_path=$R/data/gsm8k_dev.json \
  model=small model.length=512 algo=mdlm sampler=ancestral sampler.steps=128 sampler.temperature=0.1 \
  loader.eval_batch_size=32 loader.num_workers=4 trainer.num_nodes=1 trainer.devices=1 \
  gsm8k.output_dir=$OUT +wandb.offline=True hydra.run.dir=$OUT 2>&1 | tee $OUT.log | grep --line-buffered -E "accuracy|Error"
python -I /root/autodl-tmp/analysis/grade_v2.py $OUT | tail -1
for t in 0.1 1.0; do
  OUT=$R/eval_runs/dev_conf/s128_t$t
  echo "=== confidence sampler base model t$t $(date +%T)"
  python -u -m main mode=gsm8k_eval seed=1 eval.checkpoint_path=$R/checkpoints/tinygsm/mdlm.ckpt \
    data=gsm8k-test data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir=$R/data_cache_dev data.data_path=$R/data/gsm8k_dev.json \
    model=small model.length=512 algo=mdlm sampler=confidence sampler.steps=128 sampler.temperature=$t \
    loader.eval_batch_size=16 loader.num_workers=4 trainer.num_nodes=1 trainer.devices=1 \
    gsm8k.output_dir=$OUT +wandb.offline=True hydra.run.dir=$OUT 2>&1 | tee $OUT.log | grep --line-buffered -E "accuracy|Error"
  python -I /root/autodl-tmp/analysis/grade_v2.py $OUT
done
bash /root/autodl-tmp/analysis/sft_run.sh b_ans_lr1e-5 $R/data/sft/b_ans.jsonl 1e-5 5
echo "ALL DONE $(date +%T)"
read -p "Press Enter to close"
