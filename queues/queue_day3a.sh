#!/bin/bash
set -eo pipefail
D=/root/autodl-tmp/s-flm/data/sft; O=/root/autodl-tmp/s-flm/outputs; A=/root/autodl-tmp/analysis
bash $A/sft_run.sh a_tinygsm_lr3e-5 $D/a_tinygsm.jsonl 3e-5 10 0 330,660,990,1320,1650
bash $A/sft_run.sh c_omi_lr3e-5 $D/c_omi.jsonl 3e-5 10 0 330,660,990,1320,1650
bash $A/eval_seeds.sh $O/sft_b_ans_lr3e-5_e10/checkpoints/10-1650.ckpt b_ans_lr3e-5_e10 "2 3 4 5"
bash $A/eval_seeds.sh $O/sft_b_ans_lr1e-4/checkpoints/2-330.ckpt b_ans_lr1e-4_e2 "2 3 4 5"
echo "ALL DONE $(date +%T)"
read -p "Press Enter to close"
