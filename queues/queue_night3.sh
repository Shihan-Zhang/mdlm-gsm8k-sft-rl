#!/bin/bash
set -eo pipefail
D=/root/autodl-tmp/s-flm/data/sft/b_ans.jsonl
bash /root/autodl-tmp/analysis/sft_run.sh b_ans_lr1e-4 $D 1e-4 5
bash /root/autodl-tmp/analysis/sft_run.sh b_ans_lr3e-5_e10 $D 3e-5 10 660
echo "ALL DONE $(date +%T)"
read -p "Press Enter to close"
