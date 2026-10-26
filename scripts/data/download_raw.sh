#!/bin/bash
set -eo pipefail
source /etc/network_turbo
cd /root/autodl-tmp
HF_HUB_OFFLINE=0 python -I - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('TinyGSM/TinyGSM', repo_type='dataset', allow_patterns=['data/*.parquet'],
                  local_dir='/root/autodl-tmp/raw/tinygsm', max_workers=4)
print('TinyGSM done')
snapshot_download('nvidia/OpenMathInstruct-1', repo_type='dataset',
                  allow_patterns=['correct_solutions/*.jsonl'],
                  local_dir='/root/autodl-tmp/raw/omi1', max_workers=2)
print('OpenMathInstruct-1 done')
PY
du -sh /root/autodl-tmp/raw/*
echo "DOWNLOAD DONE $(date +%T)"
read -p "Press Enter to close"
