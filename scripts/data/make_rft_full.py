"""Build full RFT sampling files: every question in gsm8k_train_rest.json repeated K times, split into chunks.
Usage: python -I make_rft_full.py [--k 8] [--chunks 4]"""
import argparse, json, math, os
ap = argparse.ArgumentParser()
ap.add_argument('--src', default='/root/autodl-tmp/s-flm/data/gsm8k_train_rest.json')
ap.add_argument('--out_dir', default='/root/autodl-tmp/s-flm/data/rft_full')
ap.add_argument('--k', type=int, default=8); ap.add_argument('--chunks', type=int, default=4)
a = ap.parse_args()
rows = json.load(open(a.src))
os.makedirs(a.out_dir, exist_ok=True)
size = math.ceil(len(rows) / a.chunks)
total = 0
for c in range(a.chunks):
    part = rows[c * size:(c + 1) * size]
    out = [r for r in part for _ in range(a.k)]
    json.dump(out, open(f'{a.out_dir}/chunk{c}.json', 'w'))
    total += len(out)
    print(f'chunk{c}: {len(part)} questions x {a.k} = {len(out)} records')
json.dump({'src': a.src, 'k': a.k, 'chunks': a.chunks, 'chunk_size': size}, open(f'{a.out_dir}/meta.json', 'w'))
print(f'total {total} records from {len(rows)} questions')
