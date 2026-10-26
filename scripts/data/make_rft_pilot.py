"""Build the RFT pilot file: N questions from gsm8k_train_rest.json, each repeated K times."""
import argparse, json, random
ap = argparse.ArgumentParser()
ap.add_argument('--src', default='/root/autodl-tmp/s-flm/data/gsm8k_train_rest.json')
ap.add_argument('--out', default='/root/autodl-tmp/s-flm/data/rft_pilot.json')
ap.add_argument('--n', type=int, default=200); ap.add_argument('--k', type=int, default=8)
ap.add_argument('--seed', type=int, default=20261023)
a = ap.parse_args()
rows = json.load(open(a.src))
idx = sorted(random.Random(a.seed).sample(range(len(rows)), a.n))
out = [rows[i] for i in idx for _ in range(a.k)]
json.dump(out, open(a.out, 'w'))
json.dump({'src': a.src, 'seed': a.seed, 'k': a.k, 'indices': idx}, open(a.out.replace('.json', '_idx.json'), 'w'))
print(f'wrote {len(out)} records ({a.n} questions x {a.k}) to {a.out}')
