"""Compare models evaluated with several sampling seeds on the same questions (needs graded.jsonl in every dir).
Each model's score per question = mean correctness over its seeds. Differences vs the reference model are tested
at the question level: paired bootstrap CI (resample questions) and a sign-flip permutation test on the per-question
differences. Seeds are treated as fixed; question resampling carries the remaining noise.
Usage: python -I multi_seed_compare.py --ref base base=DIR1,DIR2 sft=DIR3,DIR4 [...]   (globs allowed in DIRs)"""
import argparse, glob, json, os, sys
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('models', nargs='+', help='NAME=DIR[,DIR...]')
ap.add_argument('--ref', required=True); ap.add_argument('--B', type=int, default=10000)
a = ap.parse_args()

def load(d):
    return {json.loads(l)['prompt']: json.loads(l)['correct'] for l in open(os.path.join(d, 'graded.jsonl'))}

M = {}
for spec in a.models:
    name, dirs = spec.split('=', 1)
    ds = sorted(d for d in sum((sorted(glob.glob(x)) for x in dirs.split(",")), []) if os.path.isdir(d))
    assert ds, f'{name}: no directories matched {dirs}'
    runs = [load(d) for d in ds]
    keys = sorted(runs[0])
    for d, r in zip(ds, runs): assert sorted(r) == keys, f'{d}: question set differs'
    M[name] = (keys, np.array([[r[k] for k in keys] for r in runs], dtype=float), ds)
keys = M[a.ref][0]
for n, (k, _, _) in M.items(): assert k == keys, f'{n}: question set differs from {a.ref}'
rng = np.random.default_rng(0)
print(f'{len(keys)} questions; per-question score = mean over seeds\n')
print(f"{'model':<22s}{'seeds':>6s}{'mean':>8s}{'seed SD':>9s}   per-seed accuracies")
for n, (_, X, ds) in M.items():
    acc = 100 * X.mean(1)
    sd = acc.std(ddof=1) if len(acc) > 1 else float('nan')
    print(f"{n:<22s}{len(acc):6d}{acc.mean():8.2f}{sd:9.2f}   " + ' '.join(f'{v:.1f}' for v in acc))
ref = M[a.ref][1].mean(0)
print(f"\nvs {a.ref}:  {'model':<20s}{'diff':>7s}{'95% CI (bootstrap over questions)':>36s}{'p (sign-flip)':>15s}")
for n, (_, X, _) in M.items():
    if n == a.ref: continue
    d = X.mean(0) - ref
    obs = d.mean()
    idx = rng.integers(0, len(d), size=(a.B, len(d)))
    boot = d[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    flips = rng.choice([-1.0, 1.0], size=(a.B, len(d)))
    perm = (flips * d).mean(1)
    p = (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (a.B + 1)
    print(f"{'':12s}{n:<20s}{100*obs:+7.2f}{'':10s}[{100*lo:+6.2f}, {100*hi:+6.2f}]{'':8s}{p:9.4f}")
print('\nnote: different seed counts per model are allowed, but compare models on matched seeds when possible '
      '(common random numbers make paired differences less noisy).')
