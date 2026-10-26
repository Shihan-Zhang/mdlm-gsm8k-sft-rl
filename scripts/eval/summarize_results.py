"""Recompute the headline test numbers from the per-question records in results/test/.

For each model: accuracy per run, and the mean over the T=0.1 seeds 101-103.
For each pair (A vs B): per-question accuracy averaged over the shared seeds, the paired mean difference,
a question-level bootstrap 95% CI and a two-sided sign-flip permutation p-value.
Usage: python -I scripts/eval/summarize_results.py [--root results/test] [--boot 10000]"""
import argparse, glob, json, os, random

ap = argparse.ArgumentParser()
ap.add_argument('--root', default=os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'test'))
ap.add_argument('--boot', type=int, default=10000)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()


def load(path):
    return {r['idx']: float(r['correct']) for r in map(json.loads, open(path))}


runs = {}
for f in sorted(glob.glob(os.path.join(a.root, '*', '*.jsonl'))):
    runs.setdefault(os.path.basename(os.path.dirname(f)), {})[os.path.basename(f)[:-6]] = load(f)

print(f"{'model':<9}" + ''.join(f'{r:>15}' for r in ['t0.1_seed101', 't0.1_seed102', 't0.1_seed103', 'mean(T=0.1)', 't1.0_seed101']))
for m, rs in runs.items():
    acc = {k: 100 * sum(v.values()) / len(v) for k, v in rs.items()}
    s01 = [acc[k] for k in ('t0.1_seed101', 't0.1_seed102', 't0.1_seed103') if k in acc]
    cells = [acc.get('t0.1_seed101'), acc.get('t0.1_seed102'), acc.get('t0.1_seed103'),
             sum(s01) / len(s01) if len(s01) == 3 else None, acc.get('t1.0_seed101')]
    print(f'{m:<9}' + ''.join(f'{c:15.2f}' if c is not None else f"{'-':>15}" for c in cells))


def per_question(m, seeds):
    qs = sorted(runs[m][seeds[0]])
    return [sum(runs[m][s][q] for s in seeds) / len(seeds) for q in qs]


def compare(b, t):
    seeds = sorted(s for s in runs[b] if s.startswith('t0.1') and s in runs[t])
    x, y = per_question(b, seeds), per_question(t, seeds)
    d = [yi - xi for xi, yi in zip(x, y)]
    n, obs = len(d), sum(d) / len(d)
    rng = random.Random(a.seed)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(a.boot))
    lo, hi = boots[int(0.025 * a.boot)], boots[int(0.975 * a.boot) - 1]
    nz = [v for v in d if v != 0]
    extreme = sum(abs(sum(v if rng.random() < 0.5 else -v for v in nz)) >= abs(sum(nz)) - 1e-12 for _ in range(a.boot))
    p = (extreme + 1) / (a.boot + 1)
    print(f'{t} - {b:<5} seeds={len(seeds)}  diff {100 * obs:+6.2f} pp  95% CI [{100 * lo:+.2f}, {100 * hi:+.2f}]  sign-flip p={p:.4f}')


print()
for b, t in [('base', 'sft'), ('sft', 'v1'), ('base', 'v1'), ('v1', 'v2'), ('sft', 'r0'), ('base', 'bq_pure'), ('base', 'a_e8'), ('base', 'c_e4')]:
    if b in runs and t in runs:
        compare(b, t)
