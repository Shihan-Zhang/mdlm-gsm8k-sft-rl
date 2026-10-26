"""pass@k analysis for the Yue et al. (2025) check: does RL raise pass@1 but not pass@k at large k?
stage make   : repeat each question of the 300-question test subset N times -> data/passk/test{NQ}x{N}.json
               (evaluate it with main.py mode=gsm8k_eval at temperature 1, 128 steps, float64: low-precision
               categorical sampling reduces sample diversity, which is exactly what pass@k measures)
stage analyze: per question c = #correct (official grader) out of n; unbiased pass@k = 1 - C(n-c,k)/C(n,k)
               (Chen et al. 2021), averaged over questions, bootstrap 95% CI over questions; paired differences vs
               --ref; coverage sets (questions solved at least once) and per-question c saved to JSON.
Usage: python -I passk.py make [--n 32] [--nq 300]
       python -I passk.py analyze --ref sft sft=DIR rl=DIR [base=DIR] [--ks 1,2,4,8,16,32]"""
import argparse, collections, contextlib, io, json, math, multiprocessing as mp, os, sys
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('stage', choices=['make', 'analyze'])
ap.add_argument('models', nargs='*', help='NAME=RUN_DIR (analyze)')
ap.add_argument('--repo', default=os.environ.get('REPO', '/root/autodl-tmp/s-flm'))
ap.add_argument('--src', default='data/gsm8k_test_sub300.json')
ap.add_argument('--n', type=int, default=32); ap.add_argument('--nq', type=int, default=300)
ap.add_argument('--ref'); ap.add_argument('--ks', default='1,2,4,8,16,32')
ap.add_argument('--B', type=int, default=5000); ap.add_argument('--workers', type=int, default=8)
a = ap.parse_args()
os.chdir(a.repo); sys.path.insert(0, a.repo)

if a.stage == 'make':
    rows = json.load(open(a.src))[:a.nq]
    os.makedirs('data/passk', exist_ok=True)
    out = f'data/passk/test{len(rows)}x{a.n}.json'
    json.dump([r for r in rows for _ in range(a.n)], open(out, 'w'))
    print(f'wrote {out}: {len(rows)} questions x {a.n} = {len(rows) * a.n} records '
          f'(~{len(rows) * a.n / 32 * 22.5 / 3600:.1f} h at float64, batch 32)')
    sys.exit()

import sandbox_gsm8k as S
def grade(rec):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return bool(S.evaluate_samples(rec[0], rec[1], 5.0))

def load_counts(d):
    recs = json.load(open(f'{d}/rank_0000.json')); recs = recs['records'] if isinstance(recs, dict) else recs
    with mp.Pool(a.workers) as pool:
        ok = pool.map(grade, [(r['response'], r['response_ground_truth']) for r in recs], chunksize=64)
    c, n = collections.Counter(), collections.Counter()
    for r, o in zip(recs, ok): n[r['prompt']] += 1; c[r['prompt']] += o
    return c, n

def pass_at_k(c, n, k):
    if n - c < k: return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

ks = [int(x) for x in a.ks.split(',')]
M = {}
for spec in a.models:
    name, d = spec.split('=', 1)
    c, n = load_counts(d)
    M[name] = (c, n)
qs = sorted(next(iter(M.values()))[1])
for name, (c, n) in M.items():
    assert sorted(n) == qs, f'{name}: question set differs'
    assert len(set(n.values())) == 1, f'{name}: unequal samples per question {set(n.values())}'
nmin = min(next(iter(n.values())) for _, n in M.values())
ks = [k for k in ks if k <= nmin]
rng = np.random.default_rng(0)
idx = rng.integers(0, len(qs), size=(a.B, len(qs)))
P = {name: np.array([[pass_at_k(c[q], n[q], k) for q in qs] for k in ks]) for name, (c, n) in M.items()}
print(f'{len(qs)} questions, n = {nmin} samples each; unbiased pass@k (%) with 95% bootstrap CI over questions\n')
print(f"{'model':<14s}" + ''.join(f'{"k=" + str(k):>20s}' for k in ks))
for name, X in P.items():
    cells = []
    for i in range(len(ks)):
        b = X[i][idx].mean(1); lo, hi = np.percentile(b, [2.5, 97.5])
        cells.append(f'{100*X[i].mean():6.1f} [{100*lo:4.1f},{100*hi:4.1f}]')
    print(f'{name:<14s}' + ''.join(f'{s:>20s}' for s in cells))
if a.ref:
    print(f'\npaired difference vs {a.ref} (pp, 95% CI); the Yue et al. pattern is a positive diff at k=1 that shrinks and turns negative at large k')
    for name, X in P.items():
        if name == a.ref: continue
        cells = []
        for i in range(len(ks)):
            d = X[i] - P[a.ref][i]; b = d[idx].mean(1); lo, hi = np.percentile(b, [2.5, 97.5])
            cells.append(f'{100*d.mean():+6.1f} [{100*lo:+4.1f},{100*hi:+4.1f}]')
        print(f'{name:<14s}' + ''.join(f'{s:>20s}' for s in cells))
    cr, nr = M[a.ref]
    print(f'\ncoverage (solved at least once in n samples) vs {a.ref}:')
    for name, (c, n) in M.items():
        if name == a.ref: continue
        only_m = [q for q in qs if c[q] > 0 and cr[q] == 0]; only_r = [q for q in qs if cr[q] > 0 and c[q] == 0]
        both = sum(1 for q in qs if c[q] > 0 and cr[q] > 0)
        print(f'  {name:<12s} solved by both {both:4d} | only {name} {len(only_m):4d} | only {a.ref} {len(only_r):4d}')
        lost = sorted(only_r, key=lambda q: -cr[q])[:3]
        for q in lost: print(f'     lost (ref c={cr[q]}/{nr[q]}): {q[:110]}')
out = 'eval_runs/passk/summary.json'
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({'ks': ks, 'models': {name: {'per_question_c': {q: c[q] for q in qs}, 'n': nmin,
           'pass_at_k': {str(k): float(P[name][i].mean()) for i, k in enumerate(ks)}} for name, (c, n) in M.items()}},
          open(out, 'w'), ensure_ascii=False)
print(f'\nsaved per-question counts to {out}')
