"""Paired comparison of runs against a baseline on the same questions (needs graded.jsonl from grade_v2.py).
Usage: python -I compare_runs.py BASELINE_DIR RUN_DIR [RUN_DIR ...]
Prints accuracy, paired difference, discordant counts, exact McNemar p, and error-category shifts."""
import collections, json, math, os, sys

def load(d):
    rows = [json.loads(l) for l in open(os.path.join(d, 'graded.jsonl'))]
    return {r['prompt']: r for r in rows}

def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

CATS = ['correct', 'wrong_answer', 'name_error@call', 'returns_None', 'syntax_error']
base = load(sys.argv[1]); keys = sorted(base)
bacc = sum(base[k]['correct'] for k in keys) / len(keys)
bcat = collections.Counter(base[k]['category'] for k in keys)
print(f"baseline {sys.argv[1]}: acc {100*bacc:.1f}%  n={len(keys)}")
print(f"{'run':<44s}{'acc':>7s}{'diff':>7s}{'+':>5s}{'-':>5s}{'p':>8s}   " + '  '.join(f'{c[:10]:>10s}' for c in CATS[1:]))
for d in sys.argv[2:]:
    r = load(d)
    assert set(r) == set(keys), f'{d}: question set differs from baseline'
    acc = sum(r[k]['correct'] for k in keys) / len(keys)
    gain = sum(r[k]['correct'] and not base[k]['correct'] for k in keys)
    loss = sum(base[k]['correct'] and not r[k]['correct'] for k in keys)
    cat = collections.Counter(r[k]['category'] for k in keys)
    shifts = '  '.join(f'{100*(cat[c]-bcat[c])/len(keys):>+9.1f}%' for c in CATS[1:])
    print(f"{d[-44:]:<44s}{100*acc:6.1f}%{100*(acc-bacc):+6.1f}{gain:5d}{loss:5d}{mcnemar_exact(gain, loss):8.3f}   {shifts}")
print('diff = run - baseline (pp); + / - = questions only the run / only the baseline got right; p = exact McNemar (two-sided).')
print('category columns = change in share of all questions vs baseline (negative is good for error categories).')
