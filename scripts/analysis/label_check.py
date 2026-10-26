"""Label-noise check for B-q: compare, bin by bin of the base pass count c (out of 8), how often the base model's
MAJORITY executable answer disagrees with the reference answer, on
  GSM8K train_rest (reference = verified gold; disagreement = model errors only)  vs
  TinyGSM          (reference = TinyGSM's own code, unverified).
Within the same difficulty bin, excess disagreement on TinyGSM is a rough upper-bound signal of wrong TinyGSM labels.
Only questions with c >= 1 (those that can enter SFT data) are counted.
Usage: python -I label_check.py [--workers 8]"""
import argparse, collections, contextlib, glob, io, json, multiprocessing as mp, os, sys
ap = argparse.ArgumentParser()
ap.add_argument('--repo', default='/root/autodl-tmp/s-flm'); ap.add_argument('--workers', type=int, default=8)
a = ap.parse_args()
os.chdir(a.repo); sys.path.insert(0, a.repo)
import sandbox_gsm8k as S

def one(rec):
    resp, gt = rec
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ok = bool(S.evaluate_samples(resp, gt, 5.0))
        val = None
        try:
            with S._time_limit(1.0):
                ns = S._safe_exec_no_timer(S._extract_code(resp))
                fn = ns.get('simple_math_problem')
                out = fn() if fn else None
            if not isinstance(out, str): val = S._to_number(out)
        except BaseException:
            pass
    return ok, (round(float(val), 3) if val is not None else None)

def table(pattern, name):
    recs = []
    for d in sorted(glob.glob(pattern)):
        f = f'{d}/rank_0000.json'
        if os.path.isfile(f):
            r = json.load(open(f)); recs += r['records'] if isinstance(r, dict) else r
    with mp.Pool(a.workers) as pool:
        res = pool.map(one, [(r['response'], r['response_ground_truth']) for r in recs], chunksize=256)
    byq = collections.defaultdict(lambda: {'c': 0, 'vals': collections.Counter(), 'gt': None})
    for r, (ok, val) in zip(recs, res):
        q = byq[r['prompt']]; q['c'] += ok; q['gt'] = r['response_ground_truth']
        if val is not None: q['vals'][val] += 1
    bins = collections.defaultdict(lambda: [0, 0])
    for q in byq.values():
        if q['c'] == 0: continue
        gold = S._extract_gold_answer(q['gt'])
        maj = q['vals'].most_common(1)[0][0] if q['vals'] else None
        dis = maj is None or not S._numbers_equal(S._to_number(maj), gold)
        bins[q['c']][0] += 1; bins[q['c']][1] += dis
    return name, bins, len(recs), len(byq)

out = [table('eval_runs/rft_full/chunk*', 'GSM8K (gold)'), table('eval_runs/bq/chunk*', 'TinyGSM (unverified)')]
for name, bins, nr, nq in out:
    tot = sum(v[0] for v in bins.values()); dis = sum(v[1] for v in bins.values())
    print(f'{name}: {nr} samples, {nq} questions, {tot} with c>=1, majority disagrees {dis} ({100*dis/max(tot,1):.1f}%)')
print(f"\n{'c (of 8)':>9s}" + ''.join(f'{n:>34s}' for n, *_ in out))
for c in range(1, 9):
    cells = []
    for _, bins, *_ in out:
        n, d = bins.get(c, [0, 0])
        cells.append(f'n={n:5d}  disagree {100*d/max(n,1):5.1f}%')
    print(f'{c:>9d}' + ''.join(f'{s:>34s}' for s in cells))
g, t = out[0][1], out[1][1]
tn = sum(v[0] for v in t.values())
exp = sum(t[c][0] * (g[c][1] / g[c][0]) for c in t if g.get(c, [0, 0])[0])
obs = sum(v[1] for v in t.values())
print(f'\nTinyGSM: observed disagreement {obs} ({100*obs/tn:.1f}%); expected from GSM8K rates at the same difficulty '
      f'{exp:.0f} ({100*exp/tn:.1f}%); excess {100*(obs-exp)/tn:+.1f} pp (rough signal of wrong TinyGSM labels)')
