"""Generalisation sets: GSM-Plus (8 perturbations of each GSM8K test question) and GSM-Symbolic (templated variants of
100 GSM8K test questions; main / p1 / p2 = 0 / +1 / +2 extra clauses).
stage download: snapshot both datasets to raw/ (run after `source /etc/network_turbo`)
stage peek    : print the files found and the keys + one record of each, so the field mapping can be checked
stage make    : GSM-Plus  -> --plus_seeds random seed questions (seeded) x every perturbation type except
                             critical thinking (its answer is "none": not gradable by a numeric grader)
                GSM-Symbolic -> --sym_k instances per template for main, p1, p2
                writes data/gen/{gsmplus,gsmsym}.json (prompt, response_ground_truth='#### x') + *_meta.jsonl
stage analyze : accuracy per perturbation type / subset with bootstrap CIs, and the paired drop against the SAME model's
                accuracy on the original GSM8K test question (from its test run)
Usage: python -I prep_gen.py download | peek | make [--plus_seeds 150 --sym_k 5]
       python -I prep_gen.py analyze --models base=GEN_DIR_PREFIX:TEST_DIR sft=... """
import argparse, collections, glob, json, os, random, re, sys
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('stage', choices=['download', 'peek', 'make', 'analyze'])
ap.add_argument('--repo', default='/root/autodl-tmp/s-flm'); ap.add_argument('--raw', default='/root/autodl-tmp/raw')
ap.add_argument('--plus_seeds', type=int, default=150); ap.add_argument('--sym_k', type=int, default=5)
ap.add_argument('--seed', type=int, default=20261026)
ap.add_argument('--models', nargs='*', default=[], help='NAME=EVAL_ROOT:TEST_RUN_DIR (EVAL_ROOT holds gsmplus/ and gsmsym/)')
a = ap.parse_args()
a.raw = os.path.abspath(a.raw)
PLUS, SYM = f'{a.raw}/gsmplus', f'{a.raw}/gsmsym'
norm = lambda q: re.sub(r'\s+', ' ', str(q)).strip()

def read_any(path):
    if path.endswith('.parquet'):
        import pandas as pd
        return pd.read_parquet(path).to_dict('records')
    if path.endswith('.jsonl'):
        return [json.loads(l) for l in open(path) if l.strip()]
    d = json.load(open(path))
    return d if isinstance(d, list) else d.get('data', d)

def files(root):
    return sorted(f for ext in ('jsonl', 'json', 'parquet') for f in glob.glob(f'{root}/**/*.{ext}', recursive=True)
                  if '.cache' not in f)

def pick(rec, *names):
    for n in names:
        if n in rec and rec[n] is not None: return rec[n]
    raise KeyError(f'none of {names} in record keys {sorted(rec)}')

if a.stage == 'download':
    os.environ['HF_HUB_OFFLINE'] = '0'
    from huggingface_hub import snapshot_download
    snapshot_download('qintongli/GSM-Plus', repo_type='dataset', local_dir=PLUS, max_workers=2)
    snapshot_download('apple/GSM-Symbolic', repo_type='dataset', local_dir=SYM, max_workers=2)
    print('downloaded:', files(PLUS), files(SYM)); sys.exit()

if a.stage == 'peek':
    for root in (PLUS, SYM):
        for f in files(root):
            recs = read_any(f)
            print(f'== {f}: {len(recs)} records; keys {sorted(recs[0])}')
            print(json.dumps({k: (str(v)[:200]) for k, v in recs[0].items()}, ensure_ascii=False, indent=1))
    sys.exit()

os.chdir(a.repo); sys.path.insert(0, a.repo)
test = json.load(open('data/gsm8k_test.json'))
test_by_q = {norm(r['prompt']): r for r in test}

def num_answer(x):
    s = str(x).strip().replace(',', '').replace('$', '').rstrip('%').strip()
    if re.fullmatch(r'[-+]?\d+(\.\d+)?', s): return s
    m = re.fullmatch(r'([-+]?\d+)\s*/\s*(\d+)', s)          # fractions such as 3/4 -> 0.75 (grader tolerance 1e-3)
    if m and int(m.group(2)) != 0: return f'{int(m.group(1)) / int(m.group(2)):.6f}'.rstrip('0').rstrip('.')
    return None

if a.stage == 'make':
    os.makedirs('data/gen', exist_ok=True)
    # ---- GSM-Plus ----
    pf = [f for f in files(PLUS) if 'test' in os.path.basename(f).lower() and 'mini' not in f.lower()] or files(PLUS)
    recs = [r for f in pf for r in read_any(f)]
    types = collections.Counter(str(pick(r, 'perturbation_type', 'type')).strip().lower() for r in recs)
    assert recs, f'no GSM-Plus records under {PLUS} - run download / peek first'
    print(f'GSM-Plus files {pf}: {len(recs)} records; types {dict(types)}')
    byseed = collections.defaultdict(list)
    for r in recs: byseed[norm(pick(r, 'seed_question', 'original_question'))].append(r)
    seeds = sorted(q for q in byseed if q in test_by_q)
    print(f'seed questions matched to our GSM8K test file: {len(seeds)} / {len(byseed)}')
    random.Random(a.seed).shuffle(seeds)
    rows, meta, skipped = [], [], collections.Counter()
    for q in seeds[:a.plus_seeds]:
        for r in byseed[q]:
            t = str(pick(r, 'perturbation_type', 'type')).strip().lower()
            if 'critical' in t: skipped['critical_thinking'] += 1; continue
            ans = num_answer(pick(r, 'answer'))
            if ans is None: skipped['non_numeric_answer'] += 1; continue
            p = str(pick(r, 'question')).strip()
            rows.append({'prompt': p, 'response_ground_truth': '#### ' + ans})
            meta.append({'prompt': p, 'type': t, 'seed_question': test_by_q[q]['prompt']})
    dup = len(rows) - len({r['prompt'] for r in rows})
    json.dump(rows, open('data/gen/gsmplus.json', 'w'))
    with open('data/gen/gsmplus_meta.jsonl', 'w') as f:
        for m in meta: f.write(json.dumps(m, ensure_ascii=False) + '\n')
    print(f'GSM-Plus subset: {len(rows)} questions from {min(a.plus_seeds, len(seeds))} seeds; skipped {dict(skipped)}; '
          f'per type {dict(collections.Counter(m["type"] for m in meta))}; duplicate prompts {dup}')
    # ---- GSM-Symbolic ----
    rows, meta = [], []
    for sub in ('main', 'p1', 'p2'):
        sf = [f for f in files(SYM) if f'/{sub}/' in f or os.path.basename(f).startswith(sub)]
        recs = [r for f in sf for r in read_any(f)]
        if not recs: print(f'GSM-Symbolic {sub}: no files found'); continue
        bytpl = collections.defaultdict(list)
        for r in recs: bytpl[pick(r, 'id', 'original_id')].append(r)
        n0 = len(rows)
        for tid, rs in sorted(bytpl.items(), key=lambda kv: str(kv[0])):
            rs = sorted(rs, key=lambda r: int(pick(r, 'instance')) if 'instance' in r else 0)[:a.sym_k]
            for r in rs:
                gt = str(pick(r, 'answer'))
                if '####' not in gt:
                    ans = num_answer(gt)
                    if ans is None: continue
                    gt = '#### ' + ans
                p = str(pick(r, 'question')).strip()
                oq = norm(r.get('original_question', ''))
                rows.append({'prompt': p, 'response_ground_truth': gt})
                meta.append({'prompt': p, 'subset': sub, 'template': str(tid),
                             'seed_question': test_by_q[oq]['prompt'] if oq in test_by_q else None})
        print(f'GSM-Symbolic {sub}: {len(recs)} records, {len(bytpl)} templates -> {len(rows) - n0} questions; '
              f'original question matched for {sum(m["seed_question"] is not None for m in meta[n0:])}')
    assert rows, f'no GSM-Symbolic records under {SYM} - run download / peek first'
    json.dump(rows, open('data/gen/gsmsym.json', 'w'))
    with open('data/gen/gsmsym_meta.jsonl', 'w') as f:
        for m in meta: f.write(json.dumps(m, ensure_ascii=False) + '\n')
    print(f'GSM-Symbolic subset: {len(rows)} questions; duplicate prompts {len(rows) - len({r["prompt"] for r in rows})}')
    sys.exit()

# ---- analyze ----
def graded(d): return {json.loads(l)['prompt']: json.loads(l)['correct'] for l in open(f'{d}/graded.jsonl')}
rng = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x, float)
    if len(x) == 0: return float('nan'), float('nan'), float('nan')
    b = x[rng.integers(0, len(x), (4000, len(x)))].mean(1)
    return 100 * x.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)
for setname, key in (('gsmplus', 'type'), ('gsmsym', 'subset')):
    meta = [json.loads(l) for l in open(f'data/gen/{setname}_meta.jsonl')]
    groups = sorted({m[key] for m in meta})
    print(f'\n==== {setname}: accuracy % [95% CI]; orig = same model on the original test question, averaged over its test seeds; change = acc - orig (pp)')
    print(f"{'model':<10s}{'group':<38s}{'n':>5s}{'acc':>8s}{'CI':>16s}{'orig':>8s}{'change':>9s}")
    for spec in a.models:
        name, rest = spec.split('=', 1); root, testdir = rest.split(':', 1)
        tds = sorted(d for pat in testdir.split(',') for d in glob.glob(pat) if os.path.isdir(d))
        assert tds, f'no test run dirs match {testdir}'
        tl = [graded(d) for d in tds]
        g, t = graded(f'{root}/{setname}'), {q: float(np.mean([x[q] for x in tl])) for q in tl[0]}   # original = mean over test seeds
        for grp in groups + ['ALL']:
            ms = [m for m in meta if (grp == 'ALL' or m[key] == grp) and m['prompt'] in g]
            acc, lo, hi = ci([g[m['prompt']] for m in ms])
            pairs = [(g[m['prompt']], t[m['seed_question']]) for m in ms if m.get('seed_question') in t]
            o = 100 * np.mean([p[1] for p in pairs]) if pairs else float('nan')
            chg = 100 * np.mean([p[0] - p[1] for p in pairs]) if pairs else float('nan')
            print(f'{name:<10s}{grp[:37]:<38s}{len(ms):5d}{acc:8.1f}   [{lo:5.1f},{hi:5.1f}]{o:8.1f}{chg:+9.1f}')
