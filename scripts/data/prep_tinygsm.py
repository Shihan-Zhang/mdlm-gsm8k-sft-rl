"""TinyGSM pool for SFT group A (and B-q candidates), with seed-attribution decontamination.
TinyGSM was generated from GSM8K *train* questions, and our dev set is cut from GSM8K train, so roughly
500/7473 = 6.7% of TinyGSM should be variants of dev questions. Variants swap names, nouns and numbers, so an
absolute similarity threshold misses many of them. Instead each TinyGSM question is attributed to its most
similar GSM8K-train question (dev U train_rest); if that seed is a dev question, it is excluded.
GSM8K test was not a seed, so test questions use an absolute threshold.
stage pool  : sample a uniform pool, score TF-IDF cosine (digits normalised) to nearest dev / test / train_rest.
stage make_a: draw N rows (default = size of B-ans) that are not dev-attributed, test similarity < --tau_test,
              token length <= 512.
Usage: python -I prep_tinygsm.py pool [--pool 60000]
       python -I prep_tinygsm.py make_a [--tau_test 0.5]"""
import argparse, glob, json, os, random, re
import numpy as np, pyarrow.parquet as pq
ap = argparse.ArgumentParser()
ap.add_argument('stage', choices=['pool', 'make_a'])
ap.add_argument('--raw', default='/root/autodl-tmp/raw/tinygsm/data')
ap.add_argument('--data', default='/root/autodl-tmp/s-flm/data')
ap.add_argument('--pool', type=int, default=60000); ap.add_argument('--seed', type=int, default=20261025)
ap.add_argument('--tau_test', type=float, default=0.5); ap.add_argument('--n', type=int)
ap.add_argument('--margin', type=float, default=0.0)
ap.add_argument('--no_tok', action='store_true')
a = ap.parse_args()
POOL = f'{a.data}/tinygsm_pool.jsonl'

def norm(q):
    q = q.lower()
    q = re.sub(r'\d[\d,]*(\.\d+)?', ' <num> ', q)
    return re.sub(r'\s+', ' ', q).strip()

def dev_attributed(r):
    return r['sim_dev'] > r['sim_train'] - a.margin

if a.stage == 'pool':
    files = sorted(glob.glob(f'{a.raw}/*.parquet'))
    counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
    total = sum(counts)
    print(f'{len(files)} shards, {total:,} rows')
    rng = np.random.default_rng(a.seed)
    pick = np.sort(rng.choice(total, size=a.pool, replace=False))
    pool, offset = [], 0
    for f, c in zip(files, counts):
        local = pick[(pick >= offset) & (pick < offset + c)] - offset
        if len(local):
            t = pq.read_table(f, columns=['question', 'code']).take(local)
            for q, code, li in zip(t.column('question').to_pylist(), t.column('code').to_pylist(), local):
                pool.append({'question': q, 'code': code, 'src': f'{os.path.basename(f)}:{int(li)}'})
        offset += c
    print(f'pool: {len(pool):,} rows sampled uniformly (seed {a.seed})')
    from sklearn.feature_extraction.text import TfidfVectorizer
    refs = {'dev': [r['prompt'] for r in json.load(open(f'{a.data}/gsm8k_dev.json'))],
            'test': [r['prompt'] for r in json.load(open(f'{a.data}/gsm8k_test.json'))],
            'train': [r['prompt'] for r in json.load(open(f'{a.data}/gsm8k_train_rest.json'))]}
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    vec.fit([norm(r['question']) for r in pool] + [norm(q) for v in refs.values() for q in v])
    P = vec.transform([norm(r['question']) for r in pool])
    for name, ref in refs.items():
        R = vec.transform([norm(q) for q in ref])
        best, arg = np.zeros(P.shape[0]), np.zeros(P.shape[0], dtype=int)
        for s in range(0, P.shape[0], 5000):
            S = (P[s:s + 5000] @ R.T).toarray()
            best[s:s + 5000] = S.max(1); arg[s:s + 5000] = S.argmax(1)
        for i, r in enumerate(pool):
            r[f'sim_{name}'] = round(float(best[i]), 4); r[f'nn_{name}'] = ref[arg[i]]
    with open(POOL, 'w') as f:
        for r in pool: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    n_dev = sum(dev_attributed(r) for r in pool)
    expect = len(refs['dev']) / (len(refs['dev']) + len(refs['train']))
    print(f'\nattributed to a dev seed (sim_dev > sim_train): {n_dev} = {100*n_dev/len(pool):.2f}%   '
          f'(expected if attribution were perfect: {100*expect:.2f}%)')
    rnd = random.Random(0)
    for lo, hi in ((0.5, 1.01), (0.3, 0.5), (0.0, 0.3)):
        band = [r for r in pool if dev_attributed(r) and lo <= r['sim_dev'] < hi]
        print(f'\n===== dev-attributed, sim_dev in [{lo}, {min(hi,1)}): {len(band)} rows, 3 random examples =====')
        for r in rnd.sample(band, min(3, len(band))):
            print(f"  sim_dev={r['sim_dev']:.3f}  sim_train={r['sim_train']:.3f}\n  TinyGSM  : {r['question'][:260]}\n"
                  f"  dev seed : {r['nn_dev'][:260]}\n  best train_rest: {r['nn_train'][:200]}\n")
    st = np.array([r['sim_test'] for r in pool])
    print('\nmax cosine to test, distribution:')
    for lo, hi in ((0.7, 1.01), (0.6, 0.7), (0.5, 0.6), (0.4, 0.5), (0.0, 0.4)):
        print(f'  [{lo}, {min(hi,1)})  {int(((st >= lo) & (st < hi)).sum()):6d}')
    for lo, hi in ((0.6, 1.01), (0.5, 0.6), (0.4, 0.5)):
        band = [r for r in pool if lo <= r['sim_test'] < hi]
        print(f'\n===== sim_test in [{lo}, {min(hi,1)}): {len(band)} rows, 3 random examples =====')
        for r in rnd.sample(band, min(3, len(band))):
            print(f"  sim_test={r['sim_test']:.3f}\n  TinyGSM : {r['question'][:260]}\n  test    : {r['nn_test'][:260]}\n")
    print(f'wrote {POOL}')

else:
    n = a.n or sum(1 for _ in open(f'{a.data}/sft/b_ans.jsonl'))
    pool = [json.loads(l) for l in open(POOL)]
    clean = [r for r in pool if not dev_attributed(r) and r['sim_test'] < a.tau_test]
    print(f'pool {len(pool):,}; dev-attributed {sum(dev_attributed(r) for r in pool):,}; '
          f'test sim >= {a.tau_test}: {sum(r["sim_test"] >= a.tau_test for r in pool):,}; clean {len(clean):,}')
    random.Random(a.seed).shuffle(clean)
    tok = None
    if not a.no_tok:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-135M')
    out, too_long = [], 0
    for r in clean:
        if tok is not None and 2 + len(tok(r['question'])['input_ids']) + len(tok('\n')['input_ids']) + len(tok(r['code'])['input_ids']) > 512:
            too_long += 1; continue
        out.append({'question': r['question'], 'code': r['code']})
        if len(out) == n: break
    assert len(out) == n, f'only {len(out)} rows available, need {n}'
    os.makedirs(f'{a.data}/sft', exist_ok=True)
    with open(f'{a.data}/sft/a_tinygsm.jsonl', 'w') as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    json.dump({'n': n, 'rule': 'exclude if sim_dev > sim_train (dev-seed attribution) or sim_test >= tau_test',
               'tau_test': a.tau_test, 'margin': a.margin, 'pool': len(pool), 'clean': len(clean), 'too_long_skipped': too_long, 'seed': a.seed},
              open(f'{a.data}/sft/a_tinygsm_stats.json', 'w'), indent=1)
    print(f'wrote {len(out)} rows to {a.data}/sft/a_tinygsm.jsonl (skipped {too_long} too long)')
