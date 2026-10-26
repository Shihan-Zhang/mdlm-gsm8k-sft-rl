"""SFT group B-q (question diversity at fixed N) from TinyGSM questions answered by the base model itself.
stage make : take the first --nq questions of SFT group A (already decontaminated: no dev-seed attribution, test
             similarity < 0.5), compute the TinyGSM reference answer by running TinyGSM's own code in the official
             sandbox, and write sampling files (each question repeated K times, split into chunks).
             (Questions shared with A on purpose: on these questions A = TinyGSM's answers, B-q = the model's own.)
stage build: mirror B-ans exactly (official grader vs the reference answer, cleaned target = _extract_code, strict
             checks, AST dedup, token length <= 512), then assemble
               bq_mix  = every B-ans question once (4,790 rows; 1 GSM8K solution each)
                         + TinyGSM questions once each, stratified so their base pass-rate histogram (c out of 8)
                           matches the B-ans questions' histogram; total N = size of B-ans
               bq_pure = (optional, --pure) TinyGSM questions only, up to M solutions each, A order, N rows
                         (the self-answer x TinyGSM-question cell of the 2x2 data design)
             and reports label diagnostics: share of kept TinyGSM questions whose majority executable answer among
             the K samples disagrees with the TinyGSM reference (TinyGSM answers are unverified).
Usage: python -I bq.py make  [--nq 8000] [--k 8] [--chunks 4]
       python -I bq.py build [--allow_partial] [--pure]"""
import argparse, ast, builtins, collections, contextlib, glob, hashlib, io, json, math, os, random, sys
ap = argparse.ArgumentParser()
ap.add_argument('stage', choices=['make', 'build'])
ap.add_argument('--repo', default=os.environ.get('REPO', '/root/autodl-tmp/s-flm'))
ap.add_argument('--nq', type=int, default=8000); ap.add_argument('--k', type=int, default=8)
ap.add_argument('--chunks', type=int, default=4); ap.add_argument('--m', type=int, default=4)
ap.add_argument('--seed', type=int, default=20261026); ap.add_argument('--timeout', type=float, default=5.0)
ap.add_argument('--allow_partial', action='store_true'); ap.add_argument('--pure', action='store_true')
ap.add_argument('--no_tok', action='store_true'); ap.add_argument('--max_len', type=int, default=512)
a = ap.parse_args()
os.chdir(a.repo); sys.path.insert(0, a.repo)
import sandbox_gsm8k as S
DATA, RUNS = 'data/bq', 'eval_runs/bq'
_BUILTINS = set(dir(builtins))

def quiet_run(code, timeout=None):
    """Run code in the official sandbox; return the numeric value of simple_math_problem() or None."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with S._time_limit(timeout or a.timeout):
                ns = S._safe_exec_no_timer(code)
                fn = ns.get('simple_math_problem')
                if fn is None: return None
                out = fn()
    except BaseException:
        return None
    if isinstance(out, str): return None
    return S._to_number(out)

def fmt(x):
    if isinstance(x, int): return str(x)
    s = f'{x:.6f}'.rstrip('0').rstrip('.')
    return s if s not in ('', '-0') else '0'

if a.stage == 'make':
    rows = [json.loads(l) for l in open('data/sft/a_tinygsm.jsonl')][:a.nq]
    dev_test = {r['prompt'].strip() for f in ('gsm8k_dev.json', 'gsm8k_test.json') for r in json.load(open(f'data/{f}'))}
    keep, why = [], collections.Counter()
    for r in rows:
        if r['question'].strip() in dev_test: why['exact_dev_or_test'] += 1; continue
        v = quiet_run(r['code'])
        if v is None: why['reference_code_fails_or_non_numeric'] += 1; continue
        gt = '#### ' + fmt(v)
        if not S._numbers_equal(S._extract_gold_answer(gt), v): why['answer_format_roundtrip_failed'] += 1; continue
        keep.append({'prompt': r['question'], 'response_ground_truth': gt, 'tinygsm_code': r['code']})
    os.makedirs(DATA, exist_ok=True)
    size = math.ceil(len(keep) / a.chunks)
    for c in range(a.chunks):
        part = keep[c * size:(c + 1) * size]
        json.dump([{'prompt': r['prompt'], 'response_ground_truth': r['response_ground_truth']} for r in part for _ in range(a.k)],
                  open(f'{DATA}/chunk{c}.json', 'w'))
        print(f'chunk{c}: {len(part)} questions x {a.k} = {len(part) * a.k} records')
    json.dump({'nq_requested': a.nq, 'kept': len(keep), 'dropped': dict(why), 'k': a.k, 'chunks': a.chunks},
              open(f'{DATA}/meta.json', 'w'), indent=1)
    with open(f'{DATA}/references.jsonl', 'w') as f:
        for r in keep: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'kept {len(keep)} / {len(rows)} questions; dropped {dict(why)}')
    sys.exit()

# ---------------- build ----------------
class _Norm(ast.NodeTransformer):
    def __init__(self): self.m = {}
    def _n(self, x):
        if x == 'simple_math_problem': return x
        if x not in self.m: self.m[x] = f'v{len(self.m)}'
        return self.m[x]
    def visit_Name(self, node):
        if node.id == 'math' or node.id in _BUILTINS: return node
        node.id = self._n(node.id); return node
    def visit_arg(self, node):
        node.arg = self._n(node.arg); node.annotation = None; return node
    def visit_FunctionDef(self, node):
        node.name = self._n(node.name); node.returns = None
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None), ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
        self.generic_visit(node); return node

def strict_ok(code, answer):
    try: tree = ast.parse(code)
    except SyntaxError: return False, 'target_unparsable', None
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(fns) != 1 or fns[0].name != 'simple_math_problem' or any(not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef)) for n in tree.body):
        return False, 'extra_top_level_code', None
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'print' for n in ast.walk(tree)):
        return False, 'uses_print', None
    v = quiet_run(code)
    if v is None: return False, 'non_numeric_or_fails', None
    if not S._numbers_equal(v, S._extract_gold_answer(answer)): return False, 'target_wrong', None
    return True, 'ok', tree

meta = json.load(open(f'{DATA}/meta.json'))
runs = sorted(d for d in glob.glob(f'{RUNS}/chunk*') if os.path.isfile(f'{d}/rank_0000.json'))
if len(runs) != meta['chunks'] and not a.allow_partial:
    sys.exit(f'{len(runs)}/{meta["chunks"]} chunks finished (use --allow_partial for a feasibility preview)')
ref_order = [json.loads(l)['prompt'] for l in open(f'{DATA}/references.jsonl')]
byq = collections.OrderedDict()
for d in runs:
    recs = json.load(open(f'{d}/rank_0000.json')); recs = recs['records'] if isinstance(recs, dict) else recs
    for r in recs:
        byq.setdefault(r['prompt'], {'answer': r['response_ground_truth'], 'responses': []})['responses'].append(r['response'])
tok = None
if not a.no_tok:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-135M')
def fits(q, code):
    if tok is None: return True
    return 1 + len(tok(q)['input_ids']) + len(tok('\n')['input_ids']) + len(tok(code)['input_ids']) + 1 <= a.max_len

reasons, tg, label_disagree = collections.Counter(), collections.OrderedDict(), 0
pass_hist_tg = collections.Counter()
for q in ref_order:
    if q not in byq: continue
    v = byq[q]; c, uniq, answers = 0, collections.OrderedDict(), collections.Counter()
    for resp in v['responses']:
        val = quiet_run(S._extract_code(resp), timeout=1.0)   # label diagnostic only
        if val is not None: answers[round(float(val), 3)] += 1
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            official = S.evaluate_samples(resp, v['answer'], a.timeout)
        if not official: reasons['official_wrong'] += 1; continue
        c += 1
        code = S._extract_code(resp)
        ok, why, tree = strict_ok(code, v['answer'])
        if not ok: reasons[why] += 1; continue
        key = ast.dump(_Norm().visit(tree), annotate_fields=False)
        if key in uniq: reasons['duplicate_structure'] += 1; continue
        uniq[key] = code
    codes = list(uniq.values())
    random.Random(int(hashlib.md5((str(a.seed) + q).encode()).hexdigest(), 16)).shuffle(codes)
    codes = [x for x in codes if fits(q, x)]
    if codes:
        ref = S._extract_gold_answer(v['answer'])
        maj = answers.most_common(1)[0][0] if answers else None
        disagree = maj is None or not S._numbers_equal(S._to_number(maj), ref)
        label_disagree += disagree
        tg[q] = {'c': c, 'k': len(v['responses']), 'codes': codes, 'majority_disagrees': disagree}
        pass_hist_tg[c] += 1

# GSM8K part: every B-ans question once (first row per question; B-ans rows are already seeded-shuffled per question)
bans = [json.loads(l) for l in open('data/sft/b_ans.jsonl')]
N = len(bans)
first = collections.OrderedDict()
for r in bans: first.setdefault(r['question'], r)
pr = json.load(open('data/sft/b_ans_pass_rates.json'))
target = collections.Counter(pr[q]['correct'] for q in first)
n_tg = N - len(first)
quota = {c: n_tg * target[c] / len(first) for c in target}
alloc = {c: int(math.floor(x)) for c, x in quota.items()}
for c in sorted(quota, key=lambda c: quota[c] - alloc[c], reverse=True)[:n_tg - sum(alloc.values())]: alloc[c] += 1
rnd = random.Random(a.seed)
bins = collections.defaultdict(list)
for q, d in tg.items(): bins[d['c']].append(q)
for c in bins: rnd.shuffle(bins[c])
chosen, shortfall = [], {}
for c in sorted(alloc):
    take = bins[c][:alloc[c]]; chosen += take
    if len(take) < alloc[c]: shortfall[c] = alloc[c] - len(take)
left = sum(shortfall.values())
if left:   # fill from the nearest bins that still have questions (recorded as a deviation from the target histogram)
    for c in sorted(shortfall):
        need = shortfall[c]
        for dist in range(1, 9):
            for cc in (c - dist, c + dist):
                spare = [q for q in bins.get(cc, []) if q not in set(chosen)]
                t = spare[:need]; chosen += t; need -= len(t)
                if need == 0: break
            if need == 0: break
achieved = collections.Counter(tg[q]['c'] for q in chosen)
tv = 0.5 * sum(abs(achieved[c] / max(len(chosen), 1) - target[c] / len(first)) for c in range(1, 9))
mix = [{'question': q, 'code': r['code'], 'src': 'gsm8k'} for q, r in first.items()] + \
      [{'question': q, 'code': tg[q]['codes'][0], 'src': 'tinygsm'} for q in chosen]
random.Random(a.seed).shuffle(mix)
stats = {'chunks_used': runs, 'tinygsm_questions_sampled': len([q for q in ref_order if q in byq]),
         'tinygsm_questions_with_target': len(tg), 'drop_reasons': dict(reasons),
         'tinygsm_pass_hist_(c_of_k)': dict(sorted(pass_hist_tg.items())),
         'gsm8k_target_hist': dict(sorted(target.items())), 'tinygsm_alloc': dict(sorted(alloc.items())),
         'shortfall_by_bin': shortfall, 'achieved_tinygsm_hist': dict(sorted(achieved.items())),
         'tv_distance_to_target': round(tv, 4), 'N': N, 'gsm8k_rows': len(first), 'tinygsm_rows_needed': n_tg,
         'tinygsm_rows_written': len(chosen),
         'label_check_majority_disagrees_share': round(label_disagree / max(len(tg), 1), 4),
         'label_check_majority_disagrees_share_in_bq_mix': round(sum(tg[q]['majority_disagrees'] for q in chosen) / max(len(chosen), 1), 4)}
feasible = len(chosen) == n_tg
if feasible:
    with open('data/sft/bq_mix.jsonl', 'w') as f:
        for r in mix: f.write(json.dumps({'question': r['question'], 'code': r['code']}, ensure_ascii=False) + '\n')
    stats['bq_mix_written'] = len(mix)
if a.pure:
    pure = []
    for q, d in tg.items():
        for code in d['codes'][:a.m]:
            if len(pure) < N: pure.append({'question': q, 'code': code})
    stats['bq_pure_rows_available'] = len(pure)
    stats['bq_pure_questions'] = len({r['question'] for r in pure})
    if len(pure) == N:
        with open('data/sft/bq_pure.jsonl', 'w') as f:
            for r in pure: f.write(json.dumps(r, ensure_ascii=False) + '\n')
json.dump(stats, open('data/sft/bq_stats.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(stats, indent=1, ensure_ascii=False))
print('\nbq_mix', 'WRITTEN' if feasible else f'NOT written: only {len(chosen)} of {n_tg} TinyGSM rows available')
if tv > 0.05: print(f'WARNING: difficulty histogram deviates from B-ans (TV = {tv:.3f}); see shortfall_by_bin')
