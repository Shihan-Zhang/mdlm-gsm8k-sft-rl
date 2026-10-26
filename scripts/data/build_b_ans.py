"""Build the B-ans SFT set (RFT, answer diversity) from full RFT sampling outputs.
Pipeline per sample: official grader correct -> target = code the grader actually executes (_extract_code)
-> strict checks (only imports + one simple_math_problem at top level; numeric return, no print/string fallback)
-> per-question structural dedup (AST, identifiers renamed) -> keep at most M per question (seeded random)
-> token-length filter (<= model length) -> write {question, code} jsonl + stats + per-question pass rates.
Usage: python -I build_b_ans.py [--runs 'eval_runs/rft_full/chunk*'] [--m 4] [--no_tok] [--allow_partial]"""
import argparse, ast, builtins, collections, contextlib, glob, hashlib, io, json, os, random, sys
ap = argparse.ArgumentParser()
ap.add_argument('--repo', default=os.environ.get('REPO', '/root/autodl-tmp/s-flm'))
ap.add_argument('--runs', default='eval_runs/rft_full/chunk*')
ap.add_argument('--expect_chunks', type=int, default=4); ap.add_argument('--allow_partial', action='store_true')
ap.add_argument('--dev', default='data/gsm8k_dev.json'); ap.add_argument('--test', default='data/gsm8k_test.json')
ap.add_argument('--out', default='data/sft/b_ans.jsonl')
ap.add_argument('--m', type=int, default=4); ap.add_argument('--seed', type=int, default=20261024)
ap.add_argument('--max_len', type=int, default=512); ap.add_argument('--no_tok', action='store_true')
ap.add_argument('--timeout', type=float, default=5.0)
a = ap.parse_args()
a.repo = os.path.abspath(a.repo); os.chdir(a.repo); sys.path.insert(0, a.repo)
import sandbox_gsm8k as S
_BUILTINS = set(dir(builtins))

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

def struct_key(tree):
    return ast.dump(_Norm().visit(tree), annotate_fields=False, include_attributes=False)

def strict_ok(code, answer):
    """Return (ok, reason). Stricter than the official grader so training targets never rely on its leniencies."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, 'target_unparsable', None
    body = tree.body
    fns = [n for n in body if isinstance(n, ast.FunctionDef)]
    if len(fns) != 1 or fns[0].name != 'simple_math_problem' or any(not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef)) for n in body):
        return False, 'extra_top_level_code', None
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'print' for n in ast.walk(tree)):
        return False, 'uses_print', None
    try:
        with S._time_limit(a.timeout):
            ns = S._safe_exec_no_timer(code)
            out = ns['simple_math_problem']()
    except Exception:
        return False, 'target_fails_to_run', None
    if isinstance(out, str) or S._to_number(out) is None:
        return False, 'non_numeric_return', None
    if not S._numbers_equal(S._to_number(out), S._extract_gold_answer(answer)):
        return False, 'target_wrong', None
    return True, 'ok', tree

runs = sorted(d for d in glob.glob(a.runs) if os.path.isfile(os.path.join(d, 'rank_0000.json')))
if len(runs) != a.expect_chunks and not a.allow_partial:
    sys.exit(f'found {len(runs)} finished chunks, expected {a.expect_chunks} (use --allow_partial for a preview)')
dev_q = {r['prompt'].strip() for r in json.load(open(a.dev))}
test_q = {r['prompt'].strip() for r in json.load(open(a.test))}

byq = collections.OrderedDict()
for d in runs:
    recs = json.load(open(os.path.join(d, 'rank_0000.json')))
    recs = recs['records'] if isinstance(recs, dict) else recs
    for r in recs:
        byq.setdefault(r['prompt'], {'answer': r['response_ground_truth'], 'responses': []})['responses'].append(r['response'])
leak = [q for q in byq if q.strip() in dev_q or q.strip() in test_q]
assert not leak, f'{len(leak)} questions overlap dev/test - abort'

tok = None
if not a.no_tok:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-135M')

reasons, rows, pass_rates = collections.Counter(), [], {}
n_prefix_cut, distinct_hist, kept_hist = 0, collections.Counter(), collections.Counter()
for q, v in byq.items():
    k = len(v['responses']); c = 0; uniq = collections.OrderedDict()
    for resp in v['responses']:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            official = S.evaluate_samples(resp, v['answer'], a.timeout)
            if not official:
                reasons['official_wrong'] += 1; continue
            c += 1
            code = S._extract_code(resp)
            ok, why, tree = strict_ok(code, v['answer'])
        if not ok:
            reasons[why] += 1; continue
        stripped = resp[resp.find('def '):].strip() if 'def ' in resp else resp.strip()
        if code != stripped: n_prefix_cut += 1
        key = struct_key(tree)
        if key in uniq: reasons['duplicate_structure'] += 1; continue
        uniq[key] = code
    pass_rates[q] = {'correct': c, 'k': k}
    distinct_hist[len(uniq)] += 1
    codes = list(uniq.values())
    rnd = random.Random(int(hashlib.md5((str(a.seed) + q).encode()).hexdigest(), 16))
    rnd.shuffle(codes)
    if len(codes) > a.m: reasons['over_cap_m'] += len(codes) - a.m
    kept = 0
    for code in codes[:a.m]:
        if tok is not None:
            n_tok = 1 + len(tok(q)['input_ids']) + len(tok('\n')['input_ids']) + len(tok(code)['input_ids']) + 1
            if n_tok > a.max_len: reasons['too_long'] += 1; continue
        rows.append({'question': q, 'code': code}); kept += 1
    kept_hist[kept] += 1

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, 'w') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
nq = len(byq); covered = sum(1 for v in pass_rates.values() if v['correct'] > 0)
stats = {'chunks': runs, 'questions': nq, 'samples': sum(v['k'] for v in pass_rates.values()),
         'official_correct': sum(v['correct'] for v in pass_rates.values()),
         'coverage_any_correct': covered / nq, 'questions_with_kept_target': nq - kept_hist[0],
         'kept_rows': len(rows), 'mean_kept_per_question': len(rows) / nq,
         'targets_cleaned_by_prefix_cut': n_prefix_cut, 'drop_reasons': dict(reasons),
         'distinct_correct_hist': dict(sorted(distinct_hist.items())), 'kept_per_question_hist': dict(sorted(kept_hist.items())),
         'm': a.m, 'seed': a.seed, 'max_len': a.max_len, 'tokenized_length_check': tok is not None}
json.dump(stats, open(a.out.replace('.jsonl', '_stats.json'), 'w'), indent=1, ensure_ascii=False)
json.dump(pass_rates, open(a.out.replace('.jsonl', '_pass_rates.json'), 'w'), ensure_ascii=False)
print(json.dumps(stats, indent=1, ensure_ascii=False))
