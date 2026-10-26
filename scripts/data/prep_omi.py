"""SFT group C from OpenMathInstruct-1 (correct_solutions, GSM8K part).
stage peek : print field names, value counts and two raw GSM8K examples (check the format before converting).
stage build: keep GSM8K solutions with exactly one <llm-code> block, convert the block into
             def simple_math_problem() -> int: (docstring = question; final expression / print(x) -> return x),
             drop imports other than math, verify with the OFFICIAL grader against our GSM8K gold answers,
             keep train_rest questions only (dev/test excluded by exact text), structural dedup, <= M per question,
             token length <= 512, then sample N rows (default = size of B-ans).
Usage: python -I prep_omi.py peek
       python -I prep_omi.py build [--m 4]"""
import argparse, ast, builtins, collections, contextlib, glob, hashlib, io, json, os, random, re, sys
ap = argparse.ArgumentParser()
ap.add_argument('stage', choices=['peek', 'build'])
ap.add_argument('--raw', default='/root/autodl-tmp/raw/omi1/correct_solutions')
ap.add_argument('--repo', default='/root/autodl-tmp/s-flm')
ap.add_argument('--m', type=int, default=4); ap.add_argument('--n', type=int)
ap.add_argument('--seed', type=int, default=20261025); ap.add_argument('--no_tok', action='store_true')
a = ap.parse_args()
FILES = sorted(glob.glob(f'{a.raw}/*.jsonl'))
CODE_RE = re.compile(r'<llm-code>(.*?)</llm-code>', re.S)

def is_gsm(r):
    return str(r.get('dataset', '')).lower().startswith('gsm')

if a.stage == 'peek':
    keys, ds, gt, shown = collections.Counter(), collections.Counter(), collections.Counter(), 0
    for f in FILES:
        for i, line in enumerate(open(f)):
            r = json.loads(line); keys.update(r.keys()); ds[r.get('dataset')] += 1; gt[r.get('generation_type')] += 1
            if is_gsm(r) and shown < 2:
                print('=' * 70); print(json.dumps(r, ensure_ascii=False, indent=1)[:2500]); shown += 1
            if i >= 200000: break
    print('fields:', dict(keys)); print('dataset:', dict(ds)); print('generation_type:', dict(gt))
    sys.exit()

sys.path.insert(0, a.repo)
import sandbox_gsm8k as S
_B = set(dir(builtins))
norm_q = lambda q: re.sub(r'\s+', ' ', q).strip()
gold = {norm_q(r['prompt']): (r['prompt'], r['response_ground_truth'])
        for r in json.load(open(f'{a.repo}/data/gsm8k_train_rest.json'))}
banned = {norm_q(r['prompt']) for f in ('gsm8k_dev.json', 'gsm8k_test.json') for r in json.load(open(f'{a.repo}/data/{f}'))}

def to_function(question, block):
    """Return function source or (None, reason)."""
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return None, 'block_unparsable'
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = [al.name for al in n.names] if isinstance(n, ast.Import) else [n.module or '']
            if any(m.split('.')[0] != 'math' for m in mods): return None, 'non_math_import'
    body = tree.body
    if not body: return None, 'empty'
    last = body[-1]
    if isinstance(last, ast.Expr) and isinstance(last.value, ast.Call) and getattr(last.value.func, 'id', '') == 'print':
        if len(last.value.args) != 1: return None, 'print_multi_args'
        ret = last.value.args[0]
    elif isinstance(last, ast.Expr):
        ret = last.value
    else:
        return None, 'no_final_expression'
    if any(isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'print' for st in body[:-1] for n in ast.walk(st)):
        return None, 'extra_print'
    lines = [ast.unparse(st) for st in body[:-1]]
    doc = '\n'.join('    ' + l for l in question.strip().splitlines())
    src = 'def simple_math_problem() -> int:\n    """\n' + doc.replace('"""', "'''") + '\n    """\n'
    for l in lines:
        src += '\n'.join('    ' + x for x in l.splitlines()) + '\n'
    src += '    result = ' + ast.unparse(ret) + '\n    return result'
    return src, 'ok'

class _Norm(ast.NodeTransformer):
    def __init__(self): self.m = {}
    def _n(self, x):
        if x == 'simple_math_problem': return x
        if x not in self.m: self.m[x] = f'v{len(self.m)}'
        return self.m[x]
    def visit_Name(self, node):
        if node.id == 'math' or node.id in _B: return node
        node.id = self._n(node.id); return node
    def visit_FunctionDef(self, node):
        node.name = self._n(node.name); node.returns = None
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None), ast.Constant):
            node.body = node.body[1:] or [ast.Pass()]
        self.generic_visit(node); return node

reasons, byq = collections.Counter(), collections.defaultdict(collections.OrderedDict)
for f in FILES:
    for line in open(f):
        r = json.loads(line)
        if not is_gsm(r): continue
        reasons['gsm8k_solutions'] += 1
        q = norm_q(r['question'])
        if q in banned: reasons['dev_or_test_question'] += 1; continue
        if q not in gold: reasons['question_not_in_train_rest'] += 1; continue
        blocks = CODE_RE.findall(r['generated_solution'])
        if len(blocks) != 1: reasons[f'code_blocks_{min(len(blocks), 2)}'] += 1; continue
        src, why = to_function(gold[q][0], blocks[0])
        if src is None: reasons[why] += 1; continue
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ok = S.evaluate_samples(src, gold[q][1], 5.0)
        if not ok: reasons['official_grader_wrong'] += 1; continue
        key = ast.dump(_Norm().visit(ast.parse(src)), annotate_fields=False)
        if key in byq[q]: reasons['duplicate_structure'] += 1; continue
        byq[q][key] = src
        reasons['converted_ok_distinct'] += 1

tok = None
if not a.no_tok:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-135M')
rows = []
for q, d in byq.items():
    codes = list(d.values())
    random.Random(int(hashlib.md5((str(a.seed) + q).encode()).hexdigest(), 16)).shuffle(codes)
    kept = 0
    for c in codes:
        if kept == a.m: reasons['over_cap_m'] += 1; continue
        if tok is not None and 2 + len(tok(gold[q][0])['input_ids']) + len(tok('\n')['input_ids']) + len(tok(c)['input_ids']) > 512:
            reasons['too_long'] += 1; continue
        rows.append({'question': gold[q][0], 'code': c}); kept += 1
n = a.n or sum(1 for _ in open(f'{a.repo}/data/sft/b_ans.jsonl'))
random.Random(a.seed).shuffle(rows)
assert len(rows) >= n, f'only {len(rows)} rows, need {n}'
rows = rows[:n]
with open(f'{a.repo}/data/sft/c_omi.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
stats = {'reasons': dict(reasons), 'questions_with_solution': len(byq), 'rows_written': len(rows),
         'questions_in_output': len({r['question'] for r in rows}), 'm': a.m, 'seed': a.seed}
json.dump(stats, open(f'{a.repo}/data/sft/c_omi_stats.json', 'w'), indent=1)
print(json.dumps(stats, indent=1))
print('\nexample:\n' + rows[0]['code'])
