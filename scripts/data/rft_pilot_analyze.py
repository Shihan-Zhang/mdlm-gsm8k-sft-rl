"""Analyze RFT pilot runs (each question repeated k times in the eval file).
Usage: python -I rft_pilot_analyze.py RUN_DIR [RUN_DIR ...]
Each RUN_DIR needs rank_0000.json, graded.jsonl (from grade_v2.py); optional RUN_DIR.log for timing."""
import ast, collections, json, os, re, sys
sys.path.insert(0, os.environ.get('REPO', '/root/autodl-tmp/s-flm'))
import sandbox_gsm8k as S
import builtins
_BUILTINS = set(dir(builtins))

class _Norm(ast.NodeTransformer):
    """Rename every user identifier to v0, v1, ... in order of appearance; drop docstrings."""
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

def struct_key(response):
    code = S._extract_code(response)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 'UNPARSABLE:' + code
    return ast.dump(_Norm().visit(tree), annotate_fields=False, include_attributes=False)

def minutes(run_dir):
    for p in (run_dir.rstrip('/') + '.log', os.path.join(run_dir, 'run.log')):
        if os.path.exists(p):
            m = re.findall(r'sampling: 100%.*?\[(\d+):(\d+)(?::(\d+))?<', open(p, errors='ignore').read())
            if m:
                a, b, c = m[-1]
                return (int(a) * 60 + int(b) + int(c) / 60) if c else (int(a) + int(b) / 60)
    return None

print(f"{'run':<34s}{'k':>3s}{'yield':>8s}{'pass@k':>8s}{'mixed':>7s}{'p(1-p)':>8s}{'dist/q':>8s}{'dist<=4':>8s}{'dist_tot':>9s}{'min':>7s}{'dist/min':>9s}")
for d in sys.argv[1:]:
    recs = json.load(open(os.path.join(d, 'rank_0000.json')))
    recs = recs['records'] if isinstance(recs, dict) else recs
    graded = [json.loads(l) for l in open(os.path.join(d, 'graded.jsonl'))]
    assert len(graded) == len(recs)
    byq = collections.defaultdict(list)
    for r, g in zip(recs, graded):
        assert r['prompt'] == g['prompt']
        byq[r['prompt']].append((g['correct'], r['response']))
    ks = {len(v) for v in byq.values()}; k = max(ks)
    assert len(ks) == 1, f'unequal repeats per question: {ks}'
    nq = len(byq)
    c = [sum(ok for ok, _ in v) for v in byq.values()]
    distinct = [len({struct_key(resp) for ok, resp in v if ok}) for v in byq.values()]
    yld = sum(c) / (nq * k)
    passk = sum(x > 0 for x in c) / nq
    mixed = sum(0 < x < k for x in c) / nq
    pq = sum(x * (k - x) / (k * (k - 1)) for x in c) / nq
    dq = sum(distinct) / nq; d4 = sum(min(x, 4) for x in distinct) / nq; dt = sum(min(x, 4) for x in distinct)
    mins = minutes(d)
    print(f"{os.path.basename(d.rstrip('/')):<34s}{k:3d}{100*yld:7.1f}%{100*passk:7.1f}%{100*mixed:6.1f}%{pq:8.3f}{dq:8.2f}{d4:8.2f}{dt:9d}"
          f"{(mins if mins else float('nan')):7.1f}{(dt/mins if mins else float('nan')):9.1f}")
    hist = collections.Counter(c)
    print('   correct-per-question histogram:', ' '.join(f'{i}:{hist.get(i,0)}' for i in range(k + 1)))
