"""Post-hoc variable-name repair for generated GSM8K code (analysis only, never applied to test for selection).
For each response: take the code the official grader would execute, find names that are read but never
defined, map each to the most similar defined name, then re-grade with the official grader.
Usage: python -I name_repair.py RUN_DIR [--cutoff 0.6] [--show 5]"""
import argparse, ast, builtins, collections, contextlib, difflib, io, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument('run_dir'); ap.add_argument('--repo', default=os.environ.get('REPO', '/root/autodl-tmp/s-flm'))
ap.add_argument('--cutoff', type=float, default=0.6); ap.add_argument('--timeout', type=float, default=5.0)
ap.add_argument('--show', type=int, default=3)
a = ap.parse_args()
sys.path.insert(0, a.repo)
import sandbox_gsm8k as S
KNOWN = set(dir(builtins)) | {'math'}

def defined_names(tree):
    """Symbol table: every name the code binds (assignments, loop targets, args, functions, imports)."""
    d = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)): d.add(n.id)
        elif isinstance(n, ast.arg): d.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): d.add(n.name)
        elif isinstance(n, ast.alias): d.add((n.asname or n.name).split('.')[0])
    return d

def similarity(u, c):
    """Max of character similarity and word-set overlap, so total_apples ~ apples_total."""
    ch = difflib.SequenceMatcher(None, u, c).ratio()
    pu, pc = set(u.lower().split('_')) - {''}, set(c.lower().split('_')) - {''}
    jac = len(pu & pc) / len(pu | pc) if pu | pc else 0.0
    return max(ch, jac)

class Rename(ast.NodeTransformer):
    def __init__(self, m): self.m = m
    def visit_Name(self, n):
        if isinstance(n.ctx, ast.Load) and n.id in self.m: n.id = self.m[n.id]
        return n

def repair(code, cutoff):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, {}
    d = defined_names(tree)
    undefined = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - d - KNOWN
    m = {}
    cands = sorted(d - {'simple_math_problem'})
    for u in sorted(undefined):
        best = max(cands, key=lambda c: similarity(u, c), default=None)
        if best is not None and similarity(u, best) >= cutoff: m[u] = best
    if not m:
        return None, {}
    return ast.unparse(Rename(m).visit(tree)), m

recs = json.load(open(os.path.join(a.run_dir, 'rank_0000.json')))
recs = recs['records'] if isinstance(recs, dict) else recs
cnt, examples = collections.Counter(), []
for r in recs:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        before = S.evaluate_samples(r['response'], r['response_ground_truth'], a.timeout)
        new, m = repair(S._extract_code(r['response']), a.cutoff)
        after = S.evaluate_samples(new, r['response_ground_truth'], a.timeout) if new else before
    if before and not after: cnt['correct -> wrong (repair broke it)'] += 1
    elif not new: cnt['correct, untouched' if before else 'wrong, nothing to repair'] += 1
    elif after and not before:
        cnt['wrong -> CORRECT after repair'] += 1
        if len(examples) < a.show: examples.append(m)
    else: cnt['repaired, still ' + ('correct' if after else 'wrong')] += 1
n = len(recs); b = sum(v for k, v in cnt.items() if k.startswith('correct')) + cnt['repaired, still correct']
gain = cnt['wrong -> CORRECT after repair'] - cnt['correct -> wrong (repair broke it)']
print(f'{a.run_dir}  n={n}  cutoff={a.cutoff}')
for k, v in cnt.most_common(): print(f'  {k:<36s}{v:5d}')
print(f'  accuracy before {100*b/n:.1f}%  after {100*(b+gain)/n:.1f}%  (net {gain:+d})')
for m in examples: print('  example renames:', m)
