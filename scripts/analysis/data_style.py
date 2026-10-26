"""Style profile of SFT target code, to compare data groups (CPU only).
Usage: python -I data_style.py data/sft/b_ans.jsonl data/sft/a_tinygsm.jsonl data/sft/c_omi.jsonl"""
import ast, json, re, statistics as st, sys

def body_of(code):
    t = ast.parse(code)
    fn = next((n for n in t.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None: return None, t
    b = fn.body
    if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], 'value', None), ast.Constant): b = b[1:]
    return b, t

def prof(code):
    b, t = body_of(code)
    if b is None: return None
    names = {n.id for st_ in b for n in ast.walk(st_) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    nums = [n for st_ in b for n in ast.walk(st_) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))]
    src_body = '\n'.join(ast.unparse(s) for s in b)
    return {'body_lines': len(b), 'body_chars': len(src_body), 'assigned_names': len(names),
            'name_len': st.mean(len(x) for x in names) if names else 0,
            'num_literals': len(nums), 'float_literals': sum(isinstance(n.value, float) for n in nums),
            'uses_math': int('math.' in code), 'uses_round': int('round(' in code),
            'floor_div': int('//' in src_body), 'has_comment': int('#' in code.split('"""')[-1]),
            'ends_result_return': int(bool(re.search(r'result\s*=.*\n\s*return result\s*$', code.strip())))}

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    P = [p for p in (prof(r['code']) for r in rows) if p]
    print(f"\n{path}: {len(rows)} rows, {len({r['question'] for r in rows})} distinct questions, parsed {len(P)}")
    for k in P[0]:
        v = [p[k] for p in P]
        print(f"  {k:<20s} mean {st.mean(v):8.2f}   median {st.median(v):7.1f}")
