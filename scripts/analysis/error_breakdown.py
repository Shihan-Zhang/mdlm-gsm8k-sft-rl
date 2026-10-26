"""Classify GSM8K generations into error types (diagnosis only; official score comes from sandbox_gsm8k.py)."""
import json, sys, re, signal, math, builtins, collections

SAFE = {k: getattr(builtins, k) for k in [
    'abs', 'all', 'any', 'bool', 'dict', 'divmod', 'enumerate', 'float', 'int', 'len', 'list',
    'map', 'max', 'min', 'pow', 'print', 'range', 'reversed', 'round', 'set', 'sorted', 'str',
    'sum', 'tuple', 'zip', 'isinstance', 'ValueError', 'ZeroDivisionError', 'Exception']}

def _imp(name, *a, **k):
    if name == 'math':
        import math as m
        return m
    raise ImportError(f'import of {name} blocked')
SAFE['__import__'] = _imp

def gold(s):
    m = re.search(r'####\s*([-\d,\.]+)', s)
    return float(m.group(1).replace(',', '')) if m else None

class Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise Timeout()

def classify(resp, g):
    i = resp.find('def ')
    if i < 0:
        return 'no_function'
    code = resp[i:]
    if 'return' not in code:
        return 'no_return(truncated?)'
    try:
        compile(code, '<s>', 'exec')
    except SyntaxError:
        return 'syntax_error'
    ns = {'__builtins__': SAFE}
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(5)
    try:
        exec(code, ns)
        f = ns.get('simple_math_problem')
        if f is None:
            return 'no_function'
        r = f()
    except Timeout:
        return 'timeout'
    except NameError:
        return 'name_error'
    except Exception as e:
        return 'runtime_error:' + type(e).__name__
    finally:
        signal.alarm(0)
    if r is None:
        return 'returns_None'
    try:
        v = float(str(r).replace(',', '').replace('$', ''))
    except Exception:
        return 'non_numeric_return'
    if not math.isfinite(v):
        return 'non_numeric_return'
    return 'correct' if g is not None and abs(v - g) < 1e-3 else 'wrong_answer'

def main(path):
    recs = json.load(open(path))
    if isinstance(recs, dict):
        recs = recs['records']
    cnt = collections.Counter()
    examples = collections.defaultdict(list)
    lens = []
    for r in recs:
        c = classify(r['response'], gold(r['response_ground_truth']))
        cnt[c] += 1
        lens.append(len(r['response']))
        if len(examples[c]) < 2:
            examples[c].append(r['response'][-300:])
    n = len(recs)
    print(f'n = {n}')
    for k, v in cnt.most_common():
        print(f'{k:28s} {v:5d}  {100*v/n:5.1f}%')
    lens.sort()
    print('response chars: min %d  median %d  p90 %d  max %d' % (lens[0], lens[n//2], lens[int(0.9*n)], lens[-1]))
    if len(sys.argv) > 2 and sys.argv[2] == '--examples':
        for k, xs in examples.items():
            print('\n=====', k)
            for x in xs:
                print('...', x)

if __name__ == '__main__':
    main(sys.argv[1])
