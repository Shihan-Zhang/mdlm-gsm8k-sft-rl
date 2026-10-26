"""Failure-type breakdown that follows the official grader (sandbox_gsm8k.py) step by step.
Correctness is ALWAYS taken from the official evaluate_samples; the script asserts
that its own stage-by-stage replay agrees on every record.
Usage: python -I grade_v2.py RUN_DIR [--repo /root/autodl-tmp/s-flm] [--timeout 5.0]
Writes RUN_DIR/graded.jsonl (one line per record) and prints a summary."""
import argparse, collections, contextlib, io, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument('run_dir')
ap.add_argument('--repo', default='/root/autodl-tmp/s-flm')
ap.add_argument('--timeout', type=float, default=5.0)
args = ap.parse_args()
sys.path.insert(0, args.repo)
import sandbox_gsm8k as S  # official grader

def classify(sample, answer, timeout_s):
    flags = []
    stripped = sample
    fence = S.re.search(r"```(?:python)?\s*(.*?)```", sample, flags=S.re.DOTALL | S.re.IGNORECASE)
    if fence:
        flags.append('fence'); stripped = fence.group(1)
    i = stripped.find('def ')
    full = (stripped[i:] if i != -1 else stripped).strip()
    code = S._extract_code(sample)
    if code != full:
        flags.append('prefix_cut')          # official grader dropped trailing lines
    if not code:
        return 'empty', flags
    stage = 'exec'
    try:
        with S._time_limit(timeout_s):
            ns = S._safe_exec_no_timer(code)
            fn = ns.get('simple_math_problem', None)
            if fn is None:
                return 'no_function', flags
            stage = 'call'
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                out = fn()
    except S._Timeout:
        return f'timeout@{stage}', flags
    except SyntaxError:
        return 'syntax_error', flags
    except NameError:
        return f'name_error@{stage}', flags
    except ImportError:
        return f'import_blocked@{stage}', flags
    except Exception as e:
        return f'runtime_error@{stage}:{type(e).__name__}', flags
    gold = S._extract_gold_answer(answer)
    raw = out
    out = S._to_number(out)
    if isinstance(raw, str) and out is not None:
        flags.append('string_parsed')
    if out is None:
        out = S._extract_text_answer(stdout.getvalue())
        if out is not None:
            flags.append('stdout_fallback')   # answer taken from printed text
        elif raw is None:
            return 'returns_None', flags
        else:
            return 'non_numeric_return', flags
    return ('correct' if S._numbers_equal(out, gold) else 'wrong_answer'), flags

recs = json.load(open(os.path.join(args.run_dir, 'rank_0000.json')))
recs = recs['records'] if isinstance(recs, dict) else recs
cnt, flagcnt, mismatches = collections.Counter(), collections.Counter(), []
out_path = os.path.join(args.run_dir, 'graded.jsonl')
with open(out_path, 'w') as f:
    for k, r in enumerate(recs):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            official = S.evaluate_samples(r['response'], r['response_ground_truth'], args.timeout)
            cat, flags = classify(r['response'], r['response_ground_truth'], args.timeout)
        if (cat == 'correct') != bool(official):
            mismatches.append((k, cat, official))
        cnt[cat] += 1
        for fl in flags: flagcnt[(fl, cat == 'correct')] += 1
        f.write(json.dumps({'idx': k, 'prompt': r['prompt'], 'correct': bool(official),
                            'category': cat, 'flags': flags, 'chars': len(r['response'])}) + '\n')
n = len(recs)
print(f'{args.run_dir}\nn = {n}   official correct = {sum(1 for _ in open(out_path) if json.loads(_)["correct"])}')
for c, v in cnt.most_common():
    print(f'  {c:<40s}{v:5d}  {100*v/n:5.1f}%')
print('flags (flag, is_correct): count')
for (fl, ok), v in sorted(flagcnt.items()):
    print(f'  {fl:<16s} correct={ok!s:<5s} {v}')
if mismatches:
    print(f'!! {len(mismatches)} records where replay disagrees with official grader: {mismatches[:10]}')
else:
    print('replay agrees with official grader on all records')
