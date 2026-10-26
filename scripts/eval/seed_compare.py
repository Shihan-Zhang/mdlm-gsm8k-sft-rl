"""Compare per-question correctness of two runs of the same model/protocol with different seeds.
Usage: python -I seed_compare.py RUN_A RUN_B   (each must contain graded.jsonl from grade_v2.py)"""
import json, math, sys
def load(d):
    return {json.loads(l)['prompt']: json.loads(l)['correct'] for l in open(f'{d}/graded.jsonl')}
a, b = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(a) & set(b)); n = len(keys)
assert n == len(a) == len(b), (len(a), len(b), n)
n11 = sum(a[k] and b[k] for k in keys); n00 = sum((not a[k]) and (not b[k]) for k in keys)
n10 = sum(a[k] and not b[k] for k in keys); n01 = sum(b[k] and not a[k] for k in keys)
pa, pb = (n11 + n10) / n, (n11 + n01) / n
disc = (n10 + n01) / n
within = disc / 2
sd_seed = math.sqrt(within / n)
pbar = (pa + pb) / 2
sd_total = math.sqrt(pbar * (1 - pbar) / n)
print(f'n={n}  acc A={100*pa:.1f}%  acc B={100*pb:.1f}%')
print(f'both correct {n11}, both wrong {n00}, only A {n10}, only B {n01}  -> discordant {100*disc:.1f}%')
print(f'estimated mean p(1-p) within question = {within:.4f}  (max possible at this acc: {pbar*(1-pbar):.4f})')
print(f'share of single-sample variance due to sampling randomness = {within/(pbar*(1-pbar)):.2f}')
print(f'SD of accuracy from sampling alone: {100*sd_seed:.2f}pp; total binomial SD: {100*sd_total:.2f}pp')
for m in (1, 2, 4, 8):
    print(f'  averaging m={m} samples/question -> sampling SD {100*sd_seed/math.sqrt(m):.2f}pp')
