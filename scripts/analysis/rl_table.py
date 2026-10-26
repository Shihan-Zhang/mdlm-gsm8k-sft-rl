"""Print an RL log.jsonl as a compact table.  Usage: python -I rl_table.py OUT_DIR [OUT_DIR ...]"""
import json, sys
cols = [('step', '{:>4d}'), ('reward', '{:6.3f}'), ('official_acc', '{:6.3f}'), ('executable', '{:6.3f}'),
        ('hack_rate', '{:6.3f}'), ('mixed_groups', '{:5.2f}'), ('entropy', '{:7.4f}'), ('kl', '{:9.2e}'), ('flip', '{:6.4f}'),
        ('grad_norm', '{:7.3f}'), ('code_tokens', '{:6.1f}'), ('t_step', '{:6.1f}')]
for d in sys.argv[1:]:
    print(f'\n{d}')
    print(' '.join(f'{c[:9]:>9s}' for c, _ in cols))
    for l in open(f'{d}/log.jsonl'):
        r = json.loads(l)
        print(' '.join(f'{(f.format(r[c]) if r.get(c) is not None else "-"):>9s}' for c, f in cols))
