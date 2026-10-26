"""Demo material: the denoising trajectory of SFT and RL (V1) on a few GSM8K test questions.
Questions are picked automatically from the seed-101 test runs: (a) base fails with a name error, SFT and V1 correct;
(b) SFT wrong, V1 correct; (c) all three correct. Each trajectory is decoded every --stride steps (masked tokens shown
as a placeholder) with the eval sampler settings (ancestral, 128 steps, T=0.1, float64, seed 101).
Output: eval_runs/demo/trajectories.json
Usage: python -u demo_traj.py [--stride 4] [--per_kind 1]"""
import argparse, json, os, sys
import torch
ap = argparse.ArgumentParser()
ap.add_argument('--repo', default='/root/autodl-tmp/s-flm'); ap.add_argument('--stride', type=int, default=4)
ap.add_argument('--per_kind', type=int, default=1); ap.add_argument('--seed', type=int, default=101)
a = ap.parse_args()
os.chdir(a.repo); sys.path.insert(0, a.repo); os.environ.setdefault('HF_HUB_OFFLINE', '1')
import omegaconf
for _n, _f in (('cwd', os.getcwd), ('device_count', torch.cuda.device_count), ('eval', eval),
               ('div_up', lambda x, y: (x + y - 1) // y)):
    if not omegaconf.OmegaConf.has_resolver(_n): omegaconf.OmegaConf.register_new_resolver(_n, _f)
from hydra import compose, initialize_config_dir
import algo, dataloader, sandbox_gsm8k as S
O = 'outputs'
MODELS = {'sft': f'{O}/sft_b_ans_lr3e-5_e10/checkpoints/10-1650.ckpt', 'v1': f'{O}/rl_v1/step200.ckpt'}
def graded(m): return {json.loads(l)['prompt']: json.loads(l) for l in open(f'eval_runs/test/{m}/t0.1_seed101/graded.jsonl')}
G = {m: graded(m) for m in ('base', 'sft', 'v1')}
test = json.load(open('data/gsm8k_test.json'))
kinds = {'base_name_error_fixed': lambda q: G['base'][q]['category'].startswith('name_error') and G['sft'][q]['correct'] and G['v1'][q]['correct'],
         'sft_wrong_rl_right': lambda q: (not G['sft'][q]['correct']) and G['v1'][q]['correct'],
         'all_correct': lambda q: G['base'][q]['correct'] and G['sft'][q]['correct'] and G['v1'][q]['correct']}
picked = []
for k, f in kinds.items():
    qs = [r for r in test if f(r['prompt'])]
    qs.sort(key=lambda r: len(r['prompt']))            # prefer short questions for display
    picked += [(k, r) for r in qs[:a.per_kind]]
print('picked:', [(k, r['prompt'][:60]) for k, r in picked], flush=True)
dev = torch.device('cuda')
overrides = ['data=gsm8k-test', 'data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M', 'model=small', 'model.length=512',
             'algo=mdlm', 'sampler=ancestral', 'sampler.steps=128', 'sampler.temperature=0.1', 'sampler.use_float64=true',
             'training.ema=0', f"eval.checkpoint_path={MODELS['sft']}", 'trainer.num_nodes=1', 'trainer.devices=1', '+wandb.offline=True']
with initialize_config_dir(config_dir=f'{a.repo}/configs', version_base=None):
    cfg = compose(config_name='config', overrides=overrides)
tok = dataloader.get_tokenizer(cfg)
BOS = tok.bos_token_id; SEP = tok(cfg.data.separator, add_special_tokens=False).input_ids
out = {'settings': {'sampler': 'ancestral', 'steps': 128, 'temperature': 0.1, 'float64': True, 'seed': a.seed, 'stride': a.stride},
       'questions': [{'kind': k, 'prompt': r['prompt'], 'gold': str(S._extract_gold_answer(r['response_ground_truth'])),
                      'eval_categories': {m: G[m][r['prompt']]['category'] for m in G}, 'models': {}} for k, r in picked]}
for name, path in MODELS.items():
    model = algo.MDLM.load_from_checkpoint(path, tokenizer=tok, config=cfg, strict=True).to(dev)
    assert model.ema is None
    model.backbone.eval(); model.noise.eval()
    MASK, PAD = model.mask_index, tok.pad_token_id
    for qi, (k, r) in enumerate(picked):
        torch.manual_seed(a.seed + qi)
        p = [BOS] + tok(r['prompt'].strip(), add_special_tokens=False).input_ids + SEP
        pre = torch.tensor([p], device=dev); lens = torch.tensor([len(p)], device=dev)
        with torch.no_grad():
            st = model.sampler.init_state(model, 1, num_steps=128, eps=1e-5, prefix_tokens=pre, prefix_lengths=lens)
            frames, step = [], 0
            def show(x):
                ids = x[0, len(p):].tolist(); s = []
                for t in ids:
                    if t == MASK: s.append('░')
                    elif t == PAD or t == tok.eos_token_id: s.append('')
                    else: s.append(tok.decode([t]))
                return ''.join(s).rstrip()
            frames.append({'step': 0, 'masked': int((st.xt[0, len(p):] == MASK).sum()), 'text': show(st.xt)})
            while not st.done:
                st = model.sampler.step(model, st); step += 1
                if step % a.stride == 0 or st.done:
                    frames.append({'step': step, 'masked': int((st.xt[0, len(p):] == MASK).sum()), 'text': show(st.xt)})
        final = tok.decode(st.xt[0, len(p):].tolist(), skip_special_tokens=True)
        ok = bool(S.evaluate_samples(final, r['response_ground_truth'], 5.0))
        out['questions'][qi]['models'][name] = {'frames': frames, 'final': final, 'correct': ok}
        print(f'{name} q{qi} ({k}): correct={ok}, frames={len(frames)}', flush=True)
    del model; torch.cuda.empty_cache()
for qi, (k, r) in enumerate(picked):     # final outputs of base / sft / v1 / v2 from the actual seed-101 test runs
    for m in ('base', 'sft', 'v1', 'v2'):
        recs = json.load(open(f'eval_runs/test/{m}/t0.1_seed101/rank_0000.json'))
        recs = recs['records'] if isinstance(recs, dict) else recs
        rec = next(x for x in recs if x['prompt'] == r['prompt'])
        gm = graded(m)[r['prompt']] if m == 'v2' else G[m][r['prompt']]
        out['questions'][qi].setdefault('test_run_outputs', {})[m] = {'response': rec['response'], 'correct': gm['correct'], 'category': gm['category']}
os.makedirs('eval_runs/demo', exist_ok=True)
json.dump(out, open('eval_runs/demo/trajectories.json', 'w'), ensure_ascii=False, indent=1)
print('saved eval_runs/demo/trajectories.json')
