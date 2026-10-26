"""Policy drift of RL checkpoints relative to the RL init, on a FIXED probe set with FIXED masks.
Metric: exact per-position predictive KL(pi || ref) = sum_v pi(v|x_t) [log pi(v|x_t) - log ref(v|x_t)] at masked code
positions, averaged per sequence then over sequences (>= 0 by construction; no 1/p importance weights, no dependence
on how the probe sequences were sampled). Also reports the argmax flip rate (share of masked positions where the two
models' top-1 token differs). Probe = the init model's own dev responses (on-policy for the init).
Usage: python -u drift.py --ref INIT.ckpt v1_s50=CKPT v1_s100=CKPT ... [--probe RUN_DIR] [--n 256]"""
import argparse, json, os, sys
import numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument('models', nargs='+', help='NAME=CKPT')
ap.add_argument('--ref', required=True)
ap.add_argument('--repo', default='/root/autodl-tmp/s-flm')
ap.add_argument('--probe', default='eval_runs/final_dev/b_ans_lr3e-5_e10/t0.1_seed2')
ap.add_argument('--n', type=int, default=256); ap.add_argument('--ratios', default='0.25,0.5,0.75')
ap.add_argument('--micro', type=int, default=8); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='eval_runs/drift/summary.json')
args = ap.parse_args()

def pred_kl(lp_a, lp_b, m):
    """lp_*: [B, L, V] log-probs; m: [B, L] bool positions to score. Returns per-sequence mean KL(a||b), flip rate."""
    la, lb = lp_a.float().clamp(min=-50), lp_b.float().clamp(min=-50)
    kl_tok = (la.exp() * (la - lb)).sum(-1)
    flip = (la.argmax(-1) != lb.argmax(-1)).float()
    n = m.sum(-1).clamp(min=1)
    return (kl_tok * m).sum(-1) / n, (flip * m).sum(-1) / n

if __name__ == '__main__' and os.environ.get('DRIFT_SELFTEST') is None:
    os.chdir(args.repo); sys.path.insert(0, args.repo)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import omegaconf
    for _n, _f in (('cwd', os.getcwd), ('device_count', torch.cuda.device_count), ('eval', eval),
                   ('div_up', lambda x, y: (x + y - 1) // y)):
        if not omegaconf.OmegaConf.has_resolver(_n):
            omegaconf.OmegaConf.register_new_resolver(_n, _f)
    from hydra import compose, initialize_config_dir
    import algo, dataloader
    dev = torch.device('cuda')
    overrides = ['data=gsm8k-test', 'data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M', 'model=small',
                 'model.length=512', 'algo=mdlm', 'sampler=ancestral', 'training.ema=0',
                 f'eval.checkpoint_path={args.ref}', 'trainer.num_nodes=1', 'trainer.devices=1', '+wandb.offline=True']
    with initialize_config_dir(config_dir=f'{args.repo}/configs', version_base=None):
        cfg = compose(config_name='config', overrides=overrides)
    tok = dataloader.get_tokenizer(cfg)
    L = cfg.model.length
    def load(path):
        m = algo.MDLM.load_from_checkpoint(path, tokenizer=tok, config=cfg, strict=True).to(dev)
        assert m.ema is None
        m.backbone.eval(); m.noise.eval(); return m
    BOS, PAD = tok.bos_token_id, tok.pad_token_id
    SEP = tok(cfg.data.separator, add_special_tokens=False).input_ids
    recs = json.load(open(f'{args.probe}/rank_0000.json')); recs = recs['records'] if isinstance(recs, dict) else recs
    X, CODE = [], []
    for r in recs:
        p = [BOS] + tok(r['prompt'].strip(), add_special_tokens=False).input_ids + SEP
        c = tok(r['response'], add_special_tokens=False).input_ids[:L - len(p)]
        if len(c) < 8: continue
        X.append(p + c + [PAD] * (L - len(p) - len(c)))
        CODE.append([0] * len(p) + [1] * len(c) + [0] * (L - len(p) - len(c)))
        if len(X) == args.n: break
    X = torch.tensor(X, device=dev); CODE = torch.tensor(CODE, device=dev).bool()
    ratios = [float(r) for r in args.ratios.split(',')]
    g = torch.Generator(device='cpu').manual_seed(args.seed)
    MASKS = {r: (torch.rand(X.shape, generator=g) < r).to(dev) & CODE for r in ratios}
    MASK_ID = None
    ref = load(args.ref); MASK_ID = ref.mask_index   # ref stays on the GPU; full log-prob tensors are never stored
    @torch.no_grad()
    def drift(model, r):
        kls, fls = [], []
        for s in range(0, len(X), args.micro):
            x0, m = X[s:s + args.micro], MASKS[r][s:s + args.micro]
            xt = torch.where(m, torch.full_like(x0, MASK_ID), x0)
            sig = -torch.log(1 - m.float().mean(-1, keepdim=True).clamp(1e-4, 1 - 1e-4))
            k, f = pred_kl(model.forward(xt=xt, sigma=sig), ref.forward(xt=xt, sigma=sig), m)
            kls.append(k.cpu()); fls.append(f.cpu())
        return torch.cat(kls).numpy(), torch.cat(fls).numpy()
    res = {}
    print(f'probe: {len(X)} sequences from {args.probe}; masked code positions per ratio: ' +
          ', '.join(f'{r}: {int(MASKS[r].sum())}' for r in ratios))
    print(f"{'model':<16s}" + ''.join(f'{"KLx1e3@" + str(r):>16s}{"flip%@" + str(r):>12s}' for r in ratios))
    for spec in ['ref_selfcheck=' + args.ref] + args.models:
        name, path = spec.split('=', 1)
        if not os.path.exists(path): print(f'{name:<16s} missing {path}'); continue
        mdl = load(path); row = {}
        for r in ratios:
            k, f = drift(mdl, r)
            row[str(r)] = {'kl': float(k.mean()), 'kl_se': float(k.std(ddof=1) / np.sqrt(len(k))), 'flip': float(f.mean())}
        res[name] = row
        print(f'{name:<16s}' + ''.join(f"{1e3*row[str(r)]['kl']:10.3f}±{1e3*row[str(r)]['kl_se']:.3f}{100*row[str(r)]['flip']:10.2f}  " for r in ratios), flush=True)
        del mdl; torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({'ref': args.ref, 'probe': args.probe, 'n': len(X), 'ratios': ratios, 'results': res}, open(args.out, 'w'), indent=1)
    print(f'saved {args.out}')
