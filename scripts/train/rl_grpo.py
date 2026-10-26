"""Critic-free GRPO-family RL for the s-flm MDLM GSM8K model (design: Notion section 2.2).
  rollouts : ancestral sampler, temperature 1.0, eval-identical prompt format and decoding
  reward   : 1 iff the official grader says correct AND the answer is a direct numeric return
             (print / string fallbacks of the official grader earn nothing)
  advantage: r - group mean (Dr. GRPO / RLOO style, no std scaling); groups with identical rewards are skipped
  likelihood: ELBO with complementary (antithetic) masks. Pass A masks completion tokens w.p. p, pass B masks the
             complement, so each completion token is scored exactly once; MDLM weights (1-eps)/p and (1-eps)/(1-p)
  mu = 1   : one gradient step per rollout batch -> plain policy gradient, no ratio, no clipping, beta = 0
  variant v1: uniform credit; v2: per-token weight 1 + lambda * z(DPS) (DACA-GRPO denoising progress score);
          r0: Bernoulli(0.5) random rewards (spurious-reward control)
Usage: python -u rl_grpo.py --variant v1 --init_ckpt CKPT --out_dir DIR [--steps 200]"""
import argparse, contextlib, io, json, math, os, random, sys, time
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--variant', choices=['v1', 'v2', 'r0'], required=True)
ap.add_argument('--init_ckpt', required=True); ap.add_argument('--out_dir', required=True)
ap.add_argument('--repo', default='/root/autodl-tmp/s-flm'); ap.add_argument('--config_name', default='config')
ap.add_argument('--steps', type=int, default=200); ap.add_argument('--prompts_per_step', type=int, default=16)
ap.add_argument('--group', type=int, default=8); ap.add_argument('--gen_batch', type=int, default=32)
ap.add_argument('--micro', type=int, default=16); ap.add_argument('--sampling_steps', type=int, default=128)
ap.add_argument('--lr', type=float, default=2e-6); ap.add_argument('--grad_clip', type=float, default=1.0)
ap.add_argument('--pmin', type=float, default=1e-3)
ap.add_argument('--dps_stride', type=int, default=8); ap.add_argument('--dps_lambda', type=float, default=0.1)
ap.add_argument('--save_every', type=int, default=50); ap.add_argument('--kl_every', type=int, default=10)
ap.add_argument('--kl_n', type=int, default=64, help='rollouts used for the KL estimate')
ap.add_argument('--seed', type=int, default=1234)
ap.add_argument('--pass_rates', default='data/sft/b_ans_pass_rates.json')
ap.add_argument('--seen_file', default='data/sft/b_ans.jsonl', help='SFT training file; its questions count as "seen"')
args = ap.parse_args()

os.chdir(args.repo); sys.path.insert(0, args.repo)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import omegaconf
for _n, _f in (('cwd', os.getcwd), ('device_count', torch.cuda.device_count), ('eval', eval),
               ('div_up', lambda x, y: (x + y - 1) // y)):
    if not omegaconf.OmegaConf.has_resolver(_n):
        omegaconf.OmegaConf.register_new_resolver(_n, _f)
from hydra import compose, initialize_config_dir
import algo, dataloader, sandbox_gsm8k as S

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
os.makedirs(args.out_dir, exist_ok=True)
assert not os.path.exists(f'{args.out_dir}/log.jsonl'), 'out_dir already has a run - use a fresh directory'
dev = torch.device('cuda')
overrides = ['data=gsm8k-test', 'data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M', 'model=small',
             'model.length=512', 'algo=mdlm', 'sampler=ancestral', f'sampler.steps={args.sampling_steps}',
             'sampler.temperature=1.0', 'sampler.use_float64=false', 'training.ema=0',
             f'eval.checkpoint_path={args.init_ckpt}', 'trainer.num_nodes=1', 'trainer.devices=1', '+wandb.offline=True']
with initialize_config_dir(config_dir=f'{args.repo}/configs', version_base=None):
    cfg = compose(config_name=args.config_name, overrides=overrides)
tok = dataloader.get_tokenizer(cfg)

def load_model(path):
    m = algo.MDLM.load_from_checkpoint(path, tokenizer=tok, config=cfg, strict=True).to(dev)
    assert m.ema is None
    m.backbone.eval(); m.noise.eval()          # dropout off for both rollouts and likelihoods
    return m

model = load_model(args.init_ckpt)
ref = load_model(args.init_ckpt) if args.kl_every > 0 else None
if ref is not None:
    for p_ in ref.parameters(): p_.requires_grad_(False)
MASK = model.mask_index
kl_gen = torch.Generator(device=dev); kl_gen.manual_seed(args.seed + 7)
EPS = float(cfg.noise.eps)
opt = torch.optim.AdamW([p_ for p_ in model.backbone.parameters() if p_.requires_grad],
                        lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)

# ---------------- prompts (eval-identical format: [BOS] + question + separator) ----------------
BOS = tok.bos_token_id
SEP = tok(cfg.data.separator, add_special_tokens=False).input_ids
def encode(q): return [BOS] + tok(q.strip(), add_special_tokens=False).input_ids + SEP
rows = json.load(open('data/gsm8k_train_rest.json'))
pr = json.load(open(args.pass_rates))
seen = {json.loads(l)['question'] for l in open(args.seen_file)} if args.seen_file else set()
pool = [r for r in rows if not (r['prompt'] in pr and pr[r['prompt']]['correct'] == pr[r['prompt']]['k'])]
rng_prompts = random.Random(args.seed)
order = list(range(len(pool))); rng_prompts.shuffle(order); cursor = 0
print(f'prompt pool {len(pool)} (excluded {len(rows) - len(pool)} base-model 8/8 questions); seen-in-SFT {sum(r["prompt"] in seen for r in pool)}', flush=True)

# ---------------- reward ----------------
def grade(resp, gold):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        official = bool(S.evaluate_samples(resp, gold, 5.0))
        strict, executable = False, False
        try:
            with S._time_limit(5.0):
                ns = S._safe_exec_no_timer(S._extract_code(resp))
                fn = ns.get('simple_math_problem')
                if fn is not None:
                    out = fn(); executable = True
                    if not isinstance(out, str) and S._to_number(out) is not None:
                        strict = S._numbers_equal(S._to_number(out), S._extract_gold_answer(gold))
        except BaseException:
            pass
    return official, official and strict, executable

# ---------------- generation (optionally with stride snapshots for DPS) ----------------
@torch.no_grad()
def generate(prefixes, stride):
    B, Lp = len(prefixes), max(len(p) for p in prefixes)
    padded = torch.zeros(B, Lp, dtype=torch.long, device=dev)
    for i, p in enumerate(prefixes): padded[i, :len(p)] = torch.tensor(p, device=dev)
    lengths = torch.tensor([len(p) for p in prefixes], device=dev)
    sampler = model.sampler
    state = sampler.init_state(model, B, num_steps=args.sampling_steps, eps=1e-5,
                               prefix_tokens=padded, prefix_lengths=lengths)
    snaps, k = ([state.xt.clone()] if stride else None), 0
    while not state.done:
        state = sampler.step(model, state); k += 1
        if stride and (k % stride == 0 or state.done) and not torch.equal(snaps[-1], state.xt):
            snaps.append(state.xt.clone())
    return state.xt.clone(), lengths, snaps

def comp_mask_of(x, lengths):
    # completion positions; positions holding the mask id are never scored (guard: PAD and MASK may share an id)
    return (torch.arange(x.shape[1], device=x.device)[None, :] >= lengths[:, None]) & (x != MASK)

PAD_ID, EOS_ID = tok.pad_token_id, tok.eos_token_id
def code_mask_of(x, lengths):
    # completion positions before the first PAD / EOS token = the code itself (excludes the padding tail)
    cm = comp_mask_of(x, lengths)
    end = ((x == PAD_ID) | (x == EOS_ID)) & cm
    return cm & ~(end.cumsum(-1) > 0)

def masked_mean(v, m):
    s, n = (v * m).sum(-1), m.sum(-1)
    return torch.where(n > 0, s / n.clamp(min=1), torch.zeros_like(s))

@torch.no_grad()
def dps_weights(snaps, final, cmask):
    """DACA-GRPO denoising progress score at stride snapshots. Tokens revealed between snapshot k and k+1
    get delta_k = S(k+1, k+1) - S(k, k+1), S(a, b) = mean log-prob under state a of the final tokens at positions
    still masked at state b. Deltas are z-scored across the batch per interval; weight = 1 + lambda * z."""
    K = len(snaps)
    masked = [(s == MASK) & cmask for s in snaps]
    belief = []
    for s in snaps[:-1]:
        lp = model.forward(xt=s, sigma=torch.zeros(s.shape[0], 1, device=dev))
        belief.append(lp.gather(-1, final[..., None]).squeeze(-1).float().clamp(min=-50))
        del lp
    deltas = []
    for k in range(K - 1):
        if k + 1 < K - 1:
            tgt = masked[k + 1].float()
            deltas.append(masked_mean(belief[k + 1], tgt) - masked_mean(belief[k], tgt))
        else:
            deltas.append(deltas[-1] if deltas else torch.zeros(final.shape[0], device=dev))  # extrapolate last
    w = torch.ones_like(final, dtype=torch.float32)
    for k in range(K - 1):
        d = deltas[k]; z = (d - d.mean()) / (d.std(unbiased=False) + 1e-6)
        revealed = masked[k] & ~masked[k + 1]
        w = torch.where(revealed, 1 + args.dps_lambda * z[:, None], w)
    return w.clamp(min=0.0)

# ---------------- ELBO with complementary masks ----------------
def pass_loglik(m_model, x0, m, coef, w, ent_mask=None):
    """Weighted ELBO contribution of the masked positions m (each scored once overall)."""
    xt = torch.where(m, torch.full_like(x0, MASK), x0)
    alpha = 1 - m.float().mean(-1, keepdim=True).clamp(1e-4, 1 - 1e-4)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        lp = m_model.forward(xt=xt, sigma=-torch.log(alpha))
    logp = lp.gather(-1, x0[..., None]).squeeze(-1).float()
    ent = None
    if ent_mask is not None:                     # entropy at masked CODE positions only
        with torch.no_grad():
            pr_ = lp.float().exp()
            ent_tok = -(pr_ * lp.float().clamp(min=-50)).sum(-1)
            ent = masked_mean(ent_tok, (m & ent_mask).float())
            del pr_, ent_tok
    return (coef[:, None] * logp * m * w).sum(-1) / 2, ent

def make_masks(x0, cmask, p):
    u = torch.rand(x0.shape, device=x0.device)
    mA = (u < p[:, None]) & cmask
    mB = (~mA) & cmask
    return mA, mB

def strat_p(n):
    u = (torch.rand(1).item() + torch.arange(n, dtype=torch.float32)) / n
    p = args.pmin + (1 - 2 * args.pmin) * u
    return p[torch.randperm(n)].to(dev)

# ---------------- checkpoint ----------------
def save(step):
    ck = torch.load(args.init_ckpt, map_location='cpu', weights_only=False)
    ck['state_dict'] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    for key in ('optimizer_states', 'lr_schedulers', 'ema'):
        ck.pop(key, None)
    torch.save(ck, f'{args.out_dir}/step{step}.ckpt')

# ---------------- main loop ----------------
logf = open(f'{args.out_dir}/log.jsonl', 'w'); samplef = open(f'{args.out_dir}/samples.jsonl', 'w')
json.dump(vars(args), open(f'{args.out_dir}/args.json', 'w'), indent=1)
L_CONST = cfg.model.length
for step in range(1, args.steps + 1):
    t0 = time.time()
    prompts = []
    for _ in range(args.prompts_per_step):
        if cursor == len(order):
            rng_prompts.shuffle(order); cursor = 0
        prompts.append(pool[order[cursor]]); cursor += 1
    items = [(pi, prompts[pi]) for pi in range(len(prompts)) for _ in range(args.group)]
    X, LEN, W, R, OFF, EXE, HACK, CH = [], [], [], [], [], [], [], []
    for b in range(0, len(items), args.gen_batch):
        chunk = items[b:b + args.gen_batch]
        xs, lens, snaps = generate([encode(r['prompt']) for _, r in chunk], args.dps_stride if args.variant == 'v2' else 0)
        cm = comp_mask_of(xs, lens)
        W.append(dps_weights(snaps, xs, cm) if args.variant == 'v2' else torch.ones_like(xs, dtype=torch.float32))
        for i, (_, r) in enumerate(chunk):
            resp = tok.decode(xs[i, int(lens[i]):].tolist(), skip_special_tokens=True)
            off, strict, exe = grade(resp, r['response_ground_truth'])
            R.append(float(strict)); OFF.append(off); EXE.append(exe); HACK.append(off and not strict); CH.append(len(resp))
            if step % 10 == 1 and b == 0 and i < 2:
                samplef.write(json.dumps({'step': step, 'prompt': r['prompt'], 'response': resp, 'reward': float(strict)}) + '\n')
        X.append(xs); LEN.append(lens)
    X, LEN, W = torch.cat(X), torch.cat(LEN), torch.cat(W)
    t_gen = time.time() - t0
    R_true = torch.tensor(R, device=dev)
    R_used = (torch.rand(len(R), device=dev) < 0.5).float() if args.variant == 'r0' else R_true
    grp = torch.tensor([pi for pi, _ in items], device=dev)
    A = torch.zeros_like(R_used)
    for g in range(len(prompts)):
        idx = (grp == g).nonzero().flatten()
        A[idx] = R_used[idx] - R_used[idx].mean()
    train_idx = (A != 0).nonzero().flatten()
    n_tr = len(train_idx)
    # Policy drift, measured BEFORE this step's update on rollouts from the current policy: exact per-position
    # predictive KL(pi || ref) = sum_v pi(v|x_t) log(pi(v|x_t)/ref(v|x_t)) at masked CODE positions (mask ratio 0.5,
    # own RNG so measuring never changes the training randomness). >= 0 by construction, no 1/p importance weights.
    # (The earlier ELBO-difference estimator was noise-dominated and not a KL for parallel-sampler rollouts.)
    kl = flip = None
    if ref is not None and args.kl_every > 0 and step % args.kl_every == 0:
        with torch.no_grad():
            kl_ids = torch.linspace(0, len(R) - 1, min(len(R), args.kl_n), device=dev).round().long()
            ks, fs = [], []
            for s in range(0, len(kl_ids), 8):
                ids = kl_ids[s:s + 8]
                x0 = X[ids]; km = code_mask_of(x0, LEN[ids])
                m = (torch.rand(x0.shape, device=dev, generator=kl_gen) < 0.5) & km
                xt = torch.where(m, torch.full_like(x0, MASK), x0)
                sig = -torch.log(1 - m.float().mean(-1, keepdim=True).clamp(1e-4, 1 - 1e-4))
                la = model.forward(xt=xt, sigma=sig).float().clamp(min=-50)
                lb = ref.forward(xt=xt, sigma=sig).float().clamp(min=-50)
                n = m.sum(-1).clamp(min=1)
                ks.append(((la.exp() * (la - lb)).sum(-1) * m).sum(-1) / n)
                fs.append(((la.argmax(-1) != lb.argmax(-1)).float() * m).sum(-1) / n)
                del la, lb
            kl, flip = float(torch.cat(ks).mean()), float(torch.cat(fs).mean())
    opt.zero_grad(set_to_none=True)
    ent_vals, gnorm = [], 0.0
    if n_tr > 0:
        ps = strat_p(n_tr)
        norm = float(len(R) * L_CONST)                 # Dr. GRPO style constant normaliser
        for s in range(0, n_tr, args.micro):
            ids = train_idx[s:s + args.micro]
            x0, cm = X[ids], comp_mask_of(X[ids], LEN[ids])
            p = ps[s:s + args.micro]
            mA, mB = make_masks(x0, cm, p)
            km = code_mask_of(x0, LEN[ids])
            for j, (m, coef) in enumerate(((mA, (1 - EPS) / p), (mB, (1 - EPS) / (1 - p)))):
                ll, ent = pass_loglik(model, x0, m, coef, W[ids], ent_mask=(km if j == 0 else None))
                loss = -(A[ids] * ll).sum() / norm
                loss.backward()
                if ent is not None: ent_vals.append(ent.mean().item())
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), args.grad_clip))
        opt.step()
    is_seen = [prompts[pi]['prompt'] in seen for pi, _ in items]
    rs = [r for r, s_ in zip(R, is_seen) if s_]; ru = [r for r, s_ in zip(R, is_seen) if not s_]
    mixed = sum(1 for g in range(len(prompts)) if 0 < sum(R_used[grp == g].tolist()) < args.group)
    rec = {'step': step, 'reward': float(np.mean(R)), 'reward_used': float(R_used.mean()),
           'official_acc': float(np.mean(OFF)), 'executable': float(np.mean(EXE)), 'hack_rate': float(np.mean(HACK)),
           'mixed_groups': mixed / len(prompts), 'trained_rollouts': n_tr, 'entropy': float(np.mean(ent_vals)) if ent_vals else None,
           'grad_norm': gnorm, 'kl': kl, 'flip': flip, 'reward_seen': float(np.mean(rs)) if rs else None,
           'reward_unseen': float(np.mean(ru)) if ru else None, 'chars': float(np.mean(CH)),
           'code_tokens': float(code_mask_of(X, LEN).sum(-1).float().mean()),
           'dps_w_std': float(W[comp_mask_of(X, LEN)].std()) if args.variant == 'v2' else None,
           't_gen': round(t_gen, 1), 't_step': round(time.time() - t0, 1)}
    logf.write(json.dumps(rec) + '\n'); logf.flush(); samplef.flush()
    print(json.dumps(rec), flush=True)
    if step % args.save_every == 0 or step == args.steps:
        save(step)
print('RL DONE', flush=True)
