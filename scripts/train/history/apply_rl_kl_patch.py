import sys
p = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/analysis/rl_grpo.py'
s = open(p).read()
old_kl = """    kl = None
    if ref is not None and args.kl_every > 0 and step % args.kl_every == 0:
        with torch.no_grad():
            ids = torch.arange(min(len(R), args.micro), device=dev)
            x0, cm = X[ids], comp_mask_of(X[ids], LEN[ids])
            p = strat_p(len(ids)); mA, mB = make_masks(x0, cm, p)
            diff = torch.zeros(len(ids), device=dev)
            for m, coef in ((mA, (1 - EPS) / p), (mB, (1 - EPS) / (1 - p))):
                a_, _ = pass_loglik(model, x0, m, coef, torch.ones_like(x0, dtype=torch.float32))
                b_, _ = pass_loglik(ref, x0, m, coef, torch.ones_like(x0, dtype=torch.float32))
                diff += a_ - b_
            kl = float((diff / cm.sum(-1)).mean())     # ELBO-based estimate of KL(pi || ref) per completion token
"""
new_kl = """    # KL(pi || ref) per completion token, estimated BEFORE this step's update on rollouts drawn from the current
    # policy (the old version measured after opt.step() on the just-trained batch, which biases it negative because
    # the update pushes down the likelihood of the mostly negative-advantage rollouts). ELBO-based, common masks.
    kl = None
    if ref is not None and args.kl_every > 0 and step % args.kl_every == 0:
        with torch.no_grad():
            kl_ids = torch.linspace(0, len(R) - 1, min(len(R), args.kl_n), device=dev).round().long()
            vals = []
            for s in range(0, len(kl_ids), args.micro):
                ids = kl_ids[s:s + args.micro]
                x0, cm = X[ids], comp_mask_of(X[ids], LEN[ids])
                p = strat_p(len(ids)); mA, mB = make_masks(x0, cm, p)
                diff = torch.zeros(len(ids), device=dev)
                for m, coef in ((mA, (1 - EPS) / p), (mB, (1 - EPS) / (1 - p))):
                    a_, _ = pass_loglik(model, x0, m, coef, torch.ones_like(x0, dtype=torch.float32))
                    b_, _ = pass_loglik(ref, x0, m, coef, torch.ones_like(x0, dtype=torch.float32))
                    diff += a_ - b_
                vals.append(diff / cm.sum(-1).clamp(min=1))
            kl = float(torch.cat(vals).mean())
"""
anchor = "    opt.zero_grad(set_to_none=True)\n    ent_vals, gnorm = [], 0.0\n"
arg_old = "ap.add_argument('--save_every', type=int, default=50); ap.add_argument('--kl_every', type=int, default=25)\n"
arg_new = arg_old + "ap.add_argument('--kl_n', type=int, default=64, help='rollouts used for the KL estimate')\n"
for t in (old_kl, anchor, arg_old):
    assert s.count(t) == 1, f'patch target not found exactly once: {t[:60]!r}'
s = s.replace(old_kl, '').replace(anchor, new_kl + anchor).replace(arg_old, arg_new)
compile(s, p, 'exec')
open(p, 'w').write(s)
print('rl_grpo.py patched: KL measured before the update, on --kl_n rollouts')
