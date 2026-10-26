import sys
p = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/analysis/rl_grpo.py'
s = open(p).read()
i0 = s.index("    # KL(pi || ref) per completion token, estimated BEFORE")
i1 = s.index("    opt.zero_grad(set_to_none=True)\n    ent_vals, gnorm = [], 0.0\n")
new_kl = """    # Policy drift, measured BEFORE this step's update on rollouts from the current policy: exact per-position
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
"""
s = s[:i0] + new_kl + s[i1:]
reps = [
 ("ap.add_argument('--save_every', type=int, default=50); ap.add_argument('--kl_every', type=int, default=25)\n",
  "ap.add_argument('--save_every', type=int, default=50); ap.add_argument('--kl_every', type=int, default=10)\n"),
 ("MASK = model.mask_index\n", "MASK = model.mask_index\nkl_gen = torch.Generator(device=dev); kl_gen.manual_seed(args.seed + 7)\n"),
 ("'grad_norm': gnorm, 'kl': kl,", "'grad_norm': gnorm, 'kl': kl, 'flip': flip,"),
]
for a, b in reps:
    assert s.count(a) == 1, f'patch target not found exactly once: {a[:60]!r}'
    s = s.replace(a, b)
compile(s, p, 'exec')
open(p, 'w').write(s)
print('rl_grpo.py patched: in-run drift = exact predictive KL at masked code positions (+ argmax flip rate), every 10 steps')
