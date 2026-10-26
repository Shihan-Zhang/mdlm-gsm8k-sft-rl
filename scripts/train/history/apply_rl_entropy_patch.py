import sys
p = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/analysis/rl_grpo.py'
s = open(p).read()
reps = [
("def masked_mean(v, m):",
 "PAD_ID, EOS_ID = tok.pad_token_id, tok.eos_token_id\n"
 "def code_mask_of(x, lengths):\n"
 "    # completion positions before the first PAD / EOS token = the code itself (excludes the padding tail)\n"
 "    cm = comp_mask_of(x, lengths)\n"
 "    end = ((x == PAD_ID) | (x == EOS_ID)) & cm\n"
 "    return cm & ~(end.cumsum(-1) > 0)\n\n"
 "def masked_mean(v, m):"),
("def pass_loglik(m_model, x0, m, coef, w, want_ent=False):",
 "def pass_loglik(m_model, x0, m, coef, w, ent_mask=None):"),
("    if want_ent:\n", "    if ent_mask is not None:                     # entropy at masked CODE positions only\n"),
("            ent = masked_mean(ent_tok, m.float())", "            ent = masked_mean(ent_tok, (m & ent_mask).float())"),
("            mA, mB = make_masks(x0, cm, p)\n            for j,",
 "            mA, mB = make_masks(x0, cm, p)\n            km = code_mask_of(x0, LEN[ids])\n            for j,"),
("W[ids], want_ent=(j == 0))", "W[ids], ent_mask=(km if j == 0 else None))"),
("'chars': float(np.mean(CH)),",
 "'chars': float(np.mean(CH)),\n           'code_tokens': float(code_mask_of(X, LEN).sum(-1).float().mean()),"),
]
for a, b in reps:
    assert s.count(a) == 1, f'patch target not found exactly once: {a[:50]!r}'
    s = s.replace(a, b)
compile(s, p, 'exec')
open(p, 'w').write(s)
print('rl_grpo.py patched: entropy over code positions only, code_tokens logged')
