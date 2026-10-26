"""Add a confidence-ordered sampler (predictor: confidence) to s-flm without touching AncestralSampler.
Each target must occur exactly once, else abort with nothing written."""
import pathlib, sys
REPO = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/s-flm')
CLASS_CODE = '''# [confidence patch] ---------------------------------------------------------
def _confidence_update(log_probs, current_tokens, mask_index, alpha_s, alpha_t,
                       *, is_last_step, value_greedy):
  """Reveal the most confident masked positions first (LLaDA-style low-confidence remasking).

  The number revealed per sequence matches the ancestral sampler in expectation:
  k = round(#masked * (alpha_s - alpha_t) / (1 - alpha_t)); the last step reveals all.
  Only WHICH positions are revealed changes: highest probability of their drawn token first.
  """
  masked = current_tokens == mask_index
  if value_greedy:
    cand = log_probs.argmax(dim=-1)
  else:
    cand = sample_categorical(log_probs.exp())
  conf = log_probs.gather(-1, cand.unsqueeze(-1)).squeeze(-1)
  conf = conf.masked_fill(~masked, float('-inf'))
  n_masked = masked.sum(dim=-1)
  if is_last_step:
    k = n_masked
  else:
    frac = ((alpha_s - alpha_t) / (1 - alpha_t)).reshape(-1).to(conf.dtype)
    k = torch.round(n_masked.to(conf.dtype) * frac).long()
    k = torch.minimum(torch.clamp(k, min=0), n_masked)
  order = conf.argsort(dim=-1, descending=True)
  ranks = torch.empty_like(order)
  ranks.scatter_(-1, order, torch.arange(order.shape[-1], device=order.device).expand_as(order).contiguous())
  select = masked & (ranks < k.unsqueeze(-1))
  return torch.where(select, cand, current_tokens)


class ConfidenceSampler(AncestralSampler):
  """Same schedule as AncestralSampler; reveals the most confident masked positions first."""

  def __init__(self, use_float64=False, steps_policy='full',
               temperature=1.0, value_greedy=False):
    super().__init__(use_float64=use_float64, noise_removal='ancestral',
                     steps_policy=steps_policy, temperature=temperature)
    self.value_greedy = value_greedy

  def step(self, model, state):
    num_steps = len(state.timesteps) - 1
    is_last_step = (state.step_idx == num_steps - 1)
    t = state.timesteps[state.step_idx] * state.ones
    s = state.timesteps[state.step_idx + 1] * state.ones
    _, alpha_t = model.noise(t)
    if is_last_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = model.noise(s)
    sigma_t = model._sigma_from_alphat(alpha_t)
    log_probs = _maybe_cast_log_probs(
      model.forward(xt=state.xt, sigma=sigma_t,
                    context=AncestralContext(temperature=self.temperature)),
      self.use_float64)
    current_tokens = state.xt[:, state.start_idx:]
    log_probs_window = log_probs[:, state.start_idx:]
    current_tokens[:] = _confidence_update(
      log_probs_window, current_tokens, _model_mask_index(model),
      alpha_s, alpha_t, is_last_step=is_last_step,
      value_greedy=self.value_greedy)
    self._project_prefix(state.xt, state.prefix_tokens,
                         state.prefix_lengths)
    state.nfe += 1
    state.step_idx += 1
    if is_last_step:
      state.done = True
    return state
# [confidence patch] end -----------------------------------------------------


'''
PATCHES = [
  ("@dataclass(kw_only=True)\nclass ARState(BaseState):",
   CLASS_CODE + "@dataclass(kw_only=True)\nclass ARState(BaseState):"),
  ("      temperature=s.temperature)\n\n  if s.predictor == 'ar':",
   "      temperature=s.temperature)\n\n"
   "  if s.predictor == 'confidence':  # [confidence patch]\n"
   "    return ConfidenceSampler(\n"
   "      use_float64=s.use_float64,\n"
   "      steps_policy=s.steps_policy,\n"
   "      temperature=s.temperature,\n"
   "      value_greedy=s.get('value_greedy', False))\n\n"
   "  if s.predictor == 'ar':"),
]
path = REPO / 'samplers.py'
text = path.read_text()
if '[confidence patch]' in text:
    sys.exit('ABORT: samplers.py already patched.')
for old, new in PATCHES:
    n = text.count(old)
    if n != 1:
        sys.exit(f'ABORT: target found {n} times (expected 1). Nothing written.\n---\n{old}')
    text = text.replace(old, new)
for name in ('sample_categorical', '_maybe_cast_log_probs', '_model_mask_index', 'class AncestralContext', 'class AncestralSampler'):
    if name not in text:
        sys.exit(f'ABORT: helper {name} not found in samplers.py. Nothing written.')
cfg = REPO / 'configs' / 'sampler' / 'confidence.yaml'
base = (REPO / 'configs' / 'sampler' / 'ancestral.yaml').read_text()
if 'predictor: ancestral' not in base:
    sys.exit('ABORT: unexpected ancestral.yaml')
cfg_text = ('# [confidence patch] ancestral schedule, most-confident-first reveal order\n'
            + base.replace('predictor: ancestral', 'predictor: confidence', 1)
            + ('' if base.endswith('\n') else '\n') + 'value_greedy: false\n')
path.write_text(text)
cfg.write_text(cfg_text)
print('patched', path); print('wrote', cfg)
