# mdlm-gsm8k-sft-rl

Supervised fine-tuning and reinforcement learning for a 170M-parameter **masked diffusion language model** (MDLM) that solves GSM8K by writing a Python function. On the GSM8K test set, SFT on the model's own verified solutions followed by 200 steps of critic-free RL takes accuracy from **31.9% to 43.2%** (+11.3 points, mean of three fresh sampling seeds).

The repository contains every script that produced the reported numbers, two small patches to the upstream training code, the job queues in the order they were run, and per-question results for every test run.

## Results

GSM8K test, 1,319 questions. Ancestral sampler, 128 steps, float64. Accuracy is graded by the official sandbox: the generated function is executed and its return value compared with the answer.

| Model | T = 0.1, mean of seeds 101–103 | T = 1.0, seed 101 |
| --- | --- | --- |
| Base (TinyGSM checkpoint, EMA weights) | 31.92 | 12.81 |
| + SFT on self-generated solutions (RFT) | 38.59 | 22.67 |
| + RL, V1 (uniform credit) | **43.19** | 31.84 |
| + RL, V2 (denoising-progress weighting) | 42.76 | 32.37 |
| + RL, R0 (random Bernoulli(0.5) reward, 100 steps) | 39.27 (seed 101) | 22.97 |

Paired comparisons (per-question accuracy averaged over the shared seeds, question-level bootstrap CI, sign-flip test), reproducible with `scripts/eval/summarize_results.py`:

| Comparison | Difference | 95% CI | p |
| --- | --- | --- | --- |
| SFT − Base | +6.67 | [+5.2, +8.2] | < 0.001 |
| V1 − SFT | +4.60 | [+3.4, +5.8] | < 0.001 |
| V2 − V1 | −0.43 | [−1.2, +0.4] | 0.33 |
| R0 − SFT (seed 101) | +0.45 | [−0.6, +1.5] | 0.49 |

### What changed, and why

- **The main failure is parallel-decoding inconsistency.** Tokens revealed in the same denoising step are sampled independently, so one line defines `days_per_week` while another reads `days_per_omelet`. At 32 steps and T = 1, about 78% of failures are broken code rather than wrong reasoning.
- **SFT works through the source of the solutions, not the questions.** Four SFT sets of identical size (10,617 rows) form a 2 × 2 of question source × solution source. Training on the model's own verified solutions cuts `NameError` by 8.5 points. Training on TinyGSM reference solutions or OpenMathInstruct-1 solutions *lowers* accuracy below the base model (−3.4 and −10.2 points, seed 101).
- **RL sharpens more than it expands.** pass@32 on a 300-question subset is unchanged (75.3% for both SFT and V1); the questions RL gains and loses are those SFT solved only 1–2 times in 32. The gain is largest at T = 1 (+9.2 points), where the policy's distribution matters most.
- **The reward signal is doing the work.** With random rewards (R0), training reward barely moves (0.21 → 0.24 over 100 steps), entropy stays flat, drift at step 100 is about a quarter of V1's at the same step (KL 5.0e-4 vs 2.2e-3 at mask ratio 0.75), and test accuracy does not change (+0.45 vs SFT, p = 0.49).
- **Gains carry over to perturbed questions.** On GSM-Plus (7 perturbation types) and GSM-Symbolic, every model keeps a similar fraction of its original accuracy (about 65% on GSM-Plus), so training lifted the level without making the model more fragile. All models fail on inserted distractors and on GSM-Symbolic P2.

### What did not work

| Attempt | Result |
| --- | --- |
| V2: per-token weight 1 + 0.1·z(denoising progress score) | No difference from V1 (−0.43, CI [−1.2, +0.4]) |
| Confidence-ordered unmasking (base model, dev seed 1) | T = 1: 14.6 → 34.0, about the same as lowering T to 0.1. T = 0.1: 34.8 → 33.6, `NameError` 16.4% → 24.4% |
| Post-hoc variable-name repair (analysis only) | +2.3 base, +2.4 SFT, +1.3 V1. Of 190 repaired base outputs, only 30 became correct |
| In-run KL from the difference of two ELBOs | Came out negative. Replaced by exact predictive KL on masked code positions plus a flip rate |

## Method

**Model.** MDLM with the log-linear schedule α_t = ε + (1 − ε)(1 − t), ε = 10⁻³, and SUBS parameterization. The checkpoint stores EMA weights. Fine-tuning starts from θ_EMA with EMA turned off. An lr = 0 run reproduces the base outputs byte for byte on all 1,319 test questions.

**SFT (rejection-sampling fine-tuning).** Sample the base model 8 times per training question (6,973 questions, T = 0.4). Keep solutions that the official grader accepts and that return a number directly. Deduplicate by AST with identifiers renamed, and keep at most 4 per question. This gives 10,617 rows covering 4,790 questions. Train with lr 3e-5, batch 64, 20 warm-up steps and 10 epochs. The epoch and learning rate were chosen on dev with fresh seeds.

**RL.** A critic-free group-relative policy gradient:
- 16 prompts × 8 rollouts per step, sampled at T = 1. The advantage is r − mean(r) with no std scaling (Dr. GRPO / RLOO), and groups with identical rewards are skipped.
- One update per batch (μ = 1), so there is no ratio, clipping or KL penalty. lr 1e-5, 200 steps.
- The reward is 1 only if the official grader accepts the answer **and** the function returns a number directly. Print or string fallbacks earn nothing, and the hack rate stays 0 throughout.
- The sequence log-likelihood is estimated by the ELBO with **complementary masks**: draw p, mask completion tokens with probability p in one pass and the complement in a second pass, weighting by (1 − ε)/p and (1 − ε)/(1 − p). Every completion token is scored exactly once and the estimator stays unbiased.

**Evaluation protocol.** The dev set is 500 questions held out from GSM8K train and is used only for selection. Screening uses seed 1, finalists seeds 2–5, and test seeds 101–103, none of them reused. Comparisons are paired by question: exact McNemar for one seed, bootstrap plus sign-flip across seeds. On dev, two seeds of the same base model disagree on 24% of questions, so single-seed differences of a few points are noise.

## Repository layout

```
patches/            apply_sft_patch.py, apply_conf_patch.py and the resulting diff (s-flm.diff)
scripts/data/       dev split, RFT sampling and filtering (B-ans), TinyGSM / OpenMathInstruct-1 / bq_pure control sets
scripts/train/      sft_run.sh (one SFT run + dev eval per epoch), rl_grpo.py (V1 / V2 / R0); history/ holds the patches applied to rl_grpo.py during the project
scripts/eval/       multi-seed eval, re-grading, paired statistics, pass@k, GSM-Plus / GSM-Symbolic, drift probes, summarize_results.py
scripts/analysis/   error taxonomy, data style, label-noise check, variable-name repair, denoising trajectories
queues/             the shell queues exactly as run (see below)
results/test/       per-question {idx, correct, category} for every test run
results/rl_logs/    log.jsonl and args.json for every RL run, including the pilot and the aborted first V1
results/            pass@k, drift and generalization summaries
```

## Reproducing

1. Clone the upstream code `jdeschena/s-flm` and download its TinyGSM MDLM checkpoint. Apply the patch with `python patches/apply_sft_patch.py /path/to/s-flm`. The patch script refuses to write anything unless each target occurs exactly once.
2. The scripts were run on a single RTX 4090 (24 GB) and use absolute paths: `R=/root/autodl-tmp/s-flm` for the upstream repo and `A=/root/autodl-tmp/analysis` for this repository's scripts. Change them at the top of each script, or symlink those two paths.
3. Run the core path:

```bash
python -I scripts/data/make_dev_split.py
python -I scripts/data/make_rft_full.py --k 8 --chunks 4
bash queues/queue_rft_full.sh                                   # ~5 h of sampling
python -I scripts/data/build_b_ans.py --m 4
bash scripts/train/sft_run.sh b_ans_lr3e-5_e10 data/sft/b_ans.jsonl 3e-5 10   # ~19 min + dev evals
python -u scripts/train/rl_grpo.py --variant v1 --lr 1e-5 --steps 200 \
  --init_ckpt outputs/sft_b_ans_lr3e-5_e10/checkpoints/10-1650.ckpt --out_dir outputs/rl_v1   # ~2.2 h
bash queues/queue_day4a.sh                                      # test seeds 101-103, pass@k, drift
python -I scripts/eval/summarize_results.py
```

Evaluate the **base** checkpoint *without* `training.ema=0`, because its weights are the EMA. Evaluate fine-tuned checkpoints *with* it. `scripts/eval/eval_seeds.sh` hard-codes the flag and is meant only for fine-tuned checkpoints.

### Order in which the queues were run

| Queue | Purpose |
| --- | --- |
| `run_sweep.sh` (in scripts/eval), `queue_seed.sh` | Steps × temperature sweep on the base model; seed-to-seed noise |
| `queue_sanity_seed.sh` | lr = 0 acceptance test of the SFT patch |
| `queue_pilot.sh`, `queue_rft_full.sh` | RFT temperature pilot, then full RFT sampling |
| `queue_sft_b_ans.sh`, `queue_night2.sh`, `queue_night3.sh` | B-ans SFT at three learning rates, 5 and 10 epochs |
| `queue_day3a.sh` | Control sets A (TinyGSM) and C (OpenMathInstruct-1); B-ans finalists on seeds 2–5 |
| `queue_rlpilot.sh` | RL learning-rate pilot (1e-5 vs 2e-6, 20 steps each) |
| `queue_day3b.sh` | RL V1, V2, R0 and their dev evals; bq sampling; extra base seeds; pass@k of SFT |
| `queue_day4a.sh` | Drift probes, test seeds 101–103, pass@k of V1 and V2, control-arm test runs |
| `queue_day4b.sh`, `queue_day4c.sh` | bq_pure SFT; GSM-Plus and GSM-Symbolic |

## Compute

Everything ran on one rented RTX 4090. Sampling dominates the cost. An RL step takes 40 s, of which 38 s is generating 128 rollouts with 128 denoising steps. The project generated about 310k solutions in total, more of them for evaluation than for training. The core path alone (RFT → SFT → RL V1 → test) needs under 3 GPU-hours of training plus about 100k generations.

## Limitations

- One model size (170M) and one task. The dominance of decoding inconsistency may not hold at the 8B scale.
- TinyGSM was produced by rewriting GSM8K training questions, so dev, which is cut from train, is optimistic. All conclusions rest on paired test differences.
- R0, bq_pure and the A / C control sets have a single test seed, and pass@k uses 300 questions.
- The reward checks the final answer only. A solution that hard-codes the arithmetic (`result = 10 + 3`) is rewarded.

## References

- Sahoo et al. *Simple and Effective Masked Diffusion Language Models.* NeurIPS 2024.
- Deschenaux & Gulcehre. *Hyperspherical Flow Language Model.* arXiv 2605.11125 (upstream `s-flm` code).
- Liu et al. *TinyGSM.* arXiv 2312.09241. · Cobbe et al. *GSM8K.* arXiv 2110.14168. · Toshniwal et al. *OpenMathInstruct-1.* arXiv 2402.10176.
- Yuan et al. *Scaling Relationship on Learning Mathematical Reasoning with LLMs* (RFT). arXiv 2308.01825.
- Shao et al. *DeepSeekMath* (GRPO). arXiv 2402.03300. · Liu et al. *Understanding R1-Zero-Like Training* (Dr. GRPO). arXiv 2503.20783. · Ahmadian et al. *Back to Basics* (RLOO). arXiv 2402.14740.
- Zhao et al. *d1: Scaling Reasoning in Diffusion LLMs via RL.* arXiv 2504.12216. · DACA-GRPO, arXiv 2605.16342.
- Nie et al. *Large Language Diffusion Models* (LLaDA). arXiv 2502.09992.
- Yue et al. *Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?* arXiv 2504.13837. · Shao et al. *Spurious Rewards.* arXiv 2506.10947.
- Li et al. *GSM-Plus.* ACL 2024. · Mirzadeh et al. *GSM-Symbolic.* arXiv 2410.05229. · Chen et al. *Evaluating LLMs Trained on Code* (pass@k). arXiv 2107.03374.

## License

The code in this repository is released under the MIT License. `patches/` modifies the upstream `jdeschena/s-flm` code, whose own license applies to the parts it changes. Model weights and datasets are not included.
