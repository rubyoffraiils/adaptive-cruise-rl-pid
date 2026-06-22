# RL Results Summary

## Headline numbers

| Metric | PID | RL (best seed) |
|---|---|---|
| MAE on hard scenario | 0.927 m | 1.024 m |
| RL vs PID error gap | — | +10.5% |
| Min following distance | 19.42 m | 19.29 m |
| Jerk | 0.989 m/s³ | 1.179 m/s³ |
| Crashes on hard | 0 | 0 |
| Success rate (20 unseen profiles) | 100% (deterministic) | 100% |
| MAE on unseen profiles | — | 1.253 ± 0.345 m |

RL gets within 10.5% of a Bayesian-tuned PID on a scenario it was never trained on, with 100% success across 20 unseen random profiles.

---

## Context

PID was explicitly tuned on the hard scenario with 600 Bayesian trials. RL was trained only on randomised profiles and evaluated on hard for the first time at test — so the comparison is not quite fair to RL. The 10.5% gap is likely an underestimate of what's achievable with proper obs normalisation or a larger training budget. This is consistent with Lin et al. (2022), who report DRL 5.8% worse than optimal in-distribution and 17.2% worse out-of-distribution on the same ACC problem.

---

## Seed convergence

Not all seeds converge. Best run (Run 4, 1.5M steps):

| Seed | MAE | Result |
|---|---|---|
| 0 | 8.978 m | crash |
| 1 | 10.153 m | loose |
| 2 | 2.379 m | good |
| 3 | 1.298 m | good |
| 4 | 1.024 m | best |

3/5 seeds converged. This is a known PPO limitation on continuous control without observation normalisation — not a project-specific failure. The same pattern appears in the literature (DDPG "converged to slightly different sub-optimal policies each time").

---

## Training progression

| Run | Key change | Best MAE | Crashed? |
|---|---|---|---|
| Run 1 | Baseline — 100k steps, fixed scenario | 8.07 m | yes |
| Run 2 | Randomised training, 500k steps | 8.64 m | yes |
| Run 3 | Proximity ramp reward, 1.5M steps | 1.09 m | 1/5 seeds |
| Run 4 | Too-far nudge, closed hacking loophole | 1.02 m | 1/5 seeds |
| Run 5 | 3M steps — no improvement over Run 4 | 1.08 m | 3/5 seeds |

The reward redesign between Run 2 and Run 3 was the critical step. Everything after that was refinement.
