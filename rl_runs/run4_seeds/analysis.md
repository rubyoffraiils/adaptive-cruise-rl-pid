# RL Run 4 — PPO, 1.5M steps, v4 reward, 5-seed sweep

## Config
- Algorithm: PPO (MlpPolicy)
- Timesteps: 1,500,000
- Training: randomized, Eval: hard
- n_steps: 2048, batch_size: 128, lr: 3e-4, gamma: 0.99
- Reward: v4

## Change from Run 3: closed the reward-hacking loophole

Run 3 seed 2 drove away from the lead car (MAE ~96 m) because v3 had no penalty for excessive distance. v4 adds a soft nudge past 40 m:

```python
if distance > 40.0:
    reward -= 0.5 * (distance - 40.0)
```

Weight 0.5 is intentionally small. At 60 m it costs -10; the proximity ramp at 10 m costs -50. Safety still dominates — the hatch is closed without flipping the reward hierarchy.

Reward table at accel=0:
| Distance | Reward |
|---|---|
| 5 m | -145.0 |
| 15 m | -15.0 |
| 20 m | 0.0 |
| 30 m | -10.0 |
| 40 m | -20.0 |
| 60 m | -50.0 |

## VecNormalize — three failed attempts

VecNormalize normalises each obs dimension to ~N(0,1) using a running mean/variance. In theory this helps when obs dimensions span different scales (distance_error ±100, speeds 0–50). In practice it made things worse every time.

**Attempt 1: norm_obs=True, norm_reward=True** — 4/5 crashes. Reward normalisation rescaled the proximity ramp relative to tracking. The ramp values (-50 to -150) were outliers — normalising flattened them and removed the sharp safety signal.

**Attempt 2: norm_obs=True, norm_reward=False** — 3/5 crashes, non-crashing seeds had MAE 7–12 m. Obs normalisation alone still disrupted gradient flow enough to degrade performance.

**Attempt 3: norm_obs=True, norm_reward=False (repeated)** — 5/5 crashes, near-zero jerk across all seeds. Jerk ≈ 0 means constant acceleration regardless of input — a collapsed policy that stopped using observations entirely.

Root cause: the reward function is already well-scaled for this problem (proximity ramp magnitudes are deliberately large). VecNormalize's obs rescaling disrupts the implicit relationship between obs scale and reward scale that the policy learns. The instability in Run 3 was seed-dependent weight init, not observation scale — VecNormalize was solving the wrong problem.

## Seed sweep results

| Seed | MAE (m) | Min dist (m) | Jerk (m/s³) | Steps | Result |
|---|---|---|---|---|---|
| 0 | 8.978 | -0.857 | 0.668 | 73/300 | crash |
| 1 | 10.153 | 18.095 | 0.849 | 300/300 | loose |
| 2 | 2.379 | 20.907 | 1.022 | 300/300 | good |
| 3 | 1.298 | 19.826 | 1.132 | 300/300 | good |
| 4 | 1.024 | 19.293 | 1.179 | 300/300 | good |

Good seeds (2,3,4): MAE = 1.57 ± 0.71 m, min dist = 20.01 ± 0.83 m. Crashes: 1/5.

## Run comparison

| Run | Config | Converged | Best MAE | Crashes |
|---|---|---|---|---|
| Run 1 | 100k steps, fixed, v1 reward | 0/1 | 8.07 m | 1/1 |
| Run 2 | 500k steps, randomized, v1 reward | 0/1 | 8.64 m | 1/1 |
| Run 3 seeds | 1.5M steps, randomized, v3 reward | 2/5 | 1.09 m | 1/5 |
| Run 4 seeds | 1.5M steps, randomized, v4 reward | 3/5 | 1.02 m | 1/5 |
| PID (hard) | Bayesian-tuned, deterministic | 5/5 | 0.93 m | 0/5 |

Run 4 improved over Run 3: 3/5 good seeds vs 2/5, hacking loophole closed, best MAE 1.02 m. The 1/5 crash and 1/5 loose convergence are seed-dependent weight init issues — more steps didn't help (see Run 5).
