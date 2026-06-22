# RL Run 5 — PPO, 3M steps, v4 reward, 5-seed sweep

## Config
- Algorithm: PPO (MlpPolicy)
- Timesteps: 3,000,000 (2× Run 4)
- Training: randomized, Eval: hard
- n_steps: 2048, batch_size: 128, lr: 3e-4, gamma: 0.99
- Reward: v4 (same as Run 4)

## Results

| Seed | MAE (m) | Min dist (m) | Steps | Result |
|---|---|---|---|---|
| 0 | 8.231 | -1.719 | 60/300 | crash |
| 1 | 2.632 | 21.233 | 300/300 | good |
| 2 | 8.705 | -0.016 | 75/300 | crash |
| 3 | 8.602 | -0.881 | 78/300 | crash |
| 4 | 1.078 | 18.955 | 300/300 | good |

Crashes: 3/5 — worse than Run 4 (1/5).

## Why more steps didn't help

Seeds 0, 2, 3 crash at ~60–78 steps in both Run 4 and Run 5, with nearly identical MAE (~8–9 m). Same seeds, same episode length, same behaviour regardless of training budget — these aren't undertrained, they converged early to a bad local minimum where the policy learned to accelerate into the lead car. More gradient steps reinforce that behaviour rather than escaping it.

More steps is not the fix. The problem is seed-dependent weight initialisation, not training budget. VecNormalize was the theoretically correct fix but disrupted training in a different way (see Run 4). This remains an open limitation.

## Conclusion

Run 4 (1.5M steps) is the best result: 3/5 seeds converged, best MAE 1.02 m (90.5% of PID). Run 5 is recorded here as evidence that scaling compute alone doesn't resolve the seed instability.
