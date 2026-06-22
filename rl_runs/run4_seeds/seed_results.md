# run4_seeds — Seed Sweep Results

Config: PPO, 1.5M steps, randomized training + VecNormalize, eval on hard.

| Seed | MAE (m) | Min dist (m) | Jerk (m/s³) | Steps | Collision |
|---|---|---|---|---|---|
| 0 | 8.978 | -0.857 | 0.668 | 73/300 | True |
| 1 | 10.153 | 18.095 | 0.849 | 300/300 | False |
| 2 | 2.379 | 20.907 | 1.022 | 300/300 | False |
| 3 | 1.298 | 19.826 | 1.132 | 300/300 | False |
| 4 | 1.024 | 19.293 | 1.179 | 300/300 | False |
| **Mean** | **4.767** | **15.453** | **0.970** | | |
| **Std** | **3.962** | **8.205** | **0.189** | | |

Total crashes: 1/5
