# run5_seeds — Seed Sweep Results

Config: PPO, 3M steps, randomized training, v4 reward, eval on hard.

| Seed | MAE (m) | Min dist (m) | Jerk (m/s³) | Steps | Collision |
|---|---|---|---|---|---|
| 0 | 8.231 | -1.719 | 0.425 | 60/300 | True |
| 1 | 2.632 | 21.233 | 1.012 | 300/300 | False |
| 2 | 8.705 | -0.016 | 0.303 | 75/300 | True |
| 3 | 8.602 | -0.881 | 0.805 | 78/300 | True |
| 4 | 1.078 | 18.955 | 1.233 | 300/300 | False |
| **Mean** | **5.849** | **7.514** | **0.756** | | |
| **Std** | **3.302** | **10.310** | **0.349** | | |

Total crashes: 3/5
