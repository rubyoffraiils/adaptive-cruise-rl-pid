# Style: safe — Seed Sweep Results

Config: PPO, 3M steps, randomized training, style reward, eval on hard.

| Seed | MAE (m) | Mean dist (m) | Min dist (m) | Jerk (m/s³) | Steps | Collision |
|---|---|---|---|---|---|---|
| 0 | 8.030 | 24.322 | -0.081 | 0.179 | 83/300 | True |
| 1 | 2.660 | 23.912 | 15.305 | 1.227 | 300/300 | False |
| 2 | 55.097 | 79.957 | 23.805 | 0.151 | 300/300 | False |
| 3 | 2.479 | 24.468 | 16.816 | 1.104 | 300/300 | False |
| 4 | 5.538 | 26.091 | 13.111 | 1.101 | 300/300 | False |
| **Mean** | **14.761** | **35.750** | **13.791** | **0.752** | | |
| **Std** | **20.272** | **22.116** | **7.805** | **0.482** | | |

Total crashes: 1/5
