# Style: aggressive — Seed Sweep Results

Config: PPO, 3M steps, randomized training, style reward, eval on hard.

| Seed | MAE (m) | Mean dist (m) | Min dist (m) | Jerk (m/s³) | Steps | Collision |
|---|---|---|---|---|---|---|
| 0 | 13.798 | 22.491 | -0.077 | 0.169 | 108/300 | True |
| 1 | 12.749 | 22.771 | -0.537 | 0.143 | 62/300 | True |
| 2 | 13.757 | 23.711 | -0.576 | 0.042 | 71/300 | True |
| 3 | 2.968 | 12.403 | 1.800 | 1.246 | 300/300 | False |
| 4 | 9.685 | 17.242 | -0.005 | 2.037 | 54/300 | True |
| **Mean** | **10.591** | **19.724** | **0.121** | **0.727** | | |
| **Std** | **4.096** | **4.303** | **0.871** | **0.788** | | |

Total crashes: 4/5
