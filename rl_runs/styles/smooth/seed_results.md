# Style: smooth — Seed Sweep Results

Config: PPO, 3M steps, randomized training, style reward, eval on hard.

| Seed | MAE (m) | Mean dist (m) | Min dist (m) | Jerk (m/s³) | Steps | Collision |
|---|---|---|---|---|---|---|
| 0 | 8.207 | 22.629 | -0.672 | 0.317 | 64/300 | True |
| 1 | 8.682 | 22.214 | -0.690 | 0.329 | 74/300 | True |
| 2 | 4.897 | 23.898 | 14.893 | 0.874 | 300/300 | False |
| 3 | 8.811 | 23.512 | -0.805 | 0.210 | 69/300 | True |
| 4 | 9.807 | 27.158 | 11.051 | 0.784 | 300/300 | False |
| **Mean** | **8.081** | **23.882** | **4.755** | **0.503** | | |
| **Std** | **1.675** | **1.745** | **6.818** | **0.271** | | |

Total crashes: 3/5
