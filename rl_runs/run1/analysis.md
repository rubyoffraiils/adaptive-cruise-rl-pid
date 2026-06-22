# RL Run 1 — PPO, 100k steps, default scenario

## Config
- Algorithm: PPO (MlpPolicy)
- Timesteps: 100,000
- Scenario: default (fixed)

## Metrics
| Metric | Value |
|---|---|
| Episode length | 271/300 steps |
| Mean absolute error | 8.07 m |
| Min following distance | -0.69 m — collision |
| Max overshoot | +17.44 m too far |
| Max undershoot | -20.70 m too close |

## Why it failed

**Not enough training.** 100k steps / 300 per episode is ~333 episodes. PPO on a continuous-action task needs thousands of episodes just to explore the action space. The ±20 m oscillations are what a near-random policy looks like — it hasn't meaningfully started learning yet.

**Fixed scenario = memorisation, not control.** Every episode had the same brake event at t=5 s with the same magnitude. Instead of learning "brake when the gap is closing", the policy can just memorise "apply braking around step 100". Even that didn't work at 100k steps, which shows how far from convergence this was.

**Reward cliff instead of slope.** The only safety signal was a -1000 penalty at the exact moment of collision, after which the episode ends. Steps at 2 m gap and 19 m gap look identical to the agent until the final step — there's no gradient pointing toward "keep distance". The asymmetric tracking term (`-2x error` when too close) is too weak on its own to prevent the agent from drifting into dangerous territory.

## Changes for Run 2
- Randomized training scenario each episode — forces reactive control, not timing memorisation
- Longer rollouts (n_steps 1024 → 2048) so PPO sees more consequences before updating
- 5× more training (500k steps)

Note: the reward cliff was not fixed in Run 2.
