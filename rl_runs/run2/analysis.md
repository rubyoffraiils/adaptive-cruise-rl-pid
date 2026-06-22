# RL Run 2 — PPO, 500k steps, randomized training → hard eval

## Config
- Algorithm: PPO (MlpPolicy)
- Timesteps: 500,000
- Training: randomized profile each episode
- Eval: hard scenario
- n_steps: 2048, batch_size: 64, lr: 3e-4

## Metrics
| Metric | Value |
|---|---|
| Episode length | 66/300 steps — collision at ~6.6 s |
| Mean absolute error | 8.64 m |
| Min following distance | -0.35 m — collision |

## Why it still failed

`ep_rew_mean` sat at ~-6900 from start to finish — the policy didn't improve at all. Three reasons:

**Reward cliff still present (inherited from Run 1).** Collision fires -1000 once at impact. In the steps before the crash, the agent sees small negative rewards with no signal that it's in danger. PPO's gradient can't reliably attribute a single terminal penalty to actions taken 5-10 steps earlier. The fix is a continuous proximity ramp:

```python
if distance < 15:
    reward -= 10 * (15 - distance)   # ramps from 0 at 15m
if distance < 5:
    reward -= 50 * (5 - distance)    # extra pressure near impact
```

Every step spent at 8 m gap pays a large ongoing cost — the gradient unambiguously points toward "back off".

**Randomized curriculum needs more episodes.** 500k / 300 ≈ 1,667 episodes, each with a completely different lead-car profile. The policy can't reuse any episode-specific timing — it has to generalise from scratch. Empirically, PPO on randomized continuous-control tasks needs 5k–15k episodes, which means at least 1.5M–3M steps.

**Observations aren't normalised.** `distance_error` spans ±100, speeds span 0–50. Large raw values cause large activations early in training and unstable gradients. VecNormalize (running mean/variance normalisation) would stabilise this.

## Changes for Run 3
- Smooth proximity ramp in the reward
- 1.5M+ steps (~5k+ episodes)
- VecNormalize (attempted — see Run 3/4 notes)
