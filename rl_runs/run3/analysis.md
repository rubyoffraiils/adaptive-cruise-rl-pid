# RL Run 3 — PPO, 1.5M steps, proximity ramp reward

## Config
- Algorithm: PPO (MlpPolicy)
- Timesteps: 1,500,000
- Training: randomized, Eval: hard
- n_steps: 2048, batch_size: 128, lr: 3e-4, gamma: 0.99

## What changed from Run 2

### Reward redesign (v3)

Old reward penalised collision with a single -1000 at impact — no signal during approach. New reward adds a continuous proximity ramp:

```python
# asymmetric tracking: too close costs 3x more than too far
if distance_error < 0:
    reward -= 3.0 * abs(distance_error)
else:
    reward -= 1.0 * abs(distance_error)

# proximity ramp — fires every step in the danger zone
if distance < 15.0:
    reward -= 10.0 * (15.0 - distance)   # zone A: 0 at 15m, -100 at 5m
if distance < 5.0:
    reward -= 50.0 * (5.0 - distance)    # zone B: sharp extra pressure near contact

reward -= 0.05 * abs(jerk)
reward -= 0.02 * accel ** 2

if collision:
    reward -= 500.0   # backstop, no longer the primary signal
```

At 10 m gap, zone A costs -50 per step. Over 10 steps (1 second) that's -500 — equivalent to the old collision penalty, but distributed across the approach so the gradient clearly points toward "back off". The asymmetric tracking term reinforces it: 1 m too close costs 3× 1 m too far.

### 3× more training steps
1.5M steps / 300 per episode ≈ 5,000 episodes — enough to sample a diverse range of random profiles and build a general reactive policy.

### Larger batch size (64 → 128)
Reduces gradient variance per update. With a denser reward signal and more steps, this helped stabilise learning.

## Results

| Metric | Run 1 | Run 2 | Run 3 | PID |
|---|---|---|---|---|
| Episode length | 271/300 | 66/300 | **300/300** | 600/600 |
| Mean abs error | 8.07 m | 8.64 m | **1.09 m** | 0.93 m |
| Min distance | -0.69 m | -0.35 m | **18.0 m** | 19.4 m |
| Collision | Yes | Yes | **No** | No |
| Mean abs jerk | — | — | 1.20 m/s³ | 0.99 m/s³ |

First successful policy: completed all 300 steps without a collision. Mean error of 1.09 m vs PID's 0.93 m — within 0.16 m. Slightly choppier than PID (jerk 1.20 vs 0.99) but within acceptable range.

Training logs confirmed the ramp was the fix: `ep_len_mean` hit 300 in the final iterations, `explained_variance = 0.946`, `ep_rew_mean` improved from ~-6900 (Run 2) to ~-1370.

RL arguably had a harder task: PID was explicitly Bayesian-tuned on the hard scenario; RL was trained only on randomised profiles and saw hard for the first time at eval.

## Robustness

The hard scenario is deterministic, so the same model evaluated repeatedly gives identical results (MAE 1.087 ± 0.000 across 5 seeds at inference — not training seeds).

On 10 unseen random profiles never seen during training:

| Metric | Value |
|---|---|
| Mean abs error | 1.53 m |
| Mean min distance | 17.66 m |
| Crashes | 0/10 |

Clean generalisation — the policy learned reactive control, not timing memorisation.

## Remaining issue

One known loophole: because v3 has no penalty for being far away, seed 2 learned to stop and let the lead car drive off (~96 m MAE). This technically avoids collision, but it's not following. Fixed in Run 4.
