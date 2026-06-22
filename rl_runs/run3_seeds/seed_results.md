# Run 3 — Seed Sweep Results

Config: PPO, 1.5M steps, randomized training, eval on hard scenario.
Desired following distance: 20 m. Full episode = 300 steps.

| Seed | MAE (m) | Min dist (m) | Jerk (m/s³) | Steps | Collision | Verdict |
|---|---|---|---|---|---|---|
| 0 | 1.120 | 19.163 | 1.198 | 300/300 | No | ✓ Converged |
| 1 | 5.382 | 13.499 | 1.006 | 300/300 | No | ~ Partial |
| 2 | 96.607 | 30.010 | 0.160 | 300/300 | No | ✗ Reward hacked |
| 3 | 8.766 | -0.960 | 0.653 | 72/300 | Yes | ✗ Crashed |
| 4 | 1.591 | 20.099 | 1.073 | 300/300 | No | ✓ Converged |
| **Mean** | **22.69** | **16.36** | **0.82** | | | |
| **Std** | **37.06** | **10.16** | **0.38** | | | |

Good seeds only (0, 4): MAE = 1.36 ± 0.33 m, min dist = 19.63 ± 0.58 m

Total crashes: 1/5 | Converged: 2/5 | Partial: 1/5 | Reward hacked: 1/5

---

## Seed-by-seed breakdown

### Seed 0 — Converged ✓
Normal successful run. MAE 1.12 m, min distance 19.2 m. Comparable to the original
Run 3 result. The policy learned genuine reactive following behaviour.

### Seed 1 — Partial convergence
No crash, but MAE of 5.4 m and min distance 13.5 m shows the policy is following
too closely and not fully tracking the desired gap. It learned "don't crash" but
didn't fully learn "stay at 20 m". Needs more training steps or a better-shaped
reward in the 13–20 m zone.

### Seed 2 — Reward hacking ✗
MAE of 96.6 m with min distance 30 m means the ego car drove *away* from the lead
car and stayed there. The proximity ramp only penalises being too close (< 15 m) —
it has no penalty for being too far. The agent discovered it could score well by
just accelerating away, avoiding all proximity penalties at the cost of completely
ignoring the tracking objective. This is a classic reward hacking failure:
the agent found a policy that satisfies the reward function without satisfying
the intent behind it. The fix would be a two-sided penalty — punish both
being too close AND being too far beyond some threshold (e.g. > 35 m).

### Seed 3 — Crashed ✗
Collision at step 72 (~7.2 s), coinciding with the emergency brake event in the
hard scenario (lead car brakes at -4 m/s² from t=3–6 s). The policy failed to
respond to the sudden deceleration in time. This is the same failure mode as
Run 1 and Run 2 — the policy didn't fully learn to anticipate closing gaps during
hard braking events. With an unlucky weight initialisation, even 1.5M steps wasn't
enough to converge this behaviour.

### Seed 4 — Converged ✓
Clean run. MAE 1.59 m, min distance 20.1 m — actually stayed slightly beyond the
desired distance, indicating a conservative policy that errs on the side of safety.

---

## What this means

The high variance (std = 37 m on MAE) shows the training is **not yet stable**.
2/5 seeds converge to a good policy, 1 partially converges, 1 reward hacks, 1 crashes.
This is typical for PPO on continuous control without observation normalisation —
the result is sensitive to initial weight values.

**To improve stability:**
1. Add VecNormalize (observation normalisation) — the single highest-impact fix
2. Add a two-sided distance penalty to close the reward hacking loophole (seed 2)
3. Increase to 2M+ steps to help the unlucky seeds (seed 3) converge
4. Consider running more seeds (10+) to get a reliable mean

**PID comparison context:**
PID (hard scenario): MAE = 0.93 m, 0 crashes, deterministic — same result every run.
RL best seed: MAE = 1.12 m. RL mean across good seeds: 1.36 ± 0.33 m.
PID wins on consistency; RL is competitive on performance when it converges,
and unlike PID, it was never explicitly shown the hard scenario during training.
