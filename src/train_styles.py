"""
Driving style trainer — trains one RL agent per style (aggressive/safe/smooth)
using reward shaping on top of the base v4 reward.

Style reward modifications:
  aggressive  — smaller desired gap (12 m), lighter jerk/effort penalty,
                 so it chases tighter gaps with sharp inputs
  safe        — larger desired gap (28 m), doubled proximity penalties,
                 large too-far tolerance so it stays well back
  smooth      — base gap (20 m) but 5x jerk penalty and 5x effort penalty
                 to maximise ride comfort at the cost of tracking precision
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from acc_simulator import lead_acceleration_at_time, generate_random_profile


# ── Style configs ─────────────────────────────────────────────────────────────

STYLES = {
    "aggressive": dict(
        desired_distance=12.0,
        jerk_coeff=0.01,
        effort_coeff=0.005,
        prox_15_coeff=5.0,
        prox_5_coeff=25.0,
        too_far_thresh=25.0,
        too_far_coeff=1.0,
        collision_penalty=500.0,
        close_coeff=4.0,
        far_coeff=1.0,
    ),
    "safe": dict(
        desired_distance=28.0,
        jerk_coeff=0.05,
        effort_coeff=0.02,
        prox_15_coeff=20.0,
        prox_5_coeff=100.0,
        too_far_thresh=50.0,
        too_far_coeff=0.2,
        collision_penalty=1000.0,
        close_coeff=5.0,
        far_coeff=0.5,
    ),
    "smooth": dict(
        desired_distance=20.0,
        jerk_coeff=0.25,
        effort_coeff=0.10,
        prox_15_coeff=10.0,
        prox_5_coeff=50.0,
        too_far_thresh=40.0,
        too_far_coeff=0.5,
        collision_penalty=500.0,
        close_coeff=3.0,
        far_coeff=1.0,
    ),
}

SEEDS     = [0, 1, 2, 3, 4]
TIMESTEPS = 3_000_000
ROOT      = os.path.join(os.path.dirname(__file__), "..")


# ── Style-aware environment ───────────────────────────────────────────────────

class StyleEnv(gym.Env):
    """ACC env with per-style reward shaping."""

    def __init__(self, style="smooth"):
        super().__init__()
        self.cfg = STYLES[style]
        self.dt = 0.1
        self.max_time = 30.0
        self.desired_distance = self.cfg["desired_distance"]
        self.min_accel = -5.0
        self.max_accel = 3.0
        self.current_profile = None

        self.action_space = spaces.Box(
            low=np.array([self.min_accel], dtype=np.float32),
            high=np.array([self.max_accel], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=np.array([-100.0, -50.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([100.0, 50.0, 50.0, 50.0], dtype=np.float32),
        )

    def _get_obs(self):
        distance_error = (self.x_lead - self.x_ego) - self.desired_distance
        relative_speed = self.v_lead - self.v_ego
        return np.array([distance_error, relative_speed, self.v_ego, self.v_lead],
                        dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0.0
        self.x_ego = 0.0;  self.v_ego  = 20.0
        self.x_lead = 30.0; self.v_lead = 20.0
        self.prev_accel = 0.0
        rng = np.random.default_rng(seed)
        self.current_profile = generate_random_profile(total_time=self.max_time, rng=rng)
        return self._get_obs(), {}

    def step(self, action):
        cfg = self.cfg
        accel = float(np.clip(action[0], self.min_accel, self.max_accel))

        lead_accel = lead_acceleration_at_time(
            self.t, scenario="randomized", profile=self.current_profile
        )

        self.v_ego  = max(self.v_ego  + accel     * self.dt, 0.0)
        self.v_lead = max(self.v_lead + lead_accel * self.dt, 0.0)
        self.x_ego  += self.v_ego  * self.dt
        self.x_lead += self.v_lead * self.dt
        self.t      += self.dt

        distance       = self.x_lead - self.x_ego
        distance_error = distance - self.desired_distance
        jerk           = (accel - self.prev_accel) / self.dt
        self.prev_accel = accel
        collision = bool(distance <= 0.0)

        r = 0.0
        if distance_error < 0:
            r -= cfg["close_coeff"] * abs(distance_error)
        else:
            r -= cfg["far_coeff"] * abs(distance_error)

        if distance > cfg["too_far_thresh"]:
            r -= cfg["too_far_coeff"] * (distance - cfg["too_far_thresh"])

        if distance < 15.0:
            r -= cfg["prox_15_coeff"] * (15.0 - distance)
        if distance < 5.0:
            r -= cfg["prox_5_coeff"] * (5.0 - distance)

        r -= cfg["jerk_coeff"]   * abs(jerk)
        r -= cfg["effort_coeff"] * accel ** 2

        if collision:
            r -= cfg["collision_penalty"]

        terminated = bool(collision)
        truncated  = bool(self.t >= self.max_time)

        info = {
            "time": self.t, "distance": distance,
            "distance_error": distance_error,
            "relative_speed": self.v_lead - self.v_ego,
            "ego_speed": self.v_ego, "lead_speed": self.v_lead,
            "ego_accel": accel, "lead_accel": lead_accel,
            "jerk": jerk, "collision": collision,
        }
        return self._get_obs(), float(r), terminated, truncated, info


# ── Training / evaluation helpers ─────────────────────────────────────────────

def train_one(style, seed):
    out_dir = os.path.join(ROOT, "rl_runs", "styles", style)
    os.makedirs(out_dir, exist_ok=True)

    env = StyleEnv(style=style)
    env.reset(seed=seed)

    model = PPO(
        "MlpPolicy", env, seed=seed, verbose=0,
        learning_rate=3e-4, n_steps=2048, batch_size=128, gamma=0.99,
    )
    model.learn(total_timesteps=TIMESTEPS)

    model_path = os.path.join(out_dir, f"seed{seed}")
    model.save(model_path)
    return model_path, out_dir


def evaluate_one(model_path):
    from acc_env import AdaptiveCruiseControlEnv
    env   = AdaptiveCruiseControlEnv(scenario="hard")
    model = PPO.load(model_path)
    obs, _ = env.reset()
    done   = False
    errors, dists, jerks = [], [], []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        errors.append(info["distance_error"])
        dists.append(info["distance"])
        jerks.append(info["jerk"])

    return {
        "mae":       np.mean(np.abs(errors)),
        "mean_dist": float(np.mean(dists)),
        "min_dist":  float(min(dists)),
        "jerk":      np.mean(np.abs(jerks)),
        "steps":     len(dists),
        "collision": min(dists) <= 0,
    }


def run_style(style):
    print(f"\n{'='*55}")
    print(f"Style: {style.upper()}")
    print(f"{'='*55}")
    results = []
    out_dir = os.path.join(ROOT, "rl_runs", "styles", style)

    for seed in SEEDS:
        print(f"  Training seed {seed} ...", end=" ", flush=True)
        model_path, _ = train_one(style, seed)
        r = evaluate_one(model_path)
        r["seed"] = seed
        results.append(r)
        verdict = "CRASH" if r["collision"] else "ok"
        print(f"mae={r['mae']:.3f}m  min_dist={r['min_dist']:.3f}m  [{verdict}]")

    maes       = [r["mae"]       for r in results]
    mean_dists = [r["mean_dist"] for r in results]
    min_dists  = [r["min_dist"]  for r in results]
    jerks      = [r["jerk"]      for r in results]
    crashes    = sum(r["collision"] for r in results)

    print(f"\n  Style {style} — Summary:")
    print(f"  {'Seed':<6} {'MAE':<10} {'Mean dist':<12} {'Min dist':<12} {'Jerk':<8} {'Steps':<8} Crash")
    print("  " + "-" * 62)
    for r in results:
        print(f"  {r['seed']:<6} {r['mae']:<10.3f} {r['mean_dist']:<12.3f} "
              f"{r['min_dist']:<12.3f} {r['jerk']:<8.3f} {r['steps']:<8} {r['collision']}")
    print(f"  Crashes: {crashes}/{len(SEEDS)}")

    md_path = os.path.join(out_dir, "seed_results.md")
    with open(md_path, "w") as f:
        f.write(f"# Style: {style} — Seed Sweep Results\n\n")
        f.write("Config: PPO, 3M steps, randomized training, style reward, eval on hard.\n\n")
        f.write("| Seed | MAE (m) | Mean dist (m) | Min dist (m) | Jerk (m/s³) | Steps | Collision |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['seed']} | {r['mae']:.3f} | {r['mean_dist']:.3f} | "
                    f"{r['min_dist']:.3f} | {r['jerk']:.3f} | {r['steps']}/300 | {r['collision']} |\n")
        f.write(f"| **Mean** | **{np.mean(maes):.3f}** | **{np.mean(mean_dists):.3f}** | "
                f"**{np.mean(min_dists):.3f}** | **{np.mean(jerks):.3f}** | | |\n")
        f.write(f"| **Std** | **{np.std(maes):.3f}** | **{np.std(mean_dists):.3f}** | "
                f"**{np.std(min_dists):.3f}** | **{np.std(jerks):.3f}** | | |\n")
        f.write(f"\nTotal crashes: {crashes}/{len(SEEDS)}\n")
    print(f"  Results saved → {md_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--styles", nargs="+", default=list(STYLES.keys()),
                        choices=list(STYLES.keys()),
                        help="Which styles to train (default: all)")
    args = parser.parse_args()

    for style in args.styles:
        run_style(style)

    print("\nDone.")
