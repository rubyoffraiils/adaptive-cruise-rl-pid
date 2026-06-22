"""
Seed sweep trainer — runs N seeds sequentially, evaluates each, prints summary.
Used for both Run 3 and Run 4. RUN variable controls which run folder is used.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from acc_env import AdaptiveCruiseControlEnv

SEEDS     = [0, 1, 2, 3, 4]
TIMESTEPS = 3_000_000
RUN       = "run5_seeds"
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "rl_runs", RUN)
os.makedirs(OUT_DIR, exist_ok=True)


def train_one(seed):

    env = AdaptiveCruiseControlEnv(scenario="randomized")
    env.reset(seed=seed)

    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
    )
    model.learn(total_timesteps=TIMESTEPS)

    model_path = os.path.join(OUT_DIR, f"seed{seed}")
    model.save(model_path)
    return model_path


def evaluate_one(model_path):
    env = AdaptiveCruiseControlEnv(scenario="hard")
    model = PPO.load(model_path)
    obs, _ = env.reset()
    done = False
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
        "min_dist":  min(dists),
        "jerk":      np.mean(np.abs(jerks)),
        "steps":     len(dists),
        "collision": min(dists) <= 0,
    }


def main():
    results = []

    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"Training seed {seed} / {SEEDS[-1]}  ({RUN}) ...")
        print(f"{'='*50}")
        model_path = train_one(seed)
        print(f"  Done. Evaluating on hard scenario ...")
        r = evaluate_one(model_path)
        r["seed"] = seed
        results.append(r)
        verdict = "CRASH" if r["collision"] else "ok"
        print(f"  seed={seed}  mae={r['mae']:.3f}m  min_dist={r['min_dist']:.3f}m  "
              f"steps={r['steps']}/300  [{verdict}]")

    # ── summary ──────────────────────────────────────────────────────────────
    maes      = [r["mae"]      for r in results]
    min_dists = [r["min_dist"] for r in results]
    jerks     = [r["jerk"]     for r in results]
    crashes   = sum(r["collision"] for r in results)

    print(f"\n{'='*50}")
    print(f"SEED SWEEP RESULTS — {RUN}  (eval on hard)")
    print(f"{'='*50}")
    print(f"{'Seed':<6} {'MAE (m)':<12} {'Min dist (m)':<15} {'Jerk':<10} {'Steps':<8} {'Crash'}")
    print("-" * 60)
    for r in results:
        print(f"{r['seed']:<6} {r['mae']:<12.3f} {r['min_dist']:<15.3f} "
              f"{r['jerk']:<10.3f} {r['steps']:<8} {r['collision']}")
    print("-" * 60)
    print(f"{'Mean':<6} {np.mean(maes):<12.3f} {np.mean(min_dists):<15.3f} {np.mean(jerks):<10.3f}")
    print(f"{'Std':<6} {np.std(maes):<12.3f} {np.std(min_dists):<15.3f} {np.std(jerks):<10.3f}")
    print(f"\nTotal crashes: {crashes}/{len(SEEDS)}")

    # ── save markdown ─────────────────────────────────────────────────────────
    md_path = os.path.join(OUT_DIR, "seed_results.md")
    with open(md_path, "w") as f:
        f.write(f"# {RUN} — Seed Sweep Results\n\n")
        f.write("Config: PPO, 3M steps, randomized training, v4 reward, eval on hard.\n\n")
        f.write("| Seed | MAE (m) | Min dist (m) | Jerk (m/s³) | Steps | Collision |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['seed']} | {r['mae']:.3f} | {r['min_dist']:.3f} | "
                    f"{r['jerk']:.3f} | {r['steps']}/300 | {r['collision']} |\n")
        f.write(f"| **Mean** | **{np.mean(maes):.3f}** | **{np.mean(min_dists):.3f}** | "
                f"**{np.mean(jerks):.3f}** | | |\n")
        f.write(f"| **Std** | **{np.std(maes):.3f}** | **{np.std(min_dists):.3f}** | "
                f"**{np.std(jerks):.3f}** | | |\n")
        f.write(f"\nTotal crashes: {crashes}/{len(SEEDS)}\n")
    print(f"\nResults saved to {md_path}")


if __name__ == "__main__":
    main()
