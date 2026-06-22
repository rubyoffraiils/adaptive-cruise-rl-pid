"""
Two plots:
1. Reward vs distance for all reward versions (v1/v2, v3, v4) on one axes.
2. Following distance: Run 3 seed 2 (reward hacked) vs Run 4 seed 4 (fixed).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from acc_env import AdaptiveCruiseControlEnv

ROOT = os.path.join(os.path.dirname(__file__), "..")

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#cccccc",
    "axes.labelcolor":  "black",
    "axes.titlecolor":  "black",
    "xtick.color":      "black",
    "ytick.color":      "black",
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
    "text.color":       "black",
    "grid.color":       "#e0e0e0",
    "grid.linestyle":   "--",
    "grid.alpha":       0.7,
}


# ── PLOT 1: Reward shape vs distance ─────────────────────────────────────────

def reward_v1v2(distance, desired=20.0, accel=0.0, jerk=0.0):
    """Run 1 & 2 reward (cliff only)."""
    distance_error = distance - desired
    r = -abs(distance_error)
    if distance < desired:
        r -= 2.0 * abs(distance_error)
    r -= 0.05 * abs(jerk)
    r -= 0.02 * accel ** 2
    if distance <= 0.0:
        r -= 1000.0
    return r


def reward_v3(distance, desired=20.0, accel=0.0, jerk=0.0):
    """Run 3 reward (proximity ramp, no too-far nudge)."""
    distance_error = distance - desired
    r = 0.0
    if distance_error < 0:
        r -= 3.0 * abs(distance_error)
    else:
        r -= 1.0 * abs(distance_error)
    if distance < 15.0:
        r -= 10.0 * (15.0 - distance)
    if distance < 5.0:
        r -= 50.0 * (5.0 - distance)
    r -= 0.05 * abs(jerk)
    r -= 0.02 * accel ** 2
    if distance <= 0.0:
        r -= 500.0
    return r


def reward_v4(distance, desired=20.0, accel=0.0, jerk=0.0):
    """Run 4 & 5 reward (v3 + soft too-far nudge)."""
    r = reward_v3(distance, desired, accel, jerk)
    if distance > 40.0:
        r -= 0.5 * (distance - 40.0)
    return r


def plot_reward_shapes():
    plt.rcParams.update(STYLE)

    # include d=0 so the collision penalty shows; use a tiny step near 0
    distances = np.concatenate([
        np.linspace(0.001, 2.0, 200),
        np.linspace(2.0, 60.0, 800),
    ])

    r1 = [reward_v1v2(d) for d in distances]
    r3 = [reward_v3(d) for d in distances]
    r4 = [reward_v4(d) for d in distances]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.05},
    )

    for ax in (ax_top, ax_bot):
        ax.plot(distances, r1, color="#d62728", linewidth=2.0, label="v1/v2 — cliff (Runs 1–2)")
        ax.plot(distances, r3, color="#ff7f0e", linewidth=2.0, label="v3 — proximity ramp (Run 3)")
        ax.plot(distances, r4, color="#1f77b4", linewidth=2.0, linestyle="--",
                label="v4 — ramp + too-far nudge (Runs 4–5)")
        ax.axvline(20.0, color="#555555", linewidth=1.0, linestyle=":", label="Desired distance (20 m)")
        ax.axvline(15.0, color="#aaaaaa", linewidth=0.8, linestyle=":")
        ax.axvline(40.0, color="#aaaaaa", linewidth=0.8, linestyle=":")
        ax.grid(True)

    # top panel: shows the collision cliff
    ax_top.set_ylim(-1100, 5)
    ax_top.set_yticks([-1000, -500, 0])
    ax_top.text(0.4, -520, "collision\n−1000", fontsize=7.5, color="#d62728")
    ax_top.text(0.4, -120, "collision\n−500", fontsize=7.5, color="#ff7f0e")

    # bottom panel: shows the ramp detail
    ax_bot.set_ylim(-160, 5)
    ax_bot.set_xlim(0, 60)
    ax_bot.text(15.2, -150, "15 m\n(ramp start)", fontsize=7.5, color="#888888", va="bottom")
    ax_bot.text(40.2, -10, "40 m\n(nudge start)", fontsize=7.5, color="#888888", va="bottom")

    # broken axis markers
    d = 0.015
    kwargs = dict(transform=ax_top.transAxes, color="#555555", clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    ax_bot.set_xlabel("Following distance (m)", fontsize=10)
    fig.text(0.02, 0.5, "Instantaneous reward", va="center", rotation="vertical", fontsize=10)
    ax_top.set_title("Reward Function Shape vs Following Distance\n"
                     "(accel = 0, jerk = 0; tracking + safety terms only)",
                     fontsize=11)
    ax_top.legend(fontsize=9)
    fig.tight_layout()

    out = os.path.join(ROOT, "rl_runs", "reward_shape_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── PLOT 2: Run 3 seed 2 (hacked) vs Run 4 seed 4 (fixed) ───────────────────

def rollout(model_path, scenario="hard"):
    env = AdaptiveCruiseControlEnv(scenario=scenario)
    model = PPO.load(model_path)
    obs, _ = env.reset()
    times, distances = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        times.append(info["time"])
        distances.append(info["distance"])
    return np.array(times), np.array(distances)


def plot_hack_vs_fixed():
    plt.rcParams.update(STYLE)

    print("Rolling out Run 3 seed 2 (reward hacked) ...")
    t3, d3 = rollout(os.path.join(ROOT, "rl_runs/run3_seeds/seed2.zip"))
    print("Rolling out Run 4 seed 4 (fixed) ...")
    t4, d4 = rollout(os.path.join(ROOT, "rl_runs/run4_seeds/seed4.zip"))

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(t3, d3, color="#d62728", linewidth=2.0,
            label="Run 3 seed 2 — reward hacked (ego stopped, lead drove away, MAE = 96.6 m)")
    ax.plot(t4, d4, color="#1f77b4", linewidth=2.0,
            label="Run 4 seed 4 — fixed (MAE = 1.02 m)")
    ax.axhline(20.0, color="#555555", linewidth=1.0, linestyle="--", label="Desired distance (20 m)")
    ax.axhline(40.0, color="#aaaaaa", linewidth=0.8, linestyle=":",
               label="Too-far nudge threshold (40 m)")

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Following distance (m)", fontsize=10)
    ax.set_title("Reward Hacking (Run 3) vs Fixed Policy (Run 4)\n"
                 "Eval on hard scenario", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True)
    fig.tight_layout()

    out = os.path.join(ROOT, "rl_runs", "hack_vs_fixed.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── PLOT 3: Driving styles comparison ────────────────────────────────────────

STYLE_COLORS = {
    "aggressive": "#d62728",
    "safe":       "#2ca02c",
    "smooth":     "#9467bd",
    "general":    "#1f77b4",
}

STYLE_DESIRED = {
    "aggressive": 12.0,
    "safe":       28.0,
    "smooth":     20.0,
    "general":    20.0,
}


def rollout_full(model_path, scenario="hard"):
    """Return times, distances, accels, jerks arrays."""
    env = AdaptiveCruiseControlEnv(scenario=scenario)
    model = PPO.load(model_path)
    obs, _ = env.reset()
    times, distances, accels, jerks = [], [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        times.append(info["time"])
        distances.append(info["distance"])
        accels.append(info["ego_accel"])
        jerks.append(info["jerk"])
    return (np.array(times), np.array(distances),
            np.array(accels), np.array(jerks))


def _best_seed(style_dir):
    """Pick seed with min MAE that didn't crash, fallback to seed0."""
    import re
    md = os.path.join(style_dir, "seed_results.md")
    best_seed, best_mae = 0, float("inf")
    if os.path.exists(md):
        for line in open(md):
            m = re.match(r"\|\s*(\d)\s*\|\s*([\d.]+)\s*\|.*\|\s*(True|False)\s*\|", line)
            if m:
                seed, mae, crash = int(m.group(1)), float(m.group(2)), m.group(3) == "True"
                if not crash and mae < best_mae:
                    best_mae = mae
                    best_seed = seed
    return best_seed


def plot_driving_styles():
    plt.rcParams.update(STYLE)

    styles_root = os.path.join(ROOT, "rl_runs", "styles")
    general_path = os.path.join(ROOT, "rl_runs", "run5_seeds", "seed4.zip")

    rollouts = {}

    for style in ("aggressive", "safe", "smooth"):
        style_dir = os.path.join(styles_root, style)
        seed = _best_seed(style_dir)
        path = os.path.join(style_dir, f"seed{seed}.zip")
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping {style}")
            continue
        print(f"Rolling out {style} (seed {seed}) ...")
        rollouts[style] = rollout_full(path)

    if os.path.exists(general_path):
        print("Rolling out general (run5 seed4) ...")
        rollouts["general"] = rollout_full(general_path)

    if not rollouts:
        print("No style models found — skipping plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax_dist, ax_accel, ax_jerk = axes

    labels = {
        "aggressive": "Aggressive (12 m gap)",
        "safe":       "Safe (28 m gap)",
        "smooth":     "Smooth (20 m, low jerk)",
        "general":    "General (20 m)",
    }

    for name, (t, d, a, j) in rollouts.items():
        col = STYLE_COLORS[name]
        lbl = labels[name]
        ls  = "--" if name == "general" else "-"
        lw  = 1.6

        ax_dist.plot(t, d, color=col, linewidth=lw, linestyle=ls, label=lbl)
        ax_accel.plot(t, a, color=col, linewidth=lw, linestyle=ls, label=lbl, alpha=0.85)
        ax_jerk.plot(t, j, color=col, linewidth=lw, linestyle=ls, label=lbl, alpha=0.85)

    # desired distance reference lines
    for name, desired in STYLE_DESIRED.items():
        if name in rollouts:
            ax_dist.axhline(desired, color=STYLE_COLORS[name],
                            linewidth=0.7, linestyle=":", alpha=0.5)

    ax_dist.set_xlabel("Time (s)", fontsize=10)
    ax_dist.set_ylabel("Following distance (m)", fontsize=10)
    ax_dist.set_title("Following Distance\n(hard scenario)", fontsize=11)
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True)

    ax_accel.set_xlabel("Time (s)", fontsize=10)
    ax_accel.set_ylabel("Acceleration (m/s²)", fontsize=10)
    ax_accel.set_title("Ego Acceleration\n(hard scenario)", fontsize=11)
    ax_accel.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.3)
    ax_accel.grid(True)

    ax_jerk.set_xlabel("Time (s)", fontsize=10)
    ax_jerk.set_ylabel("Jerk (m/s³)", fontsize=10)
    ax_jerk.set_title("Jerk\n(hard scenario)", fontsize=11)
    ax_jerk.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.3)
    ax_jerk.grid(True)

    fig.suptitle("Driving Style Comparison — RL Agents\n"
                 "(dotted reference lines = each style's desired gap)",
                 fontsize=12)
    fig.tight_layout()

    out = os.path.join(ROOT, "rl_runs", "styles", "style_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")

    # ── metrics summary bar chart ─────────────────────────────────────────────
    metric_names = ["MAE (m)", "Mean dist (m)", "Min dist (m)", "Mean |jerk|"]
    metrics = {}
    for name, (t, d, a, j) in rollouts.items():
        desired = STYLE_DESIRED[name]
        errors = d - desired
        metrics[name] = [
            float(np.mean(np.abs(errors))),
            float(np.mean(d)),
            float(np.min(d)),
            float(np.mean(np.abs(j))),
        ]

    x = np.arange(len(rollouts))
    names = list(rollouts.keys())
    fig2, axes2 = plt.subplots(1, 4, figsize=(14, 4))

    for col_idx, mname in enumerate(metric_names):
        ax = axes2[col_idx]
        vals = [metrics[n][col_idx] for n in names]
        bars = ax.bar(x, vals, color=[STYLE_COLORS[n] for n in names], width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[n] for n in names], rotation=20, ha="right", fontsize=7.5)
        ax.set_title(mname, fontsize=9)
        ax.grid(axis="y", alpha=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    fig2.suptitle("Style Metrics — Hard Scenario\n(MAE relative to each style's own desired gap)",
                  fontsize=11)
    fig2.tight_layout()

    out2 = os.path.join(ROOT, "rl_runs", "styles", "style_metrics_bar.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {out2}")


if __name__ == "__main__":
    plot_reward_shapes()
    plot_hack_vs_fixed()
    plot_driving_styles()
    print("\nDone.")
