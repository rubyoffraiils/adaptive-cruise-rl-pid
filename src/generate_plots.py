"""
Generate and save all plots for every PID and RL run into their respective folders.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from pid_controller import PIDController
from acc_simulator import AdaptiveCruiseSimulator
from acc_env import AdaptiveCruiseControlEnv


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


def save_plots(out_dir, times, ego_speeds, lead_speeds, distances, errors,
               accelerations, desired_distance, title_prefix):
    plt.rcParams.update(STYLE)

    # 1 — Following distance
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, distances,    color="#1f77b4", label="Actual distance")
    ax.axhline(desired_distance, color="#d62728", linestyle="--", label="Desired (20 m)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title(f"{title_prefix} — Following Distance")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "following_distance.png"), dpi=150)
    plt.close(fig)

    # 2 — Distance error
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, errors, color="#2ca02c")
    ax.axhline(0, color="#d62728", linestyle="--", label="Zero error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance error (m)")
    ax.set_title(f"{title_prefix} — Distance Error")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "distance_error.png"), dpi=150)
    plt.close(fig)

    # 3 — Vehicle speeds
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, ego_speeds,  color="#1f77b4", label="Ego speed")
    ax.plot(times, lead_speeds, color="#d62728", label="Lead speed")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title(f"{title_prefix} — Vehicle Speeds")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "vehicle_speeds.png"), dpi=150)
    plt.close(fig)

    # 4 — Ego acceleration
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, accelerations, color="#ff7f0e")
    ax.axhline(0, color="#7f7f7f", linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title(f"{title_prefix} — Ego Acceleration")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ego_acceleration.png"), dpi=150)
    plt.close(fig)

    print(f"  Saved 4 plots → {out_dir}")


# ── PID runs ─────────────────────────────────────────────────────────────────

def run_pid(scenario):
    controller = PIDController(kp=4.0, ki=0.00002, kd=7.9995)
    sim = AdaptiveCruiseSimulator(
        desired_distance=20.0, dt=0.05, total_time=30.0, scenario=scenario
    )
    r = sim.run(controller)
    return r


def generate_pid_plots(run_dir, scenario, title_prefix):
    r = run_pid(scenario)
    save_plots(
        out_dir=run_dir,
        times=r["times"],
        ego_speeds=r["ego_speeds"],
        lead_speeds=r["lead_speeds"],
        distances=r["distances"],
        errors=r["distance_errors"],
        accelerations=r["ego_accelerations"],
        desired_distance=r["desired_distance"],
        title_prefix=title_prefix,
    )


# ── RL runs ──────────────────────────────────────────────────────────────────

def run_rl(model_path, scenario):
    from stable_baselines3 import PPO
    env = AdaptiveCruiseControlEnv(scenario=scenario)
    model = PPO.load(model_path)
    obs, _ = env.reset()
    done = False
    times, ego_speeds, lead_speeds = [], [], []
    distances, errors, accelerations = [], [], []
    ego_pos, lead_pos = 0.0, 30.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        ego_pos  += info["ego_speed"]  * env.dt
        lead_pos += info["lead_speed"] * env.dt
        times.append(info["time"])
        ego_speeds.append(info["ego_speed"])
        lead_speeds.append(info["lead_speed"])
        distances.append(info["distance"])
        errors.append(info["distance_error"])
        accelerations.append(info["ego_accel"])
    return (np.array(times), np.array(ego_speeds), np.array(lead_speeds),
            np.array(distances), np.array(errors), np.array(accelerations),
            env.desired_distance)


def generate_rl_plots(run_dir, model_path, scenario, title_prefix):
    times, ego_spd, lead_spd, dists, errs, accels, d_des = run_rl(model_path, scenario)
    save_plots(
        out_dir=run_dir,
        times=times,
        ego_speeds=ego_spd,
        lead_speeds=lead_spd,
        distances=dists,
        errors=errs,
        accelerations=accels,
        desired_distance=d_des,
        title_prefix=title_prefix,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    root = os.path.join(os.path.dirname(__file__), "..")

    runs = [
        # (folder,                             kind,  args...)
        ("pid_runs/run1", "pid", "default", "PID Run 1 — Default Scenario"),
        ("pid_runs/run2", "pid", "hard",    "PID Run 2 — Hard Scenario"),
        ("rl_runs/run1",  "rl",
            os.path.join(root, "rl_runs/run1/model"),
            "default", "RL Run 1 — PPO 100k, Default Scenario"),
        ("rl_runs/run2",  "rl",
            os.path.join(root, "rl_runs/run2/model"),
            "hard",    "RL Run 2 — PPO 500k Randomized → Hard Eval"),
    ]

    for entry in runs:
        folder = os.path.join(root, entry[0])
        os.makedirs(folder, exist_ok=True)
        kind = entry[1]
        print(f"\n{entry[-1]}")
        if kind == "pid":
            generate_pid_plots(folder, scenario=entry[2], title_prefix=entry[3])
        else:
            generate_rl_plots(folder, model_path=entry[2],
                              scenario=entry[3], title_prefix=entry[4])

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
