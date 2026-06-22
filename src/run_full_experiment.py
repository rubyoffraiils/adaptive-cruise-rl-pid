"""
Full experiment pipeline:

  1. Re-tune PID gains on random profiles (hard scenario held out)
  2. Retrain RL agent (general)
  3. Generate comparison plots (static PNGs) for all controllers on hard scenario
  4. Generate demo MP4 videos for all controllers on hard scenario

"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))

ROOT      = os.path.join(os.path.dirname(__file__), "..")
REPORT    = os.path.join(ROOT, "report")
os.makedirs(REPORT, exist_ok=True)

from pid_controller import PIDController
from acc_simulator import AdaptiveCruiseSimulator, generate_random_profile
from acc_env import AdaptiveCruiseControlEnv
from run_pid import compute_metrics

SEEDS     = [0, 1, 2]
TIMESTEPS = 1_000_000

PLOT_STYLE = {
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Re-tune PID
# ─────────────────────────────────────────────────────────────────────────────

def retune_pid():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n" + "="*60)
    print("STEP 1 — Re-tuning PID on 20 random profiles")
    print("="*60)

    N = 20
    profiles = [
        generate_random_profile(total_time=30.0, rng=np.random.default_rng(seed=i))
        for i in range(N)
    ]
    base_sim = AdaptiveCruiseSimulator(
        desired_distance=20.0, dt=0.05, total_time=30.0, scenario="randomized"
    )

    def score_controller(metrics, desired_distance, distances):
        score = 0.0
        score += 2.0  * metrics["mean_abs_error"]
        score += 0.5  * metrics["mean_abs_jerk"]
        score += 0.05 * metrics["control_effort"]
        overshoot = np.maximum(0, desired_distance - distances)
        score += 3.0 * float(np.mean(overshoot))
        md = metrics["min_distance"]
        if md < 15: score += 100.0  * (15 - md)
        if md < 10: score += 1000.0 * (10 - md)
        score += 10000.0 * metrics["collision_count"]
        return score

    def objective(trial):
        kp = trial.suggest_float("kp", 0.1, 4.0)
        ki = trial.suggest_float("ki", 0.0, 0.02)
        kd = trial.suggest_float("kd", 0.5, 8.0)
        total = 0.0
        for prof in profiles:
            ctrl = PIDController(kp=kp, ki=ki, kd=kd)
            res  = base_sim.run(ctrl, profile=prof)
            m    = compute_metrics(res)
            total += score_controller(m, base_sim.desired_distance, res["distances"])
        return total / N

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=200, show_progress_bar=True)

    best = study.best_trial
    kp = best.params["kp"]
    ki = best.params["ki"]
    kd = best.params["kd"]

    print(f"\nNew PID gains: Kp={kp:.4f}  Ki={ki:.6f}  Kd={kd:.4f}")
    print(f"Avg tuning score: {best.value:.4f}")

    gains = {"kp": kp, "ki": ki, "kd": kd}
    gains_path = os.path.join(REPORT, "pid_gains.json")
    with open(gains_path, "w") as f:
        json.dump(gains, f, indent=2)
    print(f"Gains saved to {gains_path}")
    return gains


# ─────────────────────────────────────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def sim_pid(kp, ki, kd, scenario="hard"):
    ctrl = PIDController(kp=kp, ki=ki, kd=kd)
    sim  = AdaptiveCruiseSimulator(
        desired_distance=20.0, dt=0.05, total_time=30.0, scenario=scenario
    )
    r = sim.run(ctrl)
    times  = r["times"]
    return {
        "times":        times,
        "ego_pos":      r["ego_positions"],
        "lead_pos":     r["lead_positions"],
        "ego_speed":    r["ego_speeds"],
        "lead_speed":   r["lead_speeds"],
        "distance":     r["distances"],
        "dist_error":   r["distance_errors"],
        "ego_accel":    r["ego_accelerations"],
        "jerk":         np.gradient(r["ego_accelerations"], times),
        "desired_dist": r["desired_distance"],
    }


def sim_rl(model_path, scenario="hard"):
    from stable_baselines3 import PPO
    env   = AdaptiveCruiseControlEnv(scenario=scenario)
    model = PPO.load(model_path)
    obs, _ = env.reset()
    done   = False
    times, ego_pos, lead_pos = [], [], []
    ego_spd, lead_spd, dists, errs, accels, jerks = [], [], [], [], [], []
    ep = 0.0; lp = 30.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        ep += info["ego_speed"]  * env.dt
        lp += info["lead_speed"] * env.dt
        times.append(info["time"])
        ego_pos.append(ep)
        lead_pos.append(lp)
        ego_spd.append(info["ego_speed"])
        lead_spd.append(info["lead_speed"])
        dists.append(info["distance"])
        errs.append(info["distance_error"])
        accels.append(info["ego_accel"])
        jerks.append(info["jerk"])
    return {
        "times":        np.array(times),
        "ego_pos":      np.array(ego_pos),
        "lead_pos":     np.array(lead_pos),
        "ego_speed":    np.array(ego_spd),
        "lead_speed":   np.array(lead_spd),
        "distance":     np.array(dists),
        "dist_error":   np.array(errs),
        "ego_accel":    np.array(accels),
        "jerk":         np.array(jerks),
        "desired_dist": env.desired_distance,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Static comparison plots
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "PID":          "#888888",
    "RL (general)": "#1f77b4",
}

def make_comparison_plots(all_data):
    """all_data: dict of label → sim dict"""
    plt.rcParams.update(PLOT_STYLE)

    labels = list(all_data.keys())

    # ── Plot 1: Following distance ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, d in all_data.items():
        ax.plot(d["times"], d["distance"], color=COLORS[label],
                label=label, linewidth=1.8 if label != "PID" else 1.4,
                linestyle="--" if label == "PID" else "-")
    ax.axhline(20.0, color="black", linewidth=0.8, linestyle=":", alpha=0.4,
               label="Desired (20 m)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Following distance (m)")
    ax.set_title("Following Distance — Hard Scenario")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT, "compare_distance.png"), dpi=150)
    plt.close(fig)

    # ── Plot 2: Distance error ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, d in all_data.items():
        ax.plot(d["times"], d["dist_error"], color=COLORS[label],
                label=label, linewidth=1.8 if label != "PID" else 1.4,
                linestyle="--" if label == "PID" else "-")
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance error (m)")
    ax.set_title("Distance Error — Hard Scenario")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT, "compare_error.png"), dpi=150)
    plt.close(fig)

    # ── Plot 3: Acceleration ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, d in all_data.items():
        ax.plot(d["times"], d["ego_accel"], color=COLORS[label],
                label=label, linewidth=1.8 if label != "PID" else 1.4,
                linestyle="--" if label == "PID" else "-", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title("Ego Acceleration — Hard Scenario")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT, "compare_acceleration.png"), dpi=150)
    plt.close(fig)

    # ── Plot 4: Jerk ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, d in all_data.items():
        ax.plot(d["times"], d["jerk"], color=COLORS[label],
                label=label, linewidth=1.8 if label != "PID" else 1.4,
                linestyle="--" if label == "PID" else "-", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Jerk (m/s³)")
    ax.set_title("Jerk — Hard Scenario")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT, "compare_jerk.png"), dpi=150)
    plt.close(fig)

    # ── Plot 5: Speed ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    # lead speed same for all — plot once
    first = next(iter(all_data.values()))
    ax.plot(first["times"], first["lead_speed"], color="black", linewidth=1.2,
            linestyle=":", label="Lead (shared)")
    for label, d in all_data.items():
        ax.plot(d["times"], d["ego_speed"], color=COLORS[label],
                label=f"{label} ego", linewidth=1.5,
                linestyle="--" if label == "PID" else "-", alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Vehicle Speeds — Hard Scenario")
    ax.legend(fontsize=7)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT, "compare_speeds.png"), dpi=150)
    plt.close(fig)

    # ── Plot 6: Metrics bar chart ─────────────────────────────────────────────
    metrics = {}
    for label, d in all_data.items():
        metrics[label] = {
            "MAE (m)":        np.mean(np.abs(d["dist_error"])),
            "Mean dist (m)":  np.mean(d["distance"]),
            "Min dist (m)":   np.min(d["distance"]),
            "Mean |jerk|":    np.mean(np.abs(d["jerk"])),
        }

    keys = list(next(iter(metrics.values())).keys())
    x = np.arange(len(labels))
    n = len(keys)
    w = 0.8 / n

    fig, axes = plt.subplots(1, n, figsize=(14, 4))
    for j, key in enumerate(keys):
        ax = axes[j]
        vals = [metrics[lb][key] for lb in labels]
        bars = ax.bar(x, vals, color=[COLORS[lb] for lb in labels], width=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(key, fontsize=9)
        ax.grid(axis="y", alpha=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Metric Comparison — Hard Scenario", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT, "compare_metrics_bar.png"), dpi=150)
    plt.close(fig)

    print(f"  Saved 6 comparison plots → {REPORT}/")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Videos
# ─────────────────────────────────────────────────────────────────────────────

STEP_STRIDE = 3
FPS         = 20
CAR_W       = 4.0
CAR_L       = 8.0
ROAD_W      = 12.0
VIEW_W      = 120.0
DASH_SPACING = 20.0
N_DASHES    = 20

def build_video(sim_data, title, out_path):
    times  = sim_data["times"]
    errors = sim_data["dist_error"]
    e_spd  = sim_data["ego_speed"]
    l_spd  = sim_data["lead_speed"]
    e_pos  = sim_data["ego_pos"]
    l_pos  = sim_data["lead_pos"]
    dists  = sim_data["distance"]
    accels = sim_data["ego_accel"]

    frame_idx = list(range(0, len(times), STEP_STRIDE))
    n_frames  = len(frame_idx)

    fig = plt.figure(figsize=(12, 7), facecolor="#1a1a2e")
    gs  = GridSpec(2, 2, figure=fig,
                   height_ratios=[1.6, 1],
                   hspace=0.38, wspace=0.32,
                   left=0.07, right=0.97, top=0.93, bottom=0.08)

    ax_road  = fig.add_subplot(gs[0, :])
    ax_speed = fig.add_subplot(gs[1, 0])
    ax_err   = fig.add_subplot(gs[1, 1])

    for ax in (ax_road, ax_speed, ax_err):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaaacc", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333366")

    fig.suptitle(title, color="#e0e0ff", fontsize=13, fontweight="bold")

    ax_road.set_ylim(-ROAD_W, ROAD_W)
    ax_road.set_aspect("equal")
    ax_road.set_yticks([])
    ax_road.set_xlabel("Road position (m)", color="#aaaacc", fontsize=8)
    ax_road.set_title("Top-down view", color="#8888bb", fontsize=9)

    for y in (-ROAD_W * 0.75, ROAD_W * 0.75):
        ax_road.axhline(y, color="#555588", linewidth=1.5, linestyle="--")

    dash_xs   = np.arange(N_DASHES) * DASH_SPACING
    dash_lines = []
    for dx in dash_xs:
        ln, = ax_road.plot([dx, dx + 8], [0, 0],
                           color="#444466", linewidth=1.2, linestyle="-")
        dash_lines.append(ln)

    ego_patch = mpatches.FancyBboxPatch(
        (0, -CAR_W/2), CAR_L, CAR_W,
        boxstyle="round,pad=0.3",
        facecolor="#4fc3f7", edgecolor="#81d4fa", linewidth=1.5, zorder=3)
    lead_patch = mpatches.FancyBboxPatch(
        (0, -CAR_W/2), CAR_L, CAR_W,
        boxstyle="round,pad=0.3",
        facecolor="#ef5350", edgecolor="#ff8a80", linewidth=1.5, zorder=3)
    ax_road.add_patch(ego_patch)
    ax_road.add_patch(lead_patch)

    gap_line, = ax_road.plot([], [], color="#ffd54f", linewidth=1.5,
                             linestyle=":", zorder=2)
    gap_text  = ax_road.text(0, ROAD_W*0.45, "", color="#ffd54f",
                              fontsize=8, ha="center", va="bottom")
    ego_label  = ax_road.text(0, -ROAD_W*0.55, "", color="#4fc3f7", fontsize=7.5, ha="center")
    lead_label = ax_road.text(0, -ROAD_W*0.55, "", color="#ef5350", fontsize=7.5, ha="center")
    time_text  = ax_road.text(0.01, 0.96, "", transform=ax_road.transAxes,
                               color="#ccccee", fontsize=9, va="top")

    ax_speed.set_xlim(times[0], times[-1])
    ax_speed.set_ylim(0, max(e_spd.max(), l_spd.max()) * 1.15 + 1)
    ax_speed.set_xlabel("Time (s)", color="#aaaacc", fontsize=8)
    ax_speed.set_ylabel("Speed (m/s)", color="#aaaacc", fontsize=8)
    ax_speed.set_title("Vehicle speeds", color="#8888bb", fontsize=9)
    ax_speed.plot(times, e_spd, color="#4fc3f7", alpha=0.15, linewidth=1)
    ax_speed.plot(times, l_spd, color="#ef5350", alpha=0.15, linewidth=1)
    spd_ego_ln,  = ax_speed.plot([], [], color="#4fc3f7", linewidth=1.8, label="Ego")
    spd_lead_ln, = ax_speed.plot([], [], color="#ef5350", linewidth=1.8, label="Lead")
    ax_speed.legend(fontsize=7, facecolor="#16213e", labelcolor="#ccccee", edgecolor="#333366")

    err_abs = np.abs(errors)
    ax_err.set_xlim(times[0], times[-1])
    lim = max(err_abs.max() * 1.2, 2.0)
    ax_err.set_ylim(-lim, lim)
    ax_err.set_xlabel("Time (s)", color="#aaaacc", fontsize=8)
    ax_err.set_ylabel("Distance error (m)", color="#aaaacc", fontsize=8)
    ax_err.set_title("Following distance error", color="#8888bb", fontsize=9)
    ax_err.axhline(0, color="#ffd54f", linewidth=1, linestyle="--")
    ax_err.plot(times, errors, color="#a5d6a7", alpha=0.15, linewidth=1)
    err_ln, = ax_err.plot([], [], color="#a5d6a7", linewidth=1.8)

    def init():
        ego_patch.set_x(-CAR_L - 10); ego_patch.set_y(-CAR_W/2)
        lead_patch.set_x(-CAR_L - 10); lead_patch.set_y(-CAR_W/2)
        gap_line.set_data([], [])
        gap_text.set_text(""); ego_label.set_text(""); lead_label.set_text("")
        time_text.set_text("")
        spd_ego_ln.set_data([], []); spd_lead_ln.set_data([], [])
        err_ln.set_data([], [])
        return (ego_patch, lead_patch, gap_line, gap_text,
                ego_label, lead_label, time_text,
                spd_ego_ln, spd_lead_ln, err_ln, *dash_lines)

    def update(fi):
        i   = frame_idx[fi]
        ex  = e_pos[i]; lx = l_pos[i]; t = times[i]
        centre = (ex + lx) / 2.0
        x_min  = centre - VIEW_W/2; x_max = centre + VIEW_W/2
        ax_road.set_xlim(x_min, x_max)
        ego_patch.set_x(ex); ego_patch.set_y(-CAR_W/2)
        lead_patch.set_x(lx); lead_patch.set_y(-CAR_W/2)
        for k, ln in enumerate(dash_lines):
            offset  = ((dash_xs[k] - x_min) % (DASH_SPACING * N_DASHES))
            wrapped = x_min + offset
            ln.set_xdata([wrapped, wrapped + 8])
        ego_front = ex + CAR_L; lead_rear = lx
        gap_mid   = (ego_front + lead_rear) / 2.0
        gap_line.set_data([ego_front, lead_rear], [CAR_W*0.6, CAR_W*0.6])
        gap_text.set_position((gap_mid, ROAD_W*0.48))
        gap_text.set_text(f"{dists[i]:.1f} m  (err {errors[i]:+.1f})")
        ego_label.set_position((ex + CAR_L/2, -ROAD_W*0.55))
        ego_label.set_text(f"{e_spd[i]:.1f} m/s")
        lead_label.set_position((lx + CAR_L/2, -ROAD_W*0.55))
        lead_label.set_text(f"{l_spd[i]:.1f} m/s")
        time_text.set_text(f"t = {t:.1f} s")
        spd_ego_ln.set_data(times[:i+1],  e_spd[:i+1])
        spd_lead_ln.set_data(times[:i+1], l_spd[:i+1])
        err_ln.set_data(times[:i+1],      errors[:i+1])
        return (ego_patch, lead_patch, gap_line, gap_text,
                ego_label, lead_label, time_text,
                spd_ego_ln, spd_lead_ln, err_ln, *dash_lines)

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=1000/FPS, blit=True
    )
    writer = animation.FFMpegWriter(fps=FPS, bitrate=1800,
                                    extra_args=["-vcodec", "libx264"])
    ani.save(out_path, writer=writer)
    plt.close(fig)
    print(f"  Saved video: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3+4 dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def generate_outputs(gains):
    print("\n" + "="*60)
    print("STEP 3+4 — Generating plots and videos")
    print("="*60)

    kp, ki, kd = gains["kp"], gains["ki"], gains["kd"]

    print("  Simulating PID on hard ...")
    all_data = {"PID": sim_pid(kp, ki, kd, scenario="hard")}

    print("  Simulating RL general (run5/best seed) on hard ...")
    run5_dir = os.path.join(ROOT, "rl_runs", "run5_seeds")
    best_run5 = os.path.join(run5_dir, "seed4")
    all_data["RL (general)"] = sim_rl(best_run5, scenario="hard")

    # plots
    print("\n  Making comparison plots ...")
    make_comparison_plots(all_data)

    # videos
    print("\n  Rendering videos ...")
    for label, d in all_data.items():
        safe_name = label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        out_path  = os.path.join(REPORT, f"demo_{safe_name}.mp4")
        print(f"  [{label}] ...", end=" ", flush=True)
        build_video(d, f"ACC — {label} — Hard Scenario", out_path)

    # print final metrics table
    print("\n  FINAL METRICS TABLE (hard scenario)\n")
    print(f"{'Controller':<16} {'MAE (m)':<10} {'Mean dist':<12} {'Min dist':<11} {'Mean |jerk|':<13} {'Collision'}")
    print("-" * 70)
    for label, d in all_data.items():
        mae  = np.mean(np.abs(d["dist_error"]))
        md   = np.mean(d["distance"])
        mnd  = np.min(d["distance"])
        jerk = np.mean(np.abs(d["jerk"]))
        coll = "YES" if mnd <= 0 else "no"
        print(f"{label:<16} {mae:<10.3f} {md:<12.3f} {mnd:<11.3f} {jerk:<13.3f} {coll}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pid-tune", action="store_true",
                        help="Skip PID re-tuning, load gains from report/pid_gains.json")
    args = parser.parse_args()

    gains_path = os.path.join(REPORT, "pid_gains.json")
    if args.skip_pid_tune and os.path.exists(gains_path):
        with open(gains_path) as f:
            gains = json.load(f)
        print(f"Loaded existing PID gains: Kp={gains['kp']:.4f} Ki={gains['ki']:.6f} Kd={gains['kd']:.4f}")
    else:
        gains = retune_pid()

    generate_outputs(gains)

    print("\nDone. All outputs in report/")
