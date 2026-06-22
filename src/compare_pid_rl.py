"""
Generate comparison plots and videos for PID vs RL on the hard scenario.

PID: gains tuned on 20 random profiles (hard held out)
RL:  run4 seed4 — trained on randomized scenario (hard held out)

Both see hard for the first time at evaluation — fair comparison.

Outputs → report/
  compare_distance.png
  compare_error.png
  compare_acceleration.png
  compare_jerk.png
  compare_speeds.png
  compare_metrics_bar.png
  demo_PID.mp4
  demo_RL.mp4
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))

ROOT   = os.path.join(os.path.dirname(__file__), "..")
REPORT = os.path.join(ROOT, "report")
os.makedirs(REPORT, exist_ok=True)

from pid_controller import PIDController
from acc_simulator import AdaptiveCruiseSimulator
from acc_env import AdaptiveCruiseControlEnv

#Tuned gains (random-profile tuning, hard held out) 
PID_KP = 4.0000
PID_KI = 0.00002
PID_KD = 7.9995

# Best RL model 
RL_MODEL = os.path.join(ROOT, "rl_runs", "run4_seeds", "seed4")

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

COLORS = {"PID": "#e67e22", "RL (PPO)": "#2980b9"}


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

def sim_pid():
    ctrl = PIDController(kp=PID_KP, ki=PID_KI, kd=PID_KD)
    sim  = AdaptiveCruiseSimulator(
        desired_distance=20.0, dt=0.05, total_time=30.0, scenario="hard"
    )
    r = sim.run(ctrl)
    times = r["times"]
    accels = r["ego_accelerations"]
    return {
        "times":        times,
        "ego_pos":      r["ego_positions"],
        "lead_pos":     r["lead_positions"],
        "ego_speed":    r["ego_speeds"],
        "lead_speed":   r["lead_speeds"],
        "distance":     r["distances"],
        "dist_error":   r["distance_errors"],
        "ego_accel":    accels,
        "jerk":         np.gradient(accels, times),
        "desired_dist": 20.0,
    }


def sim_rl():
    from stable_baselines3 import PPO
    env   = AdaptiveCruiseControlEnv(scenario="hard")
    model = PPO.load(RL_MODEL)
    obs, _ = env.reset()
    done   = False
    times, ego_pos, lead_pos = [], [], []
    ego_spd, lead_spd, dists, errs, accels, jerks = [], [], [], [], [], []
    ep = 0.0
    lp = 30.0
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
        "desired_dist": 20.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(all_data):
    plt.rcParams.update(PLOT_STYLE)

    def save(fig, name):
        path = os.path.join(REPORT, name)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")

    # 1 — Following distance
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, d in all_data.items():
        ax.plot(d["times"], d["distance"], color=COLORS[label],
                label=label, linewidth=2,
                linestyle="--" if label == "PID" else "-")
    ax.axhline(20.0, color="#555555", linewidth=1, linestyle=":", label="Desired (20 m)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Following distance (m)")
    ax.set_title("Following Distance — Hard Scenario (fair evaluation)")
    ax.legend()
    ax.grid(True)
    save(fig, "compare_distance.png")

    # 2 — Distance error
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, d in all_data.items():
        ax.plot(d["times"], d["dist_error"], color=COLORS[label],
                label=label, linewidth=2,
                linestyle="--" if label == "PID" else "-")
    ax.axhline(0, color="#555555", linewidth=1, linestyle=":", label="Zero error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance error (m)")
    ax.set_title("Distance Error — Hard Scenario")
    ax.legend()
    ax.grid(True)
    save(fig, "compare_error.png")

    # 3 — Acceleration
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, d in all_data.items():
        ax.plot(d["times"], d["ego_accel"], color=COLORS[label],
                label=label, linewidth=2,
                linestyle="--" if label == "PID" else "-", alpha=0.9)
    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title("Ego Acceleration — Hard Scenario")
    ax.legend()
    ax.grid(True)
    save(fig, "compare_acceleration.png")

    # 4 — Jerk
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, d in all_data.items():
        ax.plot(d["times"], d["jerk"], color=COLORS[label],
                label=label, linewidth=2,
                linestyle="--" if label == "PID" else "-", alpha=0.9)
    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Jerk (m/s³)")
    ax.set_title("Jerk — Hard Scenario")
    ax.legend()
    ax.grid(True)
    save(fig, "compare_jerk.png")

    # 5 — Speeds
    fig, ax = plt.subplots(figsize=(10, 4))
    first = next(iter(all_data.values()))
    ax.plot(first["times"], first["lead_speed"], color="#2ecc71",
            linewidth=1.5, linestyle=":", label="Lead car")
    for label, d in all_data.items():
        ax.plot(d["times"], d["ego_speed"], color=COLORS[label],
                label=f"{label} ego", linewidth=2,
                linestyle="--" if label == "PID" else "-", alpha=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Vehicle Speeds — Hard Scenario")
    ax.legend()
    ax.grid(True)
    save(fig, "compare_speeds.png")

    # 6 — Metrics bar chart
    metric_defs = [
        ("MAE (m)",       lambda d: np.mean(np.abs(d["dist_error"]))),
        ("Mean dist (m)", lambda d: np.mean(d["distance"])),
        ("Min dist (m)",  lambda d: np.min(d["distance"])),
        ("Mean |jerk|",   lambda d: np.mean(np.abs(d["jerk"]))),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    labels = list(all_data.keys())
    x = np.arange(len(labels))
    for ax, (mname, mfn) in zip(axes, metric_defs):
        vals  = [mfn(all_data[lb]) for lb in labels]
        bars  = ax.bar(x, vals, color=[COLORS[lb] for lb in labels], width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(mname, fontsize=9)
        ax.grid(axis="y", alpha=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Metric Comparison — Hard Scenario", fontsize=11)
    save(fig, "compare_metrics_bar.png")

    # print table
    print("\n  METRICS (hard scenario, fair evaluation)")
    print(f"  {'Controller':<12} {'MAE (m)':<10} {'Mean dist':<12} {'Min dist':<11} {'Mean |jerk|'}")
    print("  " + "-"*50)
    for label, d in all_data.items():
        print(f"  {label:<12} "
              f"{np.mean(np.abs(d['dist_error'])):<10.3f} "
              f"{np.mean(d['distance']):<12.3f} "
              f"{np.min(d['distance']):<11.3f} "
              f"{np.mean(np.abs(d['jerk'])):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Video
# ─────────────────────────────────────────────────────────────────────────────

STEP_STRIDE  = 3
FPS          = 20
CAR_W        = 4.0
CAR_L        = 8.0
ROAD_W       = 12.0
VIEW_W       = 120.0
DASH_SPACING = 20.0
N_DASHES     = 20

def build_video(d, title, out_path, car_color="#4fc3f7"):
    times  = d["times"]
    errors = d["dist_error"]
    e_spd  = d["ego_speed"]
    l_spd  = d["lead_speed"]
    e_pos  = d["ego_pos"]
    l_pos  = d["lead_pos"]
    dists  = d["distance"]

    frame_idx = list(range(0, len(times), STEP_STRIDE))

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
    ax_road.set_title("Top-down view  |  blue = ego (controlled)  |  red = lead car",
                      color="#8888bb", fontsize=9)
    for y in (-ROAD_W*0.75, ROAD_W*0.75):
        ax_road.axhline(y, color="#555588", linewidth=1.5, linestyle="--")

    dash_xs   = np.arange(N_DASHES) * DASH_SPACING
    dash_lines = []
    for dx in dash_xs:
        ln, = ax_road.plot([dx, dx+8], [0, 0], color="#444466", linewidth=1.2)
        dash_lines.append(ln)

    ego_patch = mpatches.FancyBboxPatch(
        (0, -CAR_W/2), CAR_L, CAR_W, boxstyle="round,pad=0.3",
        facecolor=car_color, edgecolor="#81d4fa", linewidth=1.5, zorder=3)
    lead_patch = mpatches.FancyBboxPatch(
        (0, -CAR_W/2), CAR_L, CAR_W, boxstyle="round,pad=0.3",
        facecolor="#ef5350", edgecolor="#ff8a80", linewidth=1.5, zorder=3)
    ax_road.add_patch(ego_patch)
    ax_road.add_patch(lead_patch)

    gap_line,  = ax_road.plot([], [], color="#ffd54f", linewidth=1.5, linestyle=":", zorder=2)
    gap_text   = ax_road.text(0, ROAD_W*0.45, "", color="#ffd54f", fontsize=8, ha="center")
    ego_label  = ax_road.text(0, -ROAD_W*0.55, "", color=car_color, fontsize=7.5, ha="center")
    lead_label = ax_road.text(0, -ROAD_W*0.55, "", color="#ef5350", fontsize=7.5, ha="center")
    time_text  = ax_road.text(0.01, 0.96, "", transform=ax_road.transAxes,
                               color="#ccccee", fontsize=9, va="top")

    ax_speed.set_xlim(times[0], times[-1])
    ax_speed.set_ylim(0, max(e_spd.max(), l_spd.max()) * 1.15 + 1)
    ax_speed.set_xlabel("Time (s)", color="#aaaacc", fontsize=8)
    ax_speed.set_ylabel("Speed (m/s)", color="#aaaacc", fontsize=8)
    ax_speed.set_title("Vehicle speeds", color="#8888bb", fontsize=9)
    ax_speed.plot(times, e_spd, color=car_color, alpha=0.15, linewidth=1)
    ax_speed.plot(times, l_spd, color="#ef5350", alpha=0.15, linewidth=1)
    spd_ego_ln,  = ax_speed.plot([], [], color=car_color,  linewidth=1.8, label="Ego")
    spd_lead_ln, = ax_speed.plot([], [], color="#ef5350", linewidth=1.8, label="Lead")
    ax_speed.legend(fontsize=7, facecolor="#16213e", labelcolor="#ccccee", edgecolor="#333366")

    ax_err.set_xlim(times[0], times[-1])
    lim = max(np.abs(errors).max() * 1.2, 2.0)
    ax_err.set_ylim(-lim, lim)
    ax_err.set_xlabel("Time (s)", color="#aaaacc", fontsize=8)
    ax_err.set_ylabel("Distance error (m)", color="#aaaacc", fontsize=8)
    ax_err.set_title("Following distance error", color="#8888bb", fontsize=9)
    ax_err.axhline(0, color="#ffd54f", linewidth=1, linestyle="--")
    ax_err.plot(times, errors, color="#a5d6a7", alpha=0.15, linewidth=1)
    err_ln, = ax_err.plot([], [], color="#a5d6a7", linewidth=1.8)

    def init():
        ego_patch.set_x(-CAR_L-10); lead_patch.set_x(-CAR_L-10)
        gap_line.set_data([], [])
        gap_text.set_text(""); ego_label.set_text(""); lead_label.set_text("")
        time_text.set_text("")
        spd_ego_ln.set_data([], []); spd_lead_ln.set_data([], [])
        err_ln.set_data([], [])
        return (ego_patch, lead_patch, gap_line, gap_text,
                ego_label, lead_label, time_text,
                spd_ego_ln, spd_lead_ln, err_ln, *dash_lines)

    def update(fi):
        i  = frame_idx[fi]
        ex = e_pos[i]; lx = l_pos[i]; t = times[i]
        centre = (ex + lx) / 2.0
        ax_road.set_xlim(centre - VIEW_W/2, centre + VIEW_W/2)
        x_min = centre - VIEW_W/2
        ego_patch.set_x(ex);  ego_patch.set_y(-CAR_W/2)
        lead_patch.set_x(lx); lead_patch.set_y(-CAR_W/2)
        for k, ln in enumerate(dash_lines):
            wrapped = x_min + ((dash_xs[k] - x_min) % (DASH_SPACING * N_DASHES))
            ln.set_xdata([wrapped, wrapped+8])
        gap_mid = (ex + CAR_L + lx) / 2.0
        gap_line.set_data([ex+CAR_L, lx], [CAR_W*0.6, CAR_W*0.6])
        gap_text.set_position((gap_mid, ROAD_W*0.48))
        gap_text.set_text(f"{dists[i]:.1f} m  (err {errors[i]:+.1f})")
        ego_label.set_position((ex+CAR_L/2, -ROAD_W*0.55))
        ego_label.set_text(f"{e_spd[i]:.1f} m/s")
        lead_label.set_position((lx+CAR_L/2, -ROAD_W*0.55))
        lead_label.set_text(f"{l_spd[i]:.1f} m/s")
        time_text.set_text(f"t = {t:.1f} s")
        spd_ego_ln.set_data(times[:i+1],  e_spd[:i+1])
        spd_lead_ln.set_data(times[:i+1], l_spd[:i+1])
        err_ln.set_data(times[:i+1],      errors[:i+1])
        return (ego_patch, lead_patch, gap_line, gap_text,
                ego_label, lead_label, time_text,
                spd_ego_ln, spd_lead_ln, err_ln, *dash_lines)

    ani = animation.FuncAnimation(
        fig, update, frames=len(frame_idx), init_func=init,
        interval=1000/FPS, blit=True
    )
    writer = animation.FFMpegWriter(fps=FPS, bitrate=1800,
                                    extra_args=["-vcodec", "libx264"])
    ani.save(out_path, writer=writer)
    plt.close(fig)
    print(f"  Saved video: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Simulating PID on hard scenario ...")
    pid_data = sim_pid()

    print("Simulating RL (run4/seed4) on hard scenario ...")
    rl_data = sim_rl()

    all_data = {"PID": pid_data, "RL (PPO)": rl_data}

    print("\nGenerating comparison plots ...")
    make_plots(all_data)

    print("\nRendering PID video ...")
    build_video(pid_data, "PID Adaptive Cruise Control — Hard Scenario",
                os.path.join(REPORT, "demo_PID.mp4"), car_color="#e67e22")

    print("Rendering RL video ...")
    build_video(rl_data, "RL (PPO) Adaptive Cruise Control — Hard Scenario",
                os.path.join(REPORT, "demo_RL.mp4"), car_color="#4fc3f7")

    print("\nDone. All outputs in report/")
