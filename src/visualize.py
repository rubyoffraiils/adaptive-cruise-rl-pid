"""
Produce demo_pid.mp4 and demo_rl.mp4 — identical layout, both controllers.

Layout (per frame):
  Top    : top-down road view with scrolling dashes, two car rectangles, gap label
  Bottom-left  : live speed plot (ego vs lead)
  Bottom-right : live distance-error plot
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

# ── make sure src/ is importable when called from the repo root ──────────────
sys.path.insert(0, os.path.dirname(__file__))

from pid_controller import PIDController
from acc_simulator import AdaptiveCruiseSimulator
from acc_env import AdaptiveCruiseControlEnv


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Collect simulation data
# ─────────────────────────────────────────────────────────────────────────────

def collect_pid_data(scenario="default"):
    controller = PIDController(kp=4.0, ki=0.00001, kd=7.9995)
    sim = AdaptiveCruiseSimulator(
        desired_distance=20.0, dt=0.05, total_time=30.0, scenario=scenario
    )
    results = sim.run(controller)

    records = []
    for i in range(len(results["times"])):
        records.append({
            "time":           results["times"][i],
            "ego_pos":        results["ego_positions"][i],
            "lead_pos":       results["lead_positions"][i],
            "ego_speed":      results["ego_speeds"][i],
            "lead_speed":     results["lead_speeds"][i],
            "distance":       results["distances"][i],
            "distance_error": results["distance_errors"][i],
            "ego_accel":      results["ego_accelerations"][i],
        })
    return records


def collect_rl_data(model_path="ppo_acc_default", scenario="default"):
    from stable_baselines3 import PPO

    # look for the zip next to this file or in the repo root
    candidates = [
        model_path,
        os.path.join(os.path.dirname(__file__), "..", model_path),
        os.path.join(os.path.dirname(__file__), "..", model_path + ".zip"),
    ]
    found = None
    for c in candidates:
        if os.path.exists(c) or os.path.exists(c + ".zip"):
            found = c
            break
    if found is None:
        raise FileNotFoundError(
            f"Could not find model '{model_path}'. "
            "Run src/train_rl.py first to produce ppo_acc_default.zip."
        )

    env = AdaptiveCruiseControlEnv(scenario=scenario)
    model = PPO.load(found)

    obs, _ = env.reset()
    done = False
    records = []

    # mirror the starting positions used by the PID simulator
    ego_pos = 0.0
    lead_pos = 30.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # reconstruct positions by integrating speeds (env doesn't expose them)
        ego_pos  += info["ego_speed"]  * env.dt
        lead_pos += info["lead_speed"] * env.dt

        records.append({
            "time":           info["time"],
            "ego_pos":        ego_pos,
            "lead_pos":       lead_pos,
            "ego_speed":      info["ego_speed"],
            "lead_speed":     info["lead_speed"],
            "distance":       info["distance"],
            "distance_error": info["distance_error"],
            "ego_accel":      info["ego_accel"],
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Animation builder
# ─────────────────────────────────────────────────────────────────────────────

# How many simulation steps per animation frame (tune for smoothness vs speed)
STEP_STRIDE = 2          # every 2nd step → ~25 fps for dt=0.05, 10 fps for dt=0.1
FPS         = 20

# Road / car geometry (metres → pixels are handled by the axes transform)
CAR_W  = 4.0   # car width  (m)
CAR_L  = 8.0   # car length (m)
ROAD_W = 12.0  # road half-width for drawing

def build_animation(records, title, out_path, run_label=None, analysis_lines=None):
    times  = np.array([r["time"]           for r in records])
    errors = np.array([r["distance_error"] for r in records])
    e_spd  = np.array([r["ego_speed"]      for r in records])
    l_spd  = np.array([r["lead_speed"]     for r in records])
    dists  = np.array([r["distance"]       for r in records])

    # subsample frames
    frame_idx = list(range(0, len(records), STEP_STRIDE))
    n_frames  = len(frame_idx)

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 7), facecolor="#1a1a2e")
    gs  = GridSpec(2, 2, figure=fig,
                   height_ratios=[1.6, 1],
                   hspace=0.38, wspace=0.32,
                   left=0.07, right=0.97, top=0.93, bottom=0.08)

    ax_road  = fig.add_subplot(gs[0, :])   # full-width road view
    ax_speed = fig.add_subplot(gs[1, 0])   # speed plot
    ax_err   = fig.add_subplot(gs[1, 1])   # distance error plot

    style = dict(facecolor="#1a1a2e", edgecolor="none")
    for ax in (ax_road, ax_speed, ax_err):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaaacc", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333366")

    fig.suptitle(title, color="#e0e0ff", fontsize=13, fontweight="bold")

    # ── road panel ──────────────────────────────────────────────────────────
    ax_road.set_ylim(-ROAD_W, ROAD_W)
    ax_road.set_aspect("equal")
    ax_road.set_yticks([])
    ax_road.set_xlabel("Road position (m)", color="#aaaacc", fontsize=8)
    ax_road.set_title("Top-down view", color="#8888bb", fontsize=9)

    # static road edges
    for y in (-ROAD_W * 0.75, ROAD_W * 0.75):
        ax_road.axhline(y, color="#555588", linewidth=1.5, linestyle="--")

    # dashed centre line (will scroll)
    DASH_SPACING = 20.0
    n_dashes = 20
    dash_xs   = np.arange(n_dashes) * DASH_SPACING
    dash_lines = []
    for dx in dash_xs:
        ln, = ax_road.plot([dx, dx + 8], [0, 0],
                           color="#444466", linewidth=1.2, linestyle="-")
        dash_lines.append(ln)

    # car patches
    ego_patch  = mpatches.FancyBboxPatch(
        (0, -CAR_W / 2), CAR_L, CAR_W,
        boxstyle="round,pad=0.3",
        facecolor="#4fc3f7", edgecolor="#81d4fa", linewidth=1.5, zorder=3)
    lead_patch = mpatches.FancyBboxPatch(
        (0, -CAR_W / 2), CAR_L, CAR_W,
        boxstyle="round,pad=0.3",
        facecolor="#ef5350", edgecolor="#ff8a80", linewidth=1.5, zorder=3)
    ax_road.add_patch(ego_patch)
    ax_road.add_patch(lead_patch)

    # gap annotation
    gap_line, = ax_road.plot([], [], color="#ffd54f", linewidth=1.5,
                             linestyle=":", zorder=2)
    gap_text  = ax_road.text(0, ROAD_W * 0.45, "", color="#ffd54f",
                              fontsize=8, ha="center", va="bottom")

    # speed / label overlays on cars
    ego_label  = ax_road.text(0, -ROAD_W * 0.55, "", color="#4fc3f7",
                               fontsize=7.5, ha="center")
    lead_label = ax_road.text(0, -ROAD_W * 0.55, "", color="#ef5350",
                               fontsize=7.5, ha="center")

    time_text = ax_road.text(0.01, 0.96, "", transform=ax_road.transAxes,
                              color="#ccccee", fontsize=9, va="top")

    # ── speed panel ──────────────────────────────────────────────────────────
    ax_speed.set_xlim(times[0], times[-1])
    ax_speed.set_ylim(0, max(e_spd.max(), l_spd.max()) * 1.15 + 1)
    ax_speed.set_xlabel("Time (s)", color="#aaaacc", fontsize=8)
    ax_speed.set_ylabel("Speed (m/s)", color="#aaaacc", fontsize=8)
    ax_speed.set_title("Vehicle speeds", color="#8888bb", fontsize=9)
    ax_speed.axhline(0, color="#333366", linewidth=0.5)

    spd_ego_bg,  = ax_speed.plot(times, e_spd, color="#4fc3f7",
                                  alpha=0.15, linewidth=1)
    spd_lead_bg, = ax_speed.plot(times, l_spd, color="#ef5350",
                                  alpha=0.15, linewidth=1)
    spd_ego_ln,  = ax_speed.plot([], [], color="#4fc3f7",
                                  linewidth=1.8, label="Ego")
    spd_lead_ln, = ax_speed.plot([], [], color="#ef5350",
                                  linewidth=1.8, label="Lead")
    ax_speed.legend(fontsize=7, facecolor="#16213e",
                    labelcolor="#ccccee", edgecolor="#333366")

    # ── error panel ──────────────────────────────────────────────────────────
    err_abs = np.abs(errors)
    ax_err.set_xlim(times[0], times[-1])
    lim = max(err_abs.max() * 1.2, 2.0)
    ax_err.set_ylim(-lim, lim)
    ax_err.set_xlabel("Time (s)", color="#aaaacc", fontsize=8)
    ax_err.set_ylabel("Distance error (m)", color="#aaaacc", fontsize=8)
    ax_err.set_title("Following distance error", color="#8888bb", fontsize=9)
    ax_err.axhline(0, color="#ffd54f", linewidth=1, linestyle="--")

    err_bg, = ax_err.plot(times, errors, color="#a5d6a7", alpha=0.15, linewidth=1)
    err_ln, = ax_err.plot([], [], color="#a5d6a7", linewidth=1.8)

    # ── view window width for the road (metres shown at once) ────────────────
    VIEW_W = 120.0   # metres of road visible

    def init():
        ego_patch.set_x(-CAR_L - 10)
        ego_patch.set_y(-CAR_W / 2)
        lead_patch.set_x(-CAR_L - 10)
        lead_patch.set_y(-CAR_W / 2)
        gap_line.set_data([], [])
        gap_text.set_text("")
        ego_label.set_text("")
        lead_label.set_text("")
        time_text.set_text("")
        spd_ego_ln.set_data([], [])
        spd_lead_ln.set_data([], [])
        err_ln.set_data([], [])
        return (ego_patch, lead_patch, gap_line, gap_text,
                ego_label, lead_label, time_text,
                spd_ego_ln, spd_lead_ln, err_ln, *dash_lines)

    def update(fi):
        i   = frame_idx[fi]
        rec = records[i]

        ego_x  = rec["ego_pos"]
        lead_x = rec["lead_pos"]
        t      = rec["time"]

        # scroll the road view centred between the two cars
        centre = (ego_x + lead_x) / 2.0
        x_min  = centre - VIEW_W / 2
        x_max  = centre + VIEW_W / 2
        ax_road.set_xlim(x_min, x_max)

        # place car patches (rear-left corner)
        ego_patch.set_x(ego_x)
        ego_patch.set_y(-CAR_W / 2)
        lead_patch.set_x(lead_x)
        lead_patch.set_y(-CAR_W / 2)

        # scroll centre dashes
        for k, ln in enumerate(dash_lines):
            raw_x = dash_xs[k]
            # wrap dash to stay near the view
            offset = ((raw_x - x_min) % (DASH_SPACING * n_dashes))
            wrapped = x_min + offset
            ln.set_xdata([wrapped, wrapped + 8])

        # gap indicator
        ego_front  = ego_x + CAR_L
        lead_rear  = lead_x
        gap_mid    = (ego_front + lead_rear) / 2.0
        gap_line.set_data([ego_front, lead_rear], [CAR_W * 0.6, CAR_W * 0.6])
        gap_text.set_position((gap_mid, ROAD_W * 0.48))
        gap_text.set_text(f"{rec['distance']:.1f} m  (err {rec['distance_error']:+.1f})")

        # car speed labels
        ego_label.set_position((ego_x + CAR_L / 2, -ROAD_W * 0.55))
        ego_label.set_text(f"{rec['ego_speed']:.1f} m/s")
        lead_label.set_position((lead_x + CAR_L / 2, -ROAD_W * 0.55))
        lead_label.set_text(f"{rec['lead_speed']:.1f} m/s")

        time_text.set_text(f"t = {t:.1f} s")

        # live plots (up to current frame)
        spd_ego_ln.set_data(times[:i+1],  e_spd[:i+1])
        spd_lead_ln.set_data(times[:i+1], l_spd[:i+1])
        err_ln.set_data(times[:i+1],      errors[:i+1])

        return (ego_patch, lead_patch, gap_line, gap_text,
                ego_label, lead_label, time_text,
                spd_ego_ln, spd_lead_ln, err_ln, *dash_lines)

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=1000 / FPS, blit=True
    )

    writer = animation.FFMpegWriter(fps=FPS, bitrate=1800,
                                    extra_args=["-vcodec", "libx264"])
    ani.save(out_path, writer=writer)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..")

    print("Collecting PID data …")
    pid_records = collect_pid_data(scenario="default")
    pid_out = os.path.join(out_dir, "demo_pid.mp4")
    print(f"Rendering PID animation ({len(pid_records)} steps) …")
    build_animation(pid_records, "PID Adaptive Cruise Control", pid_out)

    print("Collecting RL data …")
    try:
        rl_records = collect_rl_data(model_path="ppo_acc_default", scenario="default")
        rl_out = os.path.join(out_dir, "demo_rl.mp4")
        print(f"Rendering RL animation ({len(rl_records)} steps) …")
        build_animation(rl_records, "RL (PPO) Adaptive Cruise Control", rl_out)
    except FileNotFoundError as e:
        print(f"Skipping RL video: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
