"""
Generate a demo video of a naive (pre-tuning) PID controller on the hard scenario.

Gains are intentionally bad guesses — the kind you'd try before any systematic tuning:
  Kp=1.0, Ki=0.0, Kd=1.0
This produces sluggish tracking and crashes on the emergency brake event.

Output → report/demo_PID_naive.mp4
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

# Naive gains — P-only, no derivative damping.
# A natural first guess before understanding why Kd matters.
# Oscillates around desired distance and crashes on hard braking.
NAIVE_KP = 2.0
NAIVE_KI = 0.0
NAIVE_KD = 0.0

STEP_STRIDE  = 3
FPS          = 20
CAR_W        = 4.0
CAR_L        = 8.0
ROAD_W       = 12.0
VIEW_W       = 120.0
DASH_SPACING = 20.0
N_DASHES     = 20


def sim_naive_pid():
    ctrl = PIDController(kp=NAIVE_KP, ki=NAIVE_KI, kd=NAIVE_KD)
    sim  = AdaptiveCruiseSimulator(
        desired_distance=20.0, dt=0.05, total_time=30.0, scenario="hard"
    )
    r = sim.run(ctrl)
    times  = r["times"]
    accels = r["ego_accelerations"]
    return {
        "times":      times,
        "ego_pos":    r["ego_positions"],
        "lead_pos":   r["lead_positions"],
        "ego_speed":  r["ego_speeds"],
        "lead_speed": r["lead_speeds"],
        "distance":   r["distances"],
        "dist_error": r["distance_errors"],
        "ego_accel":  accels,
        "jerk":       np.gradient(accels, times),
    }


def build_video(d, title, out_path):
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
    ax_road.set_title("Top-down view  |  blue = ego  |  red = lead car",
                      color="#8888bb", fontsize=9)
    for y in (-ROAD_W*0.75, ROAD_W*0.75):
        ax_road.axhline(y, color="#555588", linewidth=1.5, linestyle="--")

    dash_xs    = np.arange(N_DASHES) * DASH_SPACING
    dash_lines = []
    for dx in dash_xs:
        ln, = ax_road.plot([dx, dx+8], [0, 0], color="#444466", linewidth=1.2)
        dash_lines.append(ln)

    ego_patch = mpatches.FancyBboxPatch(
        (0, -CAR_W/2), CAR_L, CAR_W, boxstyle="round,pad=0.3",
        facecolor="#4fc3f7", edgecolor="#81d4fa", linewidth=1.5, zorder=3)
    lead_patch = mpatches.FancyBboxPatch(
        (0, -CAR_W/2), CAR_L, CAR_W, boxstyle="round,pad=0.3",
        facecolor="#ef5350", edgecolor="#ff8a80", linewidth=1.5, zorder=3)
    ax_road.add_patch(ego_patch)
    ax_road.add_patch(lead_patch)

    gap_line,  = ax_road.plot([], [], color="#ffd54f", linewidth=1.5, linestyle=":", zorder=2)
    gap_text   = ax_road.text(0, ROAD_W*0.45, "", color="#ffd54f", fontsize=8, ha="center")
    ego_label  = ax_road.text(0, -ROAD_W*0.55, "", color="#4fc3f7", fontsize=7.5, ha="center")
    lead_label = ax_road.text(0, -ROAD_W*0.55, "", color="#ef5350", fontsize=7.5, ha="center")
    time_text  = ax_road.text(0.01, 0.96, "", transform=ax_road.transAxes,
                               color="#ccccee", fontsize=9, va="top")
    crash_text = ax_road.text(0.5, 0.5, "", transform=ax_road.transAxes,
                               color="#ff4444", fontsize=18, fontweight="bold",
                               ha="center", va="center")

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

    ax_err.set_xlim(times[0], times[-1])
    lim = max(np.abs(errors).max() * 1.2, 2.0)
    ax_err.set_ylim(-lim, lim)
    ax_err.set_xlabel("Time (s)", color="#aaaacc", fontsize=8)
    ax_err.set_ylabel("Distance error (m)", color="#aaaacc", fontsize=8)
    ax_err.set_title("Following distance error", color="#8888bb", fontsize=9)
    ax_err.axhline(0, color="#ffd54f", linewidth=1, linestyle="--")
    ax_err.plot(times, errors, color="#ff6b6b", alpha=0.15, linewidth=1)
    err_ln, = ax_err.plot([], [], color="#ff6b6b", linewidth=1.8)

    crashed = False

    def init():
        ego_patch.set_x(-CAR_L-10); lead_patch.set_x(-CAR_L-10)
        gap_line.set_data([], [])
        gap_text.set_text(""); ego_label.set_text(""); lead_label.set_text("")
        time_text.set_text(""); crash_text.set_text("")
        spd_ego_ln.set_data([], []); spd_lead_ln.set_data([], [])
        err_ln.set_data([], [])
        return (ego_patch, lead_patch, gap_line, gap_text,
                ego_label, lead_label, time_text, crash_text,
                spd_ego_ln, spd_lead_ln, err_ln, *dash_lines)

    def update(fi):
        nonlocal crashed
        i  = frame_idx[fi]
        ex = e_pos[i]; lx = l_pos[i]; t = times[i]
        centre = (ex + lx) / 2.0
        x_min  = centre - VIEW_W/2
        ax_road.set_xlim(x_min, centre + VIEW_W/2)
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

        if dists[i] <= 0 and not crashed:
            crashed = True
            crash_text.set_text("COLLISION")
            ego_patch.set_facecolor("#ff4444")

        spd_ego_ln.set_data(times[:i+1],  e_spd[:i+1])
        spd_lead_ln.set_data(times[:i+1], l_spd[:i+1])
        err_ln.set_data(times[:i+1],      errors[:i+1])
        return (ego_patch, lead_patch, gap_line, gap_text,
                ego_label, lead_label, time_text, crash_text,
                spd_ego_ln, spd_lead_ln, err_ln, *dash_lines)

    ani = animation.FuncAnimation(
        fig, update, frames=len(frame_idx), init_func=init,
        interval=1000/FPS, blit=True
    )
    writer = animation.FFMpegWriter(fps=FPS, bitrate=1800,
                                    extra_args=["-vcodec", "libx264"])
    ani.save(out_path, writer=writer)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    print(f"Simulating naive PID (Kp={NAIVE_KP}, Ki={NAIVE_KI}, Kd={NAIVE_KD}) on hard scenario ...")
    d = sim_naive_pid()

    collision = any(d["distance"] <= 0)
    print(f"  MAE:       {np.mean(np.abs(d['dist_error'])):.3f} m")
    print(f"  Min dist:  {np.min(d['distance']):.3f} m")
    print(f"  Collision: {collision}")
    print(f"  Steps:     {len(d['times'])}")

    out = os.path.join(ROOT, "pid_runs", "run_naive", "demo.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"\nRendering video ...")
    build_video(
        d,
        f"PID Before Tuning — P-only (Kp={NAIVE_KP}, Ki={NAIVE_KI}, Kd={NAIVE_KD}) — Hard Scenario",
        out,
    )
