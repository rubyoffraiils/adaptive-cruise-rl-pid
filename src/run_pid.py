import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pid_controller import PIDController
from acc_simulator import AdaptiveCruiseSimulator


def compute_metrics(results):
    distance_errors = results["distance_errors"]
    distances = results["distances"]
    accelerations = results["ego_accelerations"]
    dt = results["dt"]

    # jerk = rate of change of acceleration. High jerk = uncomfortable/jerky ride.
    jerk = np.diff(accelerations) / dt

    return {
        "mean_abs_error":  float(np.mean(np.abs(distance_errors))),
        "min_distance":    float(np.min(distances)),
        "mean_abs_jerk":   float(np.mean(np.abs(jerk))) if len(jerk) > 0 else 0.0,
        # control effort: total acceleration applied over the episode (energy proxy)
        "control_effort":  float(np.sum(np.abs(accelerations)) * dt),
        "collision_count": int(np.sum(results["collisions"])),
    }


def plot_results(results, title, out_dir=None):
    times = results["times"]
    desired = results["desired_distance"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)

    axes[0, 0].plot(times, results["distances"], label="Actual")
    axes[0, 0].axhline(desired, linestyle="--", label=f"Desired ({desired} m)")
    axes[0, 0].set(xlabel="Time (s)", ylabel="Distance (m)", title="Following Distance")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(times, results["ego_speeds"], label="Ego")
    axes[0, 1].plot(times, results["lead_speeds"], label="Lead")
    axes[0, 1].set(xlabel="Time (s)", ylabel="Speed (m/s)", title="Vehicle Speeds")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(times, results["ego_accelerations"])
    axes[1, 0].axhline(0, linestyle="--", alpha=0.4)
    axes[1, 0].set(xlabel="Time (s)", ylabel="Acceleration (m/s²)", title="Ego Acceleration")
    axes[1, 0].grid(True)

    axes[1, 1].plot(times, results["distance_errors"])
    axes[1, 1].axhline(0, linestyle="--", alpha=0.4)
    axes[1, 1].set(xlabel="Time (s)", ylabel="Error (m)", title="Distance Error")
    axes[1, 1].grid(True)

    fig.tight_layout()

    if out_dir:
        import os
        fig.savefig(os.path.join(out_dir, "results.png"), dpi=150)
    plt.close(fig)


def main():
    controller = PIDController(kp=4.0, ki=0.00002, kd=7.9995)
    simulator = AdaptiveCruiseSimulator(desired_distance=20.0, dt=0.05, total_time=30.0)
    results = simulator.run(controller)
    metrics = compute_metrics(results)

    print(f"Kp={controller.kp}  Ki={controller.ki}  Kd={controller.kd}")
    print(f"MAE:            {metrics['mean_abs_error']:.2f} m")
    print(f"Min distance:   {metrics['min_distance']:.2f} m")
    print(f"Mean abs jerk:  {metrics['mean_abs_jerk']:.2f} m/s³")
    print(f"Control effort: {metrics['control_effort']:.2f}")
    print(f"Collisions:     {metrics['collision_count']}")

    plot_results(results, title="PID Adaptive Cruise Control")


if __name__ == "__main__":
    main()
