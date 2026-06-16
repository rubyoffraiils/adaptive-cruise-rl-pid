import optuna

from pid_controller import PIDController
from acc_simulator import AdaptiveCruiseSimulator
from run_pid import compute_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)

SIMULATORS = [
    AdaptiveCruiseSimulator(desired_distance=20.0, dt=0.05, total_time=30.0, scenario="default"),
    AdaptiveCruiseSimulator(desired_distance=20.0, dt=0.05, total_time=30.0, scenario="hard"),
]


def score_controller(metrics, desired_distance):
    """
    Lower score is better.

    Combines safety, tracking, comfort, and control effort.
    Also penalises overshoot (getting closer than desired).
    """
    mean_error = metrics["mean_abs_error"]
    min_distance = metrics["min_distance"]
    mean_jerk = metrics["mean_abs_jerk"]
    control_effort = metrics["control_effort"]
    collision_count = metrics["collision_count"]
    distances = metrics.get("distances")

    score = 0.0

    # Tracking: stay near desired distance.
    score += 2.0 * mean_error

    # Comfort: avoid jerky acceleration.
    score += 0.5 * mean_jerk

    # Efficiency: avoid too much acceleration/braking.
    score += 0.05 * control_effort

    # Overshoot: penalise getting closer than desired (unsafe).
    if distances is not None:
        import numpy as np
        overshoot = np.maximum(0, desired_distance - distances)
        score += 3.0 * float(np.mean(overshoot))

    # Safety: heavily punish getting dangerously close.
    if min_distance < 15:
        score += 100.0 * (15 - min_distance)
    if min_distance < 10:
        score += 1000.0 * (10 - min_distance)

    # Collision is basically unacceptable.
    score += 10000.0 * collision_count

    return score


def evaluate(kp, ki, kd):
    """Run all scenarios and return the average score."""
    total = 0.0
    for sim in SIMULATORS:
        controller = PIDController(kp=kp, ki=ki, kd=kd)
        results = sim.run(controller)
        metrics = compute_metrics(results)
        metrics["distances"] = results["distances"]
        total += score_controller(metrics, sim.desired_distance)
    return total / len(SIMULATORS)


def main():
    def objective(trial):
        kp = trial.suggest_float("kp", 0.1, 4.0)
        ki = trial.suggest_float("ki", 0.0, 0.02)
        kd = trial.suggest_float("kd", 0.5, 8.0)
        return evaluate(kp, ki, kd)

    n_trials = 600
    print(f"Running Bayesian optimization ({n_trials} trials, 2 scenarios)...")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    kp = best.params["kp"]
    ki = best.params["ki"]
    kd = best.params["kd"]

    print(f"\nBest PID gains found:")
    print(f"  Kp = {kp:.4f}")
    print(f"  Ki = {ki:.5f}")
    print(f"  Kd = {kd:.4f}")
    print(f"  Avg score across scenarios: {best.value:.4f}")

    for sim in SIMULATORS:
        controller = PIDController(kp=kp, ki=ki, kd=kd)
        results = sim.run(controller)
        metrics = compute_metrics(results)
        metrics["distances"] = results["distances"]
        s = score_controller(metrics, sim.desired_distance)
        print(f"\n  [{sim.scenario}] score={s:.4f}")
        print(f"    Mean abs error: {metrics['mean_abs_error']:.2f} m")
        print(f"    Min distance:   {metrics['min_distance']:.2f} m")
        print(f"    Mean abs jerk:  {metrics['mean_abs_jerk']:.2f} m/s³")
        print(f"    Control effort: {metrics['control_effort']:.2f}")
        print(f"    Collisions:     {metrics['collision_count']}")

    print(f"\nTop 10 trials:")
    top10 = sorted(study.trials, key=lambda t: t.value)[:10]
    for i, t in enumerate(top10):
        print(
            f"  {i+1:2d}. Kp={t.params['kp']:.4f}, Ki={t.params['ki']:.5f}, Kd={t.params['kd']:.4f} | "
            f"score={t.value:.4f}"
        )


if __name__ == "__main__":
    main()
