import numpy as np
import optuna

from pid_controller import PIDController
from acc_simulator import AdaptiveCruiseSimulator, generate_random_profile
from run_pid import compute_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Tune on 20 fixed random profiles — diverse enough to generalise, fixed seeds
# so every Optuna trial sees exactly the same scenarios (fair comparison).
# Hard scenario is held out entirely as the evaluation set.
_N_TUNE_PROFILES = 20
_RANDOM_PROFILES = [
    generate_random_profile(total_time=30.0, rng=np.random.default_rng(seed=i))
    for i in range(_N_TUNE_PROFILES)
]
_BASE_SIM = AdaptiveCruiseSimulator(
    desired_distance=20.0, dt=0.05, total_time=30.0, scenario="randomized"
)


def score_controller(metrics, desired_distance, distances):
    """
    Lower is better. Weighted sum of tracking, safety, comfort, and control effort.

    Safety terms are intentionally massive (100x, 1000x) relative to tracking —
    getting close to a collision should dominate every other consideration.
    """
    score = 2.0 * metrics["mean_abs_error"]
    score += 0.5 * metrics["mean_abs_jerk"]
    score += 0.05 * metrics["control_effort"]

    # penalise undershooting the desired distance (getting closer than desired = unsafe)
    overshoot = np.maximum(0, desired_distance - distances)
    score += 3.0 * float(np.mean(overshoot))

    min_dist = metrics["min_distance"]
    if min_dist < 15:
        score += 100.0 * (15 - min_dist)
    if min_dist < 10:
        score += 1000.0 * (10 - min_dist)

    score += 10000.0 * metrics["collision_count"]
    return score


def evaluate(kp, ki, kd):
    """Average score across all tuning profiles."""
    total = 0.0
    for profile in _RANDOM_PROFILES:
        controller = PIDController(kp=kp, ki=ki, kd=kd)
        results = _BASE_SIM.run(controller, profile=profile)
        metrics = compute_metrics(results)
        total += score_controller(metrics, _BASE_SIM.desired_distance, results["distances"])
    return total / len(_RANDOM_PROFILES)


def main():
    n_trials = 600
    print(f"Bayesian optimisation — {n_trials} trials, {_N_TUNE_PROFILES} random profiles ...")

    # TPE (Tree-structured Parzen Estimator) — builds a probabilistic model of which
    # regions of (Kp, Ki, Kd) space tend to score well, then samples from there.
    # Much more efficient than random/grid search for 3 continuous parameters.
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial):
        kp = trial.suggest_float("kp", 0.1, 4.0)
        ki = trial.suggest_float("ki", 0.0, 0.02)
        kd = trial.suggest_float("kd", 0.5, 8.0)
        return evaluate(kp, ki, kd)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    kp = best.params["kp"]
    ki = best.params["ki"]
    kd = best.params["kd"]

    print(f"\nBest gains:  Kp={kp:.4f}  Ki={ki:.5f}  Kd={kd:.4f}")
    print(f"Avg score:   {best.value:.4f}")

    print("\nTop 10 trials:")
    for i, t in enumerate(sorted(study.trials, key=lambda t: t.value)[:10]):
        print(f"  {i+1:2d}. Kp={t.params['kp']:.4f} Ki={t.params['ki']:.5f} "
              f"Kd={t.params['kd']:.4f}  score={t.value:.4f}")


if __name__ == "__main__":
    main()
