import numpy as np
import matplotlib.pyplot as plt

from pid_controller import PIDController
from acc_simulator import AdaptiveCruiseSimulator


def compute_metrics(results):

    # Distance error over time.
    distance_errors = results["distance_errors"]

    # Actual distance between ego and lead car over time.
    distances = results["distances"]

    # Ego car acceleration over time.
    accelerations = results["ego_accelerations"]

    # Simulation timestep.
    dt = results["dt"]

    # METRICS
    
    # METRIC 1: MEAN ABSOLUTE ERROR
    mean_abs_error = np.mean(np.abs(distance_errors))

    # METRIC 2: MINIMUM DISTANCE
    min_distance = np.min(distances)

    # METRIC 3: JERK
    jerk = np.diff(accelerations) / dt

    # Mean absolute jerk summarizes how jerky the ride was.
    mean_abs_jerk = np.mean(np.abs(jerk)) if len(jerk) > 0 else 0.0

    # METRIC 4: CONTROL EFFORT
    control_effort = np.sum(np.abs(accelerations)) * dt

    # METRIC 5: COLLISION 
    collision_count = np.sum(results["collisions"])

    return {
        "mean_abs_error": mean_abs_error,
        "min_distance": min_distance,
        "mean_abs_jerk": mean_abs_jerk,
        "control_effort": control_effort,
        "collision_count": collision_count,
    }

def plot_results(results, title):
    """
    Plot the simulation results.

    These plots help you understand what the PID controller is doing.

    The most important plot is following distance over time.
    """

    times = results["times"]
    desired_distance = results["desired_distance"]

    # PLOT 1: FOLLOWING DISTANCE
    plt.figure()
    plt.plot(
        times,
        results["distances"],
        label="Actual distance"
    )

    # Draw a horizontal dashed line for desired distance.
    plt.axhline(
        y=desired_distance,
        linestyle="--",
        label="Desired distance"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Distance between cars (m)")
    plt.title(title + " - Following Distance")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # PLOT 2: SPEEDS
    plt.figure()
    plt.plot(
        times,
        results["ego_speeds"],
        label="Ego speed"
    )
    plt.plot(
        times,
        results["lead_speeds"],
        label="Lead speed"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title(title + " - Vehicle Speeds")
    plt.legend()
    plt.grid(True)
    plt.show()


    # PLOT 3: EGO ACCELERATION
    plt.figure()
    plt.plot(
        times,
        results["ego_accelerations"]
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title(title + " - Ego Acceleration")
    plt.grid(True)
    plt.show()


    # PLOT 4: DISTANCE ERROR
    plt.figure()
    plt.plot(
        times,
        results["distance_errors"]
    )

    # Error = 0 means perfect following distance.
    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Distance error (m)")
    plt.title(title + " - Distance Error")
    plt.grid(True)
    plt.show()


def main():
    """
        1. create one PID controller
        2. create the simulator
        3. run the simulation
        4. compute metrics
        5. plot results
    """
    # PID CONTROLLER
    controller = PIDController(
        kp=4.0000,
        ki=0.00001,
        kd=7.9995
    )

    # SIMULATOR
    simulator = AdaptiveCruiseSimulator(
        desired_distance=20.0,
        dt=0.05,
        total_time=30.0
    )


    # RUN SIMULATION
    results = simulator.run(controller)

    # COMPUTE METRICS
    metrics = compute_metrics(results)

    print("PID controller finished running.")
    print(f"Kp = {controller.kp}, Ki = {controller.ki}, Kd = {controller.kd}")
    print()

    print("Performance metrics:")
    print(f"Mean absolute distance error: {metrics['mean_abs_error']:.2f} m")
    print(f"Minimum distance: {metrics['min_distance']:.2f} m")
    print(f"Mean absolute jerk: {metrics['mean_abs_jerk']:.2f} m/s³")
    print(f"Control effort: {metrics['control_effort']:.2f}")
    print(f"Collision count: {metrics['collision_count']}")


    # PLOT RESULTS
    plot_results(
        results,
        title="PID Adaptive Cruise Control"
    )

if __name__ == "__main__":
    main()

