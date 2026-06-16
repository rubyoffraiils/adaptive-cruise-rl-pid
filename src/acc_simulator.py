import numpy as np
from pid_controller import PIDController


def lead_acceleration_at_time(t, scenario="default"):
    """
    Defines how the lead car moves.

    scenario="default"  — original gentle test
    scenario="hard"     — emergency braking, stop-and-go, sudden acceleration
    """

    if scenario == "hard":
        if t < 3:
            return 0.0       # cruise
        elif t < 6:
            return -4.0      # emergency brake
        elif t < 10:
            return 0.0       # stopped / very slow
        elif t < 13:
            return 2.0       # accelerate hard
        elif t < 18:
            return 0.0       # cruise
        elif t < 21:
            return -3.0      # hard brake again
        elif t < 25:
            return 1.5       # recover
        else:
            return -1.0      # gentle slowdown

    # default scenario
    if t < 5:
        return 0.0
    elif t < 10:
        return -1.0
    elif t < 15:
        return 1.0
    elif t < 22:
        return 0.0
    else:
        return -2.0


class AdaptiveCruiseSimulator:
    """
        This class represents the adaptive cruise control simulator in 1D (cars only move forward along a straight line). 
    """

    def __init__(
            self,
            desired_distance=20.0,
            dt=0.05,
            total_time=30.0,
            min_speed=0.0,
            max_speed=35.0,
            scenario="default",
    ):
        self.desired_distance = desired_distance
        self.dt = dt
        self.total_time = total_time
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.scenario = scenario


    def run(self, controller):
        """
            Runs a simulation using controller
            returns: a dictionary of recorded simulation data
        """
       
       # reset controller memory before new run
        controller.reset()
        steps = int(self.total_time / self.dt)

        # initialize car states

        ego_position = 0.0
        ego_speed = 20.0
        lead_position = 30.0
        lead_speed = 20.0

        # data storage
        times = []
        ego_positions = []
        lead_positions = []

        distances = []
        distance_errors = []

        ego_speeds = []
        lead_speeds = []

        ego_accelerations = []
        lead_accelerations = []

        collisions = []

        # Main Simulation Loop
       # The loop structure is:
        #   1. measure distance
        #   2. compute error
        #   3. PID chooses acceleration
        #   4. update lead car physics
        #   5. update ego car physics
        #   6. save data

        for step in range(steps):
            t = step * self.dt

            # measure distance
            actual_distance = lead_position - ego_position

            #distance error
            distance_error = actual_distance - self.desired_distance

            #pid controll`er output
            ego_acceleration = controller.compute_acceleration(distance_error, self.dt)

            #lead car action
            lead_acceleration = lead_acceleration_at_time(t, self.scenario)

            # update lead car physics
            lead_speed += lead_acceleration * self.dt
            lead_speed = np.clip(lead_speed, self.min_speed, self.max_speed)
            lead_position += lead_speed * self.dt

            # update ego car physics
            ego_speed += ego_acceleration * self.dt
            ego_speed = np.clip(ego_speed, self.min_speed, self.max_speed)
            ego_position += ego_speed * self.dt

            # collision check (happens when distance is 0 or negative)
            collision = actual_distance <= 0

            # save data
            times.append(t)
            ego_positions.append(ego_position)
            lead_positions.append(lead_position)
            distances.append(actual_distance)
            distance_errors.append(distance_error)
            ego_speeds.append(ego_speed)
            lead_speeds.append(lead_speed)
            ego_accelerations.append(ego_acceleration)
            lead_accelerations.append(lead_acceleration)
            collisions.append(collision)

            if collision:
                print(f"Collision occurred at time {t:.2f} seconds!")
                break

        # return all recorded data
        return {
            "times": np.array(times),

            "ego_positions": np.array(ego_positions),
            "lead_positions": np.array(lead_positions),

            "distances": np.array(distances),
            "distance_errors": np.array(distance_errors),

            "ego_speeds": np.array(ego_speeds),
            "lead_speeds": np.array(lead_speeds),

            "ego_accelerations": np.array(ego_accelerations),
            "lead_accelerations": np.array(lead_accelerations),

            "collisions": np.array(collisions),

            "desired_distance": self.desired_distance,
            "dt": self.dt,
        }
    
