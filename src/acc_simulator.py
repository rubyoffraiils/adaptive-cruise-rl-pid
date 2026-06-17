import numpy as np
from pid_controller import PIDController


# Fixed scenario profiles: list of (phase_end_time, acceleration) pairs.
# The last entry's end_time is ignored — it applies until episode end.
SCENARIO_PROFILES = {
    "default": [
        (5.0,   0.0),   # cruise
        (10.0, -1.0),   # gentle brake
        (15.0,  1.0),   # accelerate
        (22.0,  0.0),   # cruise
        (30.0, -2.0),   # moderate brake
    ],
    "hard": [
        (3.0,   0.0),   # cruise
        (6.0,  -4.0),   # emergency brake
        (10.0,  0.0),   # stopped / very slow
        (13.0,  2.0),   # hard acceleration
        (18.0,  0.0),   # cruise
        (21.0, -3.0),   # hard brake again
        (25.0,  1.5),   # recovery
        (30.0, -1.0),   # gentle slowdown
    ],
}


def generate_random_profile(total_time=30.0, rng=None):
    """
    Sample a random lead-car acceleration profile for one episode.

    The profile is a list of (phase_end_time, acceleration) pairs that
    lead_acceleration_at_time() can walk through.  Each call to this
    function produces a different sequence of phases, so the RL agent
    cannot memorise the timing of any particular manoeuvre.

    Design choices:
    - 5–8 phases of random duration (2–7 s each)
    - Acceleration drawn from {-4, -3, -2, -1, 0, 1, 2} m/s²
      with higher weight on 0 (cruising) so the car isn't always braking
    - No two consecutive identical accelerations (forces actual transitions)
    """
    if rng is None:
        rng = np.random.default_rng()

    accel_choices = [-4.0, -3.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0]
    n_phases = rng.integers(5, 9)   # 5–8 phases

    profile = []
    t = 0.0
    prev_a = None

    for i in range(n_phases):
        # last phase runs to total_time
        if i == n_phases - 1:
            phase_end = total_time
        else:
            remaining = total_time - t
            max_dur = min(7.0, remaining - 2.0 * (n_phases - i - 1))
            duration = rng.uniform(2.0, max(2.1, max_dur))
            t += duration
            phase_end = t

        # pick acceleration, avoiding repeating the previous phase
        a = prev_a
        while a == prev_a:
            a = float(rng.choice(accel_choices))
        prev_a = a

        profile.append((phase_end, a))

    return profile


def lead_acceleration_at_time(t, scenario="default", profile=None):
    """
    Return the lead car's acceleration at time t.

    - scenario="default" or "hard": uses the fixed SCENARIO_PROFILES lookup
    - scenario="randomized": requires a pre-built `profile` list generated
      by generate_random_profile(); raises if profile is None
    """
    if scenario == "randomized":
        if profile is None:
            raise ValueError("scenario='randomized' requires a profile argument")
        for (end_time, accel) in profile:
            if t < end_time:
                return accel
        return profile[-1][1]

    # fixed scenarios
    for (end_time, accel) in SCENARIO_PROFILES[scenario]:
        if t < end_time:
            return accel
    return SCENARIO_PROFILES[scenario][-1][1]


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


    def run(self, controller, profile=None):
        """
        Runs a simulation using controller.
        For scenario='randomized', pass a pre-built profile from generate_random_profile().
        Returns: a dictionary of recorded simulation data.
        """
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
            lead_acceleration = lead_acceleration_at_time(t, self.scenario, profile=profile)

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
    
