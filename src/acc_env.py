import gymnasium as gym
from gymnasium import spaces
import numpy as np

from acc_simulator import lead_acceleration_at_time, generate_random_profile

class AdaptiveCruiseControlEnv(gym.Env):
    """
    Gym environmetn for 1D ACC 
    (ego car follows lead car, RL agent controls ego acceleration)
    """

    def __init__(self, scenario="default"):
        super().__init__()

        self.scenario = scenario

        #simulation settings
        self.dt = 0.1
        self.max_time = 30.0

        #desired following distance
        self.desired_distance = 20.0

        #actuator limits (ego accel is clipped btwn braking + acceleration limits)
        self.min_accel = -5.0
        self.max_accel = 3.0

        self.current_profile = None

        # action: (ego acceleration)
        self.action_space = spaces.Box(
            low=np.array([self.min_accel], dtype=np.float32),
            high=np.array([self.max_accel], dtype=np.float32),
            dtype=np.float32
        )

        # observation:
        # [distance_error, relative_speed, ego_speed, lead_speed]
        self.observation_space = spaces.Box(
            low=np.array([-100.0, -50.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([100.0, 50.0, 50.0, 50.0], dtype=np.float32),
            dtype=np.float32
        )

    def _get_obs(self):
        distance = self.x_lead - self.x_ego
        distance_error = distance - self.desired_distance
        relative_speed = self.v_lead - self.v_ego

        return np.array(
            [distance_error, relative_speed, self.v_ego, self.v_lead],
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0.0
        #ego starts behind lead car
        self.x_ego = 0.0
        self.v_ego = 20.0
        #leadcar starts 30m ahead
        self.x_lead = 30.0
        self.v_lead = 20.0

        self.prev_accel = 0.0

        # generate a fresh random profile each episode so the agent cannot
        # memorise the timing; for fixed scenarios this stays None
        if self.scenario == "randomized":
            rng = np.random.default_rng(seed)
            self.current_profile = generate_random_profile(
                total_time=self.max_time, rng=rng
            )
        else:
            self.current_profile = None

        info = {}
        obs = self._get_obs()

        return obs, info
    
    def step(self, action):
        #RL chooses acceleration
        accel = float(action[0])
        accel = np.clip(accel, self.min_accel, self.max_accel)

        #lead car follows same scripted scenario as PID
        lead_accel = lead_acceleration_at_time(
            self.t, scenario=self.scenario, profile=self.current_profile
        )

        #update velocities
        self.v_ego += accel * self.dt
        self.v_lead += lead_accel * self.dt

        # prevent cars from reversing
        self.v_ego = max(self.v_ego, 0.0)
        self.v_lead = max(self.v_lead, 0.0)

        # update positions
        self.x_ego += self.v_ego * self.dt
        self.x_lead += self.v_lead * self.dt

        # update time
        self.t += self.dt

        # calculate state values
        distance = self.x_lead - self.x_ego
        distance_error = distance - self.desired_distance
        relative_speed = self.v_lead - self.v_ego

        jerk = (accel - self.prev_accel) / self.dt
        self.prev_accel = accel

        collision = bool(distance <= 0.0)


        # REWARD FUNCTION

        reward = 0.0

        # tracking: stay close to desired following distance
        reward -= abs(distance_error)

        #safety: being too close is worse than being too far
        if distance < self.desired_distance:
            reward -= 2.0 * abs(distance_error)

        
        #comfort: avoid sudden acceleration changes (jerk)
        reward -= 0.05 * abs(jerk)

        # efficiency: avoid using huge acceleration/braking all the time
        reward -= 0.02 * abs(accel ** 2)

        # collision is super duper bad
        if collision:
            reward -= 1000.0

        terminated = bool(collision)
        truncated = bool(self.t >= self.max_time)

        obs = self._get_obs()

        info = {
            "time": self.t,
            "distance": distance,
            "distance_error": distance_error,
            "relative_speed": relative_speed,
            "ego_speed": self.v_ego,
            "lead_speed": self.v_lead,
            "ego_accel": accel,
            "lead_accel": lead_accel,
            "jerk": jerk,
            "collision": collision,
        }

        return obs, float(reward), terminated, truncated, info

