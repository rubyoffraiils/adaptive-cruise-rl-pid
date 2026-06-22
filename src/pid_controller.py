import numpy as np


class PIDController:
    """
    PID controller for ACC.
    Input:  distance error (actual gap - desired gap)
    Output: ego car acceleration/braking command

    The three terms work together:
      P — reacts to the current error (how far off we are right now)
      I — reacts to accumulated error over time (corrects persistent drift)
      D — reacts to how fast the error is changing (damps oscillation)

    For ACC, Kd ends up being most important bc the derivative term is what
    lets the controller anticipate a closing gap and brake early instead of
    waiting until it's already too close.
    """

    def __init__(self, kp, ki, kd, min_accel=-5.0, max_accel=3.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_accel = min_accel
        self.max_accel = max_accel
        self.integral = 0.0
        self.previous_error = 0.0

    def reset(self):
        # clear memory between episodes so past errors don't bleed into new runs
        self.integral = 0.0
        self.previous_error = 0.0

    def compute_acceleration(self, distance_error, dt):
        self.integral += distance_error * dt
        derivative = (distance_error - self.previous_error) / dt

        acceleration = (
            self.kp * distance_error
            + self.ki * self.integral
            + self.kd * derivative
        )

        self.previous_error = distance_error
        return np.clip(acceleration, self.min_accel, self.max_accel)
