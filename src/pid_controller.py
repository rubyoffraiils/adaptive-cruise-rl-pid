import numpy as np


""" 
PID Controller
    input: distance error (difference between desired vs actual distance between bumpers)
    output: acceleration/braking command for ego car
"""
class PIDController:
    def __init__(self, kp, ki, kd, min_accel=-5.0, max_accel=3.0):
        # proportional gain (how strong the car reacts to current error)
        self.kp = kp
        #integral gain (how strong the car reacts to accumulated error over time)
        self.ki = ki
        # derivative gain (how strong the car reacts to changes in error)
        self.kd = kd

        self.min_accel = min_accel
        self.max_accel = max_accel

        self.integral = 0.0
        self.previous_error = 0.0

    # resets the controller memory of past errors and integral accumulation, for
    # starting new simulations/episodes
    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0

    """ 
    Compute the acceleration/braking command w PID
        input: distance error (difference between desired vs actual distance between bumpers)
        , dt (time step since last update)
    """
    def compute_acceleration(self, distance_error, dt):
        self.integral += distance_error * dt
        derivative = (distance_error - self.previous_error) / dt

        acceleration = (
            self.kp * distance_error
            + self.ki * self.integral
            + self.kd * derivative
        )

        self.previous_error = distance_error

        #we clip the acceleration to stay in a specified range
        return np.clip(acceleration, self.min_accel, self.max_accel)
