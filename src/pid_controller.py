class PIDController:
    def __init__(self, kp, ki, kd, min_accel=-5.0, max_accel=3.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_accel = min_accel
        self.max_accel = max_accel
        self.integral = 0.0
        self.previous_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0

    def compute_action(self, distance_error, dt):
        self.integral += distance_error * dt
        derivative = (distance_error - self.previous_error) / dt

        acceleration = (
            self.kp * distance_error
            + self.ki * self.integral
            + self.kd * derivative
        )

        self.previous_error = distance_error

        return np.clip(acceleration, self.min_accel, self.max_accel)
