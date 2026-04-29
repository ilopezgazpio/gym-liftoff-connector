import numpy as np


class CrashDetector:
    def __init__(self, vel_min=1e-2, input_min=1e-2, pos_min=0.01, crash_threshold_counter=3, alpha=1.0, beta=1.0,
                 gamma=0.1, phys_threshold=0.5):

        self.vel_min = vel_min
        self.input_min = input_min
        self.pos_min = pos_min
        self.crash_threshold_counter = crash_threshold_counter

        # parámetros de detector físico
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phys_threshold = phys_threshold

        # estado interno
        self.last_timestamp = 0.0
        self.last_pos = None
        self.last_vel = None
        self.last_gyro = None
        self.crash_counter = 0
        self.drone_reset = False

    def reset(self):
        self.last_pos = None
        self.last_vel = None
        self.last_gyro = None
        self.last_timestamp = 0.0
        self.crash_counter = 0
        self.drone_reset = False

    def detect_collision(self, prev_vel, prev_gyro, vel, gyro, inp):
        """Detector físico basado en cambios de velocidad y gyro ponderado por input"""
        delta_vel = np.linalg.norm(vel - prev_vel)
        delta_gyro = np.linalg.norm(gyro - prev_gyro)
        input_mag = np.linalg.norm(inp)
        score = self.alpha * delta_vel + self.beta * delta_gyro - self.gamma * input_mag
        return score > self.phys_threshold, score

    def is_crashed(self, info):
        pos = info["position"]
        timestamp = info["timestamp"]
        vel = info["velocity"]
        gyro = info["gyro"]
        inp = info["input"]

        if self.last_timestamp and timestamp < self.last_timestamp - 0.01:
            self.drone_reset = True

        if self.last_vel is not None and self.last_gyro is not None:
            collision, score = self.detect_collision(self.last_vel, self.last_gyro, vel, gyro, inp)
            if collision:
                self.crash_counter += 1
            else:
                self.crash_counter = 0

            if self.crash_counter >= self.crash_threshold_counter:
                self.drone_reset = True

        self.last_pos = pos
        self.last_vel = vel
        self.last_gyro = gyro
        self.last_timestamp = timestamp

        return self.drone_reset