import numpy as np

class CrashDetectorDeprecated:
    def __init__(self, vel_min=10, input_min=1e-1, crash_threshold_counter=3, attitude_threshold=-0.65):
        self.vel_min = vel_min
        self.input_min = input_min
        self.crash_threshold_counter = crash_threshold_counter
        self.attitude_threshold = attitude_threshold  # coseno del ángulo con up
        self.last_pos = None
        self.last_timestamp = 0.0
        self.crash_counter = 0
        self.drone_reset = False
        self.crash_reason = None

    def reset(self):
        self.last_pos = None
        self.last_timestamp = 0.0
        self.crash_counter = 0
        self.drone_reset = False
        self.crash_reason = None

    def is_crashed(self, info):
        pos = info["position"]
        timestamp = info["timestamp"]
        vel = info["velocity"]
        speed = np.linalg.norm(vel)

        if self.last_pos is None:
            self.last_pos = pos
            self.last_timestamp = timestamp
            return False

        if speed < self.vel_min:

            up_y = info["rotation"][1, 1]
            if up_y < self.attitude_threshold:
                self.crash_counter += 1
            else:
                self.crash_counter = 0
        elif speed < 2:
            self.crash_counter += 1
        else:
            self.crash_counter = 0

        if self.last_timestamp - 0.01 > timestamp:
            self.drone_reset = True
            self.crash_reason = 't'

        if self.crash_counter >= self.crash_threshold_counter:
            self.drone_reset = True
            self.crash_reason = 's'

        self.last_pos = pos
        self.last_timestamp = timestamp

        return self.drone_reset

import numpy as np


class CrashDetector:
    def __init__(
        self,
        vel_min=0.1,
        gyro_min=1.0,
        input_min=0.2,
        crash_threshold_counter=3,
        alpha=1.0,
        beta=0.5,
        gamma=0.2,
        phys_threshold=1.0,
        attitude_threshold=0.2,
    ):
        self.vel_min = vel_min
        self.gyro_min = gyro_min
        self.input_min = input_min
        self.crash_threshold_counter = crash_threshold_counter

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phys_threshold = phys_threshold

        self.attitude_threshold = attitude_threshold

        self.last_timestamp = None
        self.last_pos = None
        self.last_vel = None
        self.last_gyro = None

        self.crash_counter = 0
        self.drone_reset = False

    def reset(self):
        self.last_timestamp = None
        self.last_pos = None
        self.last_vel = None
        self.last_gyro = None
        self.crash_counter = 0
        self.drone_reset = False

    # =========================
    # UTILIDADES
    # =========================

    def quaternion_up_y(self, q):
        """
        Devuelve componente Y del vector 'up' del dron.
        """
        qx, qy, qz, qw = q
        return 1 - 2 * (qx**2 + qz**2)

    def detect_physical_collision(self, vel, prev_vel, gyro, prev_gyro, inp):
        delta_vel = np.linalg.norm(vel - prev_vel)
        delta_gyro = np.linalg.norm(gyro - prev_gyro)
        input_mag = np.linalg.norm(inp)

        score = (
            self.alpha * delta_vel
            + self.beta * delta_gyro
            - self.gamma * input_mag
        )

        return score > self.phys_threshold, score

    def detect_stuck(self, vel, gyro, inp):
        speed = np.linalg.norm(vel)
        gyro_norm = np.linalg.norm(gyro)
        input_mag = np.linalg.norm(inp)

        return (
            speed < self.vel_min
            and gyro_norm < self.gyro_min
            and input_mag > self.input_min
        )

    def detect_flip(self, attitude):
        up_y = self.quaternion_up_y(attitude)
        return up_y < self.attitude_threshold

    # =========================
    # MAIN
    # =========================

    def is_crashed(self, info):
        pos = info["position"]
        vel = info["velocity"]
        gyro = info["gyro"]
        attitude = info["attitude"]
        inp = info["input"]
        timestamp = info["timestamp"]

        # =========================
        # RESET DETECTION
        # =========================
        if self.last_timestamp is not None:
            if timestamp < self.last_timestamp - 0.01:
                self.drone_reset = True

        # =========================
        # PRIMER STEP
        # =========================
        if self.last_vel is None:
            self.last_pos = pos
            self.last_vel = vel
            self.last_gyro = gyro
            self.last_timestamp = timestamp
            return False

        collision, score = self.detect_physical_collision(
            vel, self.last_vel, gyro, self.last_gyro, inp
        )

        flipped = self.detect_flip(attitude)
        stuck = self.detect_stuck(vel, gyro, inp)

        crash = collision or flipped or stuck

        if crash:
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

