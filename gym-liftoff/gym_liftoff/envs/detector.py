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
from collections import deque

class CrashDetector:
    def __init__(
        self,
        kt= 1.5e-6,
        kd = 1e-6,
        mass=0.675,
        inertia_z=0.005,
        window_size=7,
        tau=0.15,
        c_acc=0,
        c_yaw=0,
    ):
        self.kt = kt
        self.kd = kd
        self.m = mass
        self.Iz = inertia_z

        self.W = window_size
        self.tau = tau

        self.c_acc = c_acc
        self.c_yaw = c_yaw

        # buffers
        self.acc_residuals = deque(maxlen=self.W)
        self.yaw_residuals = deque(maxlen=self.W)

        # estado previo
        self.prev_velocity = None
        self.prev_timestamp = None
        self.prev_omega_e = 0.0

        # filtro
        self.filtered_acc = np.zeros(2)
        self.filtered_yaw = 0.0

        self.crash = False

    def reset(self):
        self.acc_residuals.clear()
        self.yaw_residuals.clear()
        self.prev_velocity = None
        self.prev_timestamp = None
        self.prev_omega_e = 0.0
        self.filtered_acc = np.zeros(2)
        self.filtered_yaw = 0.0
        self.crash = False
        self.prev_acc = None

    def is_crashed(self, info):
        vel = np.array(info["velocity"])
        gyro = np.array(info["gyro"])
        R = np.array(info["rotation"])
        t = info["timestamp"]

        # =========================
        # dt
        # =========================
        if self.prev_timestamp is None:
            self.prev_timestamp = t
            self.prev_velocity = vel
            return False

        dt = t - self.prev_timestamp
        if dt <= 0:
            return False

        beta = dt / (self.tau + dt)

        # =========================
        # ACELERACION IMU (numérica)
        # =========================

        acc_imu = (vel - self.prev_velocity) / dt



        # =========================
        # FILTRO
        # =========================

        acc_norm = np.linalg.norm(acc_imu)
        omega_imu = gyro[2]

        if hasattr(self, "prev_omega"):
            delta_w = omega_imu - self.prev_omega
        else:
            delta_w = 0.0

        self.prev_omega = omega_imu

        yaw_abs = abs(delta_w)

        # =========================
        # BUFFER
        # =========================
        self.acc_residuals.append(acc_norm)




        # =========================
        # THRESHOLDS
        # =========================


        H_acc = 50.0
        H_yaw = 35.0
        """
        print("Metricas")
        print(acc_norm, yaw_abs)
        print(H_acc, H_yaw)
        """


        # =========================
        # DETECCIÓN
        # =========================

        crash_acc = acc_norm > H_acc
        crash_yaw = yaw_abs > H_yaw

        self.crash = crash_acc or crash_yaw

        # =========================
        # UPDATE
        # =========================
        self.prev_velocity = vel
        self.prev_timestamp = t

        return self.crash

