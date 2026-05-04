import numpy as np
from collections import deque


class CrashDetector:
    def __init__(
        self,
        kt,
        kd,
        mass=1.0,
        inertia_z=1.0,
        window_size=20,
        tau=0.05,
        c_acc=0.5,
        c_yaw=0.2,
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

    def step(self, info):
        vel = np.array(info["velocity"])
        gyro = np.array(info["angular_velocity"])
        R = np.array(info["rotation"])
        rpm = np.array(info["motorRPM"])
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
        acc_imu = 0.7 * acc_imu + 0.3 * getattr(self, "prev_acc", acc_imu)
        self.prev_acc = acc_imu

        # =========================
        # RPM → rad/s
        # =========================
        omega = rpm * (2 * np.pi / 60)

        # =========================
        # THRUST
        # =========================
        S = np.sum(omega**2)
        T = self.kt * S

        # =========================
        # ACELERACION ESPERADA
        # =========================
        g = np.array([0, -9.81, 0])  # Unity

        thrust_body = np.array([0, T, 0])
        thrust_world = R @ thrust_body

        a_expected = thrust_world / self.m + g

        # plano horizontal (Unity: X,Z)
        a_exy = np.array([a_expected[0], a_expected[2]])
        a_imu_xy = np.array([acc_imu[0], acc_imu[2]])

        # =========================
        # YAW ESPERADO
        # =========================
        yaw_term = (omega[0]**2 + omega[2]**2) - (omega[1]**2 + omega[3]**2)
        tau_z = self.kd * yaw_term

        omega_e = self.prev_omega_e + (tau_z / self.Iz) * dt
        self.prev_omega_e = omega_e

        # Unity: yaw = eje Y
        omega_imu = gyro[1]

        # =========================
        # RESIDUALES
        # =========================
        delta_a = a_exy - a_imu_xy
        delta_w = omega_e - omega_imu

        # =========================
        # FILTRO
        # =========================
        self.filtered_acc = (1 - beta) * self.filtered_acc + beta * delta_a
        self.filtered_yaw = (1 - beta) * self.filtered_yaw + beta * delta_w

        acc_norm = np.linalg.norm(self.filtered_acc)
        yaw_abs = abs(self.filtered_yaw)

        # =========================
        # BUFFER
        # =========================
        self.acc_residuals.append(acc_norm)
        self.yaw_residuals.append(yaw_abs)

        k_acc = np.mean(np.array(self.acc_residuals)**2)
        k_yaw = np.mean(np.array(self.yaw_residuals)**2)

        # =========================
        # THRESHOLDS
        # =========================

        if len(self.acc_residuals) < self.W:
            H_acc = 2.0
            H_yaw = 1.0
        else:
            H_acc = k_acc * np.linalg.norm(a_exy) + self.c_acc
            H_yaw = k_yaw * abs(omega_e) + self.c_yaw

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