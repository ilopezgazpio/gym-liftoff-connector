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

    def is_crashed(self, info):
        vel = np.array(info["velocity"])
        gyro = np.array(info["gyro"])
        R = np.array(info["rotation"])
        rpm = np.array(info["motorrpm"])
        t = info["timestamp"]
        #print(vel)

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
        # RESIDUALES
        # =========================
        delta_a = a_exy - a_imu_xy
        #delta_w = omega_e - omega_imu

        # =========================
        # FILTRO
        # =========================
        self.filtered_acc = (1 - beta) * self.filtered_acc + beta * delta_a

        acc_norm = np.linalg.norm(self.filtered_acc)
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

        if len(self.acc_residuals) < self.W:
            return False
        else:
            k_acc = np.mean(np.diff(np.array(self.acc_residuals)) ** 2)
            H_acc = k_acc * np.linalg.norm(a_exy) + self.c_acc
            H_yaw = 27.0
            print("Metricas")
            print(acc_norm, yaw_abs)
            print(H_acc, H_yaw)


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