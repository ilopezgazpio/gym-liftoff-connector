import numpy as np

class CrashDetector:
    def __init__(self,
                 vel_min=1e-2,
                 input_min=1e-1,
                 pos_min=0.01,
                 crash_threshold_counter=10,
                 upside_dot_threshold=-0.1):
        """
        vel_min: velocidad mínima para considerar que el dron no se mueve
        input_min: mínimo input activo
        crash_threshold_counter: nº de pasos consecutivos para marcar crash
        upside_dot_threshold: si el eje "arriba" del dron (transformado)
                              tiene componente vertical < threshold → invertido
        """
        self.vel_min = vel_min
        self.input_min = input_min
        self.pos_min = pos_min
        self.crash_threshold_counter = crash_threshold_counter
        self.upside_dot_threshold = upside_dot_threshold

        self.last_timestamp = 0.0
        self.last_pos = None
        self.crash_counter = 0
        self.drone_reset = False

    def reset(self):
        self.last_timestamp = 0.0
        self.last_pos = None
        self.crash_counter = 0
        self.drone_reset = False

    def is_crashed(self, info):
        pos = info["position"]
        timestamp = info["timestamp"]
        vel = info["velocity"]
        inp = info["input"]
        gyro = info["gyro"]
        att = info.get("attitude")  # cuaternion (qx,qy,qz,qw)

        if self.last_pos is not None and timestamp < self.last_timestamp - 0.01:
            self.drone_reset = True
        """
                if self.last_pos is not None:
            speed = np.linalg.norm(vel)
            input_active = np.linalg.norm(inp) > self.input_min
            gy = np.linalg.norm(gyro)
            movement = np.linalg.norm(pos - self.last_pos)

            info["speed"] = speed
            info["input_active"] = input_active
            info["movement"] = movement

            # sin movimiento y sin giro, pero con input activo
            if (speed < self.vel_min and gy < 0.3) and input_active:
                self.crash_counter += 1
            else:
                self.crash_counter = 0

            if self.crash_counter >= self.crash_threshold_counter:
                self.drone_reset = True
        """

        if att is not None:
            # att = [qx, qy, qz, qw]
            qx, qy, qz, qw = att


            gz_world = qw*qw - qx*qx - qy*qy + qz*qz
            info["up_dot_z"] = gz_world

            if gz_world < self.upside_dot_threshold and np.linalg.norm(vel) < self.vel_min:
                self.drone_reset = True

        self.last_pos = pos
        self.last_timestamp = timestamp

        return self.drone_reset