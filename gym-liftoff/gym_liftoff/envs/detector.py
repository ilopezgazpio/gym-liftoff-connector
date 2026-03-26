import numpy as np

class CrashDetector:
    def __init__(self, vel_min=1e-2, crash_threshold_counter=10, up_threshold=0.5):
        """
        vel_min: velocidad mínima para considerar que se ha detenido
        crash_threshold_counter: número de iteraciones consecutivas para considerar crash
        up_threshold: umbral para detectar dron invertido o de lado
        """
        self.vel_min = vel_min
        self.crash_threshold_counter = crash_threshold_counter
        self.up_threshold = up_threshold
        self.last_pos = None
        self.last_timestamp = 0.0
        self.crash_counter = 0
        self.drone_reset = False

    def reset(self):
        self.last_pos = None
        self.last_timestamp = 0.0
        self.crash_counter = 0
        self.drone_reset = False

    def is_crashed(self, info):
        pos = info["position"]
        timestamp = info["timestamp"]
        vel = info["velocity"]
        qx, qy, qz, qw = info["attitude"]

        speed = np.linalg.norm(vel)
        info["speed"] = speed

        # Primero revisamos si la telemetría es consistente
        if self.last_pos is not None and timestamp < self.last_timestamp - 0.01:
            self.drone_reset = True

        # Solo chequeamos crash si ya tenemos posición previa
        if self.last_pos is not None:
            # Rotamos el vector up del dron al mundo
            up_world = self.quaternion_rotate_vector(qx, qy, qz, qw, np.array([0, 1, 0]))

            print(up_world)

            # Si velocidad baja y el dron no está recto, contamos como crash
            if speed < self.vel_min and abs(up_world[1]) < self.up_threshold:
                self.crash_counter += 1
            else:
                self.crash_counter = 0

            if self.crash_counter >= self.crash_threshold_counter:
                self.drone_reset = True

        self.last_pos = pos
        self.last_timestamp = timestamp
        return self.drone_reset

    @staticmethod
    def quaternion_rotate_vector(qx, qy, qz, qw, v):
        """
        Rota el vector v usando el cuaternión q = [qx,qy,qz,qw]
        """
        q_vec = np.array([qx, qy, qz])
        uv = np.cross(q_vec, v)
        uuv = np.cross(q_vec, uv)
        return v + 2 * (qw * uv + uuv)