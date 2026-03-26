import numpy as np

class CrashDetector:
    def __init__(self, vel_min=1e-2, input_min=1e-1, crash_threshold_counter=10, attitude_threshold=0.7):
        self.vel_min = vel_min
        self.input_min = input_min
        self.crash_threshold_counter = crash_threshold_counter
        self.attitude_threshold = attitude_threshold  # coseno del ángulo con up
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
        inp = info["input"]
        attitude = info["attitude"]  # quaternion x,y,z,w

        speed = np.linalg.norm(vel)
        input_active = np.linalg.norm(inp) > self.input_min
        info["speed"] = speed
        info["input_active"] = input_active

        # ignorar la primera iteración
        if self.last_pos is None:
            self.last_pos = pos
            self.last_timestamp = timestamp
            return False

        # si velocidad baja, comprobar orientación
        if speed < self.vel_min:
            # extraer vector up del cuaternión
            qx, qy, qz, qw = attitude
            up_y = 1 - 2*(qx**2 + qz**2)  # coseno del ángulo con el eje Y global
            # up_y ~1 → recto, ~0 → de lado, ~-1 → boca abajo
            if up_y < self.attitude_threshold:
                self.crash_counter += 1
            else:
                self.crash_counter = 0
        else:
            self.crash_counter = 0

        if self.crash_counter >= self.crash_threshold_counter:
            self.drone_reset = True

        self.last_pos = pos
        self.last_timestamp = timestamp

        return self.drone_reset