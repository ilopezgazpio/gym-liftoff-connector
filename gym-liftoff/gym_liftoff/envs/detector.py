import numpy as np

class CrashDetector:
    def __init__(self, vel_min=10, input_min=1e-1, crash_threshold_counter=10, attitude_threshold=-0.65):
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
        elif speed < 0.2:
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

