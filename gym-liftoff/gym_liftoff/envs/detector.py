import socket
import struct
import numpy as np

class CrashDetector:
    def __init__(self, vel_min = 1e-3, input_min = 0.1, pos_min = 0.01, crash_threshold_counter = 10):
        self.vel_min = vel_min
        self.input_min = input_min
        self.pos_min = pos_min
        self.crash_threshold_counter = crash_threshold_counter
        #self.last_velocity = None
        self.last_timestamp = 0.0
        self.last_pos = None
        self.crash_counter = 0
        self.drone_reset = False

    def reset(self):
        self.last_input = None
        #self.last_velocity = None
        self.last_timestamp = 0.0
        self.crash_counter = 0
        self.drone_reset = False

    def is_crashed(self, info):
        pos = info["position"]
        timestamp = info["timestamp"]
        vel = info["velocity"]
        inp = info["input"]

        if self.last_pos is not None and timestamp < self.last_timestamp - 0.01:
            self.drone_reset = True

        if self.last_pos is not None:
            movement = np.linalg.norm(pos - self.last_pos)
            speed = np.linalg.norm(vel)
            input_active = np.linalg.norm(inp) > self.input_min

            #if (movement < self.pos_min or speed < self.vel_min) and input_active:
            if speed < self.vel_min and input_active:
                self.crash_counter += 1
            else:
                self.crash_counter = 0

            if self.crash_counter >= self.crash_threshold_counter:
                self.drone_reset = True

        self.last_pos = pos
        self.last_timestamp = timestamp

        return self.drone_reset
