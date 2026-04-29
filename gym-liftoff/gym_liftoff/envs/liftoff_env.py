import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import numpy as np
from gym_liftoff.main import VirtualGamepad, VideoSampler
import tkinter as tk
import time
import torch
import pyautogui
from gym_liftoff.envs.telemetry import init_udp_socket
from gym_liftoff.envs.detector import CrashDetector
import socket
import struct
import json

import logging
logger = logging.getLogger(__name__)

class Liftoff(gym.Env):

    metadata = {
        'render_modes': ['human']
    }

    def __get_curr_screen_geometry__(self):

        """
        Workaround to get the size of the main screen in a multi-screen setup.

        Returns:
            geometry (str): The standard Tk geometry string.
                [width]x[height]+[left]+[top]
        """
        root = tk.Tk()
        root.update_idletasks()
        root.attributes('-fullscreen', True)
        root.state('iconic')
        geometry = root.winfo_geometry()
        root.destroy()
        sc_h = int(geometry.split('x')[1].split('+')[0])
        sc_w = int(geometry.split('x')[0])
        return sc_h, sc_w


    def __init__(self, continuous_action_mode = False, max_episode_time = None):

        """
        Args:
            max_steps: array of 4 elements indicating the duration of the episode
                Positions:
                    0: days
                    1: hours
                    2: minutes
                    3: seconds

        """
        # TODO: Para medir el tiempo máximo que pueda estar el dron volando se necesita acceder al TimeStamp de los
        #       archivos de configuracion del juego
        """
        Liftoff Telemetry Socket
        """

        self.sock = init_udp_socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256)
        print("Actual Size of the socket: ", self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
        self.crash_detector = CrashDetector()

        self.telemetry_path = "~/.config/unity3d/LuGus Studios/Liftoff/TelemetryConfiguration.json"
        self.telemetry_len_path = "./delta_data/telemetry_len_data.json"
        self.telemetry_len, self.telemetry_points, self.telemetry_points_len = self.get_telemetry_len()

        logger.info("Initializing environment.....")

        self.virtual_gamepad = VirtualGamepad.VirtualGamepad()

        self.sc_w, self.sc_h = self.__get_curr_screen_geometry__()
        logger.info("Identified screen width..... {}".format(self.sc_w))
        logger.info("Identified screen height..... {}".format(self.sc_h))

        self.video_sampler = VideoSampler.VideoSampler(self.sc_w, self.sc_h)

        self.render_mode = 'human'
        self.current_observation = None
        self.current_action = None

        '''
        Observation space is defined as the screenshot converted to a numpy array
        '''
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(3, self.video_sampler.img_x, self.video_sampler.img_y),
                                            dtype=np.uint8)
        '''
        Action space is defined as 4 values between 0 and 1

        0: Throttle
        1: Yaw
        2: Roll
        3: Pitch
        '''
        self.action_space = spaces.Box(low=0, high=2047, shape=(4,), dtype=np.uint16)

        '''
        Environment state
        '''
        self._has_reset = False
        self.state = np.zeros((self.video_sampler.img_x, self.video_sampler.img_y), dtype=np.uint8)
        self.resetting = False
        self.consecutive_zero = 0

        '''
        Duration of each episode
        '''
        if max_episode_time is not None: self.max_episode_time = 86400*max_episode_time[0] + 3600*max_episode_time[1] + 60 * max_episode_time[2] + max_episode_time[3]
        else:self.max_episode_time = max_episode_time


        '''
        Passes the action space from a continuous space to a discretize action space [0, 2047] if activated. 
        '''
        self.continuous_action_mode = False

        self.still_counter = 0
        self.max_still = 5

        self.past_action = None
        self.penalty_threshold = 0.3

        print("Enviroment ready. Press ENTER here to start.")
        input()
        print("Starting in 5 seconds... Focus the Liftoff window now!")
        time.sleep(5)

    def _get_info(self):
        info = self.read_telemetry()
        attitude = info["attitude"]
        info["rotation"] = self.quaternion_to_rotation_matrix(attitude)
        info["gyro"] = info["gyro"] * (np.pi / 180.0)
        return info

    def observation(self):
        array = np.array(self.state)
        array = np.transpose(array, (2, 0, 1)).reshape((3, self.video_sampler.img_x, self.video_sampler.img_y))
        assert array.shape == self.observation_space.shape
        return array

    def _get_reward(self, terminated):
        if terminated:
            return float(-3)
        return 0

    def act(self, action, from_reset=False):
        self.resetting = False
        if self.resetting and not from_reset:
            return
        self.virtual_gamepad.act(action)

    def step(self, action):
        if not self._has_reset:
            raise gym.error.ResetNeeded("Cannot call env.step() before calling env.reset()")
    
        info = {}

        '''Send action to liftoff through virtual gamepad'''
        logger.info("Action performed: {}".format(action))
        self.act(action)

        ''' Sample liftoff state through video sampler'''
        # TODO: Seguramente que la línea de abajo esté mal pero por si acaso se pone comentada
        #self.state = self.video_sampler.sample(region=(1280, 0, 1920, 1080))
        self.state = self.video_sampler.sample()
        self.current_action = action
        observation = self.observation()
        info = self._get_info()
        terminated= self.__episode_terminated__(info)
        reward = self._get_reward(terminated)
        truncated = info["timestamp"] > self.max_episode_time

        if terminated or truncated:
            self._has_reset = False
        if terminated:
            info["crash_reason"] = self.crash_detector.crash_reason
        if truncated:
            info["crash_reason"] = None

        return observation, reward, terminated, truncated, info


    def reset_deprecated(self, seed=None, options=None):
        super().reset(seed=seed)  # sets Gymnasium RNG
        self._has_reset = True

        if hasattr(self, "resetting") and self.resetting:
            # already called from wrapper, skip duplication
            pass
        else:
            self.resetting = True
            self.virtual_gamepad.reset()
            pyautogui.press('r')
            time.sleep(1.5)
            self.virtual_gamepad.reset()
            self.act([900, 1024, 1024, 1024], from_reset=True)
            time.sleep(1)

        self.time = 0
        self.past_action = None
        self.state = self.video_sampler.sample(region=(1280, 0, 1920, 1080))
        observation = self.observation()
        done = self.__episode_terminated__(info)
        self.crash_detector.reset()
        logger.info("Reward obtained: {}".format(reward))

        return observation, reward, done, False, info

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self._has_reset = True

        if hasattr(self, "resetting") and self.resetting:
            # already called from wrapper, skip duplication
            pass
        else:
            self.resetting = True
            self.virtual_gamepad.reset()
            pyautogui.press('r')
            time.sleep(2)
            self.act([0, 1024, 1024, 1024], from_reset=True)

        self.time = 0
        self.past_action = None
        self.state = self.video_sampler.sample()
        observation = self.observation()
        self.crash_detector.reset()
        info = self._get_info()
        return observation, info



    def render(self, mode='human'):
        print("\n{}\n".format(self.state))


    def close(self):
        self.virtual_gamepad.close()
        self.video_sampler.close()
        return

    def _get_speed(self):
        number = self.video_sampler.get_speed()
        if number:
            self.consecutive_zero = 0
            return number
        self.consecutive_zero += 1
        return 0

    def __episode_terminated__(self, info):
        """Check if the episode is terminated"""
        # screen is black

        return self.crash_detector.is_crashed(info)

    def read_telemetry(self):
        latest = None
        _ = self.sock.recvfrom(2304)
        while True:
            try:
                data, _ = self.sock.recvfrom(128)  # leer todo lo disponible
                latest = data
                break
            except BlockingIOError:
                continue  # ya no hay más paquetes

        if latest is None:
            # no llegó nada en este ciclo
            return None

        if len(latest) < self.telemetry_len:
            # paquete demasiado corto
            return None

        telemetry = {}
        idx = 0
        for point in self.telemetry_points:
            point_data = self.telemetry_points_len[point]
            bytes = point_data["bytes"]
            if point == "MotorRPM":
                idx +=1
                bytes -=1
            telemetry[point.lower()] = np.frombuffer(latest[idx:idx + bytes], dtype=np.float32)
            idx += bytes
        return telemetry



        """
        unpacked = struct.unpack('18f', latest[:self.telemetry_len])
        timestamp = unpacked[0]
        pos = np.array(unpacked[1:4])
        att = np.array(unpacked[4:8])
        vel = np.array(unpacked[8:11])
        gyro = np.array(unpacked[11:14])
        inp = np.array(unpacked[14:18])

        return {
            'timestamp': timestamp,
            'position': pos,
            'attitude': att,
            'velocity': vel,
            'gyro': gyro,
            'input': inp
        }
        """


    def quaternion_to_rotation_matrix(self, attitude):
        qx, qy, qz, qw = attitude
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]
        ])
        return R

    def get_telemetry_len(self):
        telemetry_len = 0

        with open(self.telemetry_path, "r", encoding="utf-8") as f:
            data = json.load(f)["StreamFormat"]

        with open(self.telemetry_len_path, "r", encoding="utf-8") as f:
            data_len = json.load(f)

        for point in data:
            telemetry_len += data_len[point]["bytes"]

        return telemetry_len, data, data_len

        




