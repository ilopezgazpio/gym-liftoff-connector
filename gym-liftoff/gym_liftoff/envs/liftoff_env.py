import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import numpy as np
from ..main import VirtualGamepad, VideoSampler
import tkinter as tk
import time
import torch
import pyautogui

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


    def __init__(self):

        logger.info("Initializing environment.....")

        self.virtual_gamepad = VirtualGamepad.VirtualGamepad()

        self.sc_w, self.sc_h = self.__get_curr_screen_geometry__()
        logger.info("Identified screen width..... {}".format(self.sc_w))
        logger.info("Identified screen height..... {}".format(self.sc_h))

        self.video_sampler = VideoSampler.VideoSampler(self.sc_w, self.sc_h)

        self.render_mode = 'human'

        '''
        Observation space is defined as the screenshot converted to a numpy array
        '''
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(1, self.video_sampler.img_x, self.video_sampler.img_y),
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

        # self.reward_model = RewardModel.RewardModel()

    def _get_info(self):
        road = self.video_sampler.find_road()
        # get the center point of the road and the width and height of the road
        # road is a frame of shape (image_height, image_width, 3), having the road in green and the rest in black
        # road = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
        if road is None:
            features = {
                'road_center_x': 0,
                'road_center_y': 0,
                'road_width': 0,
                'road_height': 0,
            }
        else:
            features = road[1]
        return {
            'speed': self._get_speed(),
            'road': features,
        }

    def observation(self):
        array = np.array(self.state, dtype=np.uint8).reshape((1, self.video_sampler.img_x, self.video_sampler.img_y))
        # lower the resolution
        # array = array[::2, ::2]
        assert array.shape == self.observation_space.shape
        return array

    def _get_reward(self, action):
        # 0 if the game finishes
        if self.__episode_terminated__():
            return float(-100)
        # 1 otherwise
        return 1

    def act(self, action, from_reset=False):
        if self.resetting and not from_reset:
            return
        self.virtual_gamepad.act(action)

    def step(self, action):
        if not self._has_reset:
            raise gym.error.ResetNeeded("Cannot call env.step() before calling env.reset()")
    
        info = {}

        '''Send action to liftoff through virtual gamepad'''

        self.act(action)
        ''' Sample liftoff state through video sampler'''
        self.state = self.video_sampler.sample(region=(1280, 0, 1920, 1080))

        # self.__episode_terminated__() ???

        observation = self.observation()
        info = self._get_info()
        reward = self._get_reward(action)
        terminated = self.__episode_terminated__()
        truncated = False
        if terminated or truncated:
            self._has_reset = False
        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
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
            self.act([1400, 1024, 1024, 1024], from_reset=True)
            time.sleep(1)

        self.time = 0
        self.state = self.video_sampler.sample(region=(1280, 0, 1920, 1080))
        observation = self.observation()
        info = self._get_info()
        self.resetting = False
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

    def __episode_terminated__(self):
        """Check if the episode is terminated"""
        # screen is black
        return np.mean(self.state) < 40

