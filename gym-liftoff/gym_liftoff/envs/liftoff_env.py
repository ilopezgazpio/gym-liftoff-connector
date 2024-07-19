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
        'render.modes': ['human']
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

        self.sc_w, self.sc_h = self.__get_curr_screen_geometry__()
        '''
        Observation space is defined as the screenshot converted to a numpy array

        '''
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.sc_h, self.sc_w, 1), dtype=np.uint8)

        '''
        Action space is defined as 
        
        4 values between 0 and 1
        
        0: Throttle
        1: Yaw
        2: Roll
        3: Pitch

        '''
        self.action_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        '''
        Environment state
        '''

        self.state = np.arange(9).reshape((3, 3))

        self.virtual_gamepad = VirtualGamepad.VirtualGamepad()
        self.video_sampler = VideoSampler.VideoSampler()
        print("Screen width: ", self.sc_w)
        print("Screen height: ", self.sc_h)
        self.consecutive_zero = 0

        # self.reward_model = RewardModel.RewardModel()

    def _get_info(self):
        return {
            'speed': self._get_speed(),
            'road': self.video_sampler.find_road()
        }
    
    def _get_observation(self):
        array = np.array(self.state, dtype=np.uint8).reshape((self.sc_h, self.sc_w, 1))
        if array.shape != self.observation_space.shape:
            print(array.shape)
        return array

    def _get_reward(self, action, info):
        # 0 if the game finishes
        if self.__episode_terminated__():
            return 0
        # 1 otherwise
        return 1

    def step(self, action):

        info = {}

        '''Send action to liftoff through virtual gamepad'''
        # map the float values to the range of the gamepad as integers
        action = (action * 2047).astype(int)
        self.virtual_gamepad.act(action)
        ''' Sample liftoff state through video sampler'''
        self.state = self.video_sampler.sample(region=(0, 0, 1920, 1080))

        # self.__episode_terminated__() ???

        observation = self._get_observation()
        info = self._get_info()
        reward = self._get_reward(action, info)
        done = self.__episode_terminated__()

        return observation, reward, done, False, info


    def reset(self, seed=None, options=None):
        """Reset the state of the environment to an initial state"""
        #press R key on the keyboard to reset the game
        pyautogui.press('r')
        time.sleep(2)
        self.virtual_gamepad.reset()
        time.sleep(0.5)
        self.virtual_gamepad.act([1400, 1024, 1024, 1024])
        time.sleep(1) 
        self.time = 0
        self.state = self.video_sampler.sample(region=(0, 0, 1920, 1080))
        info = self._get_info()
        observation = self._get_observation()
        return (observation, info)

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
        return None

    def __episode_terminated__(self):
        """Check if the episode is terminated"""
        # screen is black
        return np.mean(self.state) < 10

