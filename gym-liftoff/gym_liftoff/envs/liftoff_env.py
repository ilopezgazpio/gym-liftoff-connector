import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from ..main import VirtualGamepad, VideoSampler, RewardModel
import tkinter as tk
import pyautogui
import time

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
        Observation space is defined as

        TODO

        '''
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.sc_h, self.sc_w, 1), dtype=np.uint8)

        '''
        Action space is defined as 
        
        Dictionary with the following keys:
        - 'THROTTLE'
        - 'YAW'
        - 'PITCH'
        - 'ROLL'

        Each key has a value of a discrete integer between 0 and 2047
        
        '''
        self.action_space = spaces.Dict({
            'THROTTLE': spaces.Discrete(2048),
            'YAW': spaces.Discrete(2048),
            'PITCH': spaces.Discrete(2048),
            'ROLL': spaces.Discrete(2048)
        })

        '''
        Environment state
        '''

        self.state = np.arange(9).reshape((3, 3))

        # self.virtual_gamepad = VirtualGamepad.VirtualGamepad()
        self.video_sampler = VideoSampler.VideoSampler()
        print("Screen width: ", self.sc_w)
        print("Screen height: ", self.sc_h)
        self.consecutive_zero = 0

        self.reward_model = RewardModel.RewardModel()

    def _get_info(self):
        return {
            'speed': self._get_speed()
        }
    
    def _get_observation(self):
        return self.state

    def _get_reward(self, action, info):
        return self.reward_model.reward(self.state, action, info)

    def step(self, action):

        info = {}

        '''Send action to liftoff through virtual gamepad'''
        self.virtual_gamepad.act(action)
        ''' Sample liftoff state through video sampler'''
        self.state = self.video_sampler.sample(region=(0, 0, 1920, 1080))

        # self.__episode_terminated__() ???

        observation = self._get_observation()
        info = self._get_info()
        reward = self._get_reward(action, info)
        done = self.consecutive_zero > 10

        return observation, reward, done, info


    def reset(self, seed=None, options=None):
        """Reset the state of the environment to an initial state"""
        time.sleep(2)
        self.state = self.video_sampler.sample(region=(0, 0, 1920, 1080))
        self.virtual_gamepad.reset()
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
        speed = self.video_sampler.get_speed()
        speed = None
        return

