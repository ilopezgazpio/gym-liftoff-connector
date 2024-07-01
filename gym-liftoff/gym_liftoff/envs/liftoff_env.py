import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from ..main import VirtualGamepad, VideoSampler
import pyautogui
import time

import logging
logger = logging.getLogger(__name__)



class Liftoff(gym.Env):

    metadata = {
        'render.modes': ['human']
    }


    def __init__(self):
        '''
        Observation space is defined as

        TODO

        '''
        self.observation_space = spaces.Box(low=0, high=255, shape=(420, 270, 1), dtype=np.uint8)

        '''
        Action space is defined as 
        
        TODO
        
        
        '''
        self.action_space = spaces.Discrete(4)

        '''
        Environment state
        '''

        self.state = np.arange(9).reshape((3, 3))

        self.virtual_gamepad = VirtualGamepad.VirtualGamepad()
        self.video_sampler = VideoSampler.VideoSampler()

    def _get_info(self):
        return {
            'time': 0,
            'score': 0,
            'lives': 1,
            'level': 1
        }
    
    def _get_observation(self):
        return self.state

    def _get_reward(self):
        return 0

    def step(self, action):

        info = {}

        '''Send action to liftoff through virtual gamepad'''
        self.virtual_gamepad.act(action)
        ''' Sample liftoff state through video sampler'''
        self.state = self.video_sampler.sample(region=(0, 0, 1920, 1080))

        # self.__episode_terminated__() ???

        observation = self._get_observation()
        reward = self._get_reward()
        done = None
        info = None

        return observation, reward, done, info

    def act(self, action):
        action_type = ACTION_LOOKUP[action[0]]
        if action_type == 'THROTTLE':
            self.virtual_gamepad.throttle(action[1])
        elif action_type == 'YAW':
            self.virtual_gamepad.yaw(action[1])
        elif action_type == 'PITCH':
            self.virtual_gamepad.pitch(action[1])
        elif action_type == 'ROLL':
            self.virtual_gamepad.roll(action[1])
        else:
            print('Unrecognized action %d' % action_type)


    def reset(self, seed=None, options=None):
        """Reset the state of the environment to an initial state"""
        self.act((0, 2047))
        # press keyboard R key
        pyautogui.press('r')
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


    def __episode_terminated__(self):
        """Check if the episode is terminated"""
        speed = self.video_sampler.get_speed()
        speed = None
        return

ACTION_LOOKUP = {
    0 : 'THROTTLE',
    1 : 'YAW',
    2 : 'PITCH',
    3 : 'ROLL'

}
