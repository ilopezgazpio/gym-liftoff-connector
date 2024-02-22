import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

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
        self.observation_space = spaces.Box(low=0, high=8, shape=(3,3), dtype=np.int8)

        '''
        Action space is defined as 
        
        TODO
        
        
        '''
        self.action_space = spaces.Discrete(4)

        '''
        Environment state
        '''

        self.state = np.arange(9).reshape((3, 3))


    def step(self, action):

        info = {}

        '''Send action to liftoff through virtual gamepad'''

        ''' Sample liftoff state through video sampler'''

        # self.__episode_terminated__() ???

        observation = None
        reward = None
        done = None
        info = None

        return observation, reward, done, info


    def reset(self):

        return self.state


    def render(self, mode='human'):
        print("\n{}\n".format(self.state))


    def close(self):
        return


    def __episode_terminated__(self):
        return
