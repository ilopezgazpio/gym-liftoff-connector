import gymnasium as gym
import time
import numpy as np
import cv2
from gym_liftoff.envs.rewards import *

class LiftoffWrapStability(gym.RewardWrapper):
    def __init__(self, env):
        super(LiftoffWrapStability, self).__init__(env)