import gymnasium as gym
import time
import numpy as np
import cv2
from gym_liftoff.envs.rewards import *
from pathlib import Path
import json
import math

class LiftoffWrapStability(gym.Wrapper):
    def __init__(self, env, ponderation = 1.0):
        super(LiftoffWrapStability, self).__init__(env)
        self.past_actions = None
        self.ponderation = ponderation
        self.current_path = Path(__file__).resolve().parent
        self.max_deltas = self.read_data()
    def reset(self, seed = None):
        self.past_actions = None
        return self.env.reset(seed)
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self.past_actions is None:
            self.past_actions = np.zeros_like(action)

        continuous_action = self.discrete2continuous(action)
        delta_action = abs(continuous_action - self.past_actions)
        self.past_actions = continuous_action

        reward = reward + self.ponderation * self.get_reward(delta_action)

        return obs, reward, done, info

    def get_reward(self, delta_action):
        return - np.sum(np.where(delta_action > self.max_deltas, (delta_action - self.max_deltas) ** 2, 0)) / 4

    def discrete2continuous(self, action):
        return action * 2 / 2047 - 1
    def read_data(self):
        try:
            with open(f"{self.current_path}/delta_data/data.json", "r") as f:
                deltas = json.load(f)
            max_deltas = np.array([deltas["throttle"], deltas["yaw"], deltas["roll"], deltas["pitch"]])
        except FileNotFoundError:
            raise FileNotFoundError
        return max_deltas

class LiftoffWrapContinuousAction(gym.ActionWrapper):
    def __init__(self, env):
        super(LiftoffWrapContinuousAction, self).__init__(env)
        self.env.unwrapped.continuous_action_mode = True
    def action(self, action):
        return self.continuous2discrete(action)

    def continuous2discrete(self, action):
        action = ((action + 1) / 2 * 2047)
        return action.astype(np.uint16)

class LiftoffWrapConstantTime(gym.RewardWrapper):
    def __init__(self, env):
        super(LiftoffWrapConstantTime, self).__init__(env)
    def reward(self, reward):
        return reward + 1.0

class LiftoffWrapLogTime(gym.Wrapper):
    def __init__(self, env):
        super(LiftoffWrapLogTime, self).__init__(env)
        self.steps = 0
    def reset(self, seed = None):
        self.steps = 0
        return self.env.reset(seed)
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward += math.log1p(self.steps)
        self.steps += 1
        return obs, reward, done, info

class LiftoffWrapGyro(gym.Wrapper):
    def __init__(self, env):
        super(LiftoffWrapGyro, self).__init__(env)
    def step(self, action):
        obs, reward, done, info = self.env.step(action)


class LiftoffWrapObservation(gym.ObservationWrapper):
    def __init__(self, env, resizeX = 256, resizeY = 256, gray = False):
        super(LiftoffWrapObservation, self).__init__(env)
        # TODO: Hacerlo
