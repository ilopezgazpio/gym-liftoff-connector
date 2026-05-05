import gymnasium as gym
import time
import numpy as np
import cv2
from pathlib import Path
import json
import math


class LiftoffWrapStability(gym.Wrapper):
    def __init__(self, env, ponderation = 1.0, delta_margin = True):
        super(LiftoffWrapStability, self).__init__(env)
        self.past_actions = None
        self.ponderation = ponderation
        if delta_margin:
            self.current_path = Path(__file__).resolve().parent
            self.max_deltas = self.read_data()
        else:
            self.max_deltas = np.array([0, 0, 0, 0]).astype(np.float32)
    def reset(self, seed = None, options = None):
        self.past_actions = None
        return self.env.reset(seed = seed, options = options)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.past_actions is None:
            self.past_actions = np.zeros_like(action)


        delta_action = abs(action - self.past_actions)
        #print(f"Past Action: {self.past_actions}, Action: {action}, Delta: {delta_action}")
        self.past_actions = action.copy()

        reward = reward + self.ponderation * self.get_reward(delta_action)
        return obs, reward, terminated, truncated, info

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
        action = (((action + 1) / 2) * 2047)
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
    def reset(self, seed = None, options = None):
        self.steps = 0
        return self.env.reset(seed = seed, options = options)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward += math.log1p(self.steps)
        self.steps += 1
        return obs, reward, terminated, truncated, info

class LiftoffWrapGyro(gym.Wrapper):
    def __init__(self, env, ponderation = 1, max_gyro = 10):
        super(LiftoffWrapGyro, self).__init__(env)
        self.ponderation = ponderation
        self.max_gyro = max_gyro
    def reset(self, seed = None, options = None):
        obs, info = self.env.reset(seed = seed, options = options)
        info["gyro"] = info["gyro"] * np.pi / 180
        return obs, info
    def step(self, action):
        obs, reward, termianted, truncated, info = self.env.step(action)
        info["gyro"] = info["gyro"] * np.pi / 180
        reward += self.get_reward(info["gyro"])
        return obs, reward, termianted, truncated, info

    def get_reward(self, gyro):
        gyro_norm = np.linalg.norm(gyro) / self.max_gyro
        return - self.ponderation*(gyro_norm**2)

class LiftoffWrapAttitude(gym.Wrapper):
    def __init__(self, env, ponderation = 1.0):
        super(LiftoffWrapAttitude, self).__init__(env)
        self.ponderation = ponderation
    def step(self, action):
        obs, reward, termianted, truncated, info = self.env.step(action)
        reward += self.get_reward(info)
        return obs, reward, termianted, truncated, info
    def get_reward(self, info):
        rotation = info["rotation"]
        up_y = rotation[1, 1]

        reward_attitude = 0.0
        if up_y < 0.3:
            reward_attitude = -1.0

        return self.ponderation*reward_attitude


class LiftoffWrapRandomPosition(gym.Wrapper):
    def __init__(self, env, ponderation = 10.0, max_position = [100, 30, 100]):
        super(LiftoffWrapRandomPosition, self).__init__(env)
        self.goal_position = None
        self.ponderation = ponderation
        self.starting_position = None
        self.max_position = max_position

    def reset(self, seed = None, options = None):
        obs, info = self.env.reset(seed = seed, options = options)
        self.starting_position = np.array(info["position"], dtype=np.float32)
        info["position_norm"] = self.get_position_norm(info["position"])
        self.goal_position = self.get_goal_position()
        self.norm_goal_position = self.get_position_norm(self.goal_position)
        info["goal"] = self.goal_position
        info["goal_norm"] = self.norm_goal_position
        info["distance2goal"] = self.past_distance =self.calculate_distance(self.starting_position)
        return obs, info

    def get_position_norm(self, position):
        return (position -self.starting_position) / self.max_position
    def step(self, action):
        obs, reward, termianted, truncated, info = self.env.step(action)

        info["goal"] = self.goal_position
        info["distance2goal"] = self.calculate_distance(info["position"])

        info["goal_norm"] = self.norm_goal_position
        info["position_norm"] = self.get_position_norm(info["position"])

        if info["distance2goal"]["esc"] < 1.0 and not termianted:
            truncated = True
            reward += 3
        elif not termianted:
            pos_rew = self.get_reward(info["distance2goal"])
            print(pos_rew)
            reward += pos_rew

        self.past_distance = info["distance2goal"]

        return obs, reward, termianted, truncated, info

    def set_new_goal(self, goal = None):
        if goal is None:
            self.goal_position = self.get_goal_position()
        else:
            self.goal_position = self.get_position_norm(goal)
        return self.goal_position

    def set_max_position(self, max_position):
        self.max_position = max_position

    def get_goal_position(self):
        low = self.starting_position - self.max_position
        high = self.starting_position + self.max_position
        low[1] += 2
        return self.sample(low, high)

    def sample(self, low, high):
        return np.array([
            np.random.randint(int(l), int(h)) for l, h in zip(low, high)
        ])

    def calculate_distance(self, current):
        vec = (self.goal_position - current)
        esc = np.linalg.norm(vec)
        vec = vec  / self.max_position
        esc_norm = np.linalg.norm(vec)
        return {
            "vec": vec,
            "esc": esc,
            "esc_norm": esc_norm
        }

    def get_reward(self, distance):
        reward = self.past_distance["esc"] - distance["esc"]
        return self.ponderation*reward

class LiftoffWrapObservation(gym.ObservationWrapper):
    def __init__(self, env, resizeX = 256, resizeY = 256, gray = False):
        super(LiftoffWrapObservation, self).__init__(env)
        # TODO: Hacerlo
