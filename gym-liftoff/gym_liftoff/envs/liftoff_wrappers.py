import gymnasium as gym
import time
import numpy as np
import cv2
from pathlib import Path
import json
import math
import random


class LiftoffWrapStability(gym.Wrapper):
    def __init__(self, env, ponderation = 0.3, delta_margin = True):
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

        act_reward = self.ponderation * self.get_reward(delta_action)
        #print("Action Reward: ", act_reward)
        reward = reward + act_reward
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
    def __init__(self, env, ponderation = 0.3, max_gyro = 10):
        super(LiftoffWrapGyro, self).__init__(env)
        self.ponderation = ponderation
        self.max_gyro = max_gyro
    def reset(self, seed = None, options = None):
        obs, info = self.env.reset(seed = seed, options = options)
        return obs, info
    def step(self, action):
        obs, reward, termianted, truncated, info = self.env.step(action)
        if not termianted:
            gyro_rew = self.get_reward(info["gyro"])
            #print("Gyro Reward: ", gyro_rew)
            reward += gyro_rew
        return obs, reward, termianted, truncated, info

    def get_reward(self, gyro):
        gyro_norm = np.linalg.norm(gyro) / self.max_gyro
        #print("gyro_norm:", gyro_norm)
        return - self.ponderation*(gyro_norm**2)

class LiftoffWrapAttitude(gym.Wrapper):
    def __init__(self, env, ponderation = 1):
        super(LiftoffWrapAttitude, self).__init__(env)
        self.ponderation = ponderation
    def step(self, action):
        obs, reward, termianted, truncated, info = self.env.step(action)
        rew_attitude = self.get_reward(info)
        reward += rew_attitude
        return obs, reward, termianted, truncated, info
    def reset(self, seed = None, options = None):
        return self.env.reset(seed = seed, options = options)
    def get_reward(self, info):
        rotation = info["rotation"]
        up_y = rotation[1, 1]

        reward_attitude = up_y - 1

        return self.ponderation*reward_attitude


class LiftoffWrapRandomPosition(gym.Wrapper):
    def __init__(self, env, ponderation = 4.0, max_position = [30, 10, 30]):
        super(LiftoffWrapRandomPosition, self).__init__(env)
        self.goal_position = None
        self.ponderation = ponderation
        self.starting_position = None
        self.max_position = max_position

    def reset(self, seed = None, options = None):
        obs, info = self.env.reset(seed = seed, options = options)
        self.starting_position = np.array(info["position"], dtype=np.float32)
        #print([round(x, 2) for x in info["position"]])
        info["position_norm"] = self.get_position_norm(info["position"])

        self.goal_position = self.get_goal_position()
        self.norm_goal_position = self.get_position_norm(self.goal_position)

        info["goal"] = self.goal_position
        info["goal_norm"] = self.norm_goal_position
        info["distance2goal"] = self.past_distance =self.calculate_distance(self.starting_position)

        return obs, info

    def step(self, action):
        obs, reward, termianted, truncated, info = self.env.step(action)

        info["goal"] = self.goal_position
        info["distance2goal"] = self.calculate_distance(info["position"])
        #print([round(x, 2) for x in info["position"]])

        info["goal_norm"] = self.norm_goal_position
        info["position_norm"] = self.get_position_norm(info["position"])
        #if random.random() < 1:
            #print(info["goal_norm"], info["position_norm"])
            #print("Distance: ", info["distance2goal"]["esc"])
        if info["distance2goal"]["esc"] < 5.0 and not termianted:
            truncated = True
            reward += 3
        elif not termianted:
            pos_rew = self.get_reward(info["distance2goal"])
            #print("Position Reward:", pos_rew)
            reward += pos_rew

        self.past_distance = info["distance2goal"]

        return obs, reward, termianted, truncated, info

    def get_position_norm(self, position):
        return (position -self.starting_position) / self.max_position


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
        #print(self.past_distance["esc"], distance["esc"])
        reward = self.past_distance["esc"] - distance["esc"]
        return self.ponderation*reward

class LiftoffWrapSpeed(gym.Wrapper):
    def __init__(self, env, ponderation = 0.5):
        super(LiftoffWrapSpeed, self).__init__(env)
        self.ponderation = ponderation
    def reset(self, seed = None, options = None):
        return self.env.reset(seed = seed, options=options)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action=action)
        rew_speed = self.get_reward(info)
        reward += rew_speed
        return obs, reward, terminated, truncated, info
    def get_reward(self, info):
        return - abs(np.linalg.norm(info["velocity"]) / 20)

class LiftoffWrapHovering(gym.Wrapper):
    def __init__(self, env):
        super(LiftoffWrapHovering, self).__init__(env)

        continuous_env = LiftoffWrapContinuousAction(self.env)
        gyro_env = LiftoffWrapGyro(continuous_env, ponderation=0.3)
        act_env = LiftoffWrapStability(gyro_env, ponderation=0.2)
        speed_env = LiftoffWrapSpeed(act_env, ponderation=0.3)
        attitude_env = LiftoffWrapAttitude(speed_env, ponderation=0.5)

        self.final_env = LiftoffWrapConstantTime(attitude_env)

    def reset(self, seed = None, options = None):
        self.max_time = self.final_env.unwrapped.max_episode_time
        self.final_env.unwrapped.max_episode_time = None

        obs, info = self.start_hovering(seed = seed, options = options)

        self.hover_position = info["position"]
        self.final_env.unwrapped.max_episode_time = self.max_time + info["timestamp"][0]
        info["hover_position"] = np.zeros(3, dtype=np.float32)
        info["relative_position"] = info["hover_position"].copy()

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.final_env.step(action = action)

        terminated = terminated or self.__episode_terminated__(info)

        position_reward = self.get_reward(info)

        info["hover_position"] = np.zeros(3, dtype=np.float32)
        info["relative_position"] = info["position"] - self.hover_position

        reward += position_reward
        return obs, reward, terminated, truncated, info

    def __episode_terminated__(self, info):
        position = info["position"]
        up_y = info["rotation"][1, 1]
        delta_position = position - self.hover_position
        return np.linalg.norm(delta_position) > 1 or up_y < - 0.5

    def get_reward(self, info):
        position = info["position"]
        z = position[1]
        x = position [0]
        y = position[2]
        z_target = self.hover_position[1]
        x_target = self.hover_position[0]
        y_target = self.hover_position[2]
        z_error = (z - z_target)**2
        xy_error = (x - x_target)**2 + (y - y_target)**2
        reward = -(1*z_error + 0.5*xy_error)
        return reward

    def start_hovering(self, seed = None, options = None):
        Kp = 3.0
        Ki = 0.4
        Kd = 2
        Kp_z = 2.5

        th  = 1024  # center of the controller. TODO: ADJUST TO YOU REQUIREMENTS
        integral = 0.0
        prev_error = 0.0

        TH_MIN = 0
        TH_MAX = 2047

        _, info = self.final_env.reset(seed = seed, options = options)

        z_target = info["position"][1] + 2

        prev_timestamp = info["timestamp"]

        dt = 0.02

        episode_ready = False
        stable_frames = 0

        while not episode_ready:
            vz = info["velocity"][1]
            z = info["position"][1]

            vz_target = Kp_z * (z_target - z)

            error = vz_target - vz

            integral += error * dt
            derivative = (error - prev_error) / dt

            u = Kp * error + Ki * integral + Kd * derivative

            th = np.clip(th + u, TH_MIN, TH_MAX)

            prev_error = error

            action = [th, 1024, 1024, 1024]

            obs, _, _, _, info = self.final_env.unwrapped.step(action)

            dt = info["timestamp"] - prev_timestamp
            prev_timestamp = info["timestamp"]

            # TODO: SOLO DE PRUBA, LUEGO QUITAR
            #print("vz:", vz, "th:", th, "z:", z, z_target)

            stable_frames += 1 if abs(error) < 0.1 and abs(vz) < 0.2 else 0
            if stable_frames > 2:
                episode_ready = True

        return obs, info

class LiftoffWrapObservation(gym.ObservationWrapper):
    def __init__(self, env, resizeX = 256, resizeY = 256, gray = False):
        super(LiftoffWrapObservation, self).__init__(env)
        # TODO: Hacerlo
