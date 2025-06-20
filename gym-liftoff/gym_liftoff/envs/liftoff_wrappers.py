import time
import gymnasium as gym
import numpy as np
import pyautogui
import cv2

class LiftoffPastState(gym.ObservationWrapper):
    def __init__(self, env, past_length=4):
        super().__init__(env)
        self.past_length = past_length
        self.obs_shape = self.observation_space.shape  # e.g., (1, 256, 256)

        # New shape: (past_length * channels, H, W)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.obs_shape[0] * past_length, self.obs_shape[1], self.obs_shape[2]),
            dtype=np.uint8
        )

        self.past_observations = []

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.past_observations = [obs for _ in range(self.past_length)]
        return self.observation(obs), info

    def observation(self, obs):
        self.past_observations.append(obs)
        if len(self.past_observations) > self.past_length:
            self.past_observations.pop(0)
        return np.concatenate(self.past_observations, axis=0)

class LiftoffWrapStability(gym.RewardWrapper):

    def __init__(self, env):
        super(LiftoffWrapStability, self).__init__(env)
        self.prev_obs = None

    def reward(self, reward):
        if self.prev_obs is not None:
            prev_obs = self.prev_obs.squeeze()
            current_obs = self.env.unwrapped.observation().squeeze()
            flow = cv2.calcOpticalFlowFarneback(prev_obs, current_obs, None,
                                     pyr_scale=0.5, levels=3, winsize=15,
                                     iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

            # Average flow magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            mean_movement = np.mean(magnitude)
            reward = reward - min(0.01 * mean_movement, 0.5) 
        self.prev_obs = self.env.unwrapped.observation()
        return float(reward)

    def reset(self, seed=None, options=None):
        self.prev_obs = None
        return self.env.reset(seed = seed, options = options)

class LiftoffWrapRoad(gym.RewardWrapper):

    def __init__(self, env):
        super(LiftoffWrapRoad, self).__init__(env)
        self.prev_obs = None

    def reward(self, reward):
        if self.prev_obs is not None:
            road_features = self.env.unwrapped._get_info()['road']
            # Features: (center, width, height, angle)
            if road_features is None or len(road_features) == 0:
                # If no road features are detected, return the original reward
                return reward
            center_x, center_y = road_features[0]
            width = road_features[1]
            center_displacement = np.abs(center_x - self.env.unwrapped.sc_w / 2)
            # Normalize the center displacement
            center_displacement = center_displacement / (self.env.unwrapped.sc_w / 2)
            # Calculate the road following reward
            road_following_reward = 1 - center_displacement
            reward = reward + 0.1 * road_following_reward
        self.prev_obs = self.env.unwrapped.observation()
        return reward

    def reset(self, seed=None, options=None):
        self.prev_obs = None
        return self.env.reset(seed = seed, options = options)

class LiftoffWrapLap(gym.RewardWrapper):

    def __init__(self, env):
        super(LiftoffWrapLap, self).__init__(env)
        self.prev_obs = None

    def reward(self, reward):
        if self.prev_obs is not None:
            # measure speed and road following
            info = self.env.unwrapped._get_info()
            reward = reward + 0.1 * info['road'] + 0.1 * info['speed']
        self.prev_obs = self.env.unwrapped.observation()
        return reward

    def reset(self, seed=None, options=None):
        self.prev_obs = None
        return self.env.reset(seed = seed, options = options)

class LiftoffWrapSpeed(gym.RewardWrapper):
    def __init__(self, env):
        super(LiftoffWrapSpeed, self).__init__(env)
        self.prev_obs = None

    def reward(self, reward):
        if self.prev_obs is not None:
            speed = self.env.unwrapped._get_info()['speed']
            reward = reward + 0.1 * speed
        self.prev_obs = self.env.unwrapped.observation()
        return reward

    def reset(self, seed=None, options=None):
        self.prev_obs = None
        return self.env.reset(seed = seed, options = options)

class LiftoffWrapDiscretized(gym.ActionWrapper):

    def __init__(self, env, n_actions=11):
        super(LiftoffWrapDiscretized, self).__init__(env)
        self.action_space = gym.spaces.MultiDiscrete([n_actions, n_actions, n_actions, n_actions])

    def action(self, action):
        """Each action represents a change in throttle, yaw, roll, and pitch
        where action = action * 2047 / n_actions - 1
        """
        action = action * 2047 / (self.action_space.nvec[0] - 1)
        return action

    def reset(self, seed=None, options=None):
        return self.env.reset(seed = seed, options = options)
    
class LiftoffWrapAutoTakeOff(gym.Wrapper):
    def __init__(self, env):
        super(LiftoffWrapAutoTakeOff, self).__init__(env)
    
    def reset(self, seed=None, options=None):
        """Reset the state of the environment to an initial state"""
        #press R key on the keyboard to reset the game
        print("Resetting the game")
        self.unwrapped.resetting = True
        self.unwrapped.virtual_gamepad.reset()
        pyautogui.press('r')
        time.sleep(1.5)
        self.unwrapped.virtual_gamepad.reset()
        self.unwrapped.act([1400, 1024, 1024, 1024], from_reset=True)
        time.sleep(1) 
        return self.env.reset(seed=seed, options=options)

class LiftoffWrapNormalizedActions(gym.ActionWrapper):
    """Wrap the environment to normalize the action space to [-1, 1]
    """

    def __init__(self, env):
        super(LiftoffWrapNormalizedActions, self).__init__(env)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def action(self, action):
        """Each action represents a change in throttle, yaw, roll, and pitch
        where action = (action + 1) / 2 * 2047
        """
        action = ((action + 1) / 2 * 2047)
        # action = np.clip(action, 0, 2047)
        # convert to uint16
        return action.astype(np.uint16)
    
    def reset(self, seed=None, options=None):
        return self.env.reset(seed = seed, options = options)
    

class LiftoffFloatActions(gym.ActionWrapper):
    def __init__(self, env):
        super(LiftoffFloatActions, self).__init__(env)
        self.action_space = gym.spaces.Box(low=0, high=2047, shape=(4,), dtype=np.float32)

    def action(self, action):
        return action
