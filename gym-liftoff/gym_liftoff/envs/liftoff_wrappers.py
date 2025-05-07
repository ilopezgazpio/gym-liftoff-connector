import time
import gymnasium as gym
import numpy as np
import pyautogui

class LiftoffWrapStability(gym.RewardWrapper):
    def __init__(self, env):
        super(LiftoffWrapStability, self).__init__(env)
        self.prev_obs = None

    def reward(self, reward):
        if self.prev_obs is not None:
            reward = max(reward - 0.01 * np.mean(np.abs(self.prev_obs - self.env.unwrapped._get_observation())), 0)
        self.prev_obs = self.env.unwrapped._get_observation()
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
            center_x, center_y = road_features[0]
            width = road_features[1]
            center_displacement = np.abs(center_x - self.env.unwrapped.sc_w / 2)
            # Normalize the center displacement
            center_displacement = center_displacement / (self.env.unwrapped.sc_w / 2)
            # Calculate the road following reward
            road_following_reward = 1 - center_displacement
            reward = reward + 0.1 * road_following_reward
        self.prev_obs = self.env.unwrapped._get_observation()
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
            info = self.env._get_info()
            reward = reward + 0.1 * info['road'] + 0.1 * info['speed']
        self.prev_obs = self.env.unwrapped._get_observation()
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
            speed = self.env._get_info()['speed']
            reward = reward + 0.1 * speed
        self.prev_obs = self.env.unwrapped._get_observation()
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
        self.resetting = True
        self.virtual_gamepad.reset()
        pyautogui.press('r')
        time.sleep(1.5)
        self.virtual_gamepad.reset()
        self.act([1400, 1024, 1024, 1024], from_reset=True)
        time.sleep(1) 
        self.time = 0
        self.state = self.video_sampler.sample(region=(1280, 0, 1920, 1080))
        info = self._get_info()
        observation = self._get_observation()
        self.resetting = False
        return (observation, info)

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