import gymnasium as gym
import numpy as np

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
            reward = reward - 0.01 * np.sum(np.abs(self.prev_obs - self.env.observation)) + 0.1 * self.env._get_info()['road']
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