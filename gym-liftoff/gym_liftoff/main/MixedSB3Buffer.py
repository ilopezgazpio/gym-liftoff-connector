import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import stable_baselines3 as sb3
import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import copy
import cv2
import pandas as pd
from typing import Any, Optional, Union, Tuple
from stable_baselines3.common.type_aliases import ReplayBufferSamples
# import logger for the agent
from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat 


#custom buffer class, for stable baselines
class MixedBuffer(sb3.common.buffers.ReplayBuffer):
    """
    Buffer with 2 different types of data, one are experiences and the other are demonstrations
    custom parameters:
    - initial_demo_ratio: initial ratio of demonstrations in the batch
    - final_demo_ratio: final ratio of demonstrations in the batch
    - demo_ratio_decay: decay of the ratio of demonstrations in the batch
    """

    demo_observations: np.ndarray
    demo_actions: np.ndarray
    demo_next_observations: np.ndarray
    demo_rewards: np.ndarray
    demo_dones: np.ndarray
    demo_truncateds: np.ndarray

    def __init__(self,
        buffer_size: int, 
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = False,
        initial_demo_ratio: float = 1, 
        final_demo_ratio:float = 0.1, 
        demo_ratio_decay: float = 0.999,
        ):


        super(MixedBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)
        self.demo_ratio = initial_demo_ratio
        self.final_demo_ratio = final_demo_ratio
        self.demo_ratio_decay = demo_ratio_decay

    def set_demo_data(self, demo_observations, demo_actions, demo_next_observations, demo_rewards, demo_dones, demo_truncateds):
        assert len(demo_observations) == len(demo_actions) == len(demo_next_observations) == len(demo_rewards) == len(demo_dones) == len(demo_truncateds)
        # make sure types are correct
        assert isinstance(demo_observations, np.ndarray)
        # assert demo_observations.dtype == env.observation_space.dtype
        assert isinstance(demo_actions, np.ndarray)
        # assert demo_actions.dtype == env.action_space.dtype
        assert isinstance(demo_next_observations, np.ndarray)
        # assert demo_next_observations.dtype == env.observation_space.dtype

        # set the demo data
        self.demo_observations = demo_observations
        self.demo_actions = demo_actions
        self.demo_next_observations = demo_next_observations
        self.demo_rewards = demo_rewards
        self.demo_dones = demo_dones
        self.demo_truncateds = demo_truncateds
        assert len(self.demo_observations) == len(self.demo_actions) == len(self.demo_next_observations) == len(self.demo_rewards) == len(self.demo_dones) == len(self.demo_truncateds)

    def sample(self, batch_size: int, env: Optional[gym.Env] = None):
        """
        Sample function for the buffer
        :param batch_size: (int) Size of the batch
        :param env: (gym.Env) The environment
        :return: (dict[str, np.ndarray]) Samples
        """
        assert self.demo_observations is not None, "No demo data available"

        demo_batch_size = int(batch_size * self.demo_ratio)
        demo_batch_size = min(len(self.demo_observations), demo_batch_size)
        exp_batch_size = batch_size - demo_batch_size
        exp_batch_size = min(self.size(), exp_batch_size)
        # update the demo ratio
        self.demo_ratio = max(self.final_demo_ratio, self.demo_ratio * self.demo_ratio_decay)
        # get demo data
        demo_indices = np.random.randint(0, len(self.demo_observations), demo_batch_size)
        demo_observations = self.demo_observations[demo_indices]
        demo_actions = self.demo_actions[demo_indices]
        demo_next_observations = self.demo_next_observations[demo_indices]
        # done if done or truncated
        assert self.demo_dones.shape == self.demo_truncateds.shape
        demo_dones = self.demo_dones[demo_indices] | self.demo_truncateds[demo_indices]
        demo_rewards = self.demo_rewards[demo_indices]
        demo_data = ReplayBufferSamples(*tuple(map(self.to_torch, (demo_observations, demo_actions, demo_next_observations, demo_dones, demo_rewards))))
        # get experience data
        exp_data = super(MixedBuffer, self).sample(exp_batch_size, env)
        # join both samples
        try:
            assert demo_data.observations.shape[1] == exp_data.observations.shape[1] 
        except AssertionError:
            print(f"Demo data shape: {demo_data.observations.shape}, Experience data shape: {exp_data.observations.shape}")
        data = ReplayBufferSamples(
            observations = torch.cat((demo_data.observations, exp_data.observations), dim=0),
            actions = torch.cat((demo_data.actions, exp_data.actions), dim=0),
            next_observations = torch.cat((demo_data.next_observations, exp_data.next_observations), dim=0),
            dones = torch.cat((demo_data.dones, exp_data.dones), dim=0),
            rewards = torch.cat((demo_data.rewards, exp_data.rewards), dim=0),
        )
        return data


