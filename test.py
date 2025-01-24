import numpy as np
import gymnasium as gym
from gym_liftoff.envs.liftoff_wrappers import LiftoffWrapStability, LiftoffWrapNormalizedActions
import time
import stable_baselines3 as sb3
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import cv2
import threading

EVAL_EPISODES = 10

print('Starting test')

agent = sb3.DDPG.load('/home/bee/development/gym-liftoff-connector/models/ddpg/model_latest.zip', device='cpu')

env_test = gym.make('gym_liftoff:liftoff-v0')
env_test = LiftoffWrapStability(env_test)
env_test = LiftoffWrapNormalizedActions(env_test)

action_distribs = []

# Test the agent
agent.set_env(env_test)
rewards = []
for _ in range(EVAL_EPISODES):
    obs, info = env_test.reset()
    done = False
    episode_reward = 0
    while not done:
        try:
            action, _ = agent.predict(obs)
        except:
            action = env_test.action_space.sample()
            print('Error predicting action')
        action_distribs.append(action)
        obs, reward, done, _, info = env_test.step(action)
        episode_reward += reward
    rewards.append(episode_reward)

# Close the environments

env_test.close()


# Show mean rewards
print('Mean reward:', np.mean(rewards))

# Plot the rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards')
plt.show()

# Plot the action distributions
action_distribs = np.array(action_distribs)
plt.plot(action_distribs[:, 0], label='Throttle')
plt.plot(action_distribs[:, 1], label='Yaw')
plt.plot(action_distribs[:, 2], label='Roll')
plt.plot(action_distribs[:, 3], label='Pitch')
plt.xlabel('Step')
plt.ylabel('Action')
plt.title('Action distributions')

plt.legend()

plt.show()