import numpy as np
import gymnasium as gym
from gym_liftoff.envs.liftoff_wrappers import LiftoffWrapStability, LiftoffWrapDiscretized
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

# Timesteps per epoch
N_TIMESTEPS = 100000
EVAL_EPISODES = 10

env = gym.make('gym_liftoff:liftoff-v0')
env = LiftoffWrapStability(env)
env = LiftoffWrapDiscretized(env, n_actions=11)

print('Observation space:', env.observation_space)
print('Observation space sample:', env.observation_space.sample())
print('Action space:', env.action_space)
print('Action space sample:', env.action_space.sample())


# experiment results
rewards_total = list()
steps_total = list()
egreedy_total = list()
solved_after = 0
start_time = time.time()

num_episodes = 1000


seed = 1


# Create the agent
agent = sb3.A2C( 
                 env = env,
                 verbose=1,
                 policy='CnnPolicy',
                 policy_kwargs=dict(net_arch=[256, 512, 512, 256]),
                 learning_rate=0.00063,
                 gamma=0.99,
                 n_steps=5,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 rms_prop_eps=1e-5,
                 use_rms_prop=True,
                 use_sde=False,
                 sde_sample_freq=-1,
                 gae_lambda=1.0
                )


reward = 0
# Train the agent
stop = False

def get_stop():
    global stop
    stop = input() == 'stop'
    
    


threading.Thread(target=get_stop).start()

while reward < 10000 and not stop:
    assert not stop
    agent.learn(total_timesteps=N_TIMESTEPS)
    # Test the agent
    rewards = []
    print('Evaluating')
    for _ in range(EVAL_EPISODES):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    reward = np.mean(rewards)

# Save the agent parameters
agent.save("./models/DQN_LIFTOFF.zip")

env.close()

print('Starting test')

env_test = gym.make('gym_liftoff:liftoff-v0')
env_test = LiftoffWrapStability(env_test)
env_test = LiftoffWrapDiscretized(env_test)
# Test the agent
agent.set_env(env_test)
rewards = []
for _ in range(EVAL_EPISODES):
    obs, info = env_test.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _ = agent.predict(obs)
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
