import gymnasium as gym
from gym_liftoff.envs.liftoff_wrappers import LiftoffWrapStability, LiftoffWrapDiscretized
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import sys
import time
import matplotlib.pyplot as plt

import pandas as pd
import os
import numpy as np
import cv2
from SimpleBaselines.agent.rl_agents.MultiDQN_liftoff import MultiDQN
from SimpleBaselines.states.State import State


#launch as PYTHONPATH=/home/bee/development/UBC_GameIntelligence/ python3 /home/bee/development/gym-liftoff-connector/DQN_for_liftoff.py

seed = 1
discretize_actions = 11

#set working directory for obtaining demos
os.chdir('/home/bee/development/gym-liftoff-connector/database/csv')  

path = './RaceFlight_10FPS_ShortCircuit_SimpleQuality_2024-8-1.csv'
df = pd.read_csv(path)

# for each row in the dataframe, get (state, action, next_state, reward, terminatedm, truncated)
# state = img (256, 256, 1)
# action = (throttle, yaw, roll, pitch)
# next_state = img (256, 256, 3)
# reward = max(1 - 0.01 * np.mean(np.abs(prev_img - img)), 0)
# terminated = False
# truncated = False

exp = []

for i in range(1,len(df)-1):
    prev_state = cv2.imread(df['img'][i-1], cv2.IMREAD_GRAYSCALE)
    # resize to 256x256
    prev_state = cv2.resize(prev_state, (256, 256))
    state = cv2.imread(df['img'][i], cv2.IMREAD_GRAYSCALE)
    state = cv2.resize(state, (256, 256))
    # flatten the state
    next_state = cv2.imread(df['img'][i+1], cv2.IMREAD_GRAYSCALE)
    next_state = cv2.resize(next_state, (256, 256))
    action = np.array([df['throttle'][i], df['yaw'][i], df['roll'][i], df['pitch'][i]])
    # discretize the action space (discretize_actions for each action)
    action = np.round(action * (discretize_actions - 1) / 2047).astype(int)
    reward = max(1 - 0.01 * np.mean(np.abs(prev_state - state)), 0)
    terminated = False
    truncated = False
    exp.append((state.flatten(), action, next_state.flatten(), reward, terminated, truncated))

action = np.array([df['throttle'][len(df)-1], df['yaw'][len(df)-1], df['roll'][len(df)-1], df['pitch'][len(df)-1]])
action = np.round(action * (discretize_actions - 1) / 2047).astype(int)
reward = 0

exp.append((next_state.flatten(), action, next_state.flatten(), reward, True, False))
print('Generated experience of length:', len(exp))

#### Create the environment and agent

# Timesteps per epoch
N_EPISODES = 1000
EVAL_EPISODES = 10
OBJ_REWARD = 100000

env = gym.make('gym_liftoff:liftoff-v0')
env = LiftoffWrapStability(env)
env = LiftoffWrapDiscretized(env, n_actions=discretize_actions)
# Flatten the state
env = gym.wrappers.FlattenObservation(env)

print('Observation space:', env.observation_space)
print('Observation space sample:', env.observation_space.sample())
print('Action space:', env.action_space)
print('Action space sample:', env.action_space.sample())

# Create the agent
agent = MultiDQN(    
    env=env,
    seed=seed,
    gamma=0.99,
    nn_learning_rate=0.0001,
    egreedy=0.9,
    egreedy_final=0.02,
    egreedy_decay=2,
    hidden_layers_size=[64],
    activation_fn=nn.Tanh,
    dropout=0.0,
    use_batch_norm=False,
    loss_fn=nn.MSELoss,
    optimizer=optim.Adam,
    data_amount_start = 1.0,
    data_amount_end = 0.0,
    data_amount_decay = 0.99
)

# make sure all states in exp have shape (256*256)

agent.populate_data(exp)


# experiment results
rewards_total = list()
steps_total = list()
egreedy_total = list()
solved_after = 0
start_time = time.time()

# Train the agent
reward = 0
episode = 0
report_interval = 100
stop = False
while reward < OBJ_REWARD and not stop:
    episode += 1
    agent.reset_env(seed=seed)

    # Play the game
    agent.play(max_steps=5000000, seed=seed)

    if agent.current_state.terminated or agent.current_state.truncated:
        steps_total.append(agent.current_state.step )
        rewards_total.append(agent.final_state.cumulative_reward)
        mean_reward_100 = sum(rewards_total[-100:]) / min(len(rewards_total), 100)
        egreedy_total.append(agent.egreedy)

        if mean_reward_100 > OBJ_REWARD and solved_after == 0:
            print("*************************")
            print("SOLVED! After {} episodes".format(episode))
            print("*************************")
            solved_after = episode

        if episode % report_interval == 0:
            elapsed_time = time.time() - start_time
            print("-----------------")
            print("Episode: {}".format(episode))
            print("Average Reward [last {}]: {:.2f}".format(report_interval,
                                                            sum(rewards_total[-report_interval:]) / report_interval))
            print("Average Reward [last 100]: {:.2f}".format(sum(rewards_total[-100:]) / 100))
            print("Average Reward: {:.2f}".format(sum(rewards_total) / len(steps_total)))

            print("Average Steps [last {}]: {:.2f}".format(report_interval,
                                                           sum(steps_total[-report_interval:]) / report_interval))
            print("Average Steps [last 100]: {:.2f}".format(sum(steps_total[-100:]) / 100))
            print("Average Steps: {:.2f}".format(sum(steps_total) / len(steps_total)))

            print("Epsilon: {:.2f}".format(agent.egreedy))
            print("Frames Total: {}".format(agent.current_state.step))
            print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")



# Print the results
if solved_after > 0:
    print("Solved after {} episodes".format(solved_after))
else:
    print("Could not solve after {} episodes".format(episode))

rewards_total = np.array(rewards_total)
steps_total = np.array(steps_total)
egreedy_total = np.array(egreedy_total)

print("Average reward: {}".format( sum(rewards_total) / episode) )
print("Average reward (last 100 episodes): {}".format( sum(rewards_total[-100:]) / 100) )
print("Percent of episodes finished successfully: {}".format( sum(rewards_total > OBJ_REWARD) / episode) )
print("Percent of episodes finished successfully (last 100 episodes): {}".format( sum(rewards_total[-100:] > OBJ_REWARD) / 100) )
print("Average number of steps: {}".format( sum(steps_total)/episode) )
print("Average number of steps (last 100 episodes): {}".format( sum(steps_total[-100:])/100) )

env.close()

plt.figure(figsize=(12,5))
plt.title("Cumulative Rewards")
plt.bar(torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color='green', width=5)
plt.show()

plt.figure(figsize=(12,5))
plt.title("Steps / Episode length")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='red', width=5)
plt.show()

plt.figure(figsize=(12,5))
plt.title("Egreedy / Episode length")
plt.bar(torch.arange(len(egreedy_total)), egreedy_total, alpha=0.6, color='red', width=5)
plt.show()
