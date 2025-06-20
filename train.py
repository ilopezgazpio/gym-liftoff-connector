import numpy as np
import gymnasium as gym
from gym_liftoff.envs.liftoff_wrappers import LiftoffWrapStability, LiftoffWrapAutoTakeOff, LiftoffPastState, LiftoffWrapRoad, LiftoffWrapSpeed
import time
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TQC
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import cv2
import threading
import pandas as pd
from gym_liftoff.main.MixedSB3Buffer import MixedBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Set the random seed for reproducibility
np.random.seed(42)
import random
random.seed(42)

# Timesteps per epoch
N_TIMESTEPS = 10000
# Log interval
L_N_TIMESTEPS = 1000
EVAL_EPISODES = 10

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),   # 256 -> 62
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 62 -> 30
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), # 30 -> 14
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2), # 14 -> 6
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def make_env():
    base_env = gym.make('gym_liftoff:liftoff-v0')
    base_env = LiftoffWrapStability(base_env)
    base_env = LiftoffWrapAutoTakeOff(base_env)
    base_env = LiftoffPastState(base_env, past_length=4)
    base_env = LiftoffWrapSpeed(base_env)
    base_env = LiftoffWrapRoad(base_env)
    base_env = Monitor(base_env, filename=None, allow_early_resets=True)
    return base_env

env = DummyVecEnv([make_env]) 

print('Environment created')

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
buffer_size = 10000
stack_size = 4
img_size = (256, 256)

seed = 1

path = './database/csv/StableFlight_60FPS_ShortCircuit_HighQuality_2025-1-27.csv'
df = pd.read_csv(path)

# for each row in the dataframe, get (state, action, next_state, reward, terminatedm, truncated)
# state = img (256, 256, 1)
# action = (throttle, yaw, roll, pitch)
# next_state = img (256, 256, 3)
# reward = max(1 - 0.01 * np.mean(np.abs(prev_img - img)), 0)
# terminated = False
# truncated = False
demo_observations = []
demo_actions = []
demo_next_observations = []
demo_rewards = []
demo_dones = []
demo_truncateds = []
demo_observations = []
demo_actions = []
demo_next_observations = []
demo_rewards = []
demo_dones = []
demo_truncateds = []

demo_buffer_size = 5000


frames = []
for i in range(1, demo_buffer_size):
    img_path = df['img'][i]
    # Change the ../ to database/
    img_path = img_path.replace('../', './database/')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = cv2.resize(img, img_size)
    frames.append(img)

print("Generating experience... this may take a while")

for i in range(stack_size, demo_buffer_size -1):
    # Observation: stack of frames [i-4, i)s
    obs_stack = np.stack(frames[i-stack_size:i], axis=0)  # Shape: [4, 256, 256]
    next_obs_stack = np.stack(frames[i-stack_size+1:i+1], axis=0)

    reward = 1  # Placeholder reward, can be modified later

    action = np.array([df['throttle'][i], df['yaw'][i], df['roll'][i], df['pitch'][i]])

    demo_observations.append(obs_stack)
    demo_next_observations.append(next_obs_stack)
    demo_actions.append(action)
    demo_rewards.append(reward)
    demo_dones.append(False)
    demo_truncateds.append(False)

print('Generated experience of length:', len(demo_observations))


# set data to np arrays
demo_observations = np.array(demo_observations)
# transform action to [-1, 1]
# demo_actions = np.array(demo_actions)
demo_actions = np.array(demo_actions, dtype=np.float32)
# round the actions to the nearest integer
demo_actions = np.round(demo_actions).astype(np.uint16)
demo_next_observations = np.array(demo_next_observations)
demo_rewards = np.expand_dims(np.array(demo_rewards), -1)
demo_dones = np.expand_dims(np.array(demo_dones),-1)
demo_truncateds = np.expand_dims(np.array(demo_truncateds),-1)

# make sure none is type double
demo_observations = demo_observations.astype(np.uint8)
# demo_actions = demo_actions.astype(np.float32)
demo_next_observations = demo_next_observations.astype(np.uint8)
demo_rewards = demo_rewards.astype(np.float32)
demo_dones = demo_dones.astype(bool)
demo_truncateds = demo_truncateds.astype(bool)

# Create the buffer
# data = MixedBuffer(buffer_size=1000, initial_demo_ratio=0.99, final_demo_ratio=0.1, demo_ratio_decay=0.99999, observation_space=spaces.Box(low=0, high=255, shape=(256, 256, 1), dtype=np.uint8), action_space=spaces.Box(low=0, high=2047, shape=(4,), dtype=np.uint16))
# data.set_demo_data(demo_observations, demo_actions, demo_next_observations, demo_rewards, demo_dones, demo_truncateds)

# Create the agent
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512)
)
agent = TQC( 
            env = env,
            policy='CnnPolicy',
            policy_kwargs=policy_kwargs,
            learning_rate=0.001,
            buffer_size=buffer_size,
            batch_size=256,
            train_freq=1,
            gradient_steps=-1,
            ent_coef='auto',
            gamma=0.99,
            tau=0.005,
            verbose=1,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            replay_buffer_class=MixedBuffer,
            replay_buffer_kwargs=dict(
                initial_demo_ratio=0.99,
                final_demo_ratio=0.1,
                demo_ratio_decay=1-1e-7,
            ),
            tensorboard_log="./tensorboard_logs/tqc_road/",
        )

agent.replay_buffer.set_demo_data(demo_observations, demo_actions, demo_next_observations, demo_rewards, demo_dones, demo_truncateds)

current_steps = 0

reward = 0
# Train the agent
# stop = False

# def get_stop():
#     global stop
#     stop = input() == 'stop'

# threading.Thread(target=get_stop).start()

while True:
    agent.learn(total_timesteps=N_TIMESTEPS, log_interval=L_N_TIMESTEPS)
    current_steps += 1
    agent.save(f"./models/tqc_road/{N_TIMESTEPS * current_steps}.zip")

    # # Test the agent
    # rewards = []
    # print('Evaluating')
    # for _ in range(EVAL_EPISODES):
    #     obs, info = env.reset()
    #     done = False
    #     episode_reward = 0
    #     while not done:
    #         action, _ = agent.predict(obs)
    #         obs, reward, done, _, info = env.step(action)
    #         episode_reward += reward
    #     rewards.append(episode_reward)
    # reward = np.mean(rewards)

# Save the agent parameters
agent.save("./models/tqc/latest.zip")

env.close()

print('Starting test')

env_test = DummyVecEnv([make_env])

print('Test environment created')

# Test the agent
agent = TQC.load('./models/tqc/latest.zip', env=env_test, device='cuda' if torch.cuda.is_available() else 'cpu')
rewards = []
agent.set_env(env_test)
obs, info = env_test.reset()
done = False
rewards = []
while not done:
    try:
        action, _ = agent.predict(obs)
    except Exception as e:
        print('Error predicting action:', e)
        action = env_test.action_space.sample()
    obs, reward, done, _, info = env_test.step(action)
    rewards.append(reward)

# Close the environments
print('Test completed')

env_test.close()


# Show mean rewards
print('Mean reward:', np.mean(rewards))

# Plot the rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards')
plt.show()
