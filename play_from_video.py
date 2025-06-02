import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gym_liftoff.envs.liftoff_wrappers import LiftoffWrapStability, LiftoffWrapNormalizedActions, LiftoffFloatActions
import time
import os
import cv2
import pandas as pd

#set working directory for obtaining demos
os.chdir('database/csv')  

path = './StableFlight_60FPS_ShortCircuit_HighQuality_2025-1-27.csv'
df = pd.read_csv(path)

# for each row in the dataframe, get (state, action, next_state, reward, terminatedm, truncated)
# state = img (256, 256, 1)
# action = (throttle, yaw, roll, pitch)
# next_state = img (256, 256, 3)
# reward = max(1 - 0.01 * np.mean(np.abs(prev_img - img)), 0)
# terminated = False
# truncated = False
demo_actions = []


print("Generating experience... this may take a while")

for i in range(1,len(df)):
    action = np.array([df['throttle'][i], df['yaw'][i], df['roll'][i], df['pitch'][i]])
    # discretize the action space (discretize_actions for each action)
    ##action = np.round(action * (discretize_actions - 1) / 2047).astype(int)
    demo_actions.append(action)
    if i % 10000 == 0:
        print(i, len(df))

print('Generated experience of length:', len(demo_actions))

# set data to np arrays
# transform action to [-1, 1]
# demo_actions = (np.array(demo_actions, dtype = np.float32) / 2047) * 2 - 1 
demo_actions = np.array(demo_actions)


# make sure none is type double
demo_actions = demo_actions.astype(np.float32).tolist()


# Create the buffer
# data = MixedBuffer(buffer_size=1000, initial_demo_ratio=0.99, final_demo_ratio=0.1, demo_ratio_decay=0.99999, observation_space=spaces.Box(low=0, high=255, shape=(256, 256, 1), dtype=np.uint8), action_space=spaces.Box(low=0, high=2047, shape=(4,), dtype=np.uint16))
# data.set_demo_data(demo_observations, demo_actions, demo_next_observations, demo_rewards, demo_dones, demo_truncateds)

# Create the environment
env = gym.make('gym_liftoff:liftoff-v0')
env = LiftoffWrapStability(env)
# env = LiftoffWrapNormalizedActions(env)
env = LiftoffFloatActions(env)

print('Observation space:', env.observation_space)
print('Observation space sample:', env.observation_space.sample())
print('Action space:', env.action_space)
print('Action space sample:', env.action_space.sample())

# play one game, using the buffered actions (demo_actions)
def play_game(env):
    done = False
    obs, _info = env.reset()
    while not done:
        start_time = time.time()
        action = demo_actions.pop(0)
        obs, reward, done, truncated, info = env.step(action)
        elapsed_time = time.time() - start_time
        time.sleep(max(1.0 / 60 - elapsed_time, 0))

# play the game
if __name__ == '__main__':
    play_game(env)
    env.close()
