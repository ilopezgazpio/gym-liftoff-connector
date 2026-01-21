import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gym_liftoff.envs.liftoff_wrappers import LiftoffWrapStability, LiftoffWrapNormalizedActions, LiftoffFloatActions
import time
import os
import cv2
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,  # Or DEBUG, WARNING, etc.
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
path = './data/csv_files/demo_flight.csv'
df = pd.read_csv(path)
logger.info("Loading video at : {}".format(path))

# for each row in the dataframe, get (state, action, next_state, reward, terminatedm, truncated)
# state = img (256, 256, 1)
# action = (throttle, yaw, roll, pitch)
# next_state = img (256, 256, 3)
# reward = max(1 - 0.01 * np.mean(np.abs(prev_img - img)), 0)
# terminated = False
# truncated = False

demo_actions = []


logger.info("Generating experience... this may take a while")

for i in range(1,len(df)):
    action = np.array([df['throttle'][i], df['yaw'][i], df['roll'][i], df['pitch'][i]])
    # discretize the action space (discretize_actions for each action)
    # action = np.round(action * (discretize_actions - 1) / 2047).astype(int)
    demo_actions.append(action)

    if i % 10000 == 0:
        logger.info("Generating experiences {} / {}...".format(i, len(df)))
        logger.info("Sample action: {}".format(action))

logger.info("Generated experience of length {}".format(len(demo_actions)))

# set data to np arrays
# transform action to [-1, 1]
# demo_actions = (np.array(demo_actions, dtype = np.float32) / 2047) * 2 - 1
demo_actions = np.array(demo_actions)

# make sure none is type double
demo_actions = demo_actions.astype(np.float32).tolist()

# Create the buffer
# data = MixedBuffer(buffer_size=1000, initial_demo_ratio=0.99, final_demo_ratio=0.1, demo_ratio_decay=0.99999, observation_space=spaces.Box(low=0, high=255, shape=(256, 256, 1), dtype=np.uint8), action_space=spaces.Box(low=0, high=2047, shape=(4,), dtype=np.uint16))
# data.set_demo_data(demo_observations, demo_actions, demo_next_observations, demo_rewards, demo_dones, demo_truncateds)


##########################
# Create the environment #
##########################
def init_env():
    logger.info("Setting up gameplay....")

    env = gym.make('gym_liftoff:liftoff-v0')
    env = LiftoffWrapStability(env)
    # env = LiftoffWrapNormalizedActions(env)
    env = LiftoffFloatActions(env)

    logger.info("Printing environment information....")
    logger.info("Observation space: {}".format(env.observation_space))
    logger.info("Observation space sample: {}".format(env.observation_space.sample()))
    logger.info("Action space: {}".format(env.action_space))
    logger.info("Action space sample: {}".format(env.action_space.sample()))
    return env


############################################################
# play one game, using the buffered actions (demo_actions) #
############################################################
def play_game(env):

    logger.info("Launch liftoff get short circuit ready and press any key....")
    input()
    logger.info("Sleeping 5 seconds to change focus to liftoff....")
    time.sleep(5)

    i = 0
    done = False
    obs, _info = env.reset()

    # arm the drone
    action = np.array([0.0, 1027.0, 1027.0, 1027.0])  # throttle, yaw, roll, pitch

    while not done:
        action = demo_actions[i]
        i+= 1 * skip
        obs, reward, done, truncated, info = env.step(action)


#############
# Main Loop #
#############

# Launch with -i and execute commands in terminal

# options:
#    play_game(env)
#    env.close()


if __name__ == '__main__':
    skip = 30
    env = init_env()
