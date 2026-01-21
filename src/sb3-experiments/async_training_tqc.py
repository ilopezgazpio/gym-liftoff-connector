import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gym_liftoff.envs.liftoff_wrappers import LiftoffWrapStability, LiftoffWrapNormalizedActions, LiftoffFloatActions, LiftoffWrapAutoTakeOff, LiftoffPastState
from gym_liftoff.main.MixedSB3Buffer import MixedBuffer
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
import copy
import cv2
import pandas as pd
from typing import Any, Optional, Union, Tuple
from stable_baselines3.common.type_aliases import ReplayBufferSamples
# import logger for the agent
from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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


# modify the agent predict function
def predict(self, observation: np.ndarray, state: Optional[np.ndarray] = None, episode_start: Optional[np.ndarray] = None, deterministic: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)
        with torch.no_grad():
            actions = self._predict(obs_tensor, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions_shape = actions.shape  
        actions = actions.cpu().detach().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]
        assert actions.shape == actions_shape, f"{actions.shape} != {actions_shape}"
        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                ## print(actions.shape)
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze()
        # assert actions.shape == self.action_space.shape, f"{actions.shape} != {self.action_space.shape}"
        return actions, state  # type: ignore[return-value]


SAVE_INTERVAL = 100000
stack_size = 4
img_size = (256, 256)

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
demo_observations = []
demo_actions = []
demo_next_observations = []
demo_rewards = []
demo_dones = []
demo_truncateds = []

demo_buffer_size = 5000


frames = []
for i in range(1, demo_buffer_size):
    img = cv2.imread(df['img'][i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = cv2.resize(img, img_size)
    frames.append(img)

print("Generating experience... this may take a while")

for i in range(stack_size, demo_buffer_size -1):
    # Observation: stack of frames [i-4, i)
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

# Create the environment
def make_env():
    base_env = gym.make('gym_liftoff:liftoff-v0')
    base_env = LiftoffWrapStability(base_env)
    base_env = LiftoffWrapAutoTakeOff(base_env)
    base_env = LiftoffPastState(base_env, past_length=4)
    base_env = Monitor(base_env, filename=None, allow_early_resets=True)
    return base_env

env = DummyVecEnv([make_env]) 

print('Observation space:', env.observation_space)
print('Observation space sample:', env.observation_space.sample())
print('Action space:', env.action_space)
print('Action space sample:', env.action_space.sample())

# 2 threads, one for playing the game and one for training

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
            buffer_size=10000,
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
        )

agent.replay_buffer.set_demo_data(demo_observations, demo_actions, demo_next_observations, demo_rewards, demo_dones, demo_truncateds)

agent.set_logger(Logger('../../models/ddpg/logs', output_formats = [TensorBoardOutputFormat('../../models/ddpg/logs')]))

# # load the model if it exists
# if os.path.exists('/home/bee/development/gym-liftoff-connector/models/model_step_3600000.zip'):
#     agent = agent.load('/home/bee/development/gym-liftoff-connector/models/model_step_3600000.zip')
#     print('Model loaded')

agent.policy.predict = predict.__get__(agent.policy)

# Thread-local storage
thread_local = threading.local()

#save played actions for later analysis
actions = []

lock = threading.Lock()
weights_updated = threading.Event()
pause_event = threading.Event()
pause_event.set()  # Start unpaused

def play_game(agent, env):
    obs = env.reset()
    # Create a new policy object of the same class
    local_policy = type(agent.policy)(
        agent.policy.observation_space,
        agent.policy.action_space,
        agent.lr_schedule,
    )
    local_policy.load_state_dict(agent.policy.state_dict())
    local_policy.set_training_mode(False)  # Set to eval mode
    to_add_to_buffer = []
    while not stop_training:
        pause_event.wait()  # Wait if paused
        action, _states = local_policy.predict(obs, deterministic=True)
        
        # Ensure action is in the correct shape
        if action.ndim > 1:
            action = action.squeeze()
        
        actions.append(action)
        next_obs, reward, done, truncated, info = env.step(action)
        to_add_to_buffer.append((obs, next_obs, action, reward, done or truncated, info))
        
        if done:
            obs = env.reset()
        else:
            obs = next_obs

        # Check if weights have been updated
        if weights_updated.is_set():
            with lock:
                print("Updating weights")
                local_policy.load_state_dict(agent.policy.state_dict())  # Update local policy
                if len(to_add_to_buffer) > 0:
                    for data in to_add_to_buffer:
                        agent.replay_buffer.add(*data)
                to_add_to_buffer.clear()  # Clear the data
                weights_updated.clear()  # Reset the event
            
        torch.cuda.empty_cache()

def train_agent(agent):
    global stop_training
    steps = 0
    gradient_steps = 1000  # Number of gradient steps per training iteration
    while not stop_training:
        steps += gradient_steps
        pause_event.wait()  # Wait if paused
        # sample = agent.replay_buffer.sample(4)
        try:
            with lock:
                # print("Sample obs shape:", sample.observations.shape)
                obs = agent.replay_buffer.sample(4).observations
                print("Observation shape:", obs.shape)
                print("Total elements:", obs.numel())  # should be 4 * 4 * 256 * 256 = 1048576

                agent.train(gradient_steps=gradient_steps)
                print(f"Buffer demo ratio: {agent.replay_buffer.demo_ratio}")
                if len(actions) > 0:
                    print(f"Last action: {actions[-1]}")
                agent.save(f"../../models/tqc/model_latest.zip")
                # save every 100000 steps
                if steps % SAVE_INTERVAL == 0:
                    agent.save(f"../../models/tqc/model_step_{steps}.zip")
                weights_updated.set()  # Indicate that weights have been updated
        except Exception as e:
            print("Error during training")
            print(e)
            stop_training = True
        time.sleep(1)
        # Free up unused GPU memory
        torch.cuda.empty_cache()


stop_training = False

def listen_for_stop():
    global stop_training
    input("Press Enter to stop training...\n")
    stop_training = True
    pause_event.set()  # Ensure that the threads are not paused when stopping

def listen_for_pause():
    while not stop_training:
        input("Press 'p' to pause/resume training...\n")
        if pause_event.is_set():
            pause_event.clear()  # Pause
        else:
            pause_event.set()  # Resume

batch_size = 64
play_thread = threading.Thread(target=play_game, args=(agent, env))
train_thread = threading.Thread(target=train_agent, args=(agent,))
stop_thread = threading.Thread(target=listen_for_stop)
pause_thread = threading.Thread(target=listen_for_pause)


max_train_time = 60*60*24*5 # 5 days

stop_thread.start()
pause_thread.start()
play_thread.start()
train_thread.start()

stop_thread.join()
pause_thread.join()
play_thread.join()
train_thread.join()

agent.save("../../models/model_latest.zip")

# show the actions
actions = np.array(actions)
plt.plot(actions[:,0], label='throttle')
plt.plot(actions[:,1], label='yaw')
plt.plot(actions[:,2], label='roll')
plt.plot(actions[:,3], label='pitch')

plt.legend()

plt.show()
