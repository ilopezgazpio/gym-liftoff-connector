import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gym_liftoff.envs.liftoff_wrappers import LiftoffWrapStability, LiftoffWrapNormalizedActions, LiftoffFloatActions
import time
import stable_baselines3 as sb3
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

SAVE_INTERVAL = 100000


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
        assert demo_observations.dtype == env.observation_space.dtype
        assert isinstance(demo_actions, np.ndarray)
        assert demo_actions.dtype == env.action_space.dtype
        assert isinstance(demo_next_observations, np.ndarray)
        assert demo_next_observations.dtype == env.observation_space.dtype

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
        data = ReplayBufferSamples(
            observations = torch.cat((demo_data.observations, exp_data.observations), dim=0),
            actions = torch.cat((demo_data.actions, exp_data.actions), dim=0),
            next_observations = torch.cat((demo_data.next_observations, exp_data.next_observations), dim=0),
            dones = torch.cat((demo_data.dones, exp_data.dones), dim=0),
            rewards = torch.cat((demo_data.rewards, exp_data.rewards), dim=0),
        )
        return data




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

for i in range(1,len(df)-1):
    prev_state = cv2.imread(df['img'][i-1], cv2.IMREAD_GRAYSCALE)
    prev_state = cv2.resize(prev_state, (256, 256))
    state = cv2.imread(df['img'][i], cv2.IMREAD_GRAYSCALE)
    state = cv2.resize(state, (256, 256))

    # flatten the state
    next_state = cv2.imread(df['img'][i+1], cv2.IMREAD_GRAYSCALE)
    next_state = cv2.resize(next_state, (256, 256))

    action = np.array([df['throttle'][i], df['yaw'][i], df['roll'][i], df['pitch'][i]])
    # discretize the action space (discretize_actions for each action)
    ##action = np.round(action * (discretize_actions - 1) / 2047).astype(int)
    reward = float(max(1 - 0.01 * np.mean(np.abs(prev_state - state)), 0))
    terminated = False
    truncated = False
    demo_observations.append(state.reshape(1, 256, 256))
    demo_actions.append(action)
    demo_next_observations.append(next_state.reshape(1, 256, 256))
    demo_rewards.append(reward)
    demo_dones.append(terminated)
    demo_truncateds.append(truncated)

action = np.array([df['throttle'][len(df)-1], df['yaw'][len(df)-1], df['roll'][len(df)-1], df['pitch'][len(df)-1]])
##action = np.round(action * (discretize_actions - 1) / 2047).astype(int)
demo_observations.append(next_state.reshape(1, 256, 256))
demo_actions.append(action)
demo_next_observations.append(next_state.reshape(1, 256, 256))
demo_rewards.append(0)
demo_dones.append(True)
demo_truncateds.append(False)
print('Generated experience of length:', len(demo_observations))

# set data to np arrays
demo_observations = np.array(demo_observations)
# transform action to [-1, 1]
# demo_actions = (np.array(demo_actions, dtype = np.float32) / 2047) * 2 - 1 
demo_actions = np.array(demo_actions)
demo_next_observations = np.array(demo_next_observations)
demo_rewards = np.expand_dims(np.array(demo_rewards), -1)
demo_dones = np.expand_dims(np.array(demo_dones),-1)
demo_truncateds = np.expand_dims(np.array(demo_truncateds),-1)

# make sure none is type double
demo_observations = demo_observations.astype(np.uint8)
demo_actions = demo_actions.astype(np.float32)
demo_next_observations = demo_next_observations.astype(np.uint8)
demo_rewards = demo_rewards.astype(np.float32)
demo_dones = demo_dones.astype(bool)
demo_truncateds = demo_truncateds.astype(bool)

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

# 2 threads, one for playing the game and one for training

# Create the agent

agent = sb3.DDPG(
                    env = env,
                    verbose=1,
                    policy='CnnPolicy',
                    policy_kwargs=dict(net_arch=[256, 512, 512, 256]),
                    learning_rate=0.00063,
                    gamma=0.99,
                    # n_steps=5,
                    # ent_coef=0.01,
                    # vf_coef=0.5,
                    # max_grad_norm=0.5,
                    # rms_prop_eps=1e-5,
                    # use_rms_prop=True,
                    # use_sde=False,
                    # sde_sample_freq=-1,
                    # gae_lambda=1.0,
                    buffer_size=1000,
                    batch_size=64,
                    learning_starts=1000,
                    train_freq=1,
                    gradient_steps=1,
                    tau=0.005,
                    action_noise=None,
                    optimize_memory_usage=False,
                    replay_buffer_class=MixedBuffer,
                    replay_buffer_kwargs=dict(
                        initial_demo_ratio=0.99,
                        final_demo_ratio=0.1,
                        demo_ratio_decay=0.99999,
                    ),
                    )

agent.replay_buffer.set_demo_data(demo_observations, demo_actions, demo_next_observations, demo_rewards, demo_dones, demo_truncateds)

agent.set_logger(Logger('../../models/ddpg/logs', output_formats = [TensorBoardOutputFormat('../../models/ddpg/logs')]))

# # load the model if it exists
# if os.path.exists('/home/bee/development/gym-liftoff-connector/models/model_step_3600000.zip'):
#     agent = agent.load('/home/bee/development/gym-liftoff-connector/models/model_step_3600000.zip')
#     print('Model loaded')

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
    obs, _info = env.reset()
    local_policy = copy.deepcopy(agent.policy)  # Create a copy of the agent's policy
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
            obs, _info = env.reset()
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
    while not stop_training:
        steps += 1
        pause_event.wait()  # Wait if paused
        try:
            with lock:
                agent.train(gradient_steps=100)
                print(f"Buffer demo ratio: {agent.replay_buffer.demo_ratio}")
                if len(actions) > 0:
                    print(f"Last action: {actions[-1]}")
                agent.save(f"../../models/ddpg/model_latest.zip")
                # save every 100000 steps
                if steps % SAVE_INTERVAL == 0:
                    agent.save(f"../../models/ddpg/model_step_{steps}.zip")
                weights_updated.set()  # Indicate that weights have been updated
        except Exception as e:
            print("Error during training")
            print(e)
            stop_training = True
        time.sleep(1)
        # Free up unused GPU memory
        torch.cuda.empty_cache()

    # global stop_training
    # steps = 0   
    # policy = agent.policy
    # optimizer = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=1e-5)  # Add weight decay for L2 regularization
    # max_grad_norm = 0.5  # Gradient clipping threshold

    # while not stop_training:
    #     batch = data.sample(batch_size=batch_size)
    #     if batch:
    #         obs, actions, next_obs, rewards, dones = batch

    #         # Convert data to float tensors
    #         obs_tensor = obs.to(torch.float32)
    #         actions_tensor = actions.to(torch.float32)
    #         rewards_tensor = rewards.to(torch.float32)
    #         next_obs_tensor = next_obs.to(torch.float32)
    #         dones_tensor = dones.to(torch.float32)

    #         # Compute loss (example: policy loss + value loss)
    #         # reshape the observations to (batch_size, 1, 256, 256)
    #         obs_tensor = obs_tensor.squeeze().unsqueeze(1)
    #         next_obs_tensor = next_obs_tensor.squeeze().unsqueeze(1)
    #         try:
    #             values, log_probs, entropy = policy.evaluate_actions(obs_tensor, actions_tensor)
    #         except Exception as e:
    #             print(obs_tensor.shape)
    #         advantages = rewards_tensor - values.detach()

    #         policy_loss = -(log_probs * advantages).mean()
    #         value_loss = ((values - rewards_tensor) ** 2).mean()
    #         loss = policy_loss + value_loss - 0.01 * entropy.mean()

    #         # Backpropagation
    #         optimizer.zero_grad()
    #         loss.backward()

    #         nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)

    #         optimizer.step()

    #         steps += 1
            # if steps % SAVE_INTERVAL == 0:
            #     print(f"Step {steps}, Loss: {loss.item()}")
            #     print(f"Buffer demo ratio: {data.demo_ratio}")
            #     agent.save(f"/home/bee/development/gym-liftoff-connector/models/model_step_{steps}.zip")

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
