from gym_liftoff.envs.liftoff_env import Liftoff
from gym_liftoff.envs.liftoff_wrappers import LiftoffWrapStability, LiftoffWrapContinuousAction, LiftoffWrapRandomPosition, LiftoffWrapGyro

from src.utils.datasets import LMDBIntrinsicCuriosityDataset
from src.utils.ensemble import SmallEnsemble, StateEncoder, StateDecoder
from src.utils.sac import CriticSAC_GADP, ActorSAC_GADP, update_sac_n_steps
from src.utils.lmdb_utils import LMDBWriter
from src.utils.datasets import PPODataset
from src.utils.ReplayBuffer import LMDBReplayBuffer
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
import torch.nn.functional as F
import torch
from torchvision import transforms
import json
import lmdb
import gc
import numpy as np

current_dir = Path(__file__).resolve().parent
lmdb_path = current_dir / "lmdb_episode"
replay_buffer_path = current_dir / "replay_buffer_lmdb"
info_path = current_dir / "infos"
logs_path = current_dir.parent.parent/ "logs" / "training_sac_position_reward_logs.json"

def save_last_episode(episode:int):
    with last_episode_saving_path.open("w") as f:
        f.write(str(episode))

def read_last_episode():
    if not last_episode_saving_path.exists():
        return 0
    with last_episode_saving_path.open("r") as f:
        return int(f.read())

models_dir = current_dir / "models"
last_episode_saving_path = models_dir / "last_episode_time.txt"
models_dir.mkdir(exist_ok=True, parents=True)

env = Liftoff()
env = LiftoffWrapContinuousAction(env)
env = LiftoffWrapStability(env, delta_margin = False)
env = LiftoffWrapGyro(env)
env = LiftoffWrapRandomPosition(env)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

NUM_EPISODES = 10001
ACTION_DIM = 4
BATCH_SIZE = 32

running_mean_ir = 0.0
running_std_ir = 1.0
gamma_ir = 0.99

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

# =================
# Models
# =================


actor = ActorSAC_GADP(action_dim=ACTION_DIM)
critic = CriticSAC_GADP(action_dim=ACTION_DIM)
critic_target = CriticSAC_GADP(action_dim=ACTION_DIM)
critic_target.load_state_dict(critic.state_dict())

checkpoint = None

last_episode = 0

try:
    last_episode = read_last_episode()
    checkpoint = torch.load(models_dir / f"sac_models_optimizers_position_{last_episode}.pth")

    actor.load_state_dict(checkpoint["models"]["actor"])
    critic.load_state_dict(checkpoint["models"]["critic"])
    critic_target.load_state_dict(checkpoint["models"]["critic_target"])
except FileNotFoundError:
    pass


# =================
# Optimizers
# =================

learning_rate = 1e-4
actor_opt = Adam(actor.parameters(), lr = learning_rate)
critic_opt = Adam(critic.parameters(), lr = learning_rate)

if checkpoint:
    actor = actor.to(device)
    critic = critic.to(device)
    actor_opt.load_state_dict(checkpoint["optimizers"]["actor"])
    critic_opt.load_state_dict(checkpoint["optimizers"]["critic"])

# =================
# Initialize Worker and LMDB
# =================

replay_buffer = LMDBReplayBuffer(path = str(replay_buffer_path), obs_shape= env.observation_space.shape, act_size= ACTION_DIM, tel_size=27)
replay_buffer.writer.clear_database()
# =================
# Image Normalization
# =================

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

# =================
# Training
# =================

torch.cuda.empty_cache()

for episode in range(last_episode, NUM_EPISODES):
    print(f"\n===== EPISODE {episode} =====")

    env_rewards_list = []
    actor_losses_list = []
    critic_losses_list = []
    env_rewards = []
    infos = []

    actor = actor.to(device)

    # reset del env
    obs, info = env.reset()
    infos.append(info)
    done = False

    previous_action = torch.zeros((1, ACTION_DIM), dtype = torch.float32).to(device)
    past_distance2goal = info["distance2goal"]["vec"]

    step = 0

    torch.cuda.empty_cache()

    reached_goals = 0
    num_goals = int(episode/3000) + 1

    basic_tel = np.concatenate([
            info["velocity"]/20.0,
            info["gyro"]/10.0,
            info["rotation"].flatten()])

    pos_tel = np.concatenate([
        info["position_norm"],
        info["goal_norm"],
    ])


    telemetry = np.concatenate([
        basic_tel,
        pos_tel,
        info["distance2goal"]["vec"],
        past_distance2goal
    ])

    while not done:
        obs_tensor = torch.from_numpy(obs).float() / 255.0
        obs_tensor_norm = normalize(obs_tensor)
        obs_tensor_norm = obs_tensor_norm.unsqueeze(0)

        action, log_prob = actor.sample(obs_tensor_norm.to(device), previous_action, torch.tensor(telemetry, dtype = torch.float32).unsqueeze(0).to(device))

        next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0).detach().cpu().numpy())
        done = terminated or truncated

        t = step

        replay_data = np.concatenate([
            obs.reshape(-1),
            action.detach().cpu().numpy().reshape(-1),
            telemetry,
            np.array([reward, done, t], dtype=np.float32),
        ]).astype(np.float32)

        basic_tel = np.concatenate([
            info["velocity"]/20.0,
            info["gyro"]/10.0,
            info["rotation"].flatten()])

        pos_tel = np.concatenate([
            info["position_norm"],
            info["goal_norm"],
        ])

        telemetry = np.concatenate([
            basic_tel,
            pos_tel,
            info["distance2goal"]["vec"],
            past_distance2goal
        ])

        replay_buffer.add(replay_data)

        past_distance2goal = info["distance2goal"]["vec"]
        env_rewards.append(reward)

        previous_action = action.detach()

        infos.append(info)

        obs = next_obs

        if done:
            reached_goals += 1
            if reached_goals < num_goals and not terminated:
                env.set_new_goal()
                done = False

        step += 1

    torch.cuda.empty_cache()

    print(f"Crashed: {terminated}, Finished: {truncated}")
    replay_buffer.writer.close()
    print(f"Episode finished in {step} steps, total env reward: {sum(env_rewards):.3f}")

    critic = critic.to(device)

    actor = actor.to(device)
    critic = critic.to(device)
    critic_target = critic_target.to(device)
    replay_buffer.writer.open()
    for _ in range(50):
        critic_loss, actor_loss = update_sac_n_steps(
            actor, critic, critic_target,
            replay_buffer, actor_opt, critic_opt,
            batch_size=BATCH_SIZE, device=device, normalize=normalize
        )
        print(f"SAC update -> critic_loss: {critic_loss:.4f}, actor_loss: {actor_loss:.4f}")
        critic_losses_list.append(critic_loss)
        actor_losses_list.append(actor_loss)

    episode_log = {
        "episode": episode,
        "steps": step,
        "env_reward_total": float(sum(env_rewards_list)),
        "actor_loss_mean": float(torch.tensor(actor_losses_list).mean()),
        "actor_loss_std": float(torch.tensor(actor_losses_list).std(unbiased=False)),
        "critic_loss_mean": float(torch.tensor(critic_losses_list).mean()),
        "critic_loss_std": float(torch.tensor(critic_losses_list).std(unbiased=False))
    }

    if logs_path.exists():
        with open(logs_path, "r") as f:
            logs_data = json.load(f)
    else:
        logs_data = []

    logs_data.append(episode_log)

    with open(logs_path, "w") as f:
        json.dump(logs_data, f, indent=2)

    critic = critic.to(cpu)
    gc.collect()
    torch.cuda.empty_cache()

    if episode % 100 == 0 and episode != 0:
        all_models = {
            "models": {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "critic_target": critic_target.state_dict(),
            },
            "optimizers": {
                "actor": actor_opt.state_dict(),
                "critic": critic_opt.state_dict(),
            }
        }
        torch.save(all_models, models_dir / f"sac_models_optimizers_position_{episode}.pth")
        save_last_episode(episode=episode)