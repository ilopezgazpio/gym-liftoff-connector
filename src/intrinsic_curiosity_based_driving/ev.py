from gym_liftoff.envs.liftoff_env import Liftoff
from gym_liftoff.envs.liftoff_wrappers import LiftoffWrapStability, LiftoffWrapContinuousAction, LiftoffWrapConstantTime

from src.utils.datasets import LMDBIntrinsicCuriosityDataset
from src.utils.ensemble import SmallEnsemble, StateEncoder, StateDecoder
from src.utils.sac import ActorSAC, CriticSAC_LSTM, update_sac
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
logs_path = current_dir / "training_sac_time_reward_logs.json"

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

max_time_per_episode = [
    0, # Day
    0, # Hour
    1, # Minute
    0 # second
]

env = Liftoff(max_episode_time = max_time_per_episode)
env = LiftoffWrapContinuousAction(env)
env = LiftoffWrapStability(env)
env = LiftoffWrapConstantTime(env)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

NUM_EPISODES = 20001
NUM_ENSEMBLE_MODELS = 8
LATENT_DIM = 64
ACTION_DIM = 4
BATCH_SIZE = 32
QUEUE_MAX = 500 # Maximum numbers of elements in the queue for inserting in lmdb
LAMBDA = 0.6 # weighs the intrinsic reward in the total reward
BETA = 0.4 # weights the ensemble loss in the encoder
PPO_EPOCHS = 4
PPO_BATCH = 32

running_mean_ir = 0.0
running_std_ir = 1.0
gamma_ir = 0.99

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

# =================
# Models
# =================

encoder = StateEncoder(latent_dim=LATENT_DIM)
decoder = StateDecoder(latent_dim=LATENT_DIM)
actor = ActorSAC(action_dim=ACTION_DIM)
critic = CriticSAC_LSTM(action_dim=ACTION_DIM)
critic_target = CriticSAC_LSTM(action_dim=ACTION_DIM)
critic_target.load_state_dict(critic.state_dict())
ensemble = [SmallEnsemble(LATENT_DIM, ACTION_DIM) for _ in range(NUM_ENSEMBLE_MODELS)]

checkpoint = None

last_episode = 0

try:
    last_episode = read_last_episode()
    checkpoint = torch.load(models_dir / f"sac_models_optimizers_time_{last_episode}.pth")

    encoder.load_state_dict(checkpoint["models"]["encoder"])
    decoder.load_state_dict(checkpoint["models"]["decoder"])
    actor.load_state_dict(checkpoint["models"]["actor"])
    critic.load_state_dict(checkpoint["models"]["critic"])
    critic_target.load_state_dict(checkpoint["models"]["critic_target"])
    for model, state_dict in zip(ensemble, checkpoint["models"]["ensemble"]):
        model.load_state_dict(state_dict)
except FileNotFoundError:
    pass


# =================
# Optimizers
# =================

learning_rate = 1e-4
encoder_opt = Adam(encoder.parameters(), lr = learning_rate)
decoder_opt = Adam(decoder.parameters(), lr = learning_rate)
actor_opt = Adam(actor.parameters(), lr = 1e-4)
critic_opt = Adam(critic.parameters(), lr = 1e-4)
ensemble_opt = [Adam(ens.parameters(), lr = 1e-4) for ens in ensemble]

if checkpoint:
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    actor = actor.to(device)
    critic = critic.to(device)
    ensemble = [e.to(device) for e in ensemble]
    encoder_opt.load_state_dict(checkpoint["optimizers"]["encoder"])
    decoder_opt.load_state_dict(checkpoint["optimizers"]["decoder"])
    actor_opt.load_state_dict(checkpoint["optimizers"]["actor"])
    critic_opt.load_state_dict(checkpoint["optimizers"]["critic"])
    for opt, state in zip(ensemble_opt, checkpoint["optimizers"]["ensemble"]):
        opt.load_state_dict(state)


# =================
# Devices
# =================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

# =================
# Initialize Worker and LMDB
# =================

writer = LMDBWriter(lmdb_path=str(lmdb_path))
replay_buffer = LMDBReplayBuffer(path = str(replay_buffer_path), obs_shape= env.observation_space.shape, act_size= ACTION_DIM, tel_size=15)
#replay_buffer.writer.clear_database()

# =================
# Image Normalization
# =================

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

# =================
# Training
# =================

torch.cuda.empty_cache()

import matplotlib.pyplot as plt

# Almacena la trayectoria de todos los episodios
all_trajectories = []

for episode in range(80):
    print(f"\n===== EPISODE {episode} =====")
    obs, info = env.reset()
    previous_action = torch.zeros((1, ACTION_DIM), dtype=torch.float32).to(device)

    # Guardamos la posición inicial
    trajectory = [info["position"]]

    done = False
    step = 0

    while not done:
        obs_tensor = torch.from_numpy(obs).float() / 255.0
        obs_tensor_norm = normalize(obs_tensor)
        obs_tensor_norm = obs_tensor_norm.unsqueeze(0)
        action, log_prob = actor.sample(obs_tensor_norm.to(device), previous_action)

        next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0).detach().cpu().numpy())


        done = terminated or truncated

        # Guardar posición
        trajectory.append(info["position"])

        previous_action = action.detach()

        obs = next_obs
        step += 1

    # Guardar la trayectoria del episodio
    all_trajectories.append({
        "episode": episode,
        "trajectory": trajectory,
        "steps": len(trajectory)
    })


top3 = sorted(all_trajectories,
              key=lambda x: x["steps"],
              reverse=True)[:3]

import pickle

with open("top3_trajectories.pkl", "wb") as f:
    pickle.dump(top3, f)

plt.figure(figsize=(8, 8))

colors = ["#79A3D9", "#0B78B3", "#1F0062"]

for i, (color, ep) in enumerate(zip(colors, top3)):

    trajectory = ep["trajectory"]

    x = [p[0] for p in trajectory]
    y = [p[2] for p in trajectory]

    plt.plot(
        x,
        y,
        linewidth=2,
        color=color,
        label=f'Episodio {i}'
    )

    # Inicio
    plt.scatter(
        x[0], y[0],
        color=color,
        marker='o',
        s=70
    )

    # Fin
    plt.scatter(
        x[-1], y[-1],
        color=color,
        marker='X',
        s=100
    )

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trayectorias de 3 episodios")
plt.grid(True, linestyle="--", alpha=0.5)
plt.axis("equal")
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig("top3_trajectories.pdf", bbox_inches="tight")
plt.savefig("top3_trajectories.png", dpi=300)
plt.show()



