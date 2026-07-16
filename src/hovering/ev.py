from gym_liftoff.envs.liftoff_env import Liftoff
from gym_liftoff.envs.liftoff_wrappers import LiftoffWrapHovering

from src.utils.sac import Critic_Hovering, Actor_Hovering, update_sac_hovering
from src.utils.ReplayBuffer import LMDBReplayBuffer
from torch.optim import Adam
from pathlib import Path
import torch
import json
import gc
import numpy as np
import pandas as pd

current_dir = Path(__file__).resolve().parent
lmdb_path = current_dir / "lmdb_episode"
replay_buffer_path = current_dir / "replay_buffer_lmdb"
info_path = current_dir / "infos"
logs_path = current_dir.parent.parent/ "logs" / "training_sac_hovering.json"

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

env = Liftoff(max_episode_time=[0, 0, 0, 30])
env = LiftoffWrapHovering(env = env)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

NUM_EPISODES = 80001
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


critic = Critic_Hovering(action_dim=ACTION_DIM, telemetry_len=21)
critic_target = Critic_Hovering(action_dim=ACTION_DIM, telemetry_len=21)
critic_target.load_state_dict(critic.state_dict())

checkpoint = None

last_episode = 0

checkpoints = sorted([
    p for p in models_dir.glob("sac_models_optimizers_position_*.pth")
])
valid_checkpoints = sorted(
    [
        p for p in checkpoints
        if int(p.stem.split("_")[-1]) % 10000 == 0 or int(p.stem.split("_")[-1]) == 75000
    ],
    key=lambda p: int(p.stem.split("_")[-1])
)
print(valid_checkpoints)


# =================
# Optimizers
# =================



# =================
# Initialize Worker and LMDB
# =================

#replay_buffer.writer.clear_database()
# =================
# Image Normalization
# =================

# =================
# Training
# =================
total_time_list = []
torch.cuda.empty_cache()
for ch_path in valid_checkpoints:
    checkpoint = torch.load(ch_path, map_location=device)
    actor = Actor_Hovering(action_dim=ACTION_DIM, telemetry_len=21)
    actor.load_state_dict(checkpoint["models"]["actor"])
    actor = actor.to(device)
    actor.eval()
    time_list = []
    print("Iniciando Modelo: ", ch_path)
    for episode in range(100):
        print(f"\n===== EPISODE {episode} =====")

        env_rewards_list = []
        actor_losses_list = []
        critic_losses_list = []
        env_rewards = []
        infos = []

        actor = actor.to(device)

        # reset del env
        _, info = env.reset()
        infos.append(info)
        done = False

        previous_action = torch.zeros((1, ACTION_DIM), dtype = torch.float32).to(device)

        step = 0

        torch.cuda.empty_cache()

        reached_goals = 0

        telemetry = np.concatenate([
                info["velocity"]/20.0,
                info["gyro"]/10.0,
                info["rotation"].flatten(),
                info["hover_position"],
                info["relative_position"]
            ])
        start_time = info["timestamp"]
        last_time = 0
        while not done:

            action, log_prob = actor.sample(previous_action, torch.tensor(telemetry, dtype = torch.float32).unsqueeze(0).to(device))

            _, reward, terminated, truncated, info = env.step(action.squeeze(0).detach().cpu().numpy())
            done = terminated or truncated
            t = step


            replay_data = np.concatenate([
                action.detach().cpu().numpy().reshape(-1),
                telemetry,
                np.array([reward, done, t], dtype=np.float32),
            ]).astype(np.float32)

            telemetry = np.concatenate([
                info["velocity"]/20.0,
                info["gyro"]/10.0,
                info["rotation"].flatten(),
                info["hover_position"],
                info["relative_position"]
            ])


            env_rewards.append(reward)

            previous_action = action.detach()

            step += 1


        last_time = info["timestamp"] - start_time
        time_list.append(last_time)


    results ={
        "episode": ch_path,
        "time_mean": np.mean(time_list),
        "time_std": np.std(time_list)
    }
    total_time_list.append(results)

df = pd.DataFrame(total_time_list)
df.to_csv("hovering_ev.csv", index=False)


