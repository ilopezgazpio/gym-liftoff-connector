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


actor = Actor_Hovering(action_dim=ACTION_DIM, telemetry_len=21)
critic = Critic_Hovering(action_dim=ACTION_DIM, telemetry_len=21)
critic_target = Critic_Hovering(action_dim=ACTION_DIM, telemetry_len=21)
critic_target.load_state_dict(critic.state_dict())

checkpoint = None

last_episode = 0

try:
    last_episode = read_last_episode()
    checkpoint = torch.load(models_dir / f"sac_models_optimizers_position_{last_episode}.pth")

    print("Modelos Cargados")

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

replay_buffer = LMDBReplayBuffer(path = str(replay_buffer_path), act_size= ACTION_DIM, tel_size=21, n_steps=2, seq_len=2)
#replay_buffer.writer.clear_database()
# =================
# Image Normalization
# =================

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

        replay_buffer.add(replay_data)

        env_rewards.append(reward)

        previous_action = action.detach()

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
        critic_loss, actor_loss = update_sac_hovering(
            actor, critic, critic_target,
            replay_buffer, actor_opt, critic_opt,
            batch_size=BATCH_SIZE, device=device
        )
        print(f"SAC update -> critic_loss: {critic_loss:.4f}, actor_loss: {actor_loss:.4f}")
        critic_losses_list.append(critic_loss)
        actor_losses_list.append(actor_loss)

    episode_log = {
        "episode": episode,
        "steps": step,
        "env_reward_total": float(sum(env_rewards)),
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

    if episode % 500 == 0 and episode != 0:
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