import torch
import numpy as np
import pandas as pd
from pathlib import Path
import gc

from gym_liftoff.envs.liftoff_env import Liftoff
from gym_liftoff.envs.liftoff_wrappers import (
    LiftoffWrapStability,
    LiftoffWrapContinuousAction,
    LiftoffWrapRandomPosition,
    LiftoffWrapGyro
)

from src.utils.sac import ActorSAC_GADP

# =========================
# CONFIG
# =========================

current_dir = Path(__file__).resolve().parent
N_EPISODES = 35
models_dir = current_dir / "models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = []

# =========================
# ENV FACTORY (IMPORTANTE)
# =========================

def make_env():
    env = Liftoff()
    env = LiftoffWrapContinuousAction(env)
    env = LiftoffWrapStability(env, delta_margin=False)
    env = LiftoffWrapGyro(env)
    env = LiftoffWrapRandomPosition(env)
    return env

# =========================
# CHECKPOINTS
# =========================

checkpoints = sorted([
    p for p in models_dir.glob("sac_models_optimizers_position_*.pth")
])
valid_checkpoints = []

for p in checkpoints:
    try:
        torch.load(p, map_location="cpu")
        valid_checkpoints.append(p)
    except Exception as e:
        print(f"Skipping corrupted checkpoint: {p.name}")

checkpoints = valid_checkpoints
print("Num checkpoints:", len(checkpoints))
print(checkpoints[:5])

# =========================
# EVALUATION LOOP
# =========================
env = make_env()
for ckpt_path in checkpoints:

    print(f"Evaluating {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    actor = ActorSAC_GADP(action_dim=4)
    actor.load_state_dict(checkpoint["models"]["actor"])
    actor = actor.to(device)
    actor.eval()

    episode_rewards = []

    for ep in range(N_EPISODES):


        obs, info = env.reset()

        done = False
        total_reward = 0.0

        previous_action = torch.zeros((1, 4), device=device)

        # telemetry inicial (mínimo)
        telemetry = np.zeros(27, dtype=np.float32)

        while not done:

            obs_tensor = torch.from_numpy(obs).float() / 255.0
            obs_tensor = obs_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                action, _ = actor.sample(obs_tensor, previous_action, torch.tensor(telemetry).unsqueeze(0).to(device))

            action_np = action.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action_np)

            done = terminated or truncated

            total_reward += reward
            previous_action = action.detach()

        episode_rewards.append(total_reward)

        gc.collect()

    results.append({
        "checkpoint": ckpt_path.name,
        "steps": int(str(ckpt_path).split("_")[-1].replace(".pth", "")),
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards))
    })

    del actor
    torch.cuda.empty_cache()

# =========================
# SAVE CSV
# =========================

df = pd.DataFrame(results)
df = df.sort_values("steps")

df.to_csv("training_curve_evaluation.csv", index=False)

print("Saved: training_curve_evaluation.csv")
