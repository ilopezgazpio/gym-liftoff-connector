from gym_liftoff.envs.liftoff_env import Liftoff

from utils.datasets import LMDBIntrinsicCuriosityDataset
from .ensemble import EnsembleModel, StateAutoEncoder
from .policy import Actor, Critic, compute_gae
from .lmdb_utils import LMDBWriter
from queue import Queue
from threading import Thread
from src.utils.datasets import IntrinsicCuriosityDataset, PPODataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
import torch.nn.functional as F
import torch
from torchvision import transforms
import pickle
import lmdb

current_dir = Path(__file__).resolve().parent
lmdb_path = current_dir / "lmdb_episode_path"

def save_last_episode(episode:int):
    with last_episode_saving_path.open("w") as f:
        f.write(str(episode))

def read_last_episode():
    if not last_episode_saving_path.exists():
        return -1
    with last_episode_saving_path.open("r") as f:
        return int(f.read())

models_dir = current_dir / "models"
last_episode_saving_path = models_dir / "last_episode.txt"
models_dir.mkdir(exist_ok=True, parents=True)

env = Liftoff(continuous_action_mode=True)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

NUM_EPISODES = 10000
NUM_ENSEMBLE_MODELS = 8
LATENT_DIM = 256
ACTION_DIM = 4
BATCH_SIZE = 32
QUEUE_MAX = 500
LAMBDA = 1 # weighs the intrinsic reward in the total reward
PPO_EPOCHS = 4
PPO_BATCH = 32

# =================
# Models
# =================

autoencoder = StateAutoEncoder(latent_dim=LATENT_DIM)
actor = Actor(action_dim=ACTION_DIM)
critic = Critic(action_dim=ACTION_DIM)
ensemble = [EnsembleModel(LATENT_DIM, ACTION_DIM) for _ in range(NUM_ENSEMBLE_MODELS)]

checkpoint = None

try:
    last_episode = read_last_episode()
    checkpoint = torch.load(models_dir / f"models_optimizers_{last_episode}.pth")

    autoencoder.load_state_dict(checkpoint["models"]["autoencoder"])
    actor.load_state_dict(checkpoint["models"]["actor"])
    critic.load_state_dict(checkpoint["models"]["critic"])
    for model, state_dict in zip(ensemble, checkpoint["models"]["ensemble"]):
        model.load_state_dict(state_dict)
except FileNotFoundError:
    try:
        autoencoder.load_state_dict(torch.load(current_dir / "state_encoder.pth"))
    except FileNotFoundError:
        pass

encoder = autoencoder.encoder
decoder = autoencoder.decoder

# =================
# Optimizers
# =================

learning_rate = 1e-3
encoder_opt = Adam(encoder.parameters(), lr = learning_rate)
decoder_opt = Adam(decoder.parameters(), lr = learning_rate)
actor_opt = Adam(actor.parameters(), lr = learning_rate)
critic_opt = Adam(critic.parameters(), lr = learning_rate)
ensemble_opt = [Adam(ens.parameters(), lr = learning_rate) for ens in ensemble]

if checkpoint:
    encoder_opt.load_state_dict(checkpoint["optimizers"]["encoder"])
    decoder_opt.load_state_dict(checkpoint["optimizers"]["decoder"])
    actor_opt.load_state_dict(checkpoint["optimizers"]["actor"])
    critic_opt.load_state_dict(checkpoint["optimizers"]["critic"])
    for opt, state in zip(encoder_opt, checkpoint["optimizers"]["ensemble"]):
        opt.load_state_dict(state)

# =================
# Devices
# =================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")


# =================
# Queue thread-safe
# =================
step_queue = Queue(maxsize=QUEUE_MAX)
ppo_queue = Queue(maxsize=QUEUE_MAX)

# =================
# Initialize Worker and LMDB
# =================

writer = LMDBWriter()
writer_thread.start()

env = lmdb.open(lmdb_path, map_size=40 * 1024**3)  # 40 GB memory allocation

# =================
# Image Normalization
# =================

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

# =================
# Training
# =================

torch.cuda.empty_cache()

for episode in range(NUM_EPISODES):
    print(f"\n===== EPISODE {episode} =====")
    prev_actions = []
    dones = []
    log_probs = []
    env_rewards = []
    infos = []
    actor = actor.to(device)

    # reset del env
    obs, _ = env.reset()
    done = False

    previous_action = torch.zeros((1, ACTION_DIM), dtype = torch.float32).to(device)
    step = 0
    while not done:
        # convertir obs a tensor
        obs_tensor = torch.from_numpy(obs).float() / 255.0
        obs_tensor = normalize(obs_tensor)
        obs_tensor = obs_tensor.unsqueeze(0)
        #obs_tensor = obs_tensor.permute(2,0,1).unsqueeze(0) # H W C -> 1 C H W
        # #obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) # batch 1

        # acción del policy PPO
        action, log_prob = actor.sample_action(obs_tensor.to(device), previous_action)

        # ejecutar acción
        next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0).detach().cpu().numpy())
        done = terminated or truncated

        #print("Action:", action)
        # almacenar info para entrenamiento posterior
        step = {
            "img": obs.cpu(),
            "action": action.cpu(),
            "reward": reward.cpu(),
            "step": step
        }

        prev_actions.append(previous_action.to(cpu))
        env_rewards.append(reward.to(cpu))
        dones.append(dones)
        log_probs.append(log_prob.to(cpu))

        previous_action = action

        infos.append(info)

        # actualizar obs
        obs = next_obs
        step += 1

    for j, i in enumerate(infos):
        print("Step:", j)
        print("velocity:", i["velocity"])
        print("position", i["position"])
        print("timestamp", i["timestamp"])

    last_step = {"img": obs}


    print(f"Episode finished in {step} steps, total env reward: {sum(env_rewards):.3f}")

    intrinsic_curiosity_dataset = LMDBIntrinsicCuriosityDataset(lmdb_path=lmdb_path)
    intrinsic_curiosity_loader = DataLoader(intrinsic_curiosity_dataset, batch_size = BATCH_SIZE)


    ppo_dataset = PPODataset(log_probs = log_probs, dones= dones, past_actions=previous_action)
    ppo_loader = DataLoader(ppo_dataset, batch_size = BATCH_SIZE)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    ensemble = [e.to(device) for e in ensemble]
    critic = critic.to(device)

    final_rewards = []
    values = []

    for intrinsic_batch, ppo_batch in zip(intrinsic_curiosity_loader, ppo_loader):
        obs, act, env_reward, next_obs = intrinsic_batch.to(device)
        _, _, prev_action, _, _, _, _ = ppo_batch.to(device)


        z = autoencoder.encoder(obs)
        z_next = autoencoder.encoder(next_obs)

        reconstruct_obs = autoencoder.decoder(z)
        reconstruction_loss = F.MSELoss(reconstruct_obs, obs)

        ensemble_losses = []
        ensemble_preds = []
        for nsbl in ensemble:
            pred_next_z = nsbl(z, act)
            loss = F.mse_loss(pred_next_z, z_next.detach())
            ensemble_losses.append(loss)
            ensemble_preds.append(pred_next_z)

        ensemble_loss = torch.stack(ensemble_losses).mean()

        print(f"Reconstruction loss: {reconstruction_loss.item():.4f}, Ensemble loss: {ensemble_loss.item():.4f}")

        for nsbl in ensemble:
            for p in nsbl.parameters():
                p.requires_grad = False
        # Encoder Backprop
        encoder_opt.zero_grad()
        (total_encoder_loss := reconstruction_loss + ensemble_loss).backward(retain_graph=True)
        encoder_opt.step()

        for nsbl in ensemble:
            for p in nsbl.parameters():
                p.requires_grad = True
        # Decoder Backprop
        decoder_opt.zero_grad()
        reconstruction_loss.backward(retain_graph=True)
        decoder_opt.step()

        # Ensemble models backprop
        for pred, nsbl, optimizer in zip(ensemble_preds, ensemble, ensemble_opt):
            mask = torch.bernoulli(0.8 * torch.ones(pred.size(0), device=pred.device)).bool()

            loss = F.mse_loss(pred[mask], z_next[mask].detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        """
        ir = torch.mean(torch.var(ensemble_preds, dim=1, unbiased=False), dim=1)
        ir = (ir - ir.mean()) / (ir.std() + 1e-8)
        ir = ir.unsqueeze(-1) # [batch, 1]
        """
        ir = torch.var(ensemble_preds, dim=0, unbiased=False)
        ir = ir.mean(dim=1)  # [batch]
        print(f"Intrinsic reward mean: {ir.mean().item():.4f}, std: {ir.std().item():.4f}")
        # TODO: Ponderar si es necesario
        reward = env_reward + LAMBDA*ir
        value = critic(obs, prev_action)

        final_rewards.append(reward.to(cpu))
        values.append(value.to(cpu))
    with torch.no_grad():
        last_value = critic(obs[-1].unsqueeze(0), act[-1].unsqueeze(0))

    advantages, returns = compute_gae(final_rewards, values, dones)

    #actions = torch.cat((torch.zeros((1, 4), dtype = torch.float32), actions))
    clip_eps = 0.2

    ppo_dataset = PPODataset(log_probs, dones, prev_actions, final_rewards, advantages, values, returns)
    ppo_loader = DataLoader(ppo_dataset, batch_size = PPO_BATCH)

    intrinsic_curiosity_loader = DataLoader(intrinsic_curiosity_dataset, batch_size = PPO_BATCH)

    for _ in range(PPO_EPOCHS):
        for intrinsic_batch, ppo_batch in zip(intrinsic_curiosity_loader, ppo_loader):
            b_obs, b_acts, _, _ = intrinsic_batch.to(device)
            old_log_probs, _, b_past_acts, _, b_adv, b_val, b_return = ppo_batch.to(device)


            b_new_probs = actor.get_log_probs(b_obs, b_past_acts, b_acts)

            ratio = torch.exp(b_new_probs - old_log_probs.detach())
            surr1 = ratio*b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv

            actor_loss = -torch.mean(torch.min(surr1, surr2))

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            value_pred = critic(b_obs)
            critic_loss = F.mse_loss(value_pred, b_return)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            print(f"PPO epoch actor_loss: {actor_loss.item():.4f}, critic_loss: {critic_loss.item():.4f}")
            
    del final_rewards[:]
    del dones[:]
    del values[:]

    if episode % 100 == 0:
        all_models = {
            "models": {
                "autoencoder": autoencoder.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "ensemble": [m.state_dict() for m in ensemble]
            },
            "optimizers": {
                "encoder": encoder_opt.state_dict(),
                "decoder": decoder_opt.state_dict(),
                "actor": actor_opt.state_dict(),
                "critic": critic_opt.state_dict(),
                "ensemble": [opt.state_dict() for opt in ensemble_opt]
            }
        }
        torch.save(all_models, models_dir / f"models_optimizers_{episode}.pth")














