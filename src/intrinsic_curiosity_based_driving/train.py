from gym_liftoff.envs.liftoff_env import Liftoff

from src.utils.datasets import LMDBIntrinsicCuriosityDataset
from .ensemble import SmallEnsemble, StateEncoder, StateDecoder
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
import json
import lmdb
import gc

current_dir = Path(__file__).resolve().parent
lmdb_path = current_dir / "lmdb_episode"
info_path = current_dir / "infos"
logs_path = current_dir / "training_logs.json"

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

max_time_per_episode = [
    0, # Day
    0, # Hour
    1, # Minute
    0 # second
]

env = Liftoff(continuous_action_mode=True, max_episode_time = max_time_per_episode)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

NUM_EPISODES = 10000
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

# =================
# Models
# =================

encoder = StateEncoder(latent_dim=LATENT_DIM)
decoder = StateDecoder(latent_dim=LATENT_DIM)
actor = Actor(action_dim=ACTION_DIM)
critic = Critic(action_dim=ACTION_DIM)
ensemble = [SmallEnsemble(LATENT_DIM, ACTION_DIM) for _ in range(NUM_ENSEMBLE_MODELS)]

checkpoint = None

try:
    last_episode = read_last_episode()
    last_episode = 200
    checkpoint = torch.load(models_dir / f"models_optimizers_{last_episode}.pth")

    encoder.load_state_dict(checkpoint["models"]["encoder"])
    decoder.load_state_dict(checkpoint["models"]["decoder"])
    actor.load_state_dict(checkpoint["models"]["actor"])
    critic.load_state_dict(checkpoint["models"]["critic"])
    for model, state_dict in zip(ensemble, checkpoint["models"]["ensemble"]):
        model.load_state_dict(state_dict)
except FileNotFoundError:
    pass


# =================
# Optimizers
# =================

learning_rate = 1e-3
encoder_opt = Adam(encoder.parameters(), lr = learning_rate)
decoder_opt = Adam(decoder.parameters(), lr = learning_rate)
actor_opt = Adam(actor.parameters(), lr = 1e-4)
critic_opt = Adam(critic.parameters(), lr = 1e-4)
ensemble_opt = [Adam(ens.parameters(), lr = 1e-4) for ens in ensemble]

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
# Initialize Worker and LMDB
# =================

writer = LMDBWriter(lmdb_path=str(lmdb_path))


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

    intrinsic_rewards_list = []
    total_rewards_list = []
    env_rewards_list = []
    encoder_losses_list = []
    decoder_losses_list = []
    ensemble_losses_list = []
    actor_losses_list = []
    critic_losses_list = []

    prev_actions = []
    dones = []
    log_probs = []
    env_rewards = []
    infos = []
    actor = actor.to(device)
    writer.open()
    writer.clear_database()
    # reset del env
    obs, _ = env.reset()
    done = False

    previous_action = torch.zeros((1, ACTION_DIM), dtype = torch.float32).to(device)
    step = 0
    torch.cuda.empty_cache()
    while not done:
        # convertir obs a tensor
        obs_tensor = torch.from_numpy(obs).float() / 255.0
        obs_tensor_norm = normalize(obs_tensor)
        obs_tensor_norm = obs_tensor_norm.unsqueeze(0)

        # acción del policy PPO
        action, log_prob = actor.sample_action(obs_tensor_norm.to(device), previous_action)

        # ejecutar acción
        next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0).detach().cpu().numpy())
        done = terminated or truncated

        #print("Action:", action)
        # almacenar info para entrenamiento posterior
        t = step
        data_step = {
            "img": obs_tensor.squeeze(0).to(cpu),
            "action": action.to(cpu),
            "reward": reward,
            "step": t
        }

        writer.put(data_step)

        prev_actions.append(previous_action.detach().to(cpu))
        env_rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.detach().to(cpu))

        previous_action = action.detach()

        infos.append(info)


        # actualizar obs
        obs = next_obs
        step += 1


    last_step = {"img": obs, "step": step}
    writer.put(last_step)
    writer.close()
    del previous_action
    del reward
    del info
    del next_obs
    del obs
    torch.cuda.empty_cache()

    print(f"Episode finished in {step} steps, total env reward: {sum(env_rewards):.3f}")

    env_rewards_tensor = torch.Tensor(env_rewards)
    if env_rewards_tensor.numel() > 1:
        env_rewards_norm = (env_rewards_tensor - env_rewards_tensor.mean()) / (env_rewards_tensor.std(unbiased=False) + 1e-8)

    intrinsic_curiosity_dataset = LMDBIntrinsicCuriosityDataset(lmdb_path=str(lmdb_path))
    intrinsic_curiosity_loader = DataLoader(intrinsic_curiosity_dataset, batch_size = BATCH_SIZE)


    ppo_dataset = PPODataset(log_probs = log_probs, dones= dones, past_actions=prev_actions, rewards=env_rewards_norm)
    ppo_loader = DataLoader(ppo_dataset, batch_size = BATCH_SIZE)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    ensemble = [e.to(device) for e in ensemble]
    critic = critic.to(device)

    final_rewards = torch.Tensor().to(device)
    values = torch.Tensor().to(device)

    for intrinsic_batch, ppo_batch in zip(intrinsic_curiosity_loader, ppo_loader):
        obs, act, unnorm_reward, next_obs = [x.to(device) for x in intrinsic_batch]
        _, _, prev_action, env_reward, _, _, _ = [x.to(device) for x in ppo_batch]


        z = encoder(obs)
        z_next = encoder(next_obs)

        reconstruct_obs = decoder(z)
        reconstruction_loss = F.mse_loss(reconstruct_obs, obs)

        z_detached = z.detach()
        # Ensemble models backprop

        for nsbl, optimizer in zip(ensemble, ensemble_opt):

            pred = nsbl(z_detached, act.detach())
            mask = torch.bernoulli(0.5 * torch.ones(pred.size(0), device=pred.device)).bool()
            # print(mask.sum())
            if mask.sum() == 0:
                mask[:] = True
            # print(pred[mask])
            loss = F.mse_loss(pred[mask], z_next[mask].detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ensemble_losses = []
        ensemble_preds = []
        for nsbl in ensemble:
            pred_next_z = nsbl(z, act)
            loss = F.mse_loss(pred_next_z, z_next.detach())
            ensemble_losses.append(loss)
            ensemble_preds.append(pred_next_z)


        ensemble_loss = torch.stack(ensemble_losses).mean()
        #print(ensemble_preds)
        print(f"Reconstruction loss: {reconstruction_loss.item():.4f}, Ensemble loss: {ensemble_loss.item():.4f}")

        for nsbl in ensemble:
            for p in nsbl.parameters():
                p.requires_grad = False


        # Encoder Backprop
        encoder_opt.zero_grad()
        #(total_encoder_loss := reconstruction_loss + torch.clamp(ensemble_loss, 0, 10)).backward()
        (total_encoder_loss := reconstruction_loss + BETA*torch.clamp(ensemble_loss, 0, 5)).backward()
        encoder_opt.step()

        encoder_losses_list.append(float(total_encoder_loss.item()))
        ensemble_losses_list.append(float(ensemble_loss.item()))
        for nsbl in ensemble:
            for p in nsbl.parameters():
                p.requires_grad = True


        reconstruct_obs2 = decoder(encoder(obs).detach())
        decoder_loss = F.mse_loss(reconstruct_obs2, obs)
        # Decoder Backprop
        decoder_opt.zero_grad()
        decoder_loss.backward()
        decoder_opt.step()

        decoder_losses_list.append(decoder_loss.item())

        """
        ir = torch.mean(torch.var(ensemble_preds, dim=1, unbiased=False), dim=1)
        ir = (ir - ir.mean()) / (ir.std() + 1e-8)
        ir = ir.unsqueeze(-1) # [batch, 1]
        """

        ensemble_preds_ir = torch.stack(ensemble_preds)

        #print(ensemble_preds_ir.shape)

        ir = torch.var(ensemble_preds_ir, dim=0, unbiased=False)
        ir = ir.mean(dim=1)  # [batch]

        print(f"Intrinsic reward mean: {ir.mean().item():.4f}, std: {ir.std().item():.4f}")
        #print(f"Intrinsic Reward: ", ir)
        #print("Enviroment Reward ", env_reward)
        # TODO: Ponderar si es necesario
        batch_mean = ir.mean()
        batch_std = ir.std(unbiased=False)


        # Normalizar
        if ir.numel() > 1 and running_std_ir > 1e-8:
            running_mean_ir = gamma_ir * running_mean_ir + (1 - gamma_ir) * batch_mean.item()
            running_std_ir = gamma_ir * running_std_ir + (1 - gamma_ir) * batch_std.item()
            normalized_ir = (ir - running_mean_ir) / running_std_ir
        else:
            # Evitar NaN cuando batch es 1 o std muy pequeño
            normalized_ir = (ir - running_mean_ir) / running_std_ir

        total_reward = env_reward + LAMBDA*normalized_ir

        reward = total_reward
        value = critic(obs, prev_action)

        final_rewards = torch.cat([final_rewards, reward])
        values = torch.cat([values, value.detach()])

        total_rewards_list.extend(total_reward.detach().cpu().numpy())
        intrinsic_rewards_list.extend(normalized_ir.detach().cpu().numpy())
        env_rewards_list.extend(unnorm_reward.detach().cpu().numpy())

        torch.cuda.empty_cache()
    with torch.no_grad():
        last_value = critic(obs[-1].unsqueeze(0), act[-1].unsqueeze(0)).detach()
    advantages, returns = compute_gae(final_rewards.detach(), values.detach(), dones)

    torch.cuda.empty_cache()
    encoder = encoder.to(cpu)
    decoder = decoder.to(cpu)
    ensemble = [e.to(cpu) for e in ensemble]
    torch.cuda.empty_cache()

    clip_eps = 0.2
    ppo_dataset = PPODataset(log_probs, dones, prev_actions, rewards=final_rewards, advantages=advantages, values=values, returns=returns)
    ppo_loader = DataLoader(ppo_dataset, batch_size = PPO_BATCH)

    intrinsic_curiosity_loader = DataLoader(intrinsic_curiosity_dataset, batch_size = PPO_BATCH)
    for _ in range(PPO_EPOCHS):
        for intrinsic_batch, ppo_batch in zip(intrinsic_curiosity_loader, ppo_loader):
            b_obs, b_acts, _, _ = [x.to(device) for x in intrinsic_batch]
            old_log_probs, _, b_past_acts, _, b_adv, b_val, b_return = [x.to(device) for x in ppo_batch]

            b_obs_norm = normalize(b_obs)
            b_acts = b_acts.squeeze(1).requires_grad_(False)
            b_obs_norm = b_obs_norm.requires_grad_(False)
            b_past_acts = b_past_acts.squeeze(1)
            b_new_probs = actor.get_log_probs(b_obs_norm, b_past_acts, b_acts)

            if b_adv.numel() > 1:
                adv_std = b_adv.std(unbiased=False)
                if adv_std > 1e-8:
                    b_adv = (b_adv - b_adv.mean()) / (adv_std + 1e-8)
                else:
                    b_adv = b_adv - b_adv.mean()  # Solo centrar si no hay varianza
            else:
                b_adv = b_adv - b_adv.mean()

            ratio = torch.exp(b_new_probs - old_log_probs.detach())
            ratio = torch.clamp(ratio, 0, 10)
            surr1 = ratio*b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv

            #print("new_log_probs:", b_new_probs.min().item(), b_new_probs.max().item())
            #print("old_log_probs:", old_log_probs.min().item(), old_log_probs.max().item())
            #print("diff:", (b_new_probs - old_log_probs).min().item(), (b_new_probs - old_log_probs).max().item())

            actor_loss = -torch.mean(torch.min(surr1, surr2))

            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            actor_opt.step()
            actor_losses_list.append(float(actor_loss.item()))

            value_pred = critic(b_obs_norm, b_past_acts.detach())
            critic_loss = F.mse_loss(value_pred, b_return)
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            critic_opt.step()
            critic_losses_list.append(float(critic_loss.item()))
            print(f"PPO epoch actor_loss: {actor_loss.item():.4f}, critic_loss: {critic_loss.item():.4f}")

    episode_log = {
        "episode": episode,
        "steps": step,
        "intrinsic_reward_mean": float(torch.tensor(intrinsic_rewards_list).mean()),
        "intrinsic_reward_std": float(torch.tensor(intrinsic_rewards_list).std(unbiased=False)),
        "env_reward_total": float(sum(env_rewards_list)),
        "total_reward_mean": float(torch.tensor(total_rewards_list).mean()),
        "total_reward_std": float(torch.tensor(total_rewards_list).std(unbiased=False)),
        "encoder_loss_mean": float(torch.tensor(encoder_losses_list).mean()),
        "encoder_loss_std": float(torch.tensor(encoder_losses_list).std(unbiased=False)),
        "decoder_loss_mean": float(torch.tensor(decoder_losses_list).mean()),
        "decoder_loss_std": float(torch.tensor(decoder_losses_list).std(unbiased=False)),
        "ensemble_loss_mean": float(torch.tensor(ensemble_losses_list).mean()),
        "ensemble_loss_std": float(torch.tensor(ensemble_losses_list).std(unbiased=False)),
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
    del intrinsic_curiosity_loader
    del intrinsic_curiosity_dataset
    del final_rewards
    del dones
    del values
    gc.collect()
    torch.cuda.empty_cache()

    if episode % 100 == 0 and episode != 0:
        all_models = {
            "models": {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
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
        save_last_episode(episode=episode)

