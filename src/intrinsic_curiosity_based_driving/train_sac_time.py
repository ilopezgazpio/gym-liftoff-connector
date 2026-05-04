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

for episode in range(last_episode, NUM_EPISODES):
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
    obs, info = env.reset()
    infos.append(info)
    done = False
    previous_action = torch.zeros((1, ACTION_DIM), dtype = torch.float32).to(device)
    step = 0
    torch.cuda.empty_cache()
    while not done:
        obs_tensor = torch.from_numpy(obs).float() / 255.0
        obs_tensor_norm = normalize(obs_tensor)
        obs_tensor_norm = obs_tensor_norm.unsqueeze(0)

        action, log_prob = actor.sample(obs_tensor_norm.to(device), previous_action)

        next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0).detach().cpu().numpy())
        done = terminated or truncated
        t = step
        data_step = {
            "img": obs_tensor.squeeze(0).to(cpu),
            "action": action.to(cpu),
            "reward": reward,
            "info": info,
            "step": t
        }

        writer.put(data_step)

        prev_actions.append(previous_action.detach().to(cpu))
        env_rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.detach().to(cpu))

        previous_action = action.detach()

        infos.append(info)

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

    intrinsic_curiosity_dataset = LMDBIntrinsicCuriosityDataset(lmdb_path=str(lmdb_path))
    intrinsic_curiosity_loader = DataLoader(intrinsic_curiosity_dataset, batch_size = BATCH_SIZE)


    ppo_dataset = PPODataset(log_probs = log_probs, dones= dones, past_actions=prev_actions, rewards=env_rewards_tensor)
    ppo_loader = DataLoader(ppo_dataset, batch_size = BATCH_SIZE)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    ensemble = [e.to(device) for e in ensemble]
    critic = critic.to(device)

    final_rewards = []
    for nsbl in ensemble:
        for p in nsbl.parameters():
            p.requires_grad = False

    for intrinsic_batch, ppo_batch in zip(intrinsic_curiosity_loader, ppo_loader):
        obs, act, reward, next_obs = [x.to(device) for x in intrinsic_batch]
        _, done, prev_action, _, _, _, _ = [x.to(device) for x in ppo_batch]


        z = encoder(obs)
        z_next = encoder(next_obs)

        reconstruct_obs = decoder(z)
        reconstruction_loss = F.mse_loss(reconstruct_obs, obs)

        z_detached = z.detach()
        # Ensemble models backprop
        ensemble_losses = []
        for nsbl in ensemble:
            pred_next_z = nsbl(z_detached, act.detach())
            loss = F.mse_loss(pred_next_z, z_next.detach())
            ensemble_losses.append(loss)

        ensemble_loss = torch.stack(ensemble_losses).mean()
        print(f"Reconstruction loss: {reconstruction_loss.item():.4f}, Ensemble loss: {ensemble_loss.item():.4f}")


        # Encoder Backprop
        encoder_opt.zero_grad()
        #(total_encoder_loss := reconstruction_loss + torch.clamp(ensemble_loss, 0, 10)).backward()
        (total_encoder_loss := reconstruction_loss + BETA*torch.clamp(ensemble_loss, 0, 5)).backward()
        encoder_opt.step()

        encoder_losses_list.append(float(total_encoder_loss.item()))
        ensemble_losses_list.append(float(ensemble_loss.item()))



        reconstruct_obs2 = decoder(encoder(obs).detach())
        decoder_loss = F.mse_loss(reconstruct_obs2, obs)
        # Decoder Backprop
        decoder_opt.zero_grad()
        decoder_loss.backward()
        decoder_opt.step()

        decoder_losses_list.append(decoder_loss.item())

        torch.cuda.empty_cache()

    for nsbl in ensemble:
        for p in nsbl.parameters():
            p.requires_grad = True

    ir_episode = torch.tensor([], dtype = torch.float32).to(device)

    obs_list = torch.tensor([], dtype = torch.float32)
    act_list = torch.tensor([], dtype = torch.float32)
    done_list = torch.tensor([], dtype = torch.float32)
    reward_list = torch.tensor([], dtype = torch.float32)
    for intrinsic_batch, ppo_batch in zip(intrinsic_curiosity_loader, ppo_loader):
        obs, act, reward, next_obs = [x.to(device) for x in intrinsic_batch]
        _, done, prev_action, _, _, _, _ = [x.to(device) for x in ppo_batch]

        with torch.no_grad():
            z = encoder(obs)
            z_next = encoder(next_obs)

        for nsbl, optimizer in zip(ensemble, ensemble_opt):

            pred = nsbl(z, act.detach())
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
            ensemble_preds.append(pred_next_z)

        ensemble_preds_stack = torch.stack(ensemble_preds)
        ir = torch.var(ensemble_preds_stack, dim=0, unbiased=False).mean(dim=1)
        ir = torch.clamp(ir, 0, 5)
        ir_episode = torch.cat([ir_episode, ir])

        obs_list = torch.cat([obs_list, obs.to(cpu)])
        act_list = torch.cat([act_list, act.to(cpu)])
        reward_list = torch.cat([reward_list, reward.to(cpu)])
        done_list = torch.cat([done_list, done.to(cpu)])

    episode_mean = ir_episode.mean()
    episode_std = ir_episode.std(unbiased=False)
    normalized_ir_episode = (ir_episode - episode_mean) / (episode_std + 1e-8)

    print(f"Intrinsic reward mean: {normalized_ir_episode.mean().item():.4f}, std: {normalized_ir_episode.std().item():.4f}")

    total_reward_tensor = reward_list + LAMBDA*normalized_ir_episode.to(cpu)
    total_reward_tensor = total_reward_tensor.detach().reshape(-1)
    done_list = done_list.reshape(-1)
    normalized_ir_episode = normalized_ir_episode.reshape(-1)
    for i in range(len(obs_list)):
        info = infos[i]
        telemetry = np.concatenate([
            info["velocity"]/20.0,
            info["gyro"]/10.0,
            info["rotation"].flatten()
        ])
        replay_data = np.concatenate([
            obs_list[i].detach().cpu().numpy().reshape(-1),
            act_list[i].detach().cpu().numpy().reshape(-1),
            telemetry,
            np.array([total_reward_tensor[i], done_list[i]], dtype=np.float32),
        ]).astype(np.float32)

        replay_buffer.add(replay_data)


    replay_buffer.writer.close()


    torch.cuda.empty_cache()
    encoder = encoder.to(cpu)
    decoder = decoder.to(cpu)
    ensemble = [e.to(cpu) for e in ensemble]
    torch.cuda.empty_cache()

    actor = actor.to(device)
    critic = critic.to(device)
    critic_target = critic_target.to(device)

    replay_buffer.writer.open()


    for _ in range(50):
        critic_loss, actor_loss = update_sac(
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
        "intrinsic_reward_mean": float(normalized_ir_episode.mean()),
        "intrinsic_reward_std": float(normalized_ir_episode.std(unbiased=False)),
        "env_reward_total": float(sum(env_rewards)),
        "total_reward_mean": float(total_reward_tensor.mean()),
        "total_reward_std": float(total_reward_tensor.std(unbiased=False)),
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
    gc.collect()
    torch.cuda.empty_cache()

    if episode % 100 == 0 and episode != 0:
        all_models = {
            "models": {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "critic_target": critic_target.state_dict(),
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
        torch.save(all_models, models_dir / f"sac_models_optimizers_time_{episode}.pth")
        save_last_episode(episode=episode)



