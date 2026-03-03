from gym_liftoff.envs.liftoff_env import Liftoff
from ensemble import EnsembleModel, StateAutoEncoder
from policy import Actor, Critic, compute_gae
from src.utils.datasets import IntrinsicCuriosityDataset, PPODataset
from torch.utils.data import DataLoader
import torch.optim.Adam as Adam
from pathlib import Path

current_dir = Path(__file__).resolve().parent

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
LAMBDA = 1 # weighs the intrinsic reward in the total reward
PPO_EPOCHS = 4
PPO_BATCH = 64

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
# Training
# =================

for episode in range(NUM_EPISODES):
    observations = []
    actions = []
    intrinsic_rewards = []
    env_rewards = []
    log_probs = []
    dones = []
    values = []

    # reset del env
    obs, _, _, _, _ = env.reset()
    done = False

    previous_action = torch.zeros((1, ACTION_DIM), dtype = torch.float32)

    while not done:
        # convertir obs a tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # batch 1

        # acción del policy PPO
        action, log_prob = policy.sample_action(obs_tensor.to(device), previous_action)

        # ejecutar acción
        next_obs, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
        done = terminated or truncated

        previous_action = action.unsqueeze(0)

        # almacenar info para entrenamiento posterior
        observations.append(obs_tensor)
        actions.append(action)
        intrinsic_rewards.append(None)  # placeholder, calcular después con ensemble
        env_rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)

        # actualizar obs
        obs = next_obs
    observations.append(next_obs)

    intrinsic_curiosity_dataset = IntrinsicCuriosityDataset(obsertvations=observations, rewards=env_rewards, actions=actions)
    intrinsic_curiosity_loader = DataLoader(intrinsic_curiosity_dataset, batch_size = BATCH_SIZE)


    ppo_dataset = PPODataset(log_probs = log_probs, dones= dones, past_actions=previous_action)
    ppo_loader = DataLoader(ppo_dataset, batch_size = BATCH_SIZE)

    autoencoder = autoencoder.to(device)
    ensemble = [e.to(device) for e in ensemble]
    critic = critic.to(device)

    final_rewards = torch.Tensor()
    values = torch.Tensor()

    for intrinsic_batch, ppo_batch in zip(intrinsic_curiosity_loader, ppo_loader):
        obs, act, env_reward, next_obs = intrinsic_batch.to(device)
        log_prob, done, prev_action = ppo_batch.to(device)


        z = autoencoder.encoder(obs)
        z_next = autoencoder.encoder(next_obs)

        reconstruct_obs = autoencoder.decoder(z)
        rec_loss = F.MSELoss(reconstruct_obs, obs)

        ensemble_losses = []
        ensemble_preds = []
        for nsbl in ensembles:
            pred_next_z = nsbl(z, act)
            loss = F.mse_loss(pred_next_z, z_next.detach())
            ensemble_losses.append(loss)
            ensemble_preds.append(pred_next_z)

        ensemble_loss = torch.stack(ensemble_losses).mean()

        for nsbl in ensembles:
            for p in nsbl.parameters():
                p.requires_grad = False
        # Encoder Backprop
        encoder_opt.zero_grad()
        (total_encoder_loss := reconstruction_loss + ensemble_loss).backward(retain_graph=True)
        encoder_opt.step()

        for nsbl in ensembles:
            for p in nsbl.parameters():
                p.requires_grad = True
        # Decoder Backprop
        decoder_opt.zero_grad()
        reconstruction_loss.backward(retain_graph=True)
        decoder_opt.step()

        # Ensemble models backprop
        for pred, nsbl, optimizer in zip(ensemble_preds, ensemble, ensemble_opt):
            mask = torch.bernoulli(0.8 * torch.ones(pred.size(0), device=pred.device)).bool()

            loss = F.mse_loss(pred[mask].detach(), z_next[mask].detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ir = torch.mean(torch.var(ensemble_preds, dim=1, unbiased=False), dim=1)
        ir = (ir - ir.mean()) / (ir.std() + 1e-8)
        ir = ir.unsqueeze(-1) # [batch, 1]

        # TODO: Ponderar si es necesario
        reward = env_reward + LAMBDA*ir
        value = critic(obs, prev_action)

        final_rewards = torch.cat((final_rewards, reward), 0)
        values = torch.cat((values, value), 0)
    with torch.no_grad():
        last_value = critic(observations[-1].unsqueeze(0), actions[-1].unsqueeze(0))

    advantages, returns = compute_gae(final_rewards, values, dones)

    actions = torch.cat((torch.zeros((1, 4), dtype = torch.float32), actions))
    dataset_size = len(advantages)
    clip_eps = 0.2
    for _ in range(PPO_EPOCHS):
        for start in range(0, dataset_size, PPO_BATCH):
            end = start + PPO_BATCH
            b_obs = observations[start:end]
            b_past_acts = actions[start:end]
            b_acts = actions[start+1:end+1]
            b_adv = advantages[start:end]
            b_val = values[start:end]
            b_return = returns[start:end]
            old_log_probs = log_probs[start:end]

            b_new_probs = actor.get_log_probs(b_obs, b_past_acts, b_acts)

            ratio = torch.exp(b_new_probs - old_log_probs.detach())
            surr1 = ratio*b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv

            actor_loss = -torch.mean(torch.min(surr1, surr2))

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            value_pred = critic(b_obs)
            critic_loss = F.mse_loss(value_pred, b_returns)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

    del observations[:]
    del actions[:]
    del env_rewards[:]
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
                "actor": optimizer_actor.state_dict(),
                "critic": optimizer_critic.state_dict(),
                "ensemble": [opt.state_dict() for opt in optimizer_ensemble]
            }
        }
        torch.save(all_models, models_dir / f"models_optimizers_{episode}.pth")














