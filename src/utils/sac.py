import torch
from torchvision import models
import torch.nn as nn
import time
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
import torch.nn.functional as F

LOG_STD_MIN = -20
LOG_STD_MAX = 1
EPS = 1e-6  # Action clipping
TAU = 0.005  # Target update
GAMMA = 0.97  # Discount
ALPHA = 0.1    # Entropy coefficient

class ActorSAC(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            *(list(mobilenet.children())[:-1]),  # eliminar classifier
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU()
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(256 + 64, action_dim)
        self.log_std_layer = nn.Linear(256 + 64, action_dim)

    def forward(self, obs, prev_action):
        z_obs = self.encoder(obs)
        z_act = self.action_encoder(prev_action)
        z = torch.cat([z_obs, z_act], dim=-1)
        mu = self.mu_layer(z)
        log_std = torch.clamp(self.log_std_layer(z), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        return mu, std

    def sample(self, obs, prev_action):
        mu, std = self.forward(obs, prev_action)

        base_dist = Normal(mu, std)
        dist = TransformedDistribution(base_dist, [TanhTransform()])

        action = dist.rsample()
        action = torch.clamp(action, -1 + EPS, 1 - EPS)
        log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob




class CriticSAC_LSTM(nn.Module):
    def __init__(self, action_dim, hidden_size=256, lstm_layers=1, telemetry_len = 15):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            *(list(mobilenet.children())[:-1]),
            nn.Flatten(),
            nn.Linear(576, hidden_size),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=action_dim + telemetry_len,  # concatenamos acción
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )

        self.q1 = nn.Linear(hidden_size, 1)
        self.q2 = nn.Linear(hidden_size, 1)

    def forward(self, obs, action_seq, telemetry_seq):
        """
        obs_seq: [batch, seq_len, C, H, W]
        action_seq: [batch, seq_len, action_dim]
        telemetry_seq: [batch, seq_len, telemetry_len]
        """
        z_obs = self.encoder(obs)  # [batch*seq_len, hidden_size]

        lstm_input = torch.cat([action_seq, telemetry_seq], dim=-1)  # [batch, seq_len, hidden_size + action_dim + telemetry_len]

        lstm_out, _ = self.lstm(lstm_input)  # [batch, seq_len, hidden_size]
        lstm_out_last = lstm_out[:, -1, :]  # solo último step (para Q-target)

        z = torch.cat([z_obs, lstm_out_last], dim = -1)
        z = self.fc(z)

        return self.q1(z), self.q2(z)


# Todo: meter posicion y delta posicion con goal. Mirar chat

class CriticSAC_GADP(nn.Module): # GADP = Gyro, Attitude, Delta Actions, Position
    def __init__(self, action_dim, hidden_size=256, lstm_layers=1, telemetry_len = 21, distances = 6):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.telemetry_len = telemetry_len
        self.distances_len = distances
        self.encoder = nn.Sequential(
            *(list(mobilenet.children())[:-1]),
            nn.Flatten(),
            nn.Linear(576, hidden_size),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=action_dim + telemetry_len,  # concatenamos acción
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.distance_fc = nn.Sequential(
            nn.Linear(distances, 32),
            nn.SiLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(2*hidden_size + 32, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU()
        )

        self.q1 = nn.Linear(hidden_size, 1)
        self.q2 = nn.Linear(hidden_size, 1)

    def forward(self, obs, action_seq, telemetry_seq):
        """
        obs_seq: [batch, seq_len, C, H, W]
        action_seq: [batch, seq_len, action_dim]
        telemetry_seq: [batch, seq_len, telemetry_len]
        """
        z_obs = self.encoder(obs)  # [batch*seq_len, hidden_size]

        telemetry, distances = torch.split(telemetry_seq, [self.telemetry_len, self.distances_len], dim = -1)

        lstm_input = torch.cat([action_seq, telemetry], dim=-1)  # [batch, seq_len, hidden_size + action_dim + telemetry_len]

        lstm_out, _ = self.lstm(lstm_input)  # [batch, seq_len, hidden_size]
        lstm_out_last = lstm_out[:, -1, :]  # solo último step (para Q-target)

        z_pos = self.distance_fc(distances[:, -1])

        z = torch.cat([z_obs, lstm_out_last, z_pos], dim = -1)
        z = self.fc(z)

        return self.q1(z), self.q2(z)

class ActorSAC_GADP(nn.Module):
    def __init__(self, action_dim, telemetry_len = 15, positions_len = 6):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.telemetry_len = telemetry_len
        self.positions_len = positions_len
        self.encoder = nn.Sequential(
            *(list(mobilenet.children())[:-1]),  # eliminar classifier
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.SiLU()
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU()
        )
        self.tel_fc = nn.Sequential(
            nn.Linear(telemetry_len, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU()
        )
        self.pos_fc = nn.Sequential(
            nn.Linear(positions_len, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU()
        )
        self.mu_layer = nn.Linear(256 + 3*64, action_dim)
        self.log_std_layer = nn.Linear(256 + 3*64, action_dim)

    def forward(self, obs, prev_action, telemetry, position):
        z_obs = self.encoder(obs)
        z_act = self.action_encoder(prev_action)
        z_pos = self.pos_fc(position.reshape(position.shape[0], -1))
        z_tel = self.tel_fc(telemetry)
        z = torch.cat([z_obs, z_act, z_pos, z_tel], dim=-1)
        mu = self.mu_layer(z)
        log_std = torch.clamp(self.log_std_layer(z), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        return mu, std

    def sample(self, obs, prev_action, telemetry_positions):
        telemetry, position = torch.split(telemetry_positions[:, :21], [15, 6], dim = -1)
        mu, std = self.forward(obs, prev_action, telemetry, position)

        base_dist = Normal(mu, std)
        dist = TransformedDistribution(base_dist, [TanhTransform()])

        action = dist.rsample()
        action = torch.clamp(action, -1 + EPS, 1 - EPS)
        log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob


def update_sac(actor, critic, critic_target, buffer, actor_opt, critic_opt, batch_size=32, device='cuda', normalize = None):
    obs, act, rew, done, tel, next_obs, next_tel = buffer.sample(batch_size)
    obs = obs.to(device)
    act = act.to(device)
    rew = rew.to(device)
    tel = tel.to(device)
    done = done.to(device)
    next_obs = next_obs.to(device)
    next_tel = next_tel.to(device)

    if normalize is not None:
        next_obs = normalize(next_obs)
        obs = normalize(obs)

    with torch.no_grad():
        next_action, next_log_prob = actor.sample(next_obs, act[:, -1])

        next_act_seq = torch.cat([act[:, 1:], next_action.unsqueeze(1)], dim = 1)
        next_obs_seq = torch.cat([obs[:, 1:], next_obs.unsqueeze(1)], dim=1)
        next_tel_seq = torch.cat([tel[:, 1:], next_tel.unsqueeze(1)], dim=1)

        target_q1, target_q2 = critic_target(next_obs_seq, next_act_seq, next_tel_seq)
        target_q = torch.min(target_q1, target_q2) - ALPHA * next_log_prob

        done_t = done[:, -1].unsqueeze(-1)
        rew_t = rew[:, -1].unsqueeze(-1)

        target_q = rew_t + GAMMA * (1 - done_t) * target_q

    current_q1, current_q2 = critic(obs, act, tel)
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()


    new_action, log_prob = actor.sample(obs[:, -1], act[:, -2])
    next_act_seq = torch.cat([act[:, :-1], new_action.unsqueeze(1)], dim=1)
    q1_new, q2_new = critic(obs, next_act_seq, tel)
    actor_loss = (ALPHA * log_prob - torch.min(q1_new, q2_new)).mean()

    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    return critic_loss.item(), actor_loss.item()

def update_sac_n_steps(actor, critic, critic_target, buffer, actor_opt, critic_opt, batch_size=32, device='cuda', normalize = None, n_steps = 2, seq_len = 5):
    (obs, act, rew, done, tel), next_steps_len = buffer.sample(batch_size)
    obs = obs.to(device)
    act = act.to(device)
    rew = rew.to(device)
    tel = tel.to(device)
    done = done.to(device)

    t = seq_len - 1

    rewards = rew[:, t : t + n_steps]
    gammas = torch.tensor(
        [GAMMA ** i for i in range(n_steps)],
        device=device
    ).unsqueeze(0)

    rew_t = (rewards * gammas).sum(dim=1, keepdim=True)

    done_seq = done[:, t: t + n_steps]
    done_mask = done_seq.any(dim=1, keepdim=True).float()
    done_mask = done_mask.squeeze(-1)
    not_done_n = 1.0 - done_mask

    if normalize is not None:
        obs = normalize(obs)

    s_t = obs[:, t]
    s_tn = obs[:, t + n_steps]

    with torch.no_grad():
        next_action, next_log_prob = actor.sample(s_tn, act[:, t + n_steps - 1], tel[:, t + n_steps].squeeze(1))

        next_act_seq = torch.cat([act[:, -seq_len:-1], next_action.unsqueeze(1)], dim = 1)

        next_tel_seq = tel[:, -seq_len:]

        target_q1, target_q2 = critic_target(s_tn, next_act_seq, next_tel_seq)
        target_q = torch.min(target_q1, target_q2).squeeze(-1) - ALPHA * next_log_prob
        target_q = rew_t.squeeze(-1) + GAMMA**n_steps * not_done_n * target_q
        target_q = torch.clamp(target_q, -50, 50)


    current_q1, current_q2 = critic(s_t, act[:, :seq_len], tel[:, :seq_len])
    current_q1 = current_q1.squeeze(-1)
    current_q2 = current_q2.squeeze(-1)
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

    critic_opt.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    critic_opt.step()


    new_action, log_prob = actor.sample(s_t, act[:, t - 1], tel[:, t])

    act_seq = torch.cat([act[:, t - seq_len + 1:t], new_action.unsqueeze(1)], dim = 1)
    tel_seq = tel[:, t - seq_len + 1: t + 1]

    q1_new, q2_new = critic(s_t, act_seq, tel_seq)
    q1_new = q1_new.squeeze(-1)
    q2_new = q2_new.squeeze(-1)
    actor_loss = (ALPHA * log_prob - torch.min(q1_new, q2_new)).mean()

    actor_opt.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_opt.step()

    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    return critic_loss.item(), actor_loss.item()

