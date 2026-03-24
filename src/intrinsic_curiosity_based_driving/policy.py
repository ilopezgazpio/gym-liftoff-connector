# TODO El modelo debe utilizar tanh en la salida final para que los valor no se pasen de rango [-1, 1]
import torch
from torchvision import models
import torch.nn as nn
import time
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6

class Actor(nn.Module):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            *(list(model.children())[:-1]),
            nn.Flatten(),
            nn.Linear(576, 256)
            )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.mu_layer = nn.Linear(320, action_dim)
        self.log_std_layer = nn.Linear(320, action_dim)

    def forward(self, observation: torch.Tensor, previous_action: torch.Tensor):
        z_obs = self.encoder(observation)
        z_action = self.action_encoder(previous_action)
        z = torch.cat([z_obs, z_action], dim=-1)

        mu = self.mu_layer(z)
        mu = torch.clamp(mu, -5, 5)
        log_std = self.log_std_layer(z)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        return mu, std

    def sample_action(self, observation: torch.Tensor, previous_action: torch.Tensor):


        mu, std = self.forward(observation, previous_action)

        #print("mu:", mu.min().item(), mu.max().item())
        #print("std:", std.min().item(), std.max().item())

        base_dist = Normal(mu, std)
        dist = TransformedDistribution(base_dist, [TanhTransform()])

        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob
    def get_log_probs(self, observation: torch.Tensor, previous_action: torch.Tensor, actions: torch.Tensor):
        mu, std = self.forward(observation, previous_action)

        base_dist = Normal(mu, std)
        dist = TransformedDistribution(base_dist, [TanhTransform()])

        actions = torch.clamp(actions, -1 + EPS, 1 - EPS)
        log_prob = dist.log_prob(actions).sum(-1)

        return log_prob

class Critic(nn.Module):
    def __init__(self, action_dim):
        super(Critic, self).__init__()
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.net = nn.Sequential(
            *(list(model.children())[:-1]),
            nn.Flatten(),
            nn.Linear(576, 256),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.fc = nn.Linear(320, 1)
    def forward(self, observation: torch.Tensor, previous_action: torch.Tensor):
        z_obs = self.net(observation)
        z_action = self.action_encoder(previous_action)
        z = torch.cat([z_obs, z_action.squeeze(1)], dim=-1)
        return self.fc(z).squeeze(-1)

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_adv = 0

    for t in reversed(range(T)):
        next_value = values[t+1] if t < T-1 else 0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # TODO: comprobar que la normalización se ha puesto correctamente
    return advantages, returns

