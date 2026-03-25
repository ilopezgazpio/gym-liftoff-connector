import torch
from torchvision import models
import torch.nn as nn
import time
import torch.nn.functional as F
import torch


class EnsembleModel(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(EnsembleModel, self).__init__()
        self.input_size = latent_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.input_size*4),
            nn.ReLU(),
            nn.Linear(self.input_size*4, self.input_size*8),
            nn.ReLU(),
            nn.Linear(self.input_size*8, self.input_size*8),
            nn.ReLU(),
            nn.Linear(self.input_size*8, self.input_size*4),
            nn.ReLU(),
            # TODO: Sería interesante probar p2e con dropout en vez de bootstrapping
            #       Esto añadiría ruidp de Bernouilli a los pesos. Por tanto el entrenamiento seguramente será más ruidoso
            #nn.Dropout(p=0.5),
            nn.Linear(self.input_size*4, latent_dim)
        )
    def forward(self, latent, action):
        act = action.clone().squeeze(1)
        h = torch.cat([latent, act], dim = -1)
        return self.net(h)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class StateEncoder(nn.Module):
    def __init__(self, latent_dim, normalize_latents = True):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU(),
        )
        self.res = ResBlock(512)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 8 * 8, latent_dim)
        self.normalize_latents = normalize_latents
    def forward(self, x):
        x = self.encoder(x)
        x = self.res(x)
        x = self.flatten(x)
        z = self.fc(x)
        if self.normalize_latents:
            z = (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + 1e-8)
        else:
            z = torch.tanh(z)
        return z

class StateDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512*8*8)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 3, 3, padding=1)
        )
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 8, 8)
        x = self.decoder(x)
        return x






