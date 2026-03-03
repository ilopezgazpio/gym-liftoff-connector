import lmdb
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

from intrinsic_curiosity_based_driving.train import log_probs, previous_action


class VideoFramesDataset(Dataset):
    def __init__(self, lmdb_path, indices=None, transform=None):
        """
        lmdb_path: ruta al LMDB
        indices: lista de índices a usar (para train/dev/test split)
        transform: transformaciones de imagen (ToTensor, Resize...)
        """
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            # Todas las claves disponibles
            all_keys = [k for k, _ in txn.cursor()]
            if indices is None:
                self.keys = all_keys
            else:
                # Selecciona solo las claves indicadas
                self.keys = [all_keys[i] for i in indices]

        self.transform = transform
        self.keys = sorted(self.keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.env.begin() as txn:
            img_bytes = txn.get(key)
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        return img

class IntrinsicCuriosityDataset(Dataset):
    def __init__(self, obsertvations, actions, rewards):
        self.obsertvations = obsertvations
        self.actions = actions
        self.rewards = rewards


    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, idx):
        obs = self.obsertvations[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.obsertvations[idx + 1]
        return obs, action, reward, next_obs

class PPODataset(Dataset):
    def __init__(self, log_probs, dones, past_actions):
        self.log_probs = log_probs
        self.dones = dones
        self.past_actions = past_actions

    def __len__(self):
        return len(self.log_probs)

    def __getitem__(self, idx):
        log_prob = self.log_probs[idx]
        done = self.dones[idx]
        if idx == 0:
            prev_action = torch.zeros(4)
        else:
            prev_action = self.past_actions[idx]
        return log_prob, done, prev_action
