import lmdb
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import pickle

from intrinsic_curiosity_based_driving.train import reward, advantages


#from ..intrinsic_curiosity_based_driving.train import log_probs, previous_action


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

class LMDBIntrinsicCuriosityDataset(Dataset):
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.keys = [key for key, _ in txn.cursor()]
    def __len__(self):
        return len(self.keys) - 1
    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data = txn.get(self.keys[idx])
            next_data = txn.get(self.keys[idx+1])
        sample = pickle.loads(data)
        next_obs = torch.tensor(pickle.loads(next_data)["img"], dtype=torch.float32)

        obs = torch.tensor(sample["img"], dtype=torch.float32)
        reward = torch.tensor(sample["reward"], dtype=torch.float32)
        action = torch.tensor(sample["action"], dtype=torch.float32)

        return obs, action, reward, next_obs


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
    def __init__(self, log_probs, dones, past_actions, rewards = None, advantages = None, values = None, returns = None):
        self.log_probs = log_probs
        self.dones = dones
        self.past_actions = past_actions
        self.rewards = rewards
        if rewards is None:
            self.rewards = [0.0]*len(log_probs)
        if advantages is None:
            self.advantages = [0.0]*len(log_probs)
        if values is None:
            self.values = [0.0]*len(log_probs)
        if returns is None:
            self.returns = [0.0]*len(log_probs)

    def __len__(self):
        return len(self.log_probs)

    def __getitem__(self, idx):
        log_prob = self.log_probs[idx]
        done = self.dones[idx]
        prev_action = self.past_actions[idx]
        reward = self.rewards[idx]
        adv = self.advantages[idx]
        val = self.values[idx]
        ret = self.returns[idx]

        return log_prob, done, prev_action, reward, adv, val, ret
