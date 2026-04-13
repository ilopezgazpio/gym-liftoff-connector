import lmdb
import pickle
import torch
import random
from .lmdb_utils import LMDBWriter
import numpy as np
from collections import deque

class LMDBReplayBuffer:
    def __init__(self, path, obs_shape, act_size, tel_size, max_size = 40000,seq_len = 15, map_size=int(1e12), device='cuda', padding = True):
        self.seq_len = seq_len
        self.device = device
        self.writer = LMDBWriter(lmdb_path=path, map_size= map_size, max_size = max_size, replay_buffer= True, pickle_save=False)
        self.writer.open()
        self.max_size = max_size
        self.size = self.writer.size
        self.total_added = self.size
        self.padding = padding
        self.obs_shape = obs_shape
        self.obs_size = np.prod(self.obs_shape)
        self.act_size = act_size
        self.tel_size = tel_size


    def add(self, sample):
        self.writer.put(sample)
        self.total_added += 1
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        obs_batch = np.zeros((batch_size, self.seq_len, *self.obs_shape), dtype=np.float32)
        act_batch = np.zeros((batch_size, self.seq_len, self.act_size), dtype=np.float32)
        rew_batch = np.zeros((batch_size, self.seq_len, 1), dtype=np.float32)
        done_batch = np.zeros((batch_size, self.seq_len, 1), dtype=np.float32)
        tel_batch = np.zeros((batch_size, self.seq_len, self.tel_size), dtype=np.float32)

        next_obs_batch = np.zeros((batch_size, *self.obs_shape), dtype=np.float32)
        next_tel_batch = np.zeros((batch_size, self.tel_size), dtype=np.float32)

        max_valid_idx = self.writer.idx if self.writer.idx < self.max_size else self.max_size

        with self.writer.env.begin() as txn:
            for i in range(batch_size):
                idx = random.randint(0, max_valid_idx - 1)  # -2 para asegurar next

                seq_start = idx - self.seq_len + 1

                valid_len = 0

                for t in range(self.seq_len):
                    real_idx = (seq_start + t) % self.max_size
                    raw = txn.get(f"{real_idx:08}".encode())

                    obs, act, rew, done, tel = self.decode(raw)

                    if done > 0:
                        break

                    obs_batch[i, t] = obs
                    act_batch[i, t] = act
                    rew_batch[i, t, 0] = rew
                    done_batch[i, t, 0] = done
                    tel_batch[i, t] = tel

                    valid_len += 1

                raw_next = txn.get(f"{(idx + 1) % max_valid_idx:08}".encode())

                next_obs, _, _, _, next_tel = self.decode(raw_next)
                next_obs_batch[i] = next_obs
                next_tel_batch[i] = next_tel

                # padding si hace falta
                if self.padding and valid_len < self.seq_len:
                    pad = self.seq_len - valid_len
                    obs_batch[i] = np.roll(obs_batch[i], shift=pad, axis=0)
                    act_batch[i] = np.roll(act_batch[i], shift=pad, axis=0)
                    rew_batch[i] = np.roll(rew_batch[i], shift=pad, axis=0)
                    done_batch[i] = np.roll(done_batch[i], shift=pad, axis=0)
                    tel_batch[i] = np.roll(tel_batch[i], shift=pad, axis=0)

                    obs_batch[i, :pad] = 0
                    act_batch[i, :pad] = 0
                    rew_batch[i, :pad] = 0
                    done_batch[i, :pad] = 0
                    tel_batch[i, :pad] = 0

        return (
            torch.from_numpy(obs_batch),
            torch.from_numpy(act_batch),
            torch.from_numpy(rew_batch),
            torch.from_numpy(done_batch),
            torch.from_numpy(tel_batch),
            torch.from_numpy(next_obs_batch),
            torch.from_numpy(next_tel_batch),
        )

    def close(self):
        self.writer.close()

    def decode(self, raw_bytes):
        data = np.frombuffer(raw_bytes, dtype=np.float32)

        obs = data[:self.obs_size].reshape(self.obs_shape)
        action = data[self.obs_size:self.obs_size + self.act_size]
        telemetry = data[self.obs_size + self.act_size:self.obs_size + self.act_size + self.tel_size]

        reward = data[-2]
        done = data[-1]

        return obs, action, reward, done, telemetry







def sample(self, batch_size):
    obs_seqs, action_seqs, done_seqs, telemetry_seqs, reward_seqs, next_telemetries, next_observations = [], [], [], [], [], [], []

    for _ in range(batch_size):
        obs_seq = deque(maxlen=self.seq_len)
        action_seq = deque(maxlen=self.seq_len)
        done_seq = deque(maxlen=self.seq_len)
        telemetry_seq = deque(maxlen=self.seq_len)
        reward_seq = deque(maxlen=self.seq_len)

        max_valid_idx = max_valid_idx = self.writer.idx if self.writer.idx < self.max_size else self.max_size
        idx = random.randint(0, max_valid_idx - 1)

        with self.writer.env.begin() as txn:
            for j in range(self.seq_len - 1, 0, -1):
                real_j = (idx - j) % self.max_size
                step = pickle.loads(txn.get(f"{real_j:08}".encode()))
                if step['done'] > 0:
                    break

                obs_seq.append(step['obs'])
                action_seq.append(step['action'])
                done_seq.append(step['done'])
                reward_seq.append(step['reward'])
                telemetry_raw = step["telemetry"]
                telemetry = np.concatenate([telemetry_raw["velocity"] / 20.0,
                                            telemetry_raw["gyro"] / 20.0,
                                            telemetry_raw["rotation"].flatten()])
                telemetry_seq.append(telemetry)

            step = pickle.loads(txn.get(f"{idx:08}".encode()))
            obs_seq.append(step['obs'])
            action_seq.append(step['action'])
            done_seq.append(step['done'])
            reward_seq.append(step['reward'])
            telemetry_raw = step["telemetry"]
            telemetry = np.concatenate([telemetry_raw["velocity"] / 20.0,
                                        telemetry_raw["gyro"] / 10.0,
                                        telemetry_raw["rotation"].flatten()])
            telemetry_seq.append(telemetry)

            next_idx = (idx + 1) % self.size
            next_step = pickle.loads(txn.get(f"{next_idx:08}".encode()))
            next_obs = next_step["obs"]
            next_tel_raw = next_step["telemetry"]
            next_tel = np.concatenate([next_tel_raw["velocity"] / 20.0,
                                        next_tel_raw["gyro"] / 20.0,
                                        next_tel_raw["rotation"].flatten()])

        if self.padding:
            while len(obs_seq) < self.seq_len:
                obs_seq.appendleft(np.zeros_like(obs_seq[0], dtype=np.float32))
                action_seq.appendleft(np.zeros_like(action_seq[0], dtype=np.float32))
                done_seq.appendleft(np.zeros_like(done_seq[0], dtype=np.float32))
                reward_seq.appendleft(np.zeros_like(reward_seq[0], dtype=np.float32))
                telemetry_seq.appendleft(np.zeros_like(telemetry_seq[0], dtype=np.float32))

        obs_seqs.append(torch.stack([torch.tensor(x, dtype=torch.float32) for x in obs_seq]).unsqueeze(0))
        action_seqs.append(torch.stack([torch.tensor(x, dtype=torch.float32) for x in action_seq]).unsqueeze(0))
        done_seqs.append(torch.stack([torch.tensor(x, dtype=torch.float32) for x in done_seq]).unsqueeze(0))
        reward_seqs.append(torch.stack([torch.tensor(x, dtype=torch.float32) for x in reward_seq]).unsqueeze(0))
        telemetry_seqs.append(torch.stack([torch.tensor(x, dtype=torch.float32) for x in telemetry_seq]).unsqueeze(0))

        next_telemetries.append(torch.tensor(next_tel, dtype=torch.float32).unsqueeze(0))
        next_observations.append(torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0))



    obs_batch = torch.cat(obs_seqs, dim=0)
    action_batch = torch.cat(action_seqs, dim=0)
    done_batch = torch.cat(done_seqs, dim=0)
    rewards_batch = torch.cat(reward_seqs, dim=0)
    telemetry_batch = torch.cat(telemetry_seqs, dim=0)
    next_obs_batch = torch.cat(next_observations, dim=0)
    next_tel_batch = torch.cat(next_telemetries, dim = 0)

    return obs_batch, action_batch, rewards_batch, done_batch, telemetry_batch, next_obs_batch, next_tel_batch