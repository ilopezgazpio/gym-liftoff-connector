import lmdb
import pickle
import torch
import random
from .lmdb_utils import LMDBWriter
import numpy as np
from collections import deque

class LMDBReplayBuffer:
    def __init__(self, path, obs_shape, act_size, tel_size, max_size = 40000,seq_len = 10, n_steps = 3, map_size=int(1e12), device='cuda', padding = True):
        self.seq_len = seq_len
        self.n_steps = n_steps
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


    def add(self, sample, idx = None):
        self.writer.put(sample, idx)
        self.total_added += 1
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        obs_batch = np.zeros((batch_size, self.seq_len + self.n_steps, *self.obs_shape), dtype=np.float32)
        act_batch = np.zeros((batch_size, self.seq_len + self.n_steps, self.act_size), dtype=np.float32)
        rew_batch = np.zeros((batch_size, self.seq_len + self.n_steps), dtype=np.float32)
        done_batch = np.zeros((batch_size, self.seq_len + self.n_steps), dtype=np.float32)
        tel_batch = np.zeros((batch_size, self.seq_len + self.n_steps, self.tel_size), dtype=np.float32)

        next_batch_len = []

        max_valid_idx = self.writer.idx if self.writer.idx < self.max_size else self.max_size
        with self.writer.env.begin() as txn:
            for i in range(batch_size):
                idx = random.randint(0, max_valid_idx - 1)  # -2 para asegurar next

                future_len = 0
                for t in range(self.seq_len - 1):
                    real_idx = (idx - t - 1) % max_valid_idx
                    raw = txn.get(f"{real_idx:08}".encode())

                    obs, act, rew, done, tel = self.decode(raw)
                    if done > 0:
                        break
                    #print("done: ",done)

                    obs_batch[i, -t-2] = obs
                    act_batch[i, -t-2] = act
                    rew_batch[i, -t-2] = rew
                    done_batch[i, -t-2] = done

                    tel_batch[i, -t-2] = tel


                raw = txn.get(f"{idx:08}".encode())
                obs, act, rew, done, tel = self.decode(raw)


                obs_batch[i, self.seq_len-1] = obs
                act_batch[i, self.seq_len-1] = act
                rew_batch[i, self.seq_len-1] = rew
                done_batch[i, self.seq_len-1] = done
                tel_batch[i, self.seq_len-1] = tel

                next_done = done
                if next_done < 1:
                    for j in range(self.n_steps - 1):
                        key = (idx + 1 + j) % max_valid_idx
                        raw_next = txn.get(f"{key:08}".encode())
                        next_obs, next_act, next_rew, next_done, next_tel = self.decode(raw_next)

                        obs_batch[i, j] = next_obs
                        act_batch[i, j] = next_act
                        rew_batch[i, j] = next_rew
                        done_batch[i, j] = next_done
                        tel_batch[i, j] = next_tel

                        future_len += 1
                        if next_done > 0:
                            break

                    if next_done < 1:
                        raw_next = txn.get(f"{((idx + self.n_steps) % max_valid_idx):08}".encode())
                        next_obs, next_act, next_rew, next_done, next_tel = self.decode(raw_next)
                        j = self.seq_len + self.n_steps - 1
                        obs_batch[i, j] = next_obs
                        act_batch[i, j] = next_act
                        rew_batch[i, j] = next_rew
                        done_batch[i, j] = next_done
                        tel_batch[i, j] = next_tel

                        future_len += 1

                next_batch_len.append(future_len)


        return (
            (torch.from_numpy(obs_batch), torch.from_numpy(act_batch), torch.from_numpy(rew_batch), torch.from_numpy(done_batch), torch.from_numpy(tel_batch)),
            next_batch_len
        )

    def close(self):
        self.writer.close()

    def decode(self, raw_bytes):
        data = np.frombuffer(raw_bytes, dtype=np.float32)

        obs = data[:self.obs_size].reshape(self.obs_shape)
        action = data[self.obs_size:self.obs_size + self.act_size]
        telemetry = data[self.obs_size + self.act_size:self.obs_size + self.act_size + self.tel_size]

        reward = data[-3]
        done = data[-2]

        return obs, action, reward, done, telemetry

    def propagate_reward(self, N, decay = 0.97, penalty = -3):
        closed = self.writer.closed
        if closed:
            self.writer.open()
        max_valid_idx = self.writer.idx if self.writer.idx < self.max_size else self.max_size
        for i in range(1, N+1):
            idx = (self.writer.idx - 2 - i) % max_valid_idx
            raw = self.writer.get_item(f"{idx:08}".encode())

            obs, act, rew, done, tel = self.decode(raw)

            if done > 0:
                break

            rew += penalty * (decay**i)
            rew = np.clip(rew, -5, 5)

            data = np.concatenate([obs.reshape(-1), act.reshape(-1), tel, np.array([rew, done], dtype=np.float32)])

            self.add(data, idx = idx)
        self.writer.flush()
        if closed:
            self.writer.close()
    def get_item(self, idx):
        return self.writer.get_item(idx = idx)