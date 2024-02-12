import random  # should we use numpy or torch instead TODO

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict


class ReplayBuffer:
    def __init__(
        self, max_size: int, action_space: gym.spaces.Discrete, obs_space: gym.spaces.Box
    ):
        self.max_size = max_size
        self.actions = np.empty((max_size, 1), dtype=np.float32)
        self.obs = np.empty((max_size, obs_space.shape[0]), dtype=np.float32)
        self.rewards = np.empty((max_size, 1), dtype=np.float32)
        self.terminated = np.empty((max_size, 1), dtype=np.float32)
        self.firsts = np.empty((max_size, 1), dtype=np.float32)
        self.count = 0

    def add(self, action: np.ndarray, obs: np.ndarray, reward: float, term: bool, first: bool):
        # transition is (at_minus_1: np.ndarray, obs: np.ndarray, reward: float, terminated: bool, first:bool)
        # TODO will fail if the buffer is full, need to handle that
        self.actions[self.count] = action
        self.obs[self.count] = obs
        self.rewards[self.count] = reward
        self.terminated[self.count] = float(
            term
        )  # convert bools to floats, useful in the code later
        self.firsts[self.count] = float(first)
        self.count += 1

    def __len__(self):
        return self.count

    def sample(self, batch_size, seq_len):
        # sample batch_size trajectories of length seq_len
        trajectories = []
        for _ in range(batch_size):
            # pick a random index between 0 and len(self.buffer) - seq_len
            # TODO remove for loop, sample at once from the buffer
            idx = random.randint(0, self.count - seq_len)
            trajectory = TensorDict(
                {
                    "action": torch.tensor(self.actions[idx : idx + seq_len], dtype=torch.float32),
                    "obs": torch.tensor(self.obs[idx : idx + seq_len], dtype=torch.float32),
                    "reward": torch.tensor(self.rewards[idx : idx + seq_len], dtype=torch.float32),
                    "done": torch.tensor(
                        self.terminated[idx : idx + seq_len], dtype=torch.float32
                    ),
                    "first": torch.tensor(self.firsts[idx : idx + seq_len], dtype=torch.float32),
                },
                batch_size=[seq_len],
            )
            trajectories.append(trajectory)

        return torch.stack(trajectories, dim=0)
