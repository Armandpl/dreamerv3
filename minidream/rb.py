import random  # should we use numpy or torch instead TODO

import numpy as np
import torch
from tensordict import TensorDict


class ReplayBuffer:
    def __init__(self):
        self.actions = []
        self.obs = []
        self.rewards = []
        self.terminated = []
        self.firsts = []

    def add(self, action: np.ndarray, obs: np.ndarray, reward: float, term: bool, first: bool):
        # transition is (at_minus_1: np.ndarray, obs: np.ndarray, reward: float, terminated: bool, first:bool)
        self.actions.append(action)
        self.obs.append(obs)
        self.rewards.append(reward)
        self.terminated.append(term)
        self.firsts.append(first)

    def __len__(self):
        return len(self.actions)

    def sample(self, batch_size, seq_len):
        # sample batch_size trajectories of length seq_len
        trajectories = []
        for _ in range(batch_size):
            # pick a random index between 0 and len(self.buffer) - seq_len
            # TODO remove for loop, sample at once from the buffer
            idx = random.randint(0, len(self.actions) - seq_len)
            trajectory = TensorDict(
                {
                    "action": torch.tensor(
                        self.actions[idx : idx + seq_len], dtype=torch.float32
                    ).unsqueeze(1),
                    # TODO setup an empty mem mapped numpy array
                    # and on add, add obs to it so that creating a tensor from a slice of it isn't super slow
                    # same for actions actually!
                    "obs": torch.tensor(self.obs[idx : idx + seq_len], dtype=torch.float32),
                    "reward": torch.tensor(
                        self.rewards[idx : idx + seq_len], dtype=torch.float32
                    ).unsqueeze(1),
                    "done": torch.tensor(
                        self.terminated[idx : idx + seq_len], dtype=torch.bool
                    ).unsqueeze(1),
                    "first": torch.tensor(
                        self.firsts[idx : idx + seq_len], dtype=torch.bool
                    ).unsqueeze(1),
                },
                batch_size=[seq_len],
            )
            trajectories.append(trajectory)

        return torch.stack(trajectories, dim=0)
