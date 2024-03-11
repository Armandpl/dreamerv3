import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict


class ReplayBuffer:
    """Simple class to store transitions.

    Transitions are stored as a flat list, assuming they all come from a single env.
    """

    def __init__(self, max_size: int, obs_space: gym.spaces.Box):
        self.max_size = max_size
        # TODO make that work for continuous action spaces as well as discrete
        self.actions = np.empty((max_size, 1), dtype=np.float32)
        self.obs = np.empty((max_size, *obs_space.shape), dtype=np.float32)
        self.rewards = np.empty((max_size, 1), dtype=np.float32)
        self.terminated = np.empty((max_size, 1), dtype=np.float32)
        self.firsts = np.empty((max_size, 1), dtype=np.float32)
        self.count = 0
        self.cursor = 0

    def add(self, action: np.ndarray, obs: np.ndarray, reward: float, term: bool, first: bool):
        """Adds a transition to the replay buffer.

        Args:
            action (np.ndarray): The action taken at the previous step.
            obs (np.ndarray): The observation at the current step.
            reward (float): The reward received at the current step.
            term (bool): Whether the episode terminated at the current step.
            first (bool): Whether the current step is the first step of a new trajectory.
        """
        self.actions[self.cursor] = action
        self.obs[self.cursor] = obs
        self.rewards[self.cursor] = reward
        self.terminated[self.cursor] = float(
            term
        )  # convert bools to floats, takes a bit more ram but removes casting from the interesting training code
        self.firsts[self.cursor] = float(first)

        # make the next step the first step of the next trajectory
        # so that when the replay buffer is full, we don't mix trajectories
        # by writing over the beginning of a trajectory
        if self.cursor < self.max_size - 1:
            self.firsts[self.cursor + 1] = float(True)

        self.cursor += 1
        if self.count < self.max_size:
            self.count += 1
        self.cursor = self.cursor % self.max_size

    def __len__(self):
        return self.count

    def sample(self, batch_size: int, seq_len: int):
        """Sample <batch_size> trajectories of length <seq_len> from the replay buffer.

        Args:
            batch_size (int): The number of trajectories to sample.
            seq_len (int): The length of each trajectory.
            sample_from_first_step (bool, optional): Whether to sample from the first step of each trajectory.
                If False, samples will be taken from random positions within each trajectory.
                Defaults to False.

        Returns:
            TensorDict: A TensorDict containing the sampled trajectories, with shape (batch_size, seq_len).
        """
        start_indices = np.random.randint(0, self.count - seq_len, size=batch_size)

        # Create an array of offsets
        offsets = np.arange(seq_len)

        # Use broadcasting to add the offsets to the indices
        indices = start_indices[:, None]
        indices = indices + offsets

        # slice the arrays
        actions = self.actions[indices]
        obs = self.obs[indices]
        rewards = self.rewards[indices]
        terminated = self.terminated[indices]
        firsts = self.firsts[indices]

        # make the dict
        # TODO swap time and dict axis
        # add arg to do that? like a batch_first arg
        trajectories = TensorDict(
            {
                "action": torch.tensor(actions, dtype=torch.float32),
                "obs": torch.tensor(obs, dtype=torch.float32),
                "reward": torch.tensor(rewards, dtype=torch.float32),
                "done": torch.tensor(terminated, dtype=torch.float32),
                "first": torch.tensor(firsts, dtype=torch.float32),
            },
            batch_size=[batch_size, seq_len],
        )

        return trajectories
