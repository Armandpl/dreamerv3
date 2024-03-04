import gymnasium as gym
import numpy as np


# for some reason the obs space of cartpole has big values instead of inf?
def is_inf(a: np.ndarray):
    return np.abs(a) > 1e10


class RescaleObs(gym.Wrapper):
    """Rescale obs between -1 and 1 based on max/min value specified in the observation_space."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.high, self.low = self.env.observation_space.high, self.env.observation_space.low

        # take into account that some limits in the osbservation space can be inf
        self.high[is_inf(self.high)] = 1.0
        self.low[is_inf(self.low)] = -1.0
        self.range = self.high - self.low

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = 2 * (obs - self.low) / self.range - 1

        return obs, reward, terminated, truncated, info


class PreProcessMinatar(gym.Wrapper):
    """Transpose obs to have channel first convert from bool to float and rescale to [-1, 1]"""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        h, w, c = self.env.observation_space.shape

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(c, h, w), dtype=np.float32
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = obs.transpose(2, 0, 1)
        obs = obs.astype(np.float32) * 2 - 1

        return obs, reward, terminated, truncated, info
