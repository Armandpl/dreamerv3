import cv2
import gymnasium as gym
import numpy as np


class PreProcessMinatar(gym.Wrapper):
    """Transpose obs to have channel first, convert from bool to float and rescale to [-1, 1]"""

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


class PreProcessAtari(gym.Wrapper):
    """Transpose obs to have channel first, resize, convert from int to float and rescale to [-1,
    1]"""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        _, _, c = self.env.observation_space.shape

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(c, 64, 64), dtype=np.float32
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # TODO torch convert_image_dtype does more than / 255.0
        # do we need to worry about it?
        # using cv2 instead so that obs stays a numpy array, for consistency across envs
        obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)
        obs = obs.transpose(2, 0, 1).astype(np.float32) / 255.0
        obs = (obs * 2) - 1

        return obs, reward, terminated, truncated, info
