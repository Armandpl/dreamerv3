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
    """The wrapper expects to receive an atari env with frameskip=1 (no frame skipping).

    It then repeats the same action for 4 frames and take the maximum of the last two frames. The
    frame is then transposed to (c, h, w), resized to (64, 64), converted from int to float and
    rescaled to [-1, 1] The reward is accumulated over the 4 frames. The terminated and truncated
    flags are set to True if any of the 4 frames are terminated or truncated.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        _, _, c = self.env.observation_space.shape

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(c, 64, 64), dtype=np.float32
        )

    def step(self, action):
        total_rew = 0.0
        glob_terminated = False
        glob_truncated = False

        img_1, img_2 = None, None
        for i in range(4):
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_rew += reward
            glob_terminated = glob_terminated or terminated
            glob_truncated = glob_truncated or truncated
            if i == 2:
                img_1 = obs
            elif i == 3:
                img_2 = obs

        img = np.maximum(img_1, img_2)

        # TODO torch convert_image_dtype does more than / 255.0
        # do we need to worry about it?
        # using cv2 instead so that obs stays a numpy array, for consistency across envs
        obs = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        obs = obs.transpose(2, 0, 1).astype(np.float32) / 255.0
        obs = (obs * 2) - 1

        return obs, total_rew, glob_terminated, glob_truncated, {}
