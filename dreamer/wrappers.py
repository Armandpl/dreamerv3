from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


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


# TODO remove tmp wrappers
class DivergedSim(gym.Wrapper):
    """Check if furuta sim has diverged, if so, terminate the episode."""

    def step(self, action):
        action = np.array([action])
        action = np.clip(action, -1, 1)
        obs, reward, terminated, truncated, info = self.env.step(action)

        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            terminated = True
            obs = np.zeros_like(obs)
            reward = 0.0

        return obs, reward, terminated, truncated, info


class HistoryWrapper(gym.Wrapper):
    """Track history of observations for given amount of steps Initial steps are zero-filled."""

    def __init__(self, env: gym.Env, steps: int, use_continuity_cost: bool):
        super().__init__(env)
        assert steps > 1, "steps must be > 1"
        self.steps = steps
        self.use_continuity_cost = use_continuity_cost

        # concat obs with action
        self.step_low = np.concatenate([self.observation_space.low, self.action_space.low])
        self.step_high = np.concatenate([self.observation_space.high, self.action_space.high])

        # stack for each step
        obs_low = np.tile(self.step_low, (self.steps, 1))
        obs_high = np.tile(self.step_high, (self.steps, 1))

        self.observation_space = Box(low=obs_low.flatten(), high=obs_high.flatten())

        self.history = self._make_history()

    def _make_history(self):
        return [np.zeros_like(self.step_low) for _ in range(self.steps)]

    def _continuity_cost(self, obs):
        # TODO compute continuity cost for all steps and average?
        # and compare smoothness between training run, and viz smoothness over time
        action = obs[-1][-1]
        last_action = obs[-2][-1]
        continuity_cost = np.power((action - last_action), 2).sum()

        return continuity_cost

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.pop(0)

        obs = np.concatenate([obs, action])
        self.history.append(obs)
        obs = np.array(self.history, dtype=np.float32)

        if self.use_continuity_cost:
            continuity_cost = self._continuity_cost(obs)
            reward -= continuity_cost
            info["continuity_cost"] = continuity_cost

        return obs.flatten(), reward, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.history = self._make_history()
        self.history.pop(0)
        obs = np.concatenate([self.env.reset()[0], np.zeros_like(self.env.action_space.low)])
        self.history.append(obs)
        return np.array(self.history, dtype=np.float32).flatten(), {}
