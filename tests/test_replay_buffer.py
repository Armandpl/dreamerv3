# import pytest
import gymnasium as gym
import numpy as np

from minidream.rb import ReplayBuffer

max_size = 100
action_space = gym.spaces.Discrete(2)
obs_space = gym.spaces.Box(low=0, high=1, shape=(4,))


def test_replay_buffer_init():
    buffer = ReplayBuffer(max_size, action_space, obs_space)

    assert buffer.max_size == max_size
    assert buffer.actions.shape == (max_size, 1)
    assert buffer.obs.shape == (max_size, obs_space.shape[0])
    assert buffer.rewards.shape == (max_size, 1)
    assert buffer.terminated.shape == (max_size, 1)
    assert buffer.firsts.shape == (max_size, 1)
    assert buffer.count == 0


def test_replay_buffer_add():
    buffer = ReplayBuffer(max_size, action_space, obs_space)

    action = np.array([1])
    obs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    reward = 0.5
    term = True
    first = False

    buffer.add(action, obs, reward, term, first)

    assert np.array_equal(buffer.actions[0], action)
    assert np.array_equal(buffer.obs[0], obs), f"buffer.obs[0]: {buffer.obs[0]}, obs: {obs}"
    assert buffer.rewards[0] == reward
    assert buffer.terminated[0] == term
    assert buffer.firsts[0] == first
    assert buffer.count == 1


def test_sample():
    buffer = ReplayBuffer(max_size, action_space, obs_space)
    for i in range(max_size):
        if i % 2 == 0:
            action = np.array([1])
            obs = np.zeros(obs_space.shape[0], dtype=np.float32)
            reward = 0.5
            term = False
            first = False
        else:
            action = np.array([0])
            obs = np.ones(obs_space.shape[0], dtype=np.float32)
            reward = 0.5
            term = False
            first = False

        buffer.add(action, obs, reward, term, first)

    data = buffer.sample(16, 64)
    sampled_actions = data["action"][0].numpy()
    sampled_obs = data["obs"][0].numpy()
    assert sampled_actions[0] != sampled_actions[1]
    assert sampled_obs[0][0] != sampled_obs[1][0]
