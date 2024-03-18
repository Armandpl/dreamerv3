import gymnasium as gym
import numpy as np

from dreamer.replay_buffer import ReplayBuffer

max_size = 100
obs_space = gym.spaces.Box(low=0, high=1, shape=(4,))
disc_action_space = gym.spaces.Discrete(2)


def test_replay_buffer_init():
    buffer = ReplayBuffer(max_size, obs_space, disc_action_space)

    assert buffer.max_size == max_size
    assert buffer.actions.shape == (max_size, 1)
    assert buffer.obs.shape == (max_size, obs_space.shape[0])
    assert buffer.rewards.shape == (max_size, 1)
    assert buffer.terminated.shape == (max_size, 1)
    assert buffer.firsts.shape == (max_size, 1)
    assert buffer.count == 0


def test_replay_buffer_add():
    buffer = ReplayBuffer(max_size, obs_space, disc_action_space)

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
    """Fill the replay buffer with alternating actions and obs, then sample from it and make sure
    they are still alternating and were not shuffled."""
    buffer = ReplayBuffer(max_size, obs_space, disc_action_space)
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


def test_fifo():
    """The replay buffer is FIFO, check first transitions are overwritten when buffer is full."""
    buffer = ReplayBuffer(max_size, obs_space, disc_action_space)

    # dummy action, obs and reward
    action = np.array([1])
    obs = np.zeros(obs_space.shape[0], dtype=np.float32)
    reward = 0.5

    # fill the replay buffer
    for _ in range(max_size):
        buffer.add(action, obs, reward, False, False)

    # overwrite the first 10 transitions
    for _ in range(10):
        buffer.add(action, obs, reward, True, False)

    # the 11th index of <firsts> has been swapped to True
    # to avoid mixing trajectories
    assert buffer.firsts[10] == 1.0, f"buffer.firsts[10]: {buffer.firsts[10]}"
    # assert first 10 values for terminated are 1.0
    assert np.all(
        buffer.terminated[:10] == 1.0
    ), f"buffer.terminated[:10]: {buffer.terminated[:10]}"
