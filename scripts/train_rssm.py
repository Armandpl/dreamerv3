from pathlib import Path

import gymnasium
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from mcap_protobuf.reader import read_protobuf_messages
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.distributions import Independent
from torch.distributions.kl import kl_divergence
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorStorage
from tqdm import tqdm, trange
from train import train_step_world_model

from minidream.dist import OneHotDist as OneHotCategoricalStraightThrough
from minidream.dist import TwoHotEncodingDistribution
from minidream.functional import symlog
from minidream.networks import GRU_RECCURENT_UNITS, RSSM, STOCHASTIC_STATE_SIZE

# TODO use autocast fp16?


@hydra.main(version_base="1.3", config_path="configs", config_name="train_rssm.yaml")
def main(cfg: DictConfig):
    # load mcap data into replay buffer
    # TODO should we FIFO sample?
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=100_000, device="cpu"),
    )  # keep rb on cpu as it can become quite big

    # iterate over mcap files in cfg.data_dir
    filepaths = list(Path(cfg.data_dir).glob("*.mcap"))
    for filepath in tqdm(filepaths):
        try:
            # store messages into the replay buffer
            # batch them in trajectories of n steps
            thetas = []
            alphas = []
            actions = []
            rewards = []
            is_first = []
            is_first_flag = True
            for msg in read_protobuf_messages(filepath, log_time_order=True):
                p = msg.proto_msg
                thetas.append(p.motor_angle)
                alphas.append(p.pendulum_angle)
                actions.append(p.action)
                rewards.append(p.reward)
                is_first.append(is_first_flag)
                is_first_flag = False
                if len(actions) == cfg.seq_len:
                    thetas = torch.tensor(thetas)
                    alphas = torch.tensor(alphas)
                    replay_buffer.extend(
                        TensorDict(
                            {
                                "obs": torch.stack([thetas, alphas], dim=-1).unsqueeze(0),
                                "action": torch.tensor(actions)
                                .view(-1, 1)
                                .unsqueeze(0),  # at_minus_1,
                                "reward": torch.tensor(rewards).view(-1, 1).unsqueeze(0),
                                "done": torch.zeros(len(rewards)).view(-1, 1).unsqueeze(0),
                                "first": torch.tensor(is_first).view(-1, 1).unsqueeze(0),
                            },
                            batch_size=[1, cfg.seq_len],
                        )
                    )
                    thetas = []
                    alphas = []
                    actions = []
                    rewards = []
                    is_first = []
        except Exception as e:
            print(f"Error reading {filepath}")
            print(e)

    print(f"Replay buffer size: {len(replay_buffer)*cfg.seq_len} transitions")
    # sample one batch and print keys and shapes
    batch = replay_buffer.sample(cfg.batch_size)
    for k, v in batch.items():
        print(f"{k}: {v.shape}")

    # train world model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,))
    act_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,))
    world_model = RSSM(
        observation_space=obs_space,
        action_space=act_space,
    ).to(device)
    opt = torch.optim.Adam(world_model.parameters(), lr=cfg.lr, weight_decay=0.0)

    print(f"world model nb params: {sum(p.numel() for p in world_model.parameters())}")

    losses = {}

    for _ in trange(cfg.iterations):
        data = replay_buffer.sample(cfg.batch_size).to(device)
        _, _, loss_dict = train_step_world_model(data, world_model, opt)

        for k, v in loss_dict.items():
            losses[k] = losses.get(k, [])
            losses[k].append(v.detach().cpu().item())

    _, ax = plt.subplots(len(losses.keys()), 1, figsize=(8, 6))

    for idx, (k, v) in enumerate(losses.items()):
        ax[idx].plot(v, label=k)
        ax[idx].legend()

    plt.savefig("plot.jpg")
    # save model for use with play_robot.py!
    # TODO use safetensors
    torch.save(world_model.state_dict(), "../data/rssm.pth")


if __name__ == "__main__":
    main()
