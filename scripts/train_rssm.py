from pathlib import Path

import gymnasium
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from mcap_protobuf.reader import read_protobuf_messages
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.distributions import Independent, OneHotCategorical
from torch.distributions.kl import kl_divergence
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorStorage
from tqdm import tqdm, trange

from minidream.rssm import RSSM

# TODO use autocast fp16?


@hydra.main(version_base="1.3", config_path="configs", config_name="train_rssm.yaml")
def main(cfg: DictConfig):
    # load mcap data into replay buffer
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
            for msg in read_protobuf_messages(filepath, log_time_order=True):
                p = msg.proto_msg
                thetas.append(p.motor_angle)
                alphas.append(p.pendulum_angle)
                actions.append(p.action)
                if len(actions) == cfg.seq_len:
                    thetas = torch.tensor(thetas)
                    alphas = torch.tensor(alphas)
                    cos_thetas = torch.cos(thetas)
                    sin_thetas = torch.sin(thetas)
                    cos_alphas = torch.cos(alphas)
                    sin_alphas = torch.sin(alphas)
                    replay_buffer.extend(
                        TensorDict(
                            {
                                "obs": torch.stack(
                                    [cos_thetas, sin_thetas, cos_alphas, sin_alphas], dim=-1
                                ).unsqueeze(0),
                                "action": torch.tensor(actions).view(-1, 1).unsqueeze(0),
                            },
                            batch_size=[1, cfg.seq_len],
                        )
                    )
                    thetas = []
                    alphas = []
                    actions = []
        except:
            print(f"Error reading {filepath}")

    print(f"Replay buffer size: {len(replay_buffer)*cfg.seq_len} transitions")

    # train world model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_model = RSSM(
        observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(4,)),
        action_space=gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,)),
    ).to(device)
    opt = torch.optim.Adam(world_model.parameters(), lr=cfg.lr)

    recon_losses = []
    kl_losses = []

    for i in range(cfg.iterations):
        print(f"Training iteration {i}")
        data = replay_buffer.sample(cfg.batch_size).to(device)

        # prepare tensor to hold the ouputs
        # computed during the dynamic learning phase
        # recurrent_states = torch.zeros(cfg.batch_size, cfg.seq_len, 128, device=device)

        # initialize all the tensor to collect priors and posteriors states with their associated logits
        priors_logits = torch.empty(cfg.batch_size, cfg.seq_len, 32 * 32, device=device)
        posteriors = torch.empty(cfg.batch_size, cfg.seq_len, 32, 32, device=device)
        posteriors_logits = torch.empty(cfg.batch_size, cfg.seq_len, 32 * 32, device=device)

        reconstructed_obs = torch.empty(cfg.batch_size, cfg.seq_len, 4, device=device)

        # init first recurrent state and posterior state
        ht_minus_1 = torch.zeros(cfg.batch_size, 128)
        zt_minus_1 = torch.zeros(cfg.batch_size, 32, 32)

        # compute recurrent states
        for i in trange(cfg.seq_len):
            x_hat, _, priors_logit, ht_minus_1, zt_minus_1, posteriors_logit = world_model.step(
                data["obs"][:, i], data["action"][:, i], ht_minus_1, zt_minus_1
            )
            # recurrent_states[:, i] = ht_minus_1
            reconstructed_obs[:, i] = x_hat
            posteriors[:, i] = zt_minus_1
            posteriors_logits[:, i] = posteriors_logit
            priors_logits[:, i] = priors_logit

        priors_logits = priors_logits.view(cfg.batch_size, cfg.seq_len, 32, 32)
        posteriors_logits = posteriors_logits.view(cfg.batch_size, cfg.seq_len, 32, 32)

        opt.zero_grad()

        # compute loss
        recon_loss = (
            -Independent(
                torch.distributions.Normal(loc=reconstructed_obs, scale=1.0),
                len(reconstructed_obs.shape[2:]),
            )
            .log_prob(data["obs"])
            .mean()
        )

        # kl loss
        lhs = kl_divergence(
            Independent(OneHotCategorical(logits=posteriors_logits.detach()), 1),
            Independent(OneHotCategorical(logits=priors_logits), 1),
        ).mean()
        rhs = kl_divergence(
            Independent(
                OneHotCategorical(logits=posteriors_logits), 1
            ),  # TODO isn't this one already detached?
            Independent(OneHotCategorical(logits=priors_logits.detach()), 1),
        ).mean()
        kl_loss = cfg.kl_balancing_alpha * lhs + (1 - cfg.kl_balancing_alpha) * rhs

        loss = recon_loss + kl_loss * cfg.beta
        loss.backward()
        print(loss.item())
        recon_losses.append(recon_loss.detach().item())
        kl_losses.append(kl_loss.detach().item())

        opt.step()
        # log loss(es?)

    plt.plot(kl_losses)
    plt.show()
    plt.plot(recon_losses)
    plt.show()
    # save model for use with play_robot.py!
    # TODO use safetensors
    torch.save(world_model.state_dict(), "../data/rssm.pth")


if __name__ == "__main__":
    main()
