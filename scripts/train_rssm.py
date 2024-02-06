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

from minidream.dist import OneHotDist as OneHotCategoricalStraightThrough
from minidream.dist import TwoHotEncodingDistribution
from minidream.functional import symlog
from minidream.rssm import (
    DENSE_HIDDEN_UNITS,
    GRU_RECCURENT_UNITS,
    RSSM,
    STOCHASTIC_STATE_SIZE,
)

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
            for msg in read_protobuf_messages(filepath, log_time_order=True):
                p = msg.proto_msg
                thetas.append(p.motor_angle)
                alphas.append(p.pendulum_angle)
                actions.append(p.action)
                rewards.append(p.reward)
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
                            },
                            batch_size=[1, cfg.seq_len],
                        )
                    )
                    thetas = []
                    alphas = []
                    actions = []
                    rewards = []
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

    recon_losses = []
    kl_losses = []

    for i in trange(cfg.iterations):
        data = replay_buffer.sample(cfg.batch_size).to(device)

        # prepare tensor to hold the ouputs
        # computed during the dynamic learning phase
        # recurrent_states = torch.zeros(cfg.batch_size, cfg.seq_len, 128, device=device)

        # initialize all the tensor to collect priors and posteriors states with their associated logits
        priors_logits = torch.empty(
            cfg.batch_size, cfg.seq_len, STOCHASTIC_STATE_SIZE**2, device=device
        )
        posteriors = torch.empty(
            cfg.batch_size,
            cfg.seq_len,
            STOCHASTIC_STATE_SIZE,
            STOCHASTIC_STATE_SIZE,
            device=device,
        )
        posteriors_logits = torch.empty(
            cfg.batch_size, cfg.seq_len, STOCHASTIC_STATE_SIZE**2, device=device
        )

        reconstructed_obs = torch.empty(
            cfg.batch_size, cfg.seq_len, obs_space.shape[0], device=device
        )
        pred_rewards = torch.empty(cfg.batch_size, cfg.seq_len, 1, device=device)

        # init first recurrent state and posterior state
        ht_minus_1 = torch.zeros(cfg.batch_size, GRU_RECCURENT_UNITS, device=device)
        zt_minus_1 = torch.zeros(
            cfg.batch_size, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE, device=device
        )

        # compute recurrent states
        for i in range(cfg.seq_len):
            if i == 0:
                at_minus_1 = torch.zeros(cfg.batch_size, 1, device=device)
            else:
                at_minus_1 = data["action"][:, i]
            (
                x_hat,
                priors_logit,
                ht_minus_1,
                zt_minus_1,
                posteriors_logit,
                pred_r,
            ) = world_model.step(data["obs"][:, i], at_minus_1, ht_minus_1, zt_minus_1)
            # recurrent_states[:, i] = ht_minus_1
            reconstructed_obs[:, i] = x_hat
            posteriors[:, i] = zt_minus_1
            posteriors_logits[:, i] = posteriors_logit
            priors_logits[:, i] = priors_logit
            pred_rewards[:, i] = pred_r

        priors_logits = priors_logits.view(
            cfg.batch_size, cfg.seq_len, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE
        )
        posteriors_logits = posteriors_logits.view(
            cfg.batch_size, cfg.seq_len, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE
        )

        opt.zero_grad()

        # compute losses
        # normal distribution isn't the shit
        # recon_loss = (
        #     -Independent(
        #         torch.distributions.Normal(loc=reconstructed_obs, scale=1.0),
        #         len(reconstructed_obs.shape[2:]),
        #     )
        #     .log_prob(symlog(data["obs"]))
        #     .mean()
        # )
        distance = (reconstructed_obs - symlog(data["obs"])) ** 2
        distance = torch.where(distance < 1e-8, 0, distance)
        recon_loss = distance.sum(dim=[-1])

        # rew_loss = (
        #     -Independent(
        #         torch.distributions.Normal(loc=pred_rewards, scale=1.0),
        #         len(pred_rewards.shape[2:]),
        #     )
        #     .log_prob(data["reward"])
        #     .mean()
        # )
        rew_loss = TwoHotEncodingDistribution(pred_rewards, dims=1).log_prob(data["reward"])
        loss_pred = (recon_loss + rew_loss).mean()

        # kl loss
        free_nats = torch.tensor([1], device=device)
        loss_dyn = torch.max(
            free_nats,
            kl_divergence(
                Independent(
                    OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1
                ),
                Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1),
            ).mean(),
        )
        loss_rep = torch.max(
            free_nats,
            kl_divergence(
                Independent(  # independant is so that each batch is independant
                    OneHotCategoricalStraightThrough(logits=posteriors_logits), 1
                ),
                Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1),
            ).mean(),
        )

        loss = cfg.beta_pred * loss_pred + cfg.beta_dyn * loss_dyn + cfg.beta_rep * loss_rep
        loss.backward()
        print(loss_pred.item())
        recon_losses.append(loss_pred.detach().item())
        kl_losses.append(loss_dyn.detach().item() + loss_rep.detach().item())

        opt.step()
        # log loss(es?)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))

    ax[0].plot(kl_losses, label="KL Loss")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(recon_losses, label="Reconstruction Loss")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].set_ylim(top=0.2)

    plt.savefig("plot.jpg")
    # save model for use with play_robot.py!
    # TODO use safetensors
    torch.save(world_model.state_dict(), "../data/rssm.pth")


if __name__ == "__main__":
    main()
