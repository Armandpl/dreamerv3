from pathlib import Path

import gymnasium
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.distributions import Independent
from torch.distributions.kl import kl_divergence
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorStorage
from tqdm import tqdm, trange

from minidream.dist import OneHotDist as OneHotCategoricalStraightThrough
from minidream.dist import TwoHotEncodingDistribution
from minidream.functional import symlog
from minidream.networks import (
    DENSE_HIDDEN_UNITS,
    GRU_RECCURENT_UNITS,
    RSSM,
    STOCHASTIC_STATE_SIZE,
    PredModel,
)

# TODO use autocast fp16?
GAMMA = 0.997
BETA_PRED = 1.0
BETA_DYN = 0.5
BETA_REP = 0.1
MIN_SYMLOG_DISTANCE = 1e-8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_step_world_model(data, world_model, optim):
    """Train the world model for one step data contains a batch of replay transitions of shape (B,
    T, ...) it contains past actions, observations, rewards and dones."""
    batch_size, seq_len = data["obs"].shape[0:2]
    # initialize tensors to collect priors and posteriors states with their associated logits
    priors_logits = torch.empty(batch_size, seq_len, STOCHASTIC_STATE_SIZE**2, device=device)
    posteriors_logits = torch.empty(batch_size, seq_len, STOCHASTIC_STATE_SIZE**2, device=device)
    posteriors = torch.empty(  # sampled posteriors
        batch_size,
        seq_len,
        STOCHASTIC_STATE_SIZE,
        STOCHASTIC_STATE_SIZE,
        device=device,
    )

    reconstructed_obs = torch.empty(
        batch_size, seq_len, world_model.observation_space.shape[0], device=device
    )
    pred_rewards = torch.empty(batch_size, seq_len, 1, device=device)
    pred_continues = torch.empty(batch_size, seq_len, 1, device=device)

    # init first recurrent state and posterior state
    ht_minus_1 = torch.zeros(batch_size, GRU_RECCURENT_UNITS, device=device)
    zt_minus_1 = torch.zeros(
        batch_size, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE, device=device
    )

    # compute recurrent states
    # TODO do the preds (reward, continue and reconstruction) outside the loop bc we can and its faster
    for i in range(seq_len):
        # don't train first step TODO is that doing the right thing?
        # TODO should we use a mask instead?
        if i == 0:
            at_minus_1 = torch.zeros(batch_size, 1, device=device)
        else:
            at_minus_1 = data["action"][:, i]

        (
            x_hat,
            priors_logit,
            ht_minus_1,
            zt_minus_1,
            posteriors_logit,
            pred_r,
            pred_c,
        ) = world_model.forward(data["obs"][:, i], at_minus_1, ht_minus_1, zt_minus_1)

        # TODO make this one line somehow
        reconstructed_obs[:, i] = x_hat
        posteriors[:, i] = zt_minus_1
        posteriors_logits[:, i] = posteriors_logit
        priors_logits[:, i] = priors_logit
        pred_rewards[:, i] = pred_r
        pred_continues[:, i] = pred_c

    priors_logits = priors_logits.view(
        batch_size, seq_len, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE
    )
    posteriors_logits = posteriors_logits.view(
        batch_size, seq_len, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE
    )

    optim.zero_grad()

    # compute losses
    # TODO use the validate args arg!! get it from cfg or smth
    # TODO fold that into SymLog loss class? though it's gonna make it less readable maybe?
    distance = (reconstructed_obs - symlog(data["obs"])) ** 2
    distance = torch.where(distance < MIN_SYMLOG_DISTANCE, 0, distance)
    recon_loss = distance.sum(dim=[-1])

    rew_loss = TwoHotEncodingDistribution(pred_rewards, dims=1).log_prob(
        data["reward"]
    )  # sum over the last dim
    continue_loss = -Independent(
        torch.distributions.Bernoulli(logits=pred_continues),
        1,
    ).log_prob(data["done"])
    loss_pred = (recon_loss + rew_loss + continue_loss).mean()  # average accross batch and time

    # kl loss
    # TODO fold that into a KL loss func/class?
    free_nats = torch.tensor([1], device=device)
    loss_dyn = torch.max(
        free_nats,
        kl_divergence(
            Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1),
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

    # TODO log metrics
    # if logger is not None:
    # logger.log({}) # TODO something, maybe the logger, should keep track of the global step
    loss = BETA_PRED * loss_pred + BETA_DYN * loss_dyn + BETA_REP * loss_rep
    loss.backward()

    optim.step()
    return loss_pred, (loss_dyn + loss_rep)


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # setup logger
    # TODO

    # setup env
    env = gymnasium.make("CartPole-v1")
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.max_buffer_size, device="cpu"),
    )  # keep rb on cpu as it can become quite big

    # setup models and opts
    world_model = RSSM(
        observation_space=env.observation_space,
        action_space=env.action_space,
    ).to(device)
    actor, critic = PredModel(env.action_space.shape[0]).to(device), PredModel(1).to(device)

    wm_opt = torch.optim.Adam(world_model.parameters(), lr=cfg.lr, weight_decay=0.0)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr, weight_decay=0.0)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr, weight_decay=0.0)

    # train
    # for i in range(cfg.iterations):
    # collect transitions
    # train world model
    # imagine traj and train actor critic

    # replay_buffer.extend(
    #     TensorDict(
    #         {
    #             "obs": torch.stack([thetas, alphas], dim=-1).unsqueeze(0),
    #             "action": torch.tensor(actions)
    #             .view(-1, 1)
    #             .unsqueeze(0),  # at_minus_1,
    #             "reward": torch.tensor(rewards).view(-1, 1).unsqueeze(0),
    #         },
    #         batch_size=[1, cfg.seq_len],
    #     )
    # )

    # print(f"Replay buffer size: {len(replay_buffer)*cfg.seq_len} transitions")
    # # sample one batch and print keys and shapes
    # batch = replay_buffer.sample(cfg.batch_size)
    # for k, v in batch.items():
    #     print(f"{k}: {v.shape}")

    #     data = replay_buffer.sample(cfg.batch_size).to(device)


if __name__ == "__main__":
    main()
