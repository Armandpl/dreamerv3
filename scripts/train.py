import copy
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
from tqdm import tqdm, trange

from minidream.dist import OneHotDist as OneHotCategoricalStraightThrough
from minidream.dist import TwoHotEncodingDistribution
from minidream.functional import symlog
from minidream.networks import (
    DENSE_HIDDEN_UNITS,
    GRU_RECCURENT_UNITS,
    RSSM,
    STOCHASTIC_STATE_SIZE,
    Actor,
    Critic,
)
from minidream.rb import ReplayBuffer

# TODO use autocast fp16?

# These are the SACRED hyper-parameters
# Change them at your OWN RISK (don't)
# Actor Critic HPs
GAMMA = 0.997  # == 1 - 1/333
IMAGINE_HORIZON = 15
RETURN_LAMBDA = 0.95
ACTOR_ENTROPY = 3e-4
ACTOR_CRITIC_LR = 3e-4
ADAM_EPSILON = 1e-5
GRADIENT_CLIP = 100.0

# Losses
BETA_PRED = 1.0
BETA_DYN = 0.5
BETA_REP = 0.1
MIN_SYMLOG_DISTANCE = 1e-8

# World Model
WM_LR = 1e-4
WM_ADAM_EPSILON = 1e-8


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO use the sg() notation instead of detach??
def sg(x):
    return x.detach()


def train_step_world_model(data, world_model, optim):
    """Train the world model for one step data contains a batch of replay transitions of shape (B,
    T, ...) it contains past actions, observations, rewards and dones.

    dones = terminated, not terminated or truncated
    """
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
    recurrent_states = torch.empty(  # TODO line up the names w/ dreamer v3 paper
        batch_size,
        seq_len,
        GRU_RECCURENT_UNITS,
        device=device,
    )

    reconstructed_obs = torch.empty(
        batch_size, seq_len, world_model.observation_space.shape[0], device=device
    )
    pred_rewards = torch.empty(batch_size, seq_len, 255, device=device)
    pred_continues = torch.empty(batch_size, seq_len, 1, device=device)

    # init first recurrent state and posterior state
    ht_minus_1 = torch.zeros(batch_size, GRU_RECCURENT_UNITS, device=device)
    zt_minus_1 = torch.zeros(
        batch_size, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE, device=device
    )

    # compute recurrent states
    # TODO do the preds (reward, continue and reconstruction) outside the loop bc we can and its faster
    for i in range(seq_len):
        # don't train first step? TODO is that doing the right thing?
        # TODO should we use a mask instead?
        # if i == 0:
        #     at_minus_1 = torch.zeros(batch_size, world_model.action_space.n, device=device)
        # else:
        at_minus_1 = data["action"][:, i]

        (
            x_hat,
            priors_logit,
            ht_minus_1,
            zt_minus_1,
            posteriors_logit,
            pred_r,
            pred_c,
        ) = world_model.forward(
            data["obs"][:, i], at_minus_1, ht_minus_1, zt_minus_1, is_first=data["first"][:, i]
        )

        # TODO make this one line somehow
        # maybe fill the tensor directly?
        reconstructed_obs[:, i] = x_hat
        posteriors[:, i] = zt_minus_1
        recurrent_states[:, i] = ht_minus_1
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

    # TODO how is that working?? we should have more logits at the output of the reward net
    # also it needs to be initialized with zero weights for the output layer i think?
    rew_loss = -TwoHotEncodingDistribution(pred_rewards, dims=1).log_prob(
        data["reward"]
    )  # sum over the last dim
    continue_loss = -Independent(
        torch.distributions.Bernoulli(logits=pred_continues),
        1,
    ).log_prob(
        1.0 - data["done"].float()
    )  # TODO do we need to multiply that by 1.0?
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

    loss = BETA_PRED * loss_pred + BETA_DYN * loss_dyn + BETA_REP * loss_rep
    loss.backward()

    optim.step()
    return (
        recurrent_states,
        posteriors,
        {
            "loss/total": loss,
            "loss/pred": loss_pred,
            "loss/dyn": loss_dyn,
            "loss/rep": loss_rep,
            "loss/recon": recon_loss.mean(),
            "loss/reward": rew_loss.mean(),
            "loss/continue": continue_loss.mean(),
        },
    )


def compute_lambda_returns(rewards, values, continues):
    """Values is the output from the critic rewards are the predicted rewards continues are the
    predicted continue flags."""
    # Eq. 7
    # lambda return shape should be (B, T, 1)
    # pred_rewards is (B, T, 1)
    # pred_values is (B, T, 1)
    # continues is (B, T, 1)
    # where T is IMAGINE_HORIZON + 1
    batch_size = rewards.shape[0]
    lambda_returns = torch.empty(batch_size, IMAGINE_HORIZON, 1, device=device)

    # we compute the equation by developping it
    # eq. 7 Rt = rt + GAMMA*Ct ( (1 - LAMBDA) * Vt+1 + LAMBDA * Rt+1)
    # we first compute rt + GAMMA*Ct * ((1 - LAMBDA) * Vt+1)
    # then we go from the left to the right of the list/time dimension
    # and we compute Rt = interm[t] + Ct * GAMMA * LAMBDA * Rt+1
    # w/ the last RT+1 being the last nn values estimated
    # rt is the reward at t, Vt the output of the critic at t, Ct the output of the continue network at t
    # TODO should we offset the values to get Vt+1? -> YES
    # seems we have to offset continues too? -> yes ofc else it doesn't match the value
    discount = 1 - 1 / IMAGINE_HORIZON  # TODO what is this vs. GAMMA?
    interm = rewards[:, :-1] + discount * continues[:, 1:] * values[:, 1:] * (
        1 - RETURN_LAMBDA
    )  # (B, T, 1)

    for t in reversed(range(IMAGINE_HORIZON)):
        # don't have access to rewards after horizon so we use the estimation
        if t == (IMAGINE_HORIZON - 1):
            Rt_plus_1 = values[:, -1]
        else:
            Rt_plus_1 = lambda_returns[:, t + 1]
        lambda_returns[:, t] = (
            interm[:, t] + continues[:, t] * discount * RETURN_LAMBDA * Rt_plus_1
        )

    return lambda_returns


def train_actor_critic(
    data,
    replayed_hts,
    replayed_zts,
    world_model,
    actor,
    critic,
    slow_critic,
    actor_opt,
    critic_opt,
):
    batch_size, seq_len = data["action"].shape[0:2]

    # update the slow critic by mixing it with the critic
    for s, d in zip(critic.parameters(), slow_critic.parameters()):
        d.data = 0.02 * s.data + (1 - 0.02) * d.data

    # we need something to hold the trajectories
    # we'll have a new batch_size of old_batch_size * old_seq_len
    # and a seq_len of IMAGINE_HORIZON+1, with t=0 being 'real'/not imagined states
    hts = torch.empty(
        batch_size * seq_len, IMAGINE_HORIZON + 1, GRU_RECCURENT_UNITS, device=device
    )
    zts = torch.empty(
        batch_size * seq_len,
        IMAGINE_HORIZON + 1,
        STOCHASTIC_STATE_SIZE,
        STOCHASTIC_STATE_SIZE,
        device=device,
    )
    ats = torch.empty(batch_size * seq_len, IMAGINE_HORIZON + 1, 1, device=device)

    # imagine trajectories, starting from sampled data
    # use each step sampled from the env as the start for a new trajectory
    hts[:, 0] = replayed_hts.view(-1, GRU_RECCURENT_UNITS).detach()  # .contiguous()
    zts[:, 0] = replayed_zts.view(
        -1, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE
    ).detach()  # .contiguous()
    ats[:, 0] = actor(hts[:, 0], zts[:, 0]).sample().unsqueeze(1)

    with torch.no_grad():  # TODO torch no grad bc we not training the wm?
        for i in range(1, IMAGINE_HORIZON + 1):
            # predict the current recurrent state ht from the past
            hts[:, i] = world_model.recurrent_model(hts[:, i - 1], zts[:, i - 1], ats[:, i - 1])
            # from that predict the stochastic state, and sample from it
            zt_dist, _ = world_model.transition_model(hts[:, i])
            zts[:, i] = zt_dist.sample()
            ats[:, i] = actor(hts[:, i], zts[:, i]).sample().unsqueeze(1)

    # after doing the recurrent stuff we will do the reward, critic and continue predictions
    pred_rewards = TwoHotEncodingDistribution(
        logits=world_model.reward_model(sg(hts), sg(zts)), dims=1
    ).mean
    # values_dist = critic(sg(hts), sg(zts)) # do we need to use independant here? TODO
    values = critic(sg(hts), sg(zts)).mean  # (B, T, 1)
    # independant bc for each element of the batch, at each step the event is independant of
    # 1. the rest of the batch and 2. the rest of the traj since we predict it from the markov state
    # at least that's what I understand rn
    continues = Independent(
        torch.distributions.Bernoulli(logits=world_model.continue_model(hts, zts)), 1
    ).mode
    # make sur we don't use imagined trajectories from terminal states
    # so get the real termination flags for step=0
    true_done = data["done"].view(-1, 1).float()
    continues[:, 0] = 1.0 - true_done[:, :1]

    with torch.no_grad():
        discount = 1 - (1 / 333)
        traj_weight = torch.cumprod(discount * continues, dim=1) / discount
        traj_weight = traj_weight.squeeze(-1)

    # compute the critic target: bootstrapped lambda return
    # ignoring t=0 bc its not imagined
    # and returning B, HORIZON-1, 1 because we use the last value to bootstrap
    lambda_returns = compute_lambda_returns(pred_rewards, values, continues)  # (B, HORIZON, 1)

    # TODO should we train the actor or the critic first?
    critic_opt.zero_grad()
    # TODO re compute the value dist for traj[:, :-1]
    values_dist = critic(hts[:, :-1], zts[:, :-1])
    slow_values = slow_critic(hts[:, :-1], zts[:, :-1]).mean.detach()
    critic_loss = -values_dist.log_prob(
        sg(lambda_returns)
    )  # symlog and symexp done in the distrib
    critic_loss -= values_dist.log_prob(slow_values)
    # TODO im confused why do we discount two times???
    # ok so i think the traj weight isnt a discount thing its to stop training after receiving the continue=0 flag
    critic_loss = critic_loss * traj_weight[:, :-1]
    critic_loss = critic_loss.mean()
    critic_loss.backward()
    critic_opt.step()

    # train the actor
    # TODO norm the rewards
    actor_opt.zero_grad()
    advantage = lambda_returns - values[:, :-1]
    policy = actor(
        sg(hts), sg(zts)
    )  # TODO we've done this computation once already, can we cache it?
    logpi = policy.log_prob(sg(ats).squeeze(-1))[
        :, :-1
    ]  # again discard last action bc we bootstrap
    actor_loss = -logpi * sg(advantage.squeeze(-1))
    actor_loss = actor_loss * traj_weight[:, :-1]
    # actor_loss -= ACTOR_ENTROPY * policy.entropy()[:, :-1]
    actor_loss = actor_loss.mean()
    actor_loss.backward()
    actor_opt.step()

    return {"loss/actor": actor_loss, "loss/critic": critic_loss}


def collect_rollout(
    env, replay_buffer, actor: Actor = None, rssm: RSSM = None, deterministic=False
):
    with torch.no_grad():
        obs, _ = env.reset()
        if rssm is not None:
            ht_minus_1 = torch.zeros(1, GRU_RECCURENT_UNITS, device=device)
            zt_dist, _ = rssm.representation_model(
                torch.tensor(obs).unsqueeze(0).to(device), ht_minus_1
            )
            zt_minus_1 = zt_dist.sample()
        done = False
        first = True
        episode_return = 0
        while not done:
            if actor is not None and rssm is not None:
                act_dist = actor(ht_minus_1, zt_minus_1)
                if deterministic:
                    act = act_dist.mode
                else:
                    act = act_dist.sample()
                act = act.unsqueeze(0)
                ht_minus_1 = rssm.recurrent_model(ht_minus_1, zt_minus_1, act)
                # act = act.squeeze(0).item()
            else:
                act = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(act.squeeze(0).item())
            episode_return += reward
            if actor is not None and rssm is not None:
                zt_dist, _ = rssm.representation_model(
                    torch.tensor(obs).to(device).unsqueeze(0), ht_minus_1
                )
                zt_minus_1 = zt_dist.mode()  # TODO make mode method on all dists #sample()

            if replay_buffer is not None:
                replay_buffer.add(act, obs, reward, terminated, first)
            first = False
            done = terminated or truncated
        print(f"Episode return: {episode_return}")


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # run = wandb.init(project="minidream_dev", job_type="train")
    # setup logger
    # TODO

    # setup env
    env = gymnasium.make("CartPole-v1")
    replay_buffer = ReplayBuffer()

    # setup models and opts
    world_model = RSSM(
        observation_space=env.observation_space,
        action_space=env.action_space,
    ).to(device)
    actor = Actor(env.action_space).to(device)
    critic = Critic().to(device)
    slow_critic = copy.deepcopy(critic)

    wm_opt = torch.optim.Adam(world_model.parameters(), lr=WM_LR, weight_decay=0.0)
    actor_opt = torch.optim.Adam(
        actor.parameters(), lr=ACTOR_CRITIC_LR, weight_decay=0.0, eps=ADAM_EPSILON
    )
    critic_opt = torch.optim.Adam(
        critic.parameters(), lr=ACTOR_CRITIC_LR, weight_decay=0.0, eps=ADAM_EPSILON
    )

    # warm up
    print("Collecting initial transitions")
    while len(replay_buffer) < cfg.learning_starts:
        collect_rollout(env, replay_buffer)

    losses = {}

    for _ in trange(cfg.iterations):
        print("Collecting transitions...")
        collect_rollout(env, replay_buffer, actor, world_model)
        # TODO put this in a loop and use it for the train ratio?
        data = replay_buffer.sample(cfg.batch_size, cfg.seq_len).to(device)

        print("Training world model...")
        hts, zts, wm_loss_dict = train_step_world_model(data, world_model, wm_opt)
        print("Training actor and critic...")
        actor_critic_loss_dict = train_actor_critic(
            data, hts, zts, world_model, actor, critic, slow_critic, actor_opt, critic_opt
        )

        loss_dict = {**wm_loss_dict, **actor_critic_loss_dict}

        for k, v in loss_dict.items():
            losses[k] = losses.get(k, [])
            losses[k].append(v.detach().cpu().item())

    collect_rollout(env, replay_buffer, actor, world_model, deterministic=True)
    # plot losses and save plot.png
    _, ax = plt.subplots(len(losses.keys()), 1, figsize=(8, 6))

    for idx, (k, v) in enumerate(losses.items()):
        ax[idx].plot(v, label=k)
        ax[idx].legend()

    plt.savefig("plot.jpg")

    # log losses, use len(replay_buffer) as the global step
    #     run.log({
    #         "recon_loss": recon_loss,
    #         "kl_loss": kl_loss,
    #         "total_wm_loss": recon_loss + kl_loss,
    #         "actor_loss": actor_loss,
    #         "critic_loss": critic_loss,
    #         "global_step": len(replay_buffer),
    #     })

    # run.finish()

    # save models
    torch.save(world_model.state_dict(), "../data/world_model.pth")
    torch.save(actor.state_dict(), "../data/actor.pth")
    # TODO write inference script


if __name__ == "__main__":
    main()
    # TODO if using the real robot, we should log mcap files to wandb artifacts so that we have a dataset we can later ues to warm start the world model?
    # so that we don't break the robot again?
