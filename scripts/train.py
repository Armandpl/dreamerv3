import copy

import gymnasium
import hydra
import torch
from omegaconf import DictConfig
from torch.distributions import Independent
from torch.distributions.kl import kl_divergence
from tqdm import trange

import wandb
from minidream.dist import BernoulliSafeMode
from minidream.dist import OneHotDist as OneHotCategoricalStraightThrough
from minidream.dist import TwoHotEncodingDistribution
from minidream.functional import compute_lambda_returns, symlog
from minidream.networks import (
    GRU_RECCURENT_UNITS,
    RSSM,
    STOCHASTIC_STATE_SIZE,
    Actor,
    Critic,
)
from minidream.rb import ReplayBuffer
from minidream.utils import count_parameters, sg

DEBUG = False  # wether or not to use wandb

# Tuning the HPs = losing
# Actor Critic
GAMMA = 1 - (1 / 333)
IMAGINE_HORIZON = 15
RETURN_LAMBDA = 0.95
ACTOR_ENTROPY = 3e-4
ACTOR_CRITIC_LR = 3e-5
ADAM_EPSILON = 1e-5
ACTOR_GRADIENT_CLIP = 100.0
CRITIC_EMA_DECAY = 0.98

# World Model
BETA_PRED = 1.0
BETA_DYN = 0.5
BETA_REP = 0.1
MIN_SYMLOG_DISTANCE = 1e-8
WM_LR = 1e-4
WM_ADAM_EPSILON = 1e-8
WM_GRADIENT_CLIP = 1000.0  # TODO do we have to clip norm or values? nm512 clips norm

#
TRAIN_RATIO = 256  # ratio between real steps and imagined steps
REPLAY_CAPACITY = 1_000_000  # FIFO
BATCH_SIZE = 16
BATCH_LENGTH = 64


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_world_model(data, world_model):
    batch_size, seq_len = data["obs"].shape[0:2]
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

    # compute recurrent states
    for i in range(seq_len):
        at_minus_1 = data["action"][:, i].view(batch_size, 1)
        if i == 0:
            ht_minus_1 = torch.zeros(batch_size, GRU_RECCURENT_UNITS, device=device)
            zt_minus_1 = torch.zeros(
                batch_size, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE, device=device
            )
        else:
            ht_minus_1 = recurrent_states[:, i - 1].view(batch_size, GRU_RECCURENT_UNITS)
            zt_minus_1 = posteriors[:, i - 1].view(
                batch_size, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE
            )

        (recurrent_states[:, i], posteriors[:, i], posteriors_logits[:, i],) = world_model.forward(
            data["obs"][:, i], at_minus_1, ht_minus_1, zt_minus_1, is_first=data["first"][:, i]
        )
    return recurrent_states, posteriors, posteriors_logits


def train_step_world_model(
    data, recurrent_states, posteriors, posteriors_logits, world_model, optim
):
    """Train the world model for one step. data contains a batch of replay transitions of shape (B,
    T, ...) it contains past actions, observations, rewards and dones.

    dones = terminated, not terminated or truncated # <-- TODO actually not sure abt that
    """
    batch_size, seq_len = data["obs"].shape[0:2]

    reconstructed_obs = world_model.decoder(recurrent_states, posteriors)  # (B, T, obs_dim)
    _, priors_logits = world_model.transition_model(recurrent_states)
    pred_rewards_logits = world_model.reward_model(recurrent_states, posteriors)  # (B, 1)
    pred_continues = world_model.continue_model(recurrent_states, posteriors)  # (B, 1)

    priors_logits = priors_logits.view(
        batch_size, seq_len, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE
    )
    posteriors_logits = posteriors_logits.view(
        batch_size, seq_len, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE
    )

    optim.zero_grad()

    # compute losses
    # TODO use the validate args arg? get it from cfg or smth
    distance = 0.5 * (reconstructed_obs - symlog(data["obs"])) ** 2  # Eq. 1
    distance = torch.where(
        distance < MIN_SYMLOG_DISTANCE, 0, distance
    )  # got that from the offical implem
    recon_loss = distance.sum(dim=[-1])

    pred_rewards_distrib = TwoHotEncodingDistribution(pred_rewards_logits, dims=1)
    rew_loss = -pred_rewards_distrib.log_prob(data["reward"])  # sum over the last dim

    continue_loss = -Independent(
        BernoulliSafeMode(logits=pred_continues),
        1,
    ).log_prob(1.0 - data["done"])
    loss_pred = (recon_loss + rew_loss + continue_loss).mean()  # average accross batch and time

    # kl loss
    # TODO fold that into a KL loss func/class?
    free_nats = torch.tensor([1], dtype=torch.float32, device=device)
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
            Independent(  # independant is so that each event is independant
                OneHotCategoricalStraightThrough(logits=posteriors_logits), 1
            ),
            Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1),
        ).mean(),
    )

    loss = BETA_PRED * loss_pred + BETA_DYN * loss_dyn + BETA_REP * loss_rep
    loss.backward()
    torch.nn.utils.clip_grad_norm_(world_model.parameters(), WM_GRADIENT_CLIP)
    optim.step()
    return {
        "sampled_actions_distrib": wandb.Histogram(data["action"].view(-1).cpu().detach().numpy()),
        "sampled_reward_distrib": wandb.Histogram(data["reward"].view(-1).cpu().detach().numpy()),
        "pred_rewards_distrib": wandb.Histogram(
            pred_rewards_distrib.mean.view(-1).cpu().detach().numpy()
        ),
        "world_model/total_loss": loss,
        "world_model/pred_loss": loss_pred,
        "world_model/dyn_loss": loss_dyn,
        "world_model/rep_loss": loss_rep,
        "world_model/recon_loss": recon_loss.mean(),
        "world_model/reward_loss": rew_loss.mean(),
        "world_model/continue_loss": continue_loss.mean(),
    }


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
        BernoulliSafeMode(logits=world_model.continue_model(hts, zts)), 1
    ).mode  # TODO can we just do > 0.5?
    # make sur we don't use imagined trajectories from terminal states
    # so get the real termination flags for step=0
    true_done = data["done"].view(-1, 1)
    continues[:, 0] = 1.0 - true_done[:, :1]

    with torch.no_grad():
        traj_weight = torch.cumprod(GAMMA * continues, dim=1) / GAMMA
        traj_weight = traj_weight.squeeze(-1)

    # compute the critic target: bootstrapped lambda return
    # ignoring t=0 bc its not imagined
    # and returning B, HORIZON-1, 1 because we use the last value to bootstrap
    lambda_returns = compute_lambda_returns(
        pred_rewards, values, continues, GAMMA, RETURN_LAMBDA
    )  # (B, HORIZON, 1)

    # TODO should we train the actor or the critic first?
    critic_opt.zero_grad()
    # TODO re compute the value dist for traj[:, :-1]
    values_dist = critic(sg(hts[:, :-1]), sg(zts[:, :-1]))

    # detach the slow critic bc we don't update its weights with gradient descent
    # we mix it with the critic and we train the critic to approach the output of the slow critic
    slow_values = slow_critic(sg(hts[:, :-1]), sg(zts[:, :-1])).mean.detach()
    critic_loss = -values_dist.log_prob(
        sg(lambda_returns)
    )  # symlog and symexp done in the distrib
    critic_loss -= values_dist.log_prob(slow_values)
    # TODO im confused why do we discount two times???
    # ok so i think the traj weight isnt a discount thing its to stop training after receiving the continue=0 flag
    critic_loss = critic_loss * sg(traj_weight[:, :-1])
    critic_loss = critic_loss.mean()
    critic_loss.backward()
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), ACTOR_GRADIENT_CLIP)
    critic_opt.step()

    # update the slow critic by mixing it with the critic
    for s, d in zip(critic.parameters(), slow_critic.parameters()):
        d.data = (1 - CRITIC_EMA_DECAY) * s.data + CRITIC_EMA_DECAY * d.data

    # train the actor
    actor_opt.zero_grad()

    offset, invscale = world_model.return_ema(lambda_returns)
    normed_lambda_returns = (lambda_returns - offset) / invscale
    normed_values = (values[:, :-1] - offset) / invscale
    advantage = normed_lambda_returns - normed_values

    policy = actor(
        sg(hts[:, :-1]),
        sg(zts[:, :-1]),  # loose one bc of lambda, loose one bc lambda return is at t+1
    )  # TODO we've done this computation once already, can we cache it?
    logpi = policy.log_prob(sg(ats[:, :-1]).squeeze(-1))
    actor_loss = -logpi * sg(advantage.squeeze(-1))
    actor_entropy = policy.entropy()
    actor_loss -= ACTOR_ENTROPY * actor_entropy
    actor_loss = actor_loss * sg(traj_weight[:, :-1])
    actor_loss = actor_loss.mean()
    actor_loss.backward()
    actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), ACTOR_GRADIENT_CLIP)
    actor_opt.step()

    return {
        "agent/imagined_actions_distrib": wandb.Histogram(
            ats.view(-1).to(torch.long).cpu().detach().numpy()
        ),
        "agent/pred_values_distrib": wandb.Histogram(values.view(-1).cpu().detach().numpy()),
        "agent/pred_values_mean": values.mean(),
        "agent/lambda_values_mean": lambda_returns.mean(),
        "agent/lambda_values_distrib": wandb.Histogram(
            lambda_returns.view(-1).cpu().detach().numpy()
        ),
        "agent/actor_loss": actor_loss,
        "agent/critic_loss": critic_loss,
        "agent/advantage": advantage.mean(),
        "agent/ema_offset": offset.mean(),
        "agent/ema_invscale": invscale.mean(),
        "agent/entropy": actor_entropy.mean(),
        "agent/actor_grad_norm": actor_grad_norm,
        "agent/critic_grad_norm": critic_grad_norm,
    }


# def collect_rollout(env, replay_buffer, actor: Actor = None, rssm: RSSM = None):
#     use_actor = actor is not None and rssm is not None
#     with torch.no_grad():
#         obs, _ = env.reset()
#         if use_actor:
#             ht_minus_1 = torch.zeros(1, GRU_RECCURENT_UNITS, device=device)
#             zt_minus_1 = torch.zeros(
#                 1, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE, device=device
#             )
#             zt_dist, _ = rssm.representation_model(
#                 torch.tensor(obs).unsqueeze(0).to(device), ht_minus_1
#             )
#             zt_minus_1 = zt_dist.sample()

#         done = False
#         first = True
#         episode_return = 0

#         while not done:
#             if use_actor:
#                 act_dist = actor(ht_minus_1, zt_minus_1)
#                 act = act_dist.sample()
#                 act = act.unsqueeze(0)
#                 ht_minus_1 = rssm.recurrent_model(ht_minus_1, zt_minus_1, act)
#                 act = act.cpu()
#             else:
#                 act = env.action_space.sample()

#             obs, reward, terminated, truncated, _ = env.step(act.squeeze(0).item())
#             episode_return += reward

#             if use_actor:
#                 zt_dist, _ = rssm.representation_model(
#                     torch.tensor(obs).to(device).unsqueeze(0), ht_minus_1
#                 )
#                 zt_minus_1 = zt_dist.sample()

#             done = terminated or truncated

#             if replay_buffer is not None:
#                 # replay_buffer.add(act, obs, reward, terminated, first)
#                 replay_buffer.add(act, obs, reward, done, first)
#             first = False
#     return episode_return


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    if not DEBUG:
        run = wandb.init(project="minidream_dev", job_type="train")

    # setup env
    env = gymnasium.make("CartPole-v1")
    # env = PressTheLightUpButton()
    # env = RescaleObs(env)
    # eval_env = copy.deepcopy(env)
    replay_buffer = ReplayBuffer(REPLAY_CAPACITY, env.action_space, env.observation_space)

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

    print(f"world model size: {count_parameters(world_model)/1e6:.2f}M")
    print(f"actor size: {count_parameters(actor)/1e6:.2f}M")
    print(f"critic size: {count_parameters(critic)/1e6:.2f}M")

    print("Training")
    done = True
    episode_return = 0
    episode_len = 0
    for _ in trange(cfg.max_steps):

        if done:
            if not DEBUG:
                run.log(
                    {
                        "episode_return": episode_return,
                        "episode_len": episode_len,
                        "global_step": replay_buffer.count,
                    }
                )
            episode_return = 0
            episode_len = 0
            obs, _ = env.reset()
            first = True
            with torch.no_grad():
                ht_minus_1 = torch.zeros(1, GRU_RECCURENT_UNITS, device=device)
                zt_minus_1 = torch.zeros(
                    1, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE, device=device
                )
                # TODO no info to take first action from
                # could use the first obs to get the first zt
                # but that means we should do that when training the world model too
                # we could/should also train on the first obs from reset?

                # zt_dist, _ = world_model.representation_model(
                #     torch.tensor(obs).unsqueeze(0).to(device), ht_minus_1
                # )
                # zt_minus_1 = zt_dist.sample()

        # choose action w/ actor
        with torch.no_grad():
            act_dist = actor(ht_minus_1, zt_minus_1)
            act = act_dist.sample()
            act = act.unsqueeze(0)
            ht_minus_1 = world_model.recurrent_model(ht_minus_1, zt_minus_1, act)
            act = act.cpu().squeeze(0).item()

        obs, reward, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
        episode_return += reward
        episode_len += 1
        replay_buffer.add(act, obs, reward, done, first)
        first = False

        # encode obs
        zt_dist, _ = world_model.representation_model(
            torch.tensor(obs).to(device).unsqueeze(0), ht_minus_1
        )
        zt_minus_1 = zt_dist.sample()

        if (
            replay_buffer.count % (cfg.batch_size * cfg.seq_len // TRAIN_RATIO) == 0
        ) and replay_buffer.count > cfg.learning_starts:  # should train
            data = replay_buffer.sample(cfg.batch_size, cfg.seq_len).to(device)

            hts, zts, zts_logits = run_world_model(data, world_model)
            wm_loss_dict = train_step_world_model(data, hts, zts, zts_logits, world_model, wm_opt)
            actor_critic_loss_dict = train_actor_critic(
                data, hts, zts, world_model, actor, critic, slow_critic, actor_opt, critic_opt
            )

            loss_dict = {**wm_loss_dict, **actor_critic_loss_dict}
            run.log({**loss_dict, "global_step": replay_buffer.count})

    if not DEBUG:
        run.finish()

    # save models
    torch.save(world_model.state_dict(), "../data/world_model.pth")
    torch.save(actor.state_dict(), "../data/actor.pth")
    # TODO write inference script


if __name__ == "__main__":
    main()
    # TODO if using the real robot, we should log mcap files to wandb artifacts so that we have a dataset we can later ues to warm start the world model?
    # so that we don't break the robot again?
