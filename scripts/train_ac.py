import copy

import gymnasium as gym
import numpy as np
import torch
from tqdm import trange
from train import (
    ACTOR_CRITIC_LR,
    ACTOR_ENTROPY,
    ACTOR_GRADIENT_CLIP,
    ADAM_EPSILON,
    CRITIC_EMA_DECAY,
    GAMMA,
    compute_lambda_returns,
)

import wandb
from minidream.dist import OneHotDist, TwoHotEncodingDistribution
from minidream.ema import EMA
from minidream.networks import make_mlp, uniform_init_weights
from minidream.rb import ReplayBuffer
from minidream.wrappers import RescaleObs

DEBUG = False

TRAIN_EVERY = 32
UPDATE_CRITIC_EVERY = 25
MAX_STEPS = 100_000
BATCH_SIZE = 16
SEQ_LEN = 16
LEARNING_STARTS = 1024


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if not DEBUG:
        run = wandb.init(project="minidream_dev")
    env = gym.make("CartPole-v1")
    # env = RescaleObs(env)

    replay_buffer = ReplayBuffer(MAX_STEPS, env.action_space, env.observation_space)

    actor = make_mlp(env.observation_space.shape[0], env.action_space.n).to(device)
    critic = make_mlp(env.observation_space.shape[0], 255).to(device)
    critic[-1].apply(uniform_init_weights(0.0))
    slow_critic = copy.deepcopy(critic)  # TODO maybe we can should just intantiate a new one

    actor_opt = torch.optim.Adam(actor.parameters(), lr=ACTOR_CRITIC_LR, eps=ADAM_EPSILON)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=ACTOR_CRITIC_LR, eps=ADAM_EPSILON)
    return_ema = EMA()  # TODO should this be the return EMA or the reward ema???

    done = True
    episode_return = 0
    train_steps = 0
    for step in trange(MAX_STEPS):
        # collect experience
        if done:
            obs, _ = env.reset()
            first = True
            run.log({"episode_return": episode_return, "global_step": step})
            episode_return = 0

        with torch.no_grad():
            out = actor(torch.tensor(obs).unsqueeze(0).to(device))
            action = torch.distributions.Categorical(logits=out).sample()
        obs, reward, truncated, terminated, _ = env.step(action.cpu().item())
        episode_return += reward
        # done = terminated or truncated
        done = terminated or truncated

        replay_buffer.add(action[0].cpu(), obs, reward, done, first)
        first = False

        # train
        if step % TRAIN_EVERY == 0 and step > LEARNING_STARTS:
            train_steps += 1
            # update the slow critic by mixing it with the critic
            if train_steps % UPDATE_CRITIC_EVERY == 0:
                for s, d in zip(critic.parameters(), slow_critic.parameters()):
                    d.data = (1 - CRITIC_EMA_DECAY) * s.data + CRITIC_EMA_DECAY * d.data

            data = replay_buffer.sample(BATCH_SIZE, SEQ_LEN).to(device)
            with torch.no_grad():  # TODO dk, should i only call the critic once??
                values = TwoHotEncodingDistribution(critic(data["obs"]), dims=1).mean  # 16, 64, 1
            continues = 1.0 - data["done"]
            lambda_returns = compute_lambda_returns(data["reward"], values, continues)

            with torch.no_grad():
                traj_weight = torch.cumprod(GAMMA * continues, dim=1) / GAMMA

            critic_opt.zero_grad()
            values_dist = TwoHotEncodingDistribution(
                critic(data["obs"][:, :-1]),
                dims=1,
            )

            # detach the slow critic bc we don't update its weights with gradient descent
            # we mix it with the critic and we train the critic to approach the output of the slow critic
            slow_values = TwoHotEncodingDistribution(
                slow_critic(data["obs"][:, :-1]), dims=1
            ).mean.detach()
            critic_loss = -values_dist.log_prob(lambda_returns.detach())
            critic_loss -= values_dist.log_prob(slow_values)

            critic_loss = critic_loss * traj_weight[:, :-1]
            critic_loss = critic_loss.mean()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), ACTOR_GRADIENT_CLIP)
            critic_opt.step()

            # train the actor
            actor_opt.zero_grad()

            offset, invscale = return_ema(lambda_returns)
            normed_lambda_returns = (lambda_returns - offset) * invscale
            normed_values = (values[:, :-1] - offset) * invscale
            advantage = normed_lambda_returns - normed_values

            policy = torch.distributions.Categorical(logits=actor(data["obs"][:, :-1]))
            logpi = policy.log_prob(data["action"][:, :-1].squeeze(-1)).unsqueeze(-1)
            actor_loss = -logpi * advantage.detach()
            actor_loss -= ACTOR_ENTROPY * policy.entropy().unsqueeze(-1)
            actor_loss = actor_loss * traj_weight[:, :-1]
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), ACTOR_GRADIENT_CLIP)
            actor_opt.step()
            if not DEBUG:
                run.log(
                    {
                        "actor_loss": actor_loss.item(),
                        "critic_loss": critic_loss.item(),
                        "global_step": step,
                    }
                )

    if not DEBUG:
        run.finish()


if __name__ == "__main__":
    main()
