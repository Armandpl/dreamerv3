import time

import gymnasium as gym
import torch
from train import collect_rollout

from minidream.networks import RSSM, Actor
from minidream.wrappers import PreProcessMinatar


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # env = gym.make("CartPole-v1", render_mode="human")
    env = gym.make("MinAtar/Breakout-v1", render_mode="human")
    env = PreProcessMinatar(env)
    world_model = RSSM(env.observation_space, env.action_space).to(device)
    world_model.load_state_dict(torch.load("../data/world_model.pth", map_location=device))
    actor = Actor(env.action_space).to(device)
    actor.load_state_dict(torch.load("../data/actor.pth", map_location=device))

    for _ in range(50):
        ep_return = collect_rollout(env, None, actor, world_model)
        print(f"episode return: {ep_return}")
        time.sleep(1)


if __name__ == "__main__":
    main()
