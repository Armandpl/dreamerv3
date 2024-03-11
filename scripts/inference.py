import sys
import time

import torch
from omegaconf import DictConfig

import wandb
from dreamer.networks import RSSM, Actor, run_rollout
from dreamer.utils import load_model_from_artifact, setup_env


def main(artifact_alias: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = wandb.init(project="minidream_dev", job_type="inference")
    creator_run = run.use_artifact(artifact_alias).logged_by()

    env = setup_env(DictConfig(creator_run.config).env, render_mode="human")

    world_model = RSSM(env.observation_space, env.action_space).to(device)
    actor = Actor(env.action_space).to(device)

    load_model_from_artifact(artifact_alias, world_model, actor, device=device)

    run.finish()

    for _ in range(25):
        ep_return = run_rollout(env, actor, world_model)
        print(f"episode return: {ep_return}")
        time.sleep(1)


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please provide the model artifact alias"
    main(sys.argv[1])
