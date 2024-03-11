from pathlib import Path
from typing import Optional, Union

import gymnasium as gym
import hydra
import torch
from omegaconf import DictConfig

import wandb


def setup_env(env_cfg: DictConfig, render_mode: str = None):
    """Set up the environment and wrappers for training/inference.

    Args:
        env_cfg (DictConfig): Configuration for the environment. It should contain the following keys:
            - env_id (str): The ID of the gym environment to create.
            - kwargs (dict, optional): Additional keyword arguments to pass to the gym environment.
            - wrappers (list, optional): List of wrapper classes to apply to the environment.

        record_video (bool, optional): Whether to record video of the environment. Defaults to False.

    Returns:
        gym.Env: The configured gym environment.
    """
    kwargs = env_cfg.get("kwargs", {})
    kwargs = {**kwargs, "render_mode": render_mode}
    env = gym.make(env_cfg.env_id, **kwargs)

    if "wrappers" in env_cfg:
        for wrapper in env_cfg.wrappers:
            env = hydra.utils.instantiate(wrapper, env=env)

    return env


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_model_to_artifacts(
    run_dir: Union[str, Path],
    world_model: torch.nn.Module,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    name: str = "model",
):
    """Save the world model, actor and critic to wandb artifacts.

    Args:
        run_dir (Union[str, Path]): Directory where we'll save the model.
        world_model (torch.nn.Module): The world model to be saved.
        actor (torch.nn.Module): The actor to be saved.
        critic (torch.nn.Module): The critic to be saved.
        name (str, optional): The name of the artifact. Defaults to "model".
    """
    artifact = wandb.Artifact(name, type="model")
    if isinstance(run_dir, str):
        run_dir = Path(run_dir)

    torch.save(world_model.state_dict(), run_dir / "world_model.pth")
    torch.save(actor.state_dict(), run_dir / "actor.pth")
    torch.save(critic.state_dict(), run_dir / "critic.pth")

    artifact.add_file(run_dir / "world_model.pth")
    artifact.add_file(run_dir / "actor.pth")
    artifact.add_file(run_dir / "critic.pth")

    wandb.log_artifact(artifact)


def load_model_from_artifact(
    artifact_alias: str,
    world_model: torch.nn.Module,
    actor: torch.nn.Module,
    critic: Optional[torch.nn.Module] = None,
    device: str = "cpu",
):
    """Download the specified artifact file and return the path to the file.

    Args:
        artifact_alias (str): The alias of the wandb artifact.
        filename (str): The name of the file within the artifact.
    """
    artifact = wandb.use_artifact(artifact_alias)
    artifact_dir = Path(artifact.download())
    world_model.load_state_dict(
        torch.load(artifact_dir / "world_model.pth", map_location=torch.device(device))
    )
    actor.load_state_dict(
        torch.load(artifact_dir / "actor.pth", map_location=torch.device(device))
    )
    if critic:
        critic.load_state_dict(
            torch.load(artifact_dir / "critic.pth", map_location=torch.device(device))
        )
