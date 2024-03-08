from pathlib import Path

import torch

import wandb


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_model_to_artifacts(
    world_model: torch.nn.Module,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    name: str = "model",
):
    """Save the world model, actor and critic to wandb artifacts.

    Args:
        world_model (torch.nn.Module): The world model to be saved.
        actor (torch.nn.Module): The actor to be saved.
        critic (torch.nn.Module): The critic to be saved.
        name (str, optional): The name of the artifact. Defaults to "model".
    """
    artifact = wandb.Artifact(name, type="model")

    torch.save(world_model.state_dict(), "../data/world_model.pth")
    torch.save(actor.state_dict(), "../data/actor.pth")
    torch.save(critic.state_dict(), "../data/critic.pth")

    artifact.add_file("../data/world_model.pth")
    artifact.add_file("../data/actor.pth")
    artifact.add_file("../data/critic.pth")

    wandb.log_artifact(artifact)


def load_model_from_artifact(artifact_alias: str, world_model, actor, critic):
    """Download the specified artifact file and return the path to the file.

    Args:
        artifact_alias (str): The alias of the wandb artifact.
        filename (str): The name of the file within the artifact.
    """
    artifact = wandb.use_artifact(artifact_alias)
    artifact_dir = Path(artifact.download())
    world_model.load_state_dict(torch.load(artifact_dir / "world_model.pth"))
    actor.load_state_dict(torch.load(artifact_dir / "actor.pth"))
    critic.load_state_dict(torch.load(artifact_dir / "critic.pth"))
