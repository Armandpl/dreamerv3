import torch


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())
