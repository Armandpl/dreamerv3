import torch


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())


# TODO use the sg() notation instead of detach?? rn using both, not super clean
def sg(x):
    return x.detach()
