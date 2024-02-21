import torch

from minidream.functional import compute_lambda_returns


def test_compute_lambda_values():
    rewards = torch.tensor([[[1], [1], [1], [1], [1]]], dtype=torch.float32)
    values = torch.tensor([[[4], [3], [2], [1], [0]]], dtype=torch.float32)
    continues = torch.tensor([[[1], [1], [1], [1], [0]]], dtype=torch.float32)
    lambda_values = compute_lambda_returns(rewards, values, continues, 1.0, 0.95)
    assert lambda_values.shape == (1, 4, 1)
    assert torch.allclose(lambda_values, torch.tensor([[[4], [3], [2], [1]]], dtype=torch.float32))


if __name__ == "__main__":
    test_compute_lambda_values()
