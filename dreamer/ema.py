from typing import Any

import torch


# https://github.com/Eclectic-Sheep/sheeprl/blob/8045b2e47df9e16677dcb9062ee2990ecbc70b9e/sheeprl/algos/dreamer_v3/utils.py#L41
class EMA(torch.nn.Module):
    def __init__(
        self,
        decay: float = 0.99,
        max_: float = 1.0,
        percentile_low: float = 0.05,
        percentile_high: float = 0.95,
    ) -> None:
        super().__init__()
        self._decay = decay
        self._max = torch.tensor(max_)
        self._percentile_low = percentile_low
        self._percentile_high = percentile_high
        self.register_buffer("low", torch.zeros((), dtype=torch.float32))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Any:
        x = x.flatten().detach()  # flatten bc return is a tensor of shape (batch_size, 1)
        low = torch.quantile(x, self._percentile_low)
        high = torch.quantile(x, self._percentile_high)
        self.low = self._decay * self.low + (1 - self._decay) * low
        self.high = self._decay * self.high + (1 - self._decay) * high
        invscale = torch.max(1 / self._max, self.high - self.low)
        return self.low.detach(), invscale.detach()
