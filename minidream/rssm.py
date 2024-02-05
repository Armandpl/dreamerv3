import gymnasium
import torch
from torch import nn
from torchrl.modules import OneHotCategorical

DETERMINISTIC_STATE_SIZE = 128
STOCHASTIC_STATE_SIZE = 32


class RSSM(nn.Module):
    def __init__(
        self, observation_space: gymnasium.spaces.Box, action_space: gymnasium.spaces.Box
    ):
        super().__init__()
        self.representation_model = RepresentationModel(
            observation_space=observation_space,
            deterministic_state_size=DETERMINISTIC_STATE_SIZE,
            stochastic_state_size=STOCHASTIC_STATE_SIZE,
        )
        self.recurrent_model = RecurrentModel(
            determinstic_state_size=DETERMINISTIC_STATE_SIZE,
            stochastic_state_size=STOCHASTIC_STATE_SIZE,
            action_space=action_space,
        )
        self.transition_model = TransitionModel(
            determinstic_state_size=DETERMINISTIC_STATE_SIZE,
            stochastic_state_size=STOCHASTIC_STATE_SIZE,
        )
        self.decoder = Decoder(
            stochastic_state_size=STOCHASTIC_STATE_SIZE,
            deterministic_state_size=DETERMINISTIC_STATE_SIZE,
            observation_space=observation_space,
        )
        # self.reward_model = RewardModel(deterministic_state_size=DETERMINISTIC_STATE_SIZE, stochastic_state_size=STOCHASTIC_STATE_SIZE)
        # self.continue_model

    def step(self, x, a, ht_minus_1, zt_minus_1):
        ht = self.recurrent_model(ht_minus_1, zt_minus_1, a)
        zt, posterior_logits = self.representation_model(x, ht)
        x_hat = self.decoder(ht, zt)
        zt_hat, priors_logits = self.transition_model(ht)

        return x_hat, zt_hat, priors_logits, ht, zt, posterior_logits

    def imagine(self):
        pass


class RepresentationModel(nn.Module):
    """Encode the observation x + deterministic state into the stochastic state zt."""

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
        deterministic_state_size: int,
        stochastic_state_size: int,
    ):
        super().__init__()
        self.stochastic_state_size = stochastic_state_size

        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0] + deterministic_state_size, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, stochastic_state_size * stochastic_state_size),
        )

    def forward(self, x, ht_minus_1):
        logits = self.net(torch.cat([x, ht_minus_1], dim=-1))
        zt = OneHotCategorical(
            logits=logits.view(x.shape[0], self.stochastic_state_size, self.stochastic_state_size)
        ).sample()
        return zt, logits


class RecurrentModel(nn.Module):
    """Compute the recurrent state from the previous recurrent (=deterministic) state, the previous
    posterior state zt or zt hat, and from the previous actions."""

    def __init__(
        self,
        determinstic_state_size: int,
        stochastic_state_size: int,
        action_space: gymnasium.spaces.Box,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(stochastic_state_size**2 + action_space.shape[0], 128), nn.ELU()
        )
        self.GRU = nn.GRU(
            input_size=128,
            hidden_size=determinstic_state_size,
            batch_first=True,
        )

    def forward(self, ht_minus_1, zt_minus_1, a):
        mlp_out = self.mlp(
            torch.cat([zt_minus_1.view(zt_minus_1.shape[0], -1), a], dim=-1)
        )  # (batch_size, 128)
        mlp_out = mlp_out.unsqueeze(1)  # (batch_size, 1, 128)
        ht_minus_1 = ht_minus_1.unsqueeze(0)  # (1, batch_size, 128)
        _, ht = self.GRU(mlp_out, ht_minus_1)
        ht = ht.squeeze(0)  # (batch_size, 128)
        return ht


class TransitionModel(nn.Module):
    """
    Transition Model: predict the stochastic state from the recurrent state
    Same as the Reprensentation Model, but without the observation!
    We use it when dreaming to imagine observations.
    """

    def __init__(
        self,
        determinstic_state_size: int,
        stochastic_state_size: int,
    ):
        super().__init__()

        self.stochastic_state_size = stochastic_state_size

        self.net = nn.Sequential(
            nn.Linear(determinstic_state_size, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, stochastic_state_size * stochastic_state_size),
        )

    def forward(self, x):
        logits = self.net(x)
        zt = OneHotCategorical(
            logits=logits.view(x.shape[0], self.stochastic_state_size, self.stochastic_state_size)
        ).sample()
        return zt, logits


class Decoder(nn.Module):
    """Reconstructs observations from the latent state (ht + zt)"""

    def __init__(
        self,
        stochastic_state_size: int,
        deterministic_state_size: int,
        observation_space: gymnasium.spaces.Box,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                stochastic_state_size * stochastic_state_size + deterministic_state_size, 128
            ),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, observation_space.shape[0]),
        )

    # TODO comment shapes everywhere
    def forward(self, ht, zt):
        zt = zt.view(zt.shape[0], -1)  # flatten
        return self.net(torch.cat([ht, zt], dim=-1))


class RewardModel(nn.Module):
    """
    Reward Model: estimate rewards from the latent state (ht + zt)
    """

    def __init__(
        self,
        deterministic_state_size: int,
        stochastic_state_size: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                stochastic_state_size * stochastic_state_size + deterministic_state_size, 128
            ),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, ht, zt):
        r = self.net(torch.cat([ht, zt], dim=-1))
        return r
