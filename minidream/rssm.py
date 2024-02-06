import gymnasium
import torch
from torch import nn

# from torch.distributions import OneHotCategoricalStraightThrough
from minidream.dist import OneHotDist as OneHotCategoricalStraightThrough

GRU_RECCURENT_UNITS = 256
DENSE_HIDDEN_UNITS = 256
NLP_NB_LAYERS = 1
STOCHASTIC_STATE_SIZE = 32


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# def make_mlp(input_dim, output_dim, nb_layers=NLP_NB_LAYERS, hidden_dim=DENSE_HIDDEN_UNITS):
#     layers = []
#     for i in range(nb_layers):
#         layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
#         layers.append(nn.LayerNorm(hidden_dim))
#         layers.append(nn.SiLU())
#     layers.append(nn.Linear(hidden_dim, output_dim))
#     return nn.Sequential(*layers)


class RSSM(nn.Module):
    def __init__(
        self, observation_space: gymnasium.spaces.Box, action_space: gymnasium.spaces.Box
    ):
        super().__init__()
        self.representation_model = RepresentationModel(
            observation_space=observation_space,
        )
        self.recurrent_model = RecurrentModel(
            action_space=action_space,
        )
        self.transition_model = TransitionModel()
        self.decoder = Decoder(
            observation_space=observation_space,
        )
        self.reward_model = RewardModel()
        # self.continue_model

    def step(self, x, at_minus_1, ht_minus_1, zt_minus_1):
        ht = self.recurrent_model(ht_minus_1, zt_minus_1, at_minus_1)
        zt_dist, posterior_logits = self.representation_model(x, ht)
        zt = zt_dist.sample()
        x_hat = self.decoder(ht, zt)
        _, priors_logits = self.transition_model(ht)
        rew = self.reward_model(ht, zt)

        return x_hat, priors_logits, ht, zt, posterior_logits, rew

    def imagine(self):
        pass


class RepresentationModel(nn.Module):
    """Encode the observation x + deterministic state ht into the stochastic state zt."""

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0] + GRU_RECCURENT_UNITS, DENSE_HIDDEN_UNITS),
            nn.LayerNorm(DENSE_HIDDEN_UNITS),
            nn.SiLU(),
            nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS),
            nn.LayerNorm(DENSE_HIDDEN_UNITS),
            nn.SiLU(),
            nn.Linear(DENSE_HIDDEN_UNITS, STOCHASTIC_STATE_SIZE**2),
        )

    def forward(self, x, ht_minus_1):
        x = symlog(x)
        logits = self.net(torch.cat([x, ht_minus_1], dim=-1))
        zt_dist = OneHotCategoricalStraightThrough(
            logits=logits.view(x.shape[0], STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE)
        )
        return zt_dist, logits


class RecurrentModel(nn.Module):
    """Compute the recurrent state from the previous recurrent (=deterministic) state, the previous
    posterior state zt or zt hat, and from the previous actions."""

    def __init__(
        self,
        action_space: gymnasium.spaces.Box,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(STOCHASTIC_STATE_SIZE**2 + action_space.shape[0], GRU_RECCURENT_UNITS),
            nn.LayerNorm(GRU_RECCURENT_UNITS),
            nn.SiLU(),
        )
        self.GRU = nn.GRU(
            input_size=GRU_RECCURENT_UNITS,
            hidden_size=GRU_RECCURENT_UNITS,
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
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(GRU_RECCURENT_UNITS, DENSE_HIDDEN_UNITS),
            nn.LayerNorm(DENSE_HIDDEN_UNITS),
            nn.SiLU(),
            nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS),
            nn.LayerNorm(DENSE_HIDDEN_UNITS),
            nn.SiLU(),
            nn.Linear(DENSE_HIDDEN_UNITS, STOCHASTIC_STATE_SIZE**2),
        )

    def forward(self, x):
        logits = self.net(x)
        zt_dist = OneHotCategoricalStraightThrough(
            logits=logits.view(x.shape[0], STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE)
        )
        return zt_dist, logits


class Decoder(nn.Module):
    """Reconstructs observations from the latent state (ht + zt)"""

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STOCHASTIC_STATE_SIZE**2 + GRU_RECCURENT_UNITS, DENSE_HIDDEN_UNITS),
            nn.LayerNorm(DENSE_HIDDEN_UNITS),
            nn.SiLU(),
            nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS),
            nn.LayerNorm(DENSE_HIDDEN_UNITS),
            nn.SiLU(),
            nn.Linear(DENSE_HIDDEN_UNITS, observation_space.shape[0]),
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
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STOCHASTIC_STATE_SIZE**2 + GRU_RECCURENT_UNITS, DENSE_HIDDEN_UNITS),
            nn.LayerNorm(DENSE_HIDDEN_UNITS),
            nn.SiLU(),
            nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS),
            nn.LayerNorm(DENSE_HIDDEN_UNITS),
            nn.SiLU(),
            nn.Linear(DENSE_HIDDEN_UNITS, 1),
        )

    def forward(self, ht, zt):
        zt = zt.view(zt.shape[0], -1)  # flatten
        r = self.net(torch.cat([ht, zt], dim=-1))
        return r
