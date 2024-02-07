import gymnasium
import torch
from torch import Tensor, nn

from minidream.dist import OneHotDist as OneHotCategoricalStraightThrough
from minidream.functional import symlog

GRU_RECCURENT_UNITS = 256
DENSE_HIDDEN_UNITS = 256
NLP_NB_HIDDEN_LAYERS = 1
STOCHASTIC_STATE_SIZE = 32


def make_mlp(input_dim, output_dim, nb_layers=NLP_NB_HIDDEN_LAYERS, hidden_dim=DENSE_HIDDEN_UNITS):
    layers = []
    for i in range(nb_layers + 1):  # +1 for the input layer
        layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
    layers.append(nn.Linear(hidden_dim, output_dim))  # output layer
    return nn.Sequential(*layers)


class RSSM(nn.Module):
    def __init__(
        self, observation_space: gymnasium.spaces.Box, action_space: gymnasium.spaces.Box
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
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
        self.reward_model = PredModel(1)
        self.continue_model = PredModel(1)

    def forward(
        self,
        x: Tensor,  # obs at t, (B, obs_dim)
        at_minus_1: Tensor,  # action at t-1, (B, action_dim)
        ht_minus_1: Tensor,  # recurrent state at t-1, (B, GRU_RECCURENT_UNITS)
        zt_minus_1: Tensor,  # posterior state at t-1, (B, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE)
    ):
        ht = self.recurrent_model(ht_minus_1, zt_minus_1, at_minus_1)  # (B, GRU_RECCURENT_UNITS)
        zt_dist, posterior_logits = self.representation_model(x, ht)
        zt = zt_dist.sample()  # (B, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE)
        x_hat = self.decoder(ht, zt)  # (B, obs_dim)
        _, priors_logits = self.transition_model(ht)
        r = self.reward_model(ht, zt)  # (B, 1)
        c = self.continue_model(ht, zt)  # (B, 1)

        return x_hat, priors_logits, ht, zt, posterior_logits, r, c

    def imagine(self):
        pass


class RepresentationModel(nn.Module):
    """Encode the observation x + deterministic state ht into the stochastic state zt."""

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
    ):
        super().__init__()

        self.net = make_mlp(
            input_dim=observation_space.shape[0] + GRU_RECCURENT_UNITS,
            output_dim=STOCHASTIC_STATE_SIZE**2,
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
    posterior state zt or zt hat, and from the previous action."""

    def __init__(
        self,
        action_space: gymnasium.spaces.Box,
    ):
        super().__init__()
        self.mlp = make_mlp(
            input_dim=STOCHASTIC_STATE_SIZE**2 + action_space.shape[0],
            output_dim=GRU_RECCURENT_UNITS,
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

        self.net = make_mlp(
            input_dim=GRU_RECCURENT_UNITS,
            output_dim=STOCHASTIC_STATE_SIZE**2,
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
        self.net = make_mlp(
            input_dim=STOCHASTIC_STATE_SIZE**2 + GRU_RECCURENT_UNITS,
            output_dim=observation_space.shape[0],
        )

    def forward(self, ht, zt):
        zt = zt.view(zt.shape[0], -1)  # flatten
        return self.net(torch.cat([ht, zt], dim=-1))


class PredModel(nn.Module):
    """
    Pred Model: mlp from the latent state
    used for the actor, the critic, the reward and continue predictor
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.net = make_mlp(
            input_dim=STOCHASTIC_STATE_SIZE**2 + GRU_RECCURENT_UNITS,
            output_dim=out_dim,
        )

    def forward(self, ht, zt):
        zt = zt.view(zt.shape[0], -1)  # flatten
        return self.net(torch.cat([ht, zt], dim=-1))
