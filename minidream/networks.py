from typing import Callable, Optional

import gymnasium
import numpy as np
import torch
from torch import Tensor, nn

from minidream.distributions import (
    OneHotCategoricalStraightThroughUnimix,
    TwoHotEncodingDistribution,
)
from minidream.ema import EMA
from minidream.functional import symlog

GRU_RECCURENT_UNITS = 512
DENSE_HIDDEN_UNITS = 512
MLP_NB_HIDDEN_LAYERS = 2
STOCHASTIC_STATE_SIZE = 32

TWOHOTBUCKETS = 255

CNN_MULTIPLIER = 32
CNN_STAGES = 4


class RSSM(nn.Module):
    def __init__(
        self, observation_space: gymnasium.spaces.Box, action_space: gymnasium.spaces.Discrete
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        if len(observation_space.shape) == 3:
            img_obs = True
        elif len(observation_space.shape) == 1:
            img_obs = False
        else:
            raise ValueError(
                f"Observation space must be an image or a vector. observation_space.shape: {observation_space.shape}"
            )

        if img_obs:
            self.encoder = Encoder(observation_space)
            c, h, w = observation_space.shape
            cnn_output_dim = self.encoder.net(torch.zeros(1, c, h, w)).shape[1:]
        else:
            self.encoder = nn.Identity()
            cnn_output_dim = None

        self.representation_model = RepresentationModel(
            input_dim=np.prod(cnn_output_dim) if img_obs else observation_space.shape[0],
            symlog=(not img_obs),
        )
        self.recurrent_model = RecurrentModel(
            action_space=action_space,
        )
        self.transition_model = TransitionModel()
        self.decoder = Decoder(observation_space=observation_space, cnn_output_dim=cnn_output_dim)
        self.reward_model = PredModel(TWOHOTBUCKETS, output_init=uniform_init_weights(0.0))
        self.continue_model = PredModel(1, output_init=uniform_init_weights(1.0))
        self.return_ema = EMA()

    def forward(
        self,
        x: Tensor,  # obs at t, (B, obs_dim)
        at_minus_1: Tensor,  # action at t-1, (B, action_dim)
        ht_minus_1: Tensor,  # recurrent state at t-1, (B, GRU_RECCURENT_UNITS)
        zt_minus_1: Tensor,  # posterior state at t-1, (B, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE)
        is_first: Tensor,  # is it the first episode step (B, 1)
    ):
        # reset ht_minus_1, zt_minus_1 and action if it's the first step of an episode
        # at_minus_1 = at_minus_1 * (1 - is_first)
        # TODO is that the proper way to reset the initial state?
        # TODO do we need to stop the gradients here?
        zt_minus_1 = zt_minus_1 * (1 - is_first.unsqueeze(2))
        ht_minus_1 = ht_minus_1 * (1 - is_first)

        ht = self.recurrent_model(ht_minus_1, zt_minus_1, at_minus_1)  # (B, GRU_RECCURENT_UNITS)
        zt_dist, posterior_logits = self.representation_model(x, ht)
        zt = zt_dist.rsample()  # (B, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE)

        return ht, zt, posterior_logits


class RepresentationModel(nn.Module):  # TODO GRU w/ layer norm
    """Encode the observation x + deterministic state ht into the stochastic state zt."""

    def __init__(
        self,
        # observation_space: gymnasium.spaces.Box,
        input_dim: int,
        symlog: bool = True,
    ):
        super().__init__()
        self.symlog = symlog

        self.net = make_mlp(
            input_dim=input_dim + GRU_RECCURENT_UNITS,
            output_dim=STOCHASTIC_STATE_SIZE**2,
        )
        self.net[-1].apply(uniform_init_weights(1.0))

    def forward(self, x, ht_minus_1):
        if self.symlog:  # if not an encoded image obs
            x = symlog(x)
        logits = self.net(torch.cat([x, ht_minus_1], dim=-1))
        zt_dist = OneHotCategoricalStraightThroughUnimix(
            logits=logits.view(*list(x.shape[:-1]), STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE)
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
        self.action_space = action_space
        self.mlp = make_mlp(
            input_dim=STOCHASTIC_STATE_SIZE**2 + self.action_space.n,
            output_dim=GRU_RECCURENT_UNITS,
        )

        self.GRU = LayerNormGRUCell(
            input_size=GRU_RECCURENT_UNITS,
            hidden_size=GRU_RECCURENT_UNITS,
            batch_first=True,
            layer_norm=True,
        )

    def forward(self, ht_minus_1, zt_minus_1, a):
        # one hot encode the action
        # TODO remove the casting to long and actually pass a long tensor
        # TODO add tensor type to typing?
        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            a = torch.nn.functional.one_hot(
                a.squeeze(-1).to(torch.long), num_classes=self.action_space.n
            ).float()

        mlp_out = self.mlp(
            torch.cat([torch.flatten(zt_minus_1, start_dim=-2), a], dim=-1)
        )  # (batch_size, 128)
        mlp_out = mlp_out.unsqueeze(1)  # (batch_size, 1, 128)
        ht_minus_1 = ht_minus_1.unsqueeze(0)  # (1, batch_size, 128)
        ht = self.GRU(mlp_out, ht_minus_1.contiguous())  # TODO why do we need to call contiguous?
        ht = ht.squeeze(0)  # (batch_size, GRU_RECCURENT_UNITS)
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
        self.net[-1].apply(uniform_init_weights(1.0))

    def forward(self, x):
        logits = self.net(x)
        zt_dist = OneHotCategoricalStraightThroughUnimix(
            logits=logits.view(
                *list(logits.shape[:-1]), STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE
            )
        )
        return zt_dist, logits


class Decoder(nn.Module):
    """Reconstructs observations from the latent state (ht + zt)"""

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
        cnn_output_dim=(256, 6, 6),
    ):
        super().__init__()
        self.cnn_output_dim = cnn_output_dim
        if cnn_output_dim is not None:
            self.decnn = make_cnn(deconv=True, input_channels=observation_space.shape[0])

        self.mlp = make_mlp(
            input_dim=STOCHASTIC_STATE_SIZE**2 + GRU_RECCURENT_UNITS,
            output_dim=np.prod(cnn_output_dim)
            if cnn_output_dim is not None
            else observation_space.shape[0],
        )

    def forward(self, ht, zt):
        zt = torch.flatten(zt, start_dim=-2)
        mlp_out = self.mlp(torch.cat([ht, zt], dim=-1))

        # if obs is not an image, return the mlp output
        if self.cnn_output_dim is None:
            return mlp_out

        # if obs is an image, deconvolve the mlp output to reconstuct the image
        shape = mlp_out.shape
        # TODO aint those two lines redundant?
        to_deconv = torch.unflatten(mlp_out, dim=-1, sizes=self.cnn_output_dim)
        to_deconv = to_deconv.view(-1, *self.cnn_output_dim)
        deconved = self.decnn(to_deconv)
        deconved = deconved.view(*shape[:-1], *deconved.shape[1:])
        return deconved


class Encoder(nn.Module):
    """Encoder the observed image."""

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
    ):
        super().__init__()
        self.net = make_cnn(input_channels=observation_space.shape[0])

    def forward(self, x):
        # x is of shape (..., c, h, w)
        c, h, w = x.shape[-3:]
        # flatten the batch and time dimension as one
        encoded = self.net(x.view(-1, c, h, w))

        # unflatten the batch and time dimension
        # and flatten the output of the cnn
        return encoded.view(*x.shape[:-3], -1)


class Actor(nn.Module):
    def __init__(self, action_space: gymnasium.spaces.Discrete):
        super().__init__()
        self.action_space = action_space
        self.net = PredModel(action_space.n, output_init=uniform_init_weights(1.0))

    def forward(self, ht, zt):
        # TODO we need an actor for continuous action
        # do we model a normal dist in this case?
        output = self.net(ht, zt)
        dist = torch.distributions.Categorical(logits=output)  # TODO add unimix?
        return dist


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = PredModel(TWOHOTBUCKETS, output_init=uniform_init_weights(0.0))

    def forward(self, ht, zt):
        output = self.net(ht, zt)
        dist = TwoHotEncodingDistribution(logits=output, dims=1)
        return dist


class PredModel(nn.Module):
    """
    Pred Model: mlp from the latent state
    used for the actor, the critic, the reward and continue predictor
    """

    def __init__(self, out_dim: int, output_init: Optional[Callable] = None):
        super().__init__()
        self.net = make_mlp(
            input_dim=STOCHASTIC_STATE_SIZE**2 + GRU_RECCURENT_UNITS,
            output_dim=out_dim,
        )
        if output_init:
            output_init(self.net[-1])

    def forward(self, ht, zt):
        zt = torch.flatten(zt, start_dim=-2)  # flatten the last two dimensions
        return self.net(torch.cat([ht, zt], dim=-1))


def weight_init(m: nn.Module):
    # TODO why not use torch.init.normal_ ???
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978  # TODO why not torch.sqrt???
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_init_weights(given_scale):  # TODO same here, why not torch.nn.init.uniform_?
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


def make_mlp(
    input_dim,
    output_dim,
    nb_layers=MLP_NB_HIDDEN_LAYERS,
    hidden_dim=DENSE_HIDDEN_UNITS,
):
    """
    output_zero_init: initializes the weights of the output layer to 0 bc:
    "We further noticed that the randomly
    initialized reward predictor and critic networks at the start of training can result in large predicted
    rewards that can delay the onset of learning. We initialize the output weights of the reward predictor
    and critic to zeros, which effectively alleviates the problem and accelerates early learning." page 7
    """
    layers = []
    for i in range(nb_layers + 1):  # +1 for the input layer
        layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
    output_layer = nn.Linear(hidden_dim, output_dim)
    layers.append(output_layer)
    net = nn.Sequential(*layers)
    net.apply(weight_init)

    return net


def make_cnn(
    input_channels=4,
    stages=CNN_STAGES,
    multiplier=CNN_MULTIPLIER,
    deconv=False,
):
    layers = []
    channels = [input_channels]
    channels += [i * 2 * multiplier for i in range(1, stages + 1)]
    if deconv:
        channels = channels[::-1]
        layer = torch.nn.ConvTranspose2d
    else:
        layer = torch.nn.Conv2d

    for i in range(len(channels) - 1):
        layers.append(
            layer(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        layers.append(torch.nn.SiLU())

    return torch.nn.Sequential(*layers)


class LayerNormGRUCell(nn.Module):
    """A GRU cell with a LayerNorm, taken
    from https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/nets.py#L317.

    This particular GRU cell accepts 3-D inputs, with a sequence of length 1, and applies
    a LayerNorm after the projection of the inputs.

    Args:
        input_size (int): the input size.
        hidden_size (int): the hidden state size
        bias (bool, optional): whether to apply a bias to the input projection.
            Defaults to True.
        batch_first (bool, optional): whether the first dimension represent the batch dimension or not.
            Defaults to False.
        layer_norm (bool, optional): whether to apply a LayerNorm after the input projection.
            Defaults to False.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        batch_first: bool = False,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=self.bias)
        if layer_norm:
            self.layer_norm = torch.nn.LayerNorm(3 * hidden_size)
        else:
            self.layer_norm = nn.Identity()

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        is_3d = input.dim() == 3
        if is_3d:
            if input.shape[int(self.batch_first)] == 1:
                input = input.squeeze(int(self.batch_first))
            else:
                raise AssertionError(
                    "LayerNormGRUCell: Expected input to be 3-D with sequence length equal to 1 but received "
                    f"a sequence of length {input.shape[int(self.batch_first)]}"
                )
        if hx.dim() == 3:
            hx = hx.squeeze(0)
        assert input.dim() in (
            1,
            2,
        ), f"LayerNormGRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"

        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        input = torch.cat((hx, input), -1)
        x = self.linear(input)
        x = self.layer_norm(x)
        reset, cand, update = torch.chunk(x, 3, -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        hx = update * cand + (1 - update) * hx

        if not is_batched:
            hx = hx.squeeze(0)
        elif is_3d:
            hx = hx.unsqueeze(0)

        return hx
