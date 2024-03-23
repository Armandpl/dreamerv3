import math
from typing import Callable, Optional, Union

import gymnasium
import numpy as np
import torch
from tensordict.nn import inv_softplus
from torch import Tensor, nn
from torch.distributions import Independent

from dreamer.distributions import (
    OneHotCategoricalStraightThroughUnimix,
    TwoHotEncodingDistribution,
)
from dreamer.ema import EMA
from dreamer.functional import symlog

GRU_RECCURENT_UNITS = 512
DENSE_HIDDEN_UNITS = 512
MLP_NB_HIDDEN_LAYERS = 2
STOCHASTIC_STATE_SIZE = 32

TWOHOTBUCKETS = 255

CNN_MULTIPLIER = 32
CNN_STAGES = 4


class RSSM(nn.Module):
    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
        action_space: Union[gymnasium.spaces.Box, gymnasium.spaces.Discrete],
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
        action_space: Union[gymnasium.spaces.Box, gymnasium.spaces.Discrete],
    ):
        super().__init__()
        self.action_space = action_space
        in_action_shape = (
            action_space.shape[0]
            if isinstance(self.action_space, gymnasium.spaces.Box)
            else self.action_space.n
        )

        self.mlp = make_mlp(
            input_dim=STOCHASTIC_STATE_SIZE**2 + in_action_shape,
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
    """Reconstructs observations from the latent state (ht + zt) If cnn_output_dim is not None, the
    reconstructed observation is an image."""

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
        cnn_output_dim=(256, 6, 6),
    ):
        super().__init__()
        self.cnn_output_dim = cnn_output_dim
        if cnn_output_dim is not None:
            # TODO
            if observation_space.shape[1] == 10:
                stride = 1
            else:
                stride = 2

            # self.decnn = make_cnn(deconv=True, input_channels=observation_space.shape[0], stride=stride)
            self.decnn = make_cnn_2(deconv=True, observation_space=observation_space)

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
        # TODO ugly hardcode to have a stride of 1 for minatar
        if observation_space.shape[1] == 10:
            stride = 1
        else:
            stride = 2
        # self.net = make_cnn(input_channels=observation_space.shape[0], stride=stride)
        self.net = make_cnn_2(observation_space=observation_space)

    def forward(self, x):
        # x is of shape (..., c, h, w)
        c, h, w = x.shape[-3:]
        # flatten the batch and time dimension as one
        encoded = self.net(x.view(-1, c, h, w))

        # unflatten the batch and time dimension
        # and flatten the output of the cnn
        return encoded.view(*x.shape[:-3], -1)


class Actor(nn.Module):
    def __init__(
        self,
        action_space: Union[gymnasium.spaces.Box, gymnasium.spaces.Discrete],
        use_gsde: bool = True,
        gsde_sample_freq: int = 64,
        log_std_init: float = -3,
        clip_mean: float = 2.0,
    ):
        super().__init__()
        self.action_space = action_space
        self.use_gsde = use_gsde

        out_action_shape = (
            action_space.shape[0]
            if isinstance(action_space, gymnasium.spaces.Box)
            else action_space.n
        )
        self.net = PredModel(DENSE_HIDDEN_UNITS, output_init=uniform_init_weights(1.0))
        self.head = nn.Sequential(
            nn.SiLU(),  # TODO should gsde use the activated neuron too?
            nn.Linear(DENSE_HIDDEN_UNITS, out_action_shape),
        )
        uniform_init_weights(1.0)(self.head[-1])

        self.gsde_sample_freq = gsde_sample_freq
        log_std = torch.ones(DENSE_HIDDEN_UNITS, out_action_shape)

        # Transform it to a parameter so it can be optimized
        self.log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)

        self.count = 0  # so we know when to resample noise
        self.clip_mean = clip_mean  # for numerical stability w/ gsde

    def _sample_gsde_noise(self) -> None:
        """Sample weights for the noise exploration matrix, using a centered Gaussian distribution.

        :param batch_size:
        """
        self.weights_dist = torch.distributions.Normal(torch.zeros_like(self.log_std), 1)
        # Reparametrization trick to pass gradients
        self.exploration_mat = self.weights_dist.rsample()
        # Pre-compute matrices in case of parallel exploration
        # self.exploration_matrices = self.weights_dist.rsample((batch_size,))

    def forward(self, ht, zt, explore: bool = True):
        if self.count % self.gsde_sample_freq == 0:
            self._sample_gsde_noise()
        self.count += 1

        features = self.net(ht, zt)
        output = self.head(features)
        gsde_noise = None

        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            dist = torch.distributions.Categorical(logits=output)
            act = dist.sample().unsqueeze(-1)
        else:
            mean_action = output  # policy output
            if self.clip_mean is not None:
                mean_action = torch.functional.F.hardtanh(
                    mean_action, min_val=-self.clip_mean, max_val=self.clip_mean
                )

            std = torch.exp(self.log_std)
            if explore:
                # adding gsde noise on top
                expl_mat = self.exploration_mat.to(features.device) * std
                gsde_noise = features @ expl_mat
                act = mean_action + gsde_noise
            else:
                act = mean_action

            # variance = torch.mm(features**2, std ** 2)
            variance = (features**2) @ (std**2)
            epsilon = 1e-6
            dist = torch.distributions.Normal(mean_action, torch.sqrt(variance + epsilon))
        return dist, act, gsde_noise


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


def run_rollout(env, actor: Actor, rssm: RSSM, device: str = "cpu", render: bool = True):
    with torch.no_grad():
        obs, _ = env.reset()
        ht_minus_1 = torch.zeros(1, GRU_RECCURENT_UNITS, device=device)
        zt_minus_1 = torch.zeros(1, STOCHASTIC_STATE_SIZE, STOCHASTIC_STATE_SIZE, device=device)
        # zt_dist, _ = rssm.representation_model(
        #     torch.tensor(obs).unsqueeze(0).to(device), ht_minus_1
        # )
        # zt_minus_1 = zt_dist.sample()

        done = False
        episode_return = 0

        while not done:
            _, act, _ = actor(ht_minus_1, zt_minus_1, explore=False)
            ht_minus_1 = rssm.recurrent_model(ht_minus_1, zt_minus_1, act)
            act = act.cpu()

            obs, reward, terminated, truncated, _ = env.step(act.squeeze(0).item())
            done = terminated or truncated
            # don't render if done bc it can fail w/ diverged furuta sim
            # TODO fix the root issue
            if render and not done:
                env.render()
            episode_return += reward

            encoded_obs = rssm.encoder(torch.tensor(obs).unsqueeze(0).to(device))
            zt_dist, _ = rssm.representation_model(encoded_obs, ht_minus_1)
            zt_minus_1 = zt_dist.sample()

    return episode_return


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


def make_cnn_2(
    observation_space: gymnasium.spaces.Box,
    stages=CNN_STAGES,
    multiplier=CNN_MULTIPLIER,
    deconv=False,
):
    dummy_input = torch.zeros(1, *observation_space.shape)
    input_channels = observation_space.shape[0]

    # TODO ugly hardcode still
    # maybe have the cnn kernel, stride and padding in env configs?
    stride = 1 if observation_space.shape[1] == 10 else 2

    layers = []
    channels = [input_channels]
    channels += [i * 2 * multiplier for i in range(1, stages + 1)]
    if deconv:
        channels = channels[::-1]
        layer = torch.nn.ConvTranspose2d
        # instantiate the same cnn in conv mode, pass the dummy input to it and get the output shape
        # which will be the input shape for the de-cnn
        dummy_input = make_cnn_2(observation_space, stages, multiplier, deconv=False)(dummy_input)
    else:
        layer = torch.nn.Conv2d

    for i in range(len(channels) - 1):
        layers.append(
            layer(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=4,
                stride=stride,
                padding=1,
            )
        )
        dummy_input = layers[-1](dummy_input)
        layers.append(torch.nn.LayerNorm(dummy_input.shape[1:]))
        layers.append(torch.nn.SiLU())

    return torch.nn.Sequential(*layers)


def make_cnn(
    input_channels=4,
    stages=CNN_STAGES,
    multiplier=CNN_MULTIPLIER,
    kernel_size=4,
    stride=2,
    padding=1,
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
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
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
