from pathlib import Path
from typing import Optional, Union

import gymnasium
import numpy as np
import torch
from furuta.rl.envs.furuta_base import FurutaBase
from furuta.utils import VelocityFilter

from minidream.rssm import RSSM


class FurutaNeuralSim(FurutaBase):
    def __init__(
        self,
        rssm_path: Union[str, Path],
        control_freq=50,
        reward="cos_alpha",
        angle_limits=[None, None],
        speed_limits=[None, None],
        velocity_filter: int = 2,
        render_mode="rgb_array",
    ):

        super().__init__(control_freq, reward, angle_limits, speed_limits, render_mode)
        self.rssm = RSSM(
            observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(4,)),
            action_space=gymnasium.spaces.Box(low=-1, high=1, shape=(1,)),
        )
        torch.set_grad_enabled(False)
        self.rssm.load_state_dict(torch.load(rssm_path))

        self.velocity_filter = velocity_filter
        self._init_vel_filt()

    def _init_vel_filt(self):
        if self.velocity_filter:
            self.vel_filt = VelocityFilter(self.velocity_filter, dt=self.timing.dt)
        else:
            self.vel_filt = None

    def _init_state(self):
        self._state = 0.01 * np.float32(np.random.randn(self.state_space.shape[0]))

        # setup first recurrent state
        h0 = torch.zeros(1, 128)
        z0 = torch.zeros(1, 1024)  # init stochastic state
        a0 = torch.zeros(1, 1)  # init action
        # get rssm obs (cos_theta, sin_theta, cos_alpha, sin_alpha)
        rssm_obs = torch.tensor(
            [
                np.cos(self._state[0]),
                np.sin(self._state[0]),
                np.cos(self._state[1]),
                np.sin(self._state[1]),
            ]
        ).unsqueeze(0)

        self.ht = self.rssm.recurrent_model(ht_minus_1=h0, zt_minus_1=z0, a=a0)
        self.zt, _ = self.rssm.representation_model(x=rssm_obs, ht_minus_1=h0)

    def _update_state(self, a):
        # update the simulation state
        # get recurent (ht) from last latent (zt_hat an h_t) and action
        a = torch.tensor(a, dtype=torch.float).view(1, 1)
        self.ht = self.rssm.recurrent_model(ht_minus_1=self.ht, zt_minus_1=self.zt, a=a)
        # predict zt_hat from ht
        self.zt, _ = self.rssm.transition_model(self.ht)

        # concat zt_hat and ht, reconstruct obs
        recon = self.rssm.decoder(ht=self.ht, zt=self.zt)[0]

        # set the state
        # recon = [cos_theta, sin_theta, cos_alpha, sin_alpha]
        # reconstruct theta and alpha from cos and sin
        theta = np.arctan2(recon[1], recon[0], dtype=np.float32)
        alpha = np.arctan2(recon[3], recon[2], dtype=np.float32)
        self._state = np.array([theta, alpha, 0, 0], dtype=np.float32)

        # 2. Compute the velocities using the velocity filter
        # if self.vel_filt:
        #     self._state[2:4] = self.vel_filt(self._state[0:2])

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._init_state()
        obs = self.get_obs()
        self._init_vel_filt()
        return obs, {}
