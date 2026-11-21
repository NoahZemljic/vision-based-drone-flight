"""SB3 policy matching flightmare's ``use_tanh_act``: tanh on the action MEAN."""

import torch.nn as nn

from stable_baselines3.ppo.policies import MlpPolicy


class TanhMlpPolicy(MlpPolicy):
    """MlpPolicy with a Tanh appended to the action head.

    The Gaussian exploration noise is added AFTER the tanh (sample = tanh(mu) + sigma*eps),
    so exploration does not collapse when the mean saturates — the env clips the
    denormalized command instead. This mirrors flightmare's use_tanh_act=True.
    """

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = nn.Sequential(self.action_net, nn.Tanh())
