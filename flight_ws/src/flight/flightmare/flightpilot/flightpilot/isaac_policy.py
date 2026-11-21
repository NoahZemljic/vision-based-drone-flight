"""Loader for rpg_isaac sb3 checkpoints (``model.zip`` + ``model_vecnormalize.pkl``).

Rebuilds the TanhMlpPolicy actor as a plain torch MLP from the state dict alone —
obs dim, hidden sizes and action dim are all inferred from the weights, so neither
sb3 nor rpg_tasks is needed on the vehicle.
"""
import io
import os
import pickle
import zipfile

import numpy as np
import torch
import torch.nn as nn


def load_isaac_actor(model_zip: str) -> tuple[nn.Module, int, int]:
    """(actor, n_obs, n_act) from ``policy.pth`` inside ``model.zip``."""
    with zipfile.ZipFile(model_zip) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
    pi = {k[len("mlp_extractor.policy_net."):]: v for k, v in sd.items() if k.startswith("mlp_extractor.policy_net.")}
    head = {k[len("action_net."):]: v for k, v in sd.items() if k.startswith("action_net.")}
    hw = head["0.weight"] if "0.weight" in head else head["weight"]
    hb = head["0.bias"] if "0.bias" in head else head["bias"]
    n_layers = max(int(k.split(".")[0]) for k in pi) // 2 + 1
    n_obs, n_act = pi["0.weight"].shape[1], hw.shape[0]
    layers, d = [], n_obs
    for i in range(n_layers):
        w = pi[f"{2 * i}.weight"]
        layers += [nn.Linear(d, w.shape[0]), nn.ReLU()]
        d = w.shape[0]
    net = nn.Sequential(*layers, nn.Linear(d, n_act), nn.Tanh())
    net[-2].weight.data.copy_(hw)
    net[-2].bias.data.copy_(hb)
    for i in range(n_layers):
        net[2 * i].weight.data.copy_(pi[f"{2 * i}.weight"])
        net[2 * i].bias.data.copy_(pi[f"{2 * i}.bias"])
    return net.eval(), n_obs, n_act


def load_obs_norm(vecnorm_pkl: str, n_obs: int):
    """(mean, var, clip_obs, eps) of the sb3 VecNormalize obs stats; identity if the file is absent."""
    if not os.path.isfile(vecnorm_pkl):
        return np.zeros(n_obs), np.ones(n_obs), np.inf, 1e-8
    with open(vecnorm_pkl, "rb") as f:
        vn = pickle.load(f)  # VecNormalize.__setstate__ tolerates the missing venv
    return vn.obs_rms.mean.astype(np.float64), vn.obs_rms.var.astype(np.float64), float(vn.clip_obs), float(vn.epsilon)
