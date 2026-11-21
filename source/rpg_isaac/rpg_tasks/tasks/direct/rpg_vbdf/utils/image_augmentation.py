"""Port of flightrender ``render_utils/image_augmentation.py`` (course renderer): the
augmentations applied to every arena/winti render before it reaches the network —
transmission-noise blocks (real analog-video patches, Pareto-distributed bursts),
sigmoid contrast, random linear motion blur and the FPV logo overlay
(``offboard_small_aug.yaml``). Operates on uint8 RGB (N,H,W,3) tensors."""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aug_assets")


class ImageAugmentation:
    def __init__(
        self,
        width: int,
        height: int,
        device="cuda",
        contrast=(5.0, 6.0),
        contrast_cutoff=(0.45, 0.55),
        motionblur_k: int = 11,
        motionblur_n: int = 1000,
        transmission_noise: bool = True,
        noise_block_h=(8,),
        noise_block_w=(11, 10, 11, 10, 11, 11),
        noise_zeta_s: float = 1.1,
        add_logo: bool = True,
        aug_prob: float = 1.0,
        seed: int | None = None,
    ):
        self.w, self.h, self.device = width, height, torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.aug_prob = aug_prob
        self.contrast, self.cutoff = contrast, contrast_cutoff
        # motion blur kernels: a line through the centre, random angle, random asymmetric weighting
        k = motionblur_k if motionblur_k % 2 else motionblur_k + 1
        self.k = k
        angle = self.rng.uniform(0, 360, motionblur_n)
        direction = self.rng.uniform(-1, 1, motionblur_n) + 0.5
        kernels = np.zeros((motionblur_n, k, k))
        kernels[:, k // 2, k // 2] = 1.0
        for i in range(motionblur_n):
            if self.rng.random() < aug_prob:
                kernels[i, :, k // 2] = np.linspace(direction[i], 1.0 - direction[i], num=k)
                kernels[i] = ndimage.rotate(kernels[i], angle[i], reshape=False)
                kernels[i] /= kernels[i].sum()
        self.blur_kernels = torch.tensor(kernels, dtype=torch.float32, device=self.device)
        # transmission noise
        self.noise = transmission_noise
        if self.noise:
            bh, bw = list(noise_block_h), list(noise_block_w)
            assert width % sum(bw) == 0 and height % sum(bh) == 0, "image size must tile the noise blocks"
            ws = np.r_[0, np.cumsum(np.tile(bw, width // sum(bw)))]
            hs = np.r_[0, np.cumsum(np.tile(bh, height // sum(bh)))]
            w1, h1 = np.meshgrid(ws[:-1], hs[:-1])
            w2, h2 = np.meshgrid(ws[1:], hs[1:])
            self.blocks = np.c_[h1.ravel(), w1.ravel(), h2.ravel(), w2.ravel()].astype(int)
            self.n_blocks = len(self.blocks)
            self.patches = {}
            for h in set(bh):
                for w in set(bw):
                    p = os.path.join(ASSETS, f"patches_{h}_{w}.npz")
                    arr = np.load(p)["patches"] if os.path.isfile(p) else self.rng.integers(0, 256, (100, h, w, 3), dtype=np.uint8)
                    self.patches[h * w] = torch.as_tensor(arr, device=self.device)
            # block gaps ~ discrete Pareto (heavy tail -> bursts of corrupted blocks)
            zeta = torch.distributions.pareto.Pareto(torch.tensor([0.5]), torch.tensor([0.1]))
            samples = torch.round(zeta.sample((100000,))).flatten()
            self.zeta_samples = samples[samples < self.n_blocks].int()
            self._resample_noise()
        self.add_logo = add_logo
        if add_logo:
            logo = np.array(Image.open(os.path.join(ASSETS, "logo.png")).convert("RGB"))
            mask = 1.0 - np.array(Image.open(os.path.join(ASSETS, "logo_mask.png")).convert("RGB")) / 255.0
            full = np.zeros((720, 1280, 3), dtype=np.uint8)
            mfull = np.ones((720, 1280, 3), dtype=np.float32)
            full[: logo.shape[0], -logo.shape[1] :, :] = logo
            mfull[: mask.shape[0], -mask.shape[1] :, :] = mask
            full = np.array(Image.fromarray(full).resize((width, height), Image.BILINEAR))
            mfull = np.array(Image.fromarray((mfull * 255).astype(np.uint8)).resize((width, height), Image.BILINEAR)) / 255.0
            self.logo = torch.as_tensor(full, device=self.device).float() / 255.0 * torch.as_tensor(1 - mfull, device=self.device).float()
            self.logo_mask = torch.as_tensor(mfull, device=self.device).float()

    def _resample_noise(self):
        perm = torch.randperm(self.zeta_samples.numel())
        self.noise_idx = torch.cumsum(self.zeta_samples[perm], dim=0).cpu().numpy()
        self.noise_sel = 1

    def _sample_noise(self, img: torch.Tensor):
        """Overwrite Pareto-spaced blocks of one (H,W,3) uint8 image with real noise patches."""
        if self.noise_idx[-1] - self.noise_idx[self.noise_sel] < self.n_blocks:
            self._resample_noise()
        start = self.noise_idx[self.noise_sel - 1]
        while (blk := int(self.noise_idx[self.noise_sel] - start)) < self.n_blocks:
            blk = blk % self.n_blocks
            y0, x0, y1, x1 = self.blocks[blk]
            pat = self.patches[(y1 - y0) * (x1 - x0)]
            img[y0:y1, x0:x1, :] = pat[self.noise_sel % pat.shape[0]]
            self.noise_sel += 1

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """uint8 (N,H,W,3) -> augmented uint8 (N,H,W,3)."""
        images = images.to(self.device).clone()
        n = images.shape[0]
        if self.noise:
            for i in range(n):
                self._sample_noise(images[i])
        x = images.permute(0, 3, 1, 2).float()  # (N,3,H,W) in [0,255]
        for i in range(n):
            if self.rng.random() < self.aug_prob:
                gain = self.rng.uniform(*self.contrast)
                cutoff = self.rng.uniform(*self.cutoff)
                x[i] = 255.0 / (1.0 + torch.exp(gain * (cutoff - x[i] / 255.0)))
        x = x / 255.0
        idx = torch.randint(0, self.blur_kernels.shape[0], (n,), device=self.device).repeat_interleave(3)
        x = F.conv2d(x.reshape(1, 3 * n, self.h, self.w), self.blur_kernels[idx].unsqueeze(1), groups=3 * n, padding="same")
        x = x.reshape(n, 3, self.h, self.w).permute(0, 2, 3, 1)
        if self.add_logo:
            x = x * self.logo_mask + self.logo
        return (x.clamp(0, 1) * 255.0).round().to(torch.uint8)
