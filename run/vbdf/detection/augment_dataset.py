#!/bin/sh
''':'
# Re-exec under Isaac Sim's python so ./this_file.py works from any shell.
exec "${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}" "$0" "$@"
'''
"""Apply the course's render-time image augmentation (flightrender ImageAugmentation:
transmission-noise patches, sigmoid contrast, motion blur, FPV logo) to a generated
dataset directory: ``<in>/img_XXXXX.png`` -> ``<out>/img_XXXXX.png``, labels copied.

  python augment_dataset.py --in dataset/train --out dataset_aug/train [--device cuda:0] [--seed 0]
"""

import argparse
import glob
import importlib.util
import os
import shutil

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
AUG_PY = os.path.join(REPO, "source", "rpg_isaac", "rpg_tasks", "tasks", "direct", "rpg_vbdf", "utils", "image_augmentation.py")


def load_augmentation_module():
    spec = importlib.util.spec_from_file_location("vbdf_image_augmentation", AUG_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--no_noise", action="store_true")
    p.add_argument("--no_logo", action="store_true")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    files = sorted(glob.glob(os.path.join(a.inp, "*.png")))
    if not files:
        raise SystemExit(f"no png in {a.inp}")
    os.makedirs(a.out, exist_ok=True)
    w, h = Image.open(files[0]).size
    aug = load_augmentation_module().ImageAugmentation(
        w, h, device=a.device, transmission_noise=not a.no_noise, add_logo=not a.no_logo, seed=a.seed
    )
    for i in range(0, len(files), a.batch):
        batch = files[i : i + a.batch]
        imgs = torch.as_tensor(np.stack([np.array(Image.open(f).convert("RGB")) for f in batch]))
        out = aug(imgs).cpu().numpy()
        for f, im in zip(batch, out):
            dst = os.path.join(a.out, os.path.basename(f))
            Image.fromarray(im).save(dst)
            lbl = f.replace(".png", ".txt")
            if os.path.isfile(lbl):
                shutil.copy2(lbl, dst.replace(".png", ".txt"))
        if (i // a.batch) % 25 == 0:
            print(f"[augment_dataset] {min(i + a.batch, len(files))}/{len(files)}", flush=True)
    print(f"[augment_dataset] wrote {len(files)} images to {a.out}")


if __name__ == "__main__":
    main()
