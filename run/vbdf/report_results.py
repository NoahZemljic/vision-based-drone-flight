"""Report manifest and fleet aggregation; no simulator or checkpoint loading required."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "deliverable/models"
FLEET = REPO / "deliverable/eval_fleet"
MANIFEST = json.loads((REPO / "deliverable/report_manifest.json").read_text())
SEEDS = MANIFEST["seeds"]
CELLS = MANIFEST["cells"]
RATE_KEYS = [f"all_envs_ome{axis}_std2" for axis in "xyz"]
CRASH_KEYS = ["all_envs_crash_episodes", "all_envs_crash_n_envs", "all_envs_crash_max_per_env"]


def bundle_status(cell: dict) -> dict:
    folder = MODELS / cell["id"]
    required = ["ckpt/model.zip", "ckpt/model_vecnormalize.pkl", "command.txt",
                "params/agent.yaml", "params/env.yaml", "params/env_cfg.py"]
    missing = [name for name in required if not (folder / name).is_file()]
    return {"complete": not missing, "missing_files": missing}


def load_rows(cell: dict, out: Path = FLEET, seeds=None, *, strict=True) -> dict:
    """Read saved evidence. Strict mode enforces the published evaluation protocol."""
    rows = {}
    for seed in SEEDS if seeds is None else seeds:
        name = cell["id"].split("/")[1]
        path = out / cell["id"] / f"{name}_{seed}.json"
        if not path.is_file():
            if strict:
                raise ValueError(f"Missing report evidence: {path}")
            continue
        row = json.loads(path.read_text())
        expected = {"cell": name, "task": cell["task"], "seed": seed,
                    "num_envs": MANIFEST["num_envs"], "delay_seed": MANIFEST["delay_seed"]}
        if strict:
            expected["steps"] = cell["steps"]
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"{path}: {key}={row.get(key)!r}; expected {value!r}")
        keys = RATE_KEYS + CRASH_KEYS + ["all_envs_dist_mean_2nd_half"]
        if not cell["id"].startswith("hover/"):
            keys += ["all_envs_dist_err", "all_envs_view_angle_deg", "all_envs_bbox_lost_frac"]
        for key in keys:
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{path}: missing or non-finite {key}")
        crashes = row.get("all_envs_crash_per_env", [])
        if (len(crashes) != MANIFEST["num_envs"]
                or any(type(n) is not int or n < 0 for n in crashes)
                or sum(crashes) != row["all_envs_crash_episodes"]
                or sum(n > 0 for n in crashes) != row["all_envs_crash_n_envs"]
                or max(crashes) != row["all_envs_crash_max_per_env"]
                or len(row.get("crash_events", [])) != sum(crashes)):
            raise ValueError(f"{path}: inconsistent fleet crash counts")
        rows[str(seed)] = row
    return rows


def summarize(out: Path = FLEET, cells=None, seeds=None, *, strict=True, gpu=None) -> dict:
    cells = CELLS if cells is None else cells
    seeds = SEEDS if seeds is None else seeds
    summary = {
        "scope": "Final-report evidence" if strict else "New evaluation (may be partial or use a different horizon)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "num_envs": MANIFEST["num_envs"], "delay_seed": MANIFEST["delay_seed"], "seeds": seeds,
        "aggregation": "Arithmetic mean across seeds of saved all_envs_* aggregates.",
        "cells": {},
    }
    previous = out / "summary.json"
    if previous.is_file():
        old = json.loads(previous.read_text())
        summary["recorded_provenance"] = old.get("recorded_provenance", {
            "gpu": old.get("gpu", "not recorded per seed"),
            "previous_summary_generated_utc": old.get("generated_utc"),
            "note": "Inherited summary label; per-seed files do not independently record GPU identity.",
        })
    if gpu:
        summary["aggregation_host_gpu"] = gpu
    for cell in cells:
        rows = load_rows(cell, out, seeds, strict=strict)
        if not rows:
            continue
        keys = sorted(k for k in set.intersection(*(set(r) for r in rows.values()))
                      if k.startswith("all_envs_") and all(type(r[k]) in (int, float) for r in rows.values()))
        for key in keys:
            if not all(math.isfinite(r[key]) for r in rows.values()):
                raise ValueError(f"{cell['id']}: non-finite {key}")
        summary["cells"][cell["id"]] = {
            "report_label": cell["label"], "report_reference": cell["report_reference"],
            "n_seeds": len(rows), "bundle": bundle_status(cell),
            "mean": {k: mean(r[k] for r in rows.values()) for k in keys},
            "seeds": {s: {k: r[k] for k in keys} for s, r in rows.items()},
            "steps_by_seed": {s: r["steps"] for s, r in rows.items()},
        }
    return summary


def write_summary(out: Path, summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = ["# Report fleet results", "", summary["scope"], "",
             "Seeds: " + ", ".join(map(str, summary["seeds"])) + "; 32 environments; delay seed 42.", "",
             "Distance and measured body-rate variation use the second half of each rollout. "
             "Angular error and crash counts use the full rollout. Rates are per-environment population "
             "standard deviations, averaged over environments and then seeds. Angular error is relative "
             "to the body forward axis.", "",
             "| Cell | Seeds | Distance error (m) | Angle (deg) | Roll std | Pitch std | Yaw std | Crashes | Envs crashed | Bundle |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, cell in summary["cells"].items():
        m = cell["mean"]
        distance = "all_envs_dist_mean_2nd_half" if name.startswith("hover/") else "all_envs_dist_err"
        keys = [distance, "all_envs_view_angle_deg", *RATE_KEYS, *CRASH_KEYS[:2]]
        values = [f"{m[k]:.6f}" if k in m else "—" for k in keys]
        status = "present" if cell["bundle"]["complete"] else "MISSING"
        lines.append(f"| {name} | {cell['n_seeds']} | " + " | ".join(values) + f" | {status} |")
    lines += ["", "Hover distance is error to (0, 0, 1) m; tracking distance is |d − 1.5|. "
              "Rates are in rad/s; crash counts are averages across seeds, not probabilities.", ""]
    missing = [name for name, cell in summary["cells"].items() if not cell["bundle"]["complete"]]
    if missing:
        lines += ["Missing original checkpoint bundles: " + ", ".join(missing) + ".", ""]
    lines += ["Rebuilding this summary does not run policies or establish fresh simulation reproducibility.", ""]
    (out / "summary.md").write_text("\n".join(lines))
