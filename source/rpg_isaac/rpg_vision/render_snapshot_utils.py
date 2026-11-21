"""Record and compare everything that determines what a rendered frame looks like.

A policy trained on one renderer will not work on another, so a training run
records its rendering configuration and replays report any difference. Nothing
here reconstructs configuration: presets do that. The snapshot exists to tell
you when what you are about to fly differs from what was trained.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: Where the snapshot sits in a run's ``config.json``.
RENDER_CONFIG_SNAPSHOT_KEY = "render_config_snapshot"

#: Snapshot keys that decide the pixels. ``presets`` is excluded: it records how
#: the configuration was produced, and two tokens can resolve to the same one.
_COMPARED_KEYS = ("camera", "sim_render", "scene_host")


def selected_presets(argv: list[str] | None = None) -> str:
    """Return the ``presets=`` token this process was launched with, if any."""
    for arg in argv if argv is not None else sys.argv:
        if arg.startswith("presets="):
            return arg.split("=", 1)[1]
    return ""


def render_config_snapshot(env_cfg) -> dict:
    """Return the configuration that determines what a frame looks like.

    Normalized through JSON so a saved snapshot and a live one compare equal:
    tuples survive a round trip as lists either way.
    """
    camera = getattr(env_cfg, "onboard_camera", None)
    snapshot = {
        "presets": selected_presets(),
        "camera": camera.to_dict() if camera is not None else None,
        "sim_render": env_cfg.sim.render.to_dict(),
        "scene_host": type(env_cfg.sim.physics).__name__,
    }
    return json.loads(json.dumps(snapshot, default=str, sort_keys=True))


def save_render_snapshot(run_config: dict, env_cfg) -> None:
    """Record the rendering configuration in a run config."""
    run_config[RENDER_CONFIG_SNAPSHOT_KEY] = render_config_snapshot(env_cfg)


def load_render_snapshot(checkpoint_path: str | Path) -> dict | None:
    """Return the snapshot saved beside a policy checkpoint, or ``None``."""
    config_path = Path(checkpoint_path).expanduser().resolve().parent / "config.json"
    if not config_path.is_file():
        return None
    return json.loads(config_path.read_text()).get(RENDER_CONFIG_SNAPSHOT_KEY)


def diff_render_snapshot(saved, current, path: str = "") -> list[str]:
    """Return one ``path: saved -> current`` line per differing leaf.

    At the top level only the keys that decide the pixels are compared.
    """
    if not path:
        saved = {key: saved.get(key) for key in _COMPARED_KEYS}
        current = {key: current.get(key) for key in _COMPARED_KEYS}
    if isinstance(saved, dict) and isinstance(current, dict):
        differences = []
        for key in sorted(set(saved) | set(current)):
            child = f"{path}.{key}" if path else str(key)
            if key not in saved:
                differences.append(f"{child}: <absent> -> {current[key]!r}")
            elif key not in current:
                differences.append(f"{child}: {saved[key]!r} -> <absent>")
            else:
                differences.extend(diff_render_snapshot(saved[key], current[key], child))
        return differences
    if saved != current:
        return [f"{path or '<root>'}: {saved!r} -> {current!r}"]
    return []


def report_render_snapshot(saved: dict | None, env_cfg, source: str = "the checkpoint") -> bool:
    """Print how the live rendering configuration differs from a saved one.

    Args:
        saved: A snapshot from :func:`load_render_snapshot`, or ``None`` when the
            run predates them or was not recorded.
        env_cfg: The resolved environment configuration about to be launched.
        source: What produced ``saved``, named in the printed messages.

    Returns:
        Whether the live configuration matches ``saved``. ``False`` when
        ``saved`` is ``None``, since nothing could be confirmed.
    """
    current = render_config_snapshot(env_cfg)
    if saved is None:
        print(f"\033[1;33m[render config] {source} recorded none; cannot compare\033[0m")
        return False

    differences = diff_render_snapshot(saved, current)
    if not differences:
        print(f"\033[1;32m[render config] matches {source} ({current['presets'] or 'no preset'})\033[0m")
        return True

    print(f"\033[1;33m[render config] differs from {source} in {len(differences)} field(s):\033[0m")
    for line in differences:
        print(f"\033[1;33m[render config]   {line}\033[0m")
    return False
