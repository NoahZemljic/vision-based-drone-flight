"""VBDF course tasks (uzh-rpg/vbdf-course Task 2), ported from flightmare."""

import importlib.util

import gymnasium as gym

from . import agents

gym.register(
    id="Rpg-Vbdf-Hover-v0",
    entry_point=f"{__name__}.vbdf_hover_env:RPGVbdfHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vbdf_env_cfg:RPGVbdfHoverEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Rpg-Vbdf-Tracking-v0",
    entry_point=f"{__name__}.vbdf_tracking_env:RPGVbdfTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vbdf_env_cfg:RPGVbdfTrackingEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Rpg-Vbdf-Tracking-Viz-v0",
    entry_point=f"{__name__}.vbdf_render_env:RPGVbdfTrackingVizEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vbdf_render_env:RPGVbdfTrackingVizEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

# --- fresh Task-2 trackers (this session) ---
gym.register(
    id="Rpg-Vbdf-SB-Tracking-v0",
    entry_point=f"{__name__}.vbdf_state_tracking_env:RPGVbdfStateTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vbdf_tracking_cfg:RPGVbdfStateTrackingEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Rpg-Vbdf-BB-Tracking-v0",
    entry_point=f"{__name__}.vbdf_bbox_tracking_env:RPGVbdfBboxTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vbdf_tracking_cfg:RPGVbdfBboxTrackingEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Rpg-Vbdf-Hover-Viz-v0",
    entry_point=f"{__name__}.vbdf_render_env:RPGVbdfHoverVizEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vbdf_render_env:RPGVbdfHoverVizEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Rpg-Vbdf-SB-Tracking-Viz-v0",
    entry_point=f"{__name__}.vbdf_render_env:RPGVbdfStateTrackingVizEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vbdf_render_env:RPGVbdfStateTrackingVizEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Rpg-Vbdf-BB-Tracking-Viz-v0",
    entry_point=f"{__name__}.vbdf_render_env:RPGVbdfBboxTrackingVizEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vbdf_render_env:RPGVbdfBboxTrackingVizEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

# instructor-only environments: registered only while the *solution* modules exist
# (the student handout is produced by deleting them)
if importlib.util.find_spec(f"{__name__}.vbdf_tracking_solution_env") is not None:
    gym.register(
        id="Rpg-Vbdf-Tracking-Solution-v0",
        entry_point=f"{__name__}.vbdf_tracking_solution_env:RPGVbdfTrackingSolutionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.vbdf_tracking_solution_env:RPGVbdfTrackingSolutionEnvCfg",
            "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        },
    )

    gym.register(
        id="Rpg-Vbdf-Tracking-Solution-Viz-v0",
        entry_point=f"{__name__}.vbdf_render_solution_env:RPGVbdfTrackingSolutionVizEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.vbdf_render_solution_env:RPGVbdfTrackingSolutionVizEnvCfg",
            "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        },
    )
