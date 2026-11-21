"""Sim-level RTX pipeline settings for the RPG cameras.

A :class:`~isaaclab.sim.RenderCfg` belongs on ``sim.render``, where
``SimulationContext`` reads it; these are the settings each RTX camera in
``camera_presets.py`` was tuned against, named to match. The Newton raytracer is
kitless and needs none, so it has no entry here and reuses the task-wide
``RPG_DEFAULT_RENDER_CFG`` (see ``RPGRacingRenderCfg``).
"""

from __future__ import annotations

from isaaclab.sim import RenderCfg

_RTX_COMMON = dict(
    enable_reflections=False,
    enable_global_illumination=False,
    enable_direct_lighting=True,
    enable_shadows=True,
)

#: Isaac RTX Minimal, textured diffuse shading.
RTX_MINIMAL_TEXTURED_RENDER = RenderCfg(
    **_RTX_COMMON,
    enable_dl_denoiser=False,
    carb_settings={
        "/rtx/rendermode": "Minimal",
        "/rtx/minimal/mode": 2,
        "/rtx/pathtracing/spp": 1,
        "/rtx/pathtracing/totalSpp": 1,
        "/rtx/pathtracing/maxBounces": 2,
        "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 1,
        "/rtx/pathtracing/optixDenoiser/enabled": True,
    },
)

#: Isaac RTX Minimal, constant diffuse shading.
RTX_MINIMAL_CONSTANT_RENDER = RenderCfg(
    **_RTX_COMMON,
    enable_dl_denoiser=False,
    carb_settings={
        "/rtx/rendermode": "Minimal",
        "/rtx/minimal/mode": 1,
        "/rtx/pathtracing/spp": 1,
        "/rtx/pathtracing/totalSpp": 1,
        "/rtx/pathtracing/maxBounces": 2,
        "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 1,
        "/rtx/pathtracing/optixDenoiser/enabled": True,
    },
)

#: Isaac RTX Real-Time 2.0 with balanced DLSS ray reconstruction.
RTX_REALTIME_BALANCED_RENDER = RenderCfg(
    **_RTX_COMMON,
    enable_dl_denoiser=True,
    dlss_mode=1,
    max_bounces=2,
    carb_settings={
        "/ngx/enabled": True,
        "/persistent/rtx/modes/rt2/enabled": True,
        "/rtx/rendermode": "RealTimePathTracing",
        "/rtx/post/aa/op": 3,
        "/rtx/rtpt/maxBounces": 2,
        "/rtx/rtpt/maxSpecularAndTransmissionBounces": 1,
    },
)

#: Isaac RTX path tracing with one sample per pixel and short paths.
RTX_PATH_TRACING_LOW_SAMPLE_RENDER = RenderCfg(
    **_RTX_COMMON,
    enable_dl_denoiser=False,
    carb_settings={
        "/rtx/rendermode": "PathTracing",
        "/rtx/pathtracing/spp": 1,
        "/rtx/pathtracing/totalSpp": 1,
        "/rtx/pathtracing/maxBounces": 2,
        "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 1,
        "/rtx/pathtracing/optixDenoiser/enabled": True,
    },
)
