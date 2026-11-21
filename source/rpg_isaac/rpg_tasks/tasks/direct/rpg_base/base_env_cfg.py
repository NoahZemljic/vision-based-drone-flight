from __future__ import annotations

import copy

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from rpg_assets.falcon.falcon_articulation_cfg import FALCON_ROBOT_CFG
from rpg_assets.falcon.falcon_sim_params import FALCON_SIM_PARAMS
from rpg_assets.kolibri.kolibri_articulation_cfg import KOLIBRI_ROBOT_CFG
from rpg_assets.kolibri.kolibri_sim_params import *  # noqa: F403

# moved to rpg_simulator; re-exported here for existing task-cfg imports
from rpg_simulator.simulator_params import QuadRandomizationCfg
from rpg_vision.onboard_camera_cfg import OnboardCameraCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import RenderCfg, SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg


@configclass
class RPGPhysicsCfg(PresetCfg):
    """IsaacLab 3.0 physics-backend preset for rpg_isaac.

    Selects the physics engine at launch via Hydra. Two families of variants:

    - ``physx`` / ``newton_mjwarp`` / ``ovphysx``: the engine integrates the
      drone dynamics (rpg injects wrenches).
    - ``agisim_physx`` / ``agisim_newton_mjwarp`` / ``agisim_ovphysx``: rpg's
      own Warp integrator drives the dynamics and the named engine only hosts
      the scene (assets, rendering, sensors). These names are shared with
      :class:`RPGBackendCfg`, so one token (e.g. ``physics=agisim_physx``)
      selects the integrator and its scene host together — presets apply by
      variant name to every ``PresetCfg`` that defines it.

    ``default`` (== ``physx``) keeps the TGS solver settings tuned for the fast
    free-flying quadrotor: contacts are minimal, hence the small Newton solver
    caps and CCD/stabilization disabled on PhysX.
    """

    # PhysX (default): TGS solver tuned for a fast quadrotor.
    default: PhysxCfg = PhysxCfg(
        solver_type=1,  # TGS (Temporal Gauss-Seidel) - more accurate than PGS
        min_position_iteration_count=4,
        max_position_iteration_count=32,
        min_velocity_iteration_count=2,
        max_velocity_iteration_count=16,
        enable_ccd=False,
        enable_stabilization=False,
    )
    physx: PhysxCfg = PhysxCfg(
        solver_type=1,
        min_position_iteration_count=4,
        max_position_iteration_count=32,
        min_velocity_iteration_count=2,
        max_velocity_iteration_count=16,
        enable_ccd=False,
        enable_stabilization=False,
    )
    # Newton (MuJoCo-Warp): free-flyer with minimal contacts. njmax=45 overflowed
    # (needs 48) once drones rest on a ground plane; 64 leaves headroom.
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=64,
            nconmax=25,
            cone="pyramidal",
            integrator="implicitfast",
            impratio=1,
        ),
        num_substeps=1,
        debug_mode=False,
    )
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    # agisim-mode scene hosts: same engine settings as the variants above.
    agisim_physx: PhysxCfg = PhysxCfg(
        solver_type=1,
        min_position_iteration_count=4,
        max_position_iteration_count=32,
        min_velocity_iteration_count=2,
        max_velocity_iteration_count=16,
        enable_ccd=False,
        enable_stabilization=False,
    )
    agisim_newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=64,
            nconmax=25,
            cone="pyramidal",
            integrator="implicitfast",
            impratio=1,
        ),
        num_substeps=1,
        debug_mode=False,
    )
    agisim_ovphysx: OvPhysxCfg = OvPhysxCfg()


@configclass
class RPGBackendCfg(PresetCfg):
    """Who integrates the drone dynamics.

    ``agisim``: rpg's own Warp integrator; the physics engine only hosts the
    scene (assets, rendering, sensors). ``engine``: the engine selected by
    ``sim.physics`` integrates and rpg injects wrenches.

    The ``agisim_<engine>`` variants mirror the scene-host variants in
    :class:`RPGPhysicsCfg`; selecting ``physics=agisim_<engine>`` flips this
    preset to ``agisim`` and picks that engine as the host in one token. The
    bare engine variants flip to ``engine`` likewise, so an unprefixed token
    (e.g. ``physics=newton_mjwarp``) means engine-coupled even on tasks whose
    backend defaults to agisim (racing).
    """

    default: str = "engine"
    agisim: str = "agisim"
    engine: str = "engine"
    physx: str = "engine"
    newton_mjwarp: str = "engine"
    ovphysx: str = "engine"
    agisim_physx: str = "agisim"
    agisim_newton_mjwarp: str = "agisim"
    agisim_ovphysx: str = "agisim"


def resolve_backend(backend: RPGBackendCfg | str) -> str:
    """Resolved integration mode: "agisim" or "engine".

    Under the Hydra entry point the selected preset variant replaces the
    field with its string; on other paths the field is still the preset cfg
    (use its default). Legacy strings ("physx") mean the engine integrates.
    """
    if not isinstance(backend, str):
        backend = backend.default
    return "agisim" if backend == "agisim" else "engine"


DRONES: dict[str, tuple] = {
    "kolibri": (KOLIBRI_SIM_PARAMS, KOLIBRI_ROBOT_CFG),  # noqa: F405
    "falcon": (FALCON_SIM_PARAMS, FALCON_ROBOT_CFG),
}


def resolve_drone(cfg: RPGBaseEnvCfg) -> None:
    """Swap ``rpg_sim_params`` / ``robot`` to the airframe named by ``cfg.drone``.

    Called at env construction, after Hydra overrides are applied —
    ``__post_init__`` runs before them and cannot see a CLI ``drone=``.
    Re-applies the ``delay``/dt plumbing and keeps task-set spawn visibility.
    """
    if cfg.drone not in DRONES:
        raise ValueError(f"Unknown drone {cfg.drone!r}; available drones are {sorted(DRONES)}")
    if cfg.drone == "kolibri":
        return
    sim_params, robot_cfg = DRONES[cfg.drone]
    cfg.rpg_sim_params = copy.deepcopy(sim_params)
    cfg.rpg_sim_params.delay = cfg.delay
    cfg.rpg_sim_params.ctrl_dt = cfg.ctrl_dt
    cfg.rpg_sim_params.integrator_dt = cfg.integrator_dt
    visible = cfg.robot.spawn.visible
    cfg.robot = robot_cfg.replace(prim_path="/World/envs/env_.*/Robot")
    cfg.robot.spawn.visible = visible


RPG_DEFAULT_RENDER_CFG = RenderCfg(
    enable_reflections=True,
    enable_global_illumination=True,
    enable_direct_lighting=True,
    enable_shadows=True,
    enable_dl_denoiser=True,
    carb_settings={
        "/rtx/rendermode": "PathTracing",
        "/rtx/pathtracing/spp": 2,  # more samples = cleaner shadows
        "/rtx/pathtracing/totalSpp": 4,
        "/rtx/pathtracing/maxBounces": 6,
        "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 4,
        "/rtx/shadows/enabled": True,
    },
)
"""Sim-level RTX settings every RPG task starts from, applied by ``SimulationContext``.

Tasks that render through a specific pipeline replace this on ``sim.render``;
a kit-less scene host ignores it.
"""


@configclass
class RPGBaseEnvCfg(DirectRLEnvCfg):
    backend: RPGBackendCfg = RPGBackendCfg()
    drone: str = "kolibri"  # key into DRONES
    delay: float = 0.0
    ctrl_dt: float = 0.005
    integrator_dt: float = 0.001
    episode_length_s = 100.0
    quad_state_dtype: str = "float32"  # "float32" or "float64" for precision control

    # sim params
    rpg_sim_params: QuadrotorSimulatorParams = KOLIBRI_SIM_PARAMS  # noqa: F405

    # - spaces definition
    action_space = 4
    observation_space = 4
    state_space = 0

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        # >0 keeps PhysX broadphase pair counts linear at scale; 0.0 overlaps
        # all envs at the origin (safe only when the engine does not integrate).
        env_spacing=2.5,
        replicate_physics=True,
    )

    # robot
    robot: ArticulationCfg = KOLIBRI_ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/body",  # can be added to all other links
    )

    # A concrete OnboardCameraCfg is sufficient for the unmodified IsaacLab
    # launcher to discover and enable its renderer. None keeps non-vision tasks
    # free of camera runtime and memory costs.
    onboard_camera: OnboardCameraCfg | PresetCfg | None = None

    # Environment-owned render-scene behavior: whether this task's visible
    # geometry can move. RTX consumers use it to stop streaming geometry after
    # warm-up.
    static_render_geometry: bool = False

    # domain randomization
    deployment_mode: bool = False
    test_env: bool = False  # If true, use fixed initial state and no randomization for testing purposes
    eval_env_idx: int = 0
    quad_randomization_cfg: QuadRandomizationCfg = QuadRandomizationCfg()

    def __post_init__(self):
        ratio = self.ctrl_dt / self.integrator_dt
        assert abs(ratio - round(ratio)) < 1e-6, "Control dt must be multiple of integrator dt"
        self.decimation = int(round(ratio))

        # PhysX solver settings now live in RPGPhysicsCfg (the physics-backend preset).
        self.sim: SimulationCfg = SimulationCfg(
            dt=self.integrator_dt,  # physics dt (low-level controller frequency)
            render_interval=20,
            gravity=(0.0, 0.0, -9.8066),  # Set gravity vector
            physics=RPGPhysicsCfg(),  # backend preset: physx (default) / newton_mjwarp / ovphysx
            render=RPG_DEFAULT_RENDER_CFG,
        )

        self.rpg_sim_params.delay = self.delay
        self.rpg_sim_params.ctrl_dt = self.ctrl_dt
        self.rpg_sim_params.integrator_dt = self.integrator_dt
