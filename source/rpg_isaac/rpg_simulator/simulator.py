"""Facade over the rpg quadrotor dynamics stack.

Owns the quad state, parameter server, model pipeline, low-level controller,
command queue and integrator, and advances them together (:meth:`run_substep`).
Pure torch/warp — synchronizing the state with a host engine (root-state
reads/writes, wrench transfer) is the caller's job; for IsaacLab that adapter
is ``rpg_tasks .../rpg_base/isaac_bridge.py``.
"""

import math
import time

import torch

from rpg_simulator.low_level_controller_betaflight import LowLevelControllerBetaFlight
from rpg_simulator.low_level_controller_simple import LowLevelControllerSimple
from rpg_simulator.model_battery import ModelBattery
from rpg_simulator.model_init import ModelInit
from rpg_simulator.model_motor import ModelMotor
from rpg_simulator.model_rigid_body import ModelRigidBody
from rpg_simulator.model_rotor_simple_augmentation import ModelRotorSimpleAugmentation
from rpg_simulator.model_scale import ModelScale
from rpg_simulator.model_thrust_torque_simple import ModelThrustTorqueSimple
from rpg_simulator.quad_state import QuadState
from rpg_simulator.quadrotor_params import QuadParamServer
from rpg_simulator.simulator_params import (
    QuadRandomizationCfg,
    QuadrotorSimulatorParams,
)
from rpg_simulator.utils.cmd_queue import CmdQueue
from rpg_simulator.utils.integrator_euler import IntegratorEuler
from rpg_simulator.utils.integrator_rk4 import IntegratorRK4
from rpg_simulator.utils.math_utils import quat_from_euler_xyz, quat_mul


class QuadrotorSimulator:
    """rpg quadrotor dynamics: state + LLC + model pipeline + integrator.

    Modes:
    - agisim (``using_agisim=True``): integrates the full rigid-body state
      itself; the host engine is only ever written for rendering/sensors.
    - engine-coupled: computes the analytic wrench (``quad_state.force_torque``)
      each substep; the host engine integrates and owns the root state.
    """

    def __init__(
        self,
        params: QuadrotorSimulatorParams,
        *,
        num_envs: int,
        device: str,
        dtype: torch.dtype,
        using_agisim: bool,
        init_state: torch.Tensor | None = None,
        randomization: QuadRandomizationCfg | None = None,
    ):
        self.params = params
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype
        self.using_agisim = using_agisim
        self.delay = params.delay

        self.randomization = randomization or QuadRandomizationCfg()
        # per-env command latency jitter, added to push_command timestamps
        self.random_delay = torch.zeros(num_envs, device=device, dtype=dtype)
        rd = self.randomization

        # Dedicated RNG for the command-latency draw. When rd.delay_seed > 0 the
        # latency band is drawn from a re-seeded generator so every episode (and every
        # run / model / eval seed) sees the SAME [0, random_delay) sequence
        # ("consistent random"). When 0, latency uses the global RNG (default).
        self._delay_gen = None
        if getattr(rd, "delay_seed", 0) > 0:
            self._delay_gen = torch.Generator(device="cpu")
            self._delay_seed = int(rd.delay_seed)
        self.init_pos_rd = torch.tensor(rd.random_init_pos, device=device, dtype=dtype)
        self.init_pos_rd_scale = rd.random_init_pos_scale
        self.init_vel_rd = torch.tensor(rd.random_init_vel, device=device, dtype=dtype)
        self.init_vel_rd_scale = rd.random_init_vel_scale
        self.init_att_rd = torch.tensor(rd.random_init_att, device=device, dtype=dtype)
        self.init_att_rd_scale = rd.random_init_att_scale
        self.init_ome_rd = torch.tensor(rd.random_init_ome, device=device, dtype=dtype)
        self.init_ome_rd_scale = rd.random_init_ome_scale

        self.init_battery_voltage = params.quad_params.rest_voltage

        if init_state is not None:
            self.quad_state = QuadState.from_state(
                state=init_state,
                using_agisim=using_agisim,
                init_battery_voltage=self.init_battery_voltage,
            )
        else:
            self.quad_state = QuadState(
                num_quads=num_envs,
                using_agisim=using_agisim,
                device=device,
                dtype=dtype,
                init_battery_voltage=self.init_battery_voltage,
            )

        # default motor time constant, overridden if ModelMotor is used
        self.motor_tau = params.quad_params.motor_tau
        self.param_server = QuadParamServer(
            params.quad_params, num_envs=num_envs, device=device, dtype=dtype
        )

        self.model_pipeline = self._build_model_pipeline()
        self.integrator = self._build_integrator()

        # Size for the largest configured command latency. The extra slot
        # keeps the oldest command intact when a new control sample arrives at
        # the same instant that command becomes eligible.
        max_command_delay = params.delay + max(0.0, self.randomization.random_delay)
        self.cmd_queue = CmdQueue(
            num_envs=num_envs,
            capacity=math.ceil(max_command_delay / params.ctrl_dt) + 1,
            act_dim=4,  # collective thrust + body rates
            device=device,
            dtype=dtype,
        )
        # created after cmd_queue so it can bind the current command tensor
        self.llc = self._build_low_level_controller(
            cmd_tensor=self.cmd_queue.current_cmd
        )

        self.timer_dict = {
            "deriv_func(graph_launch)": 0.0,
            "cache_state(deriv_func)": 0.0,
            "pre_physics(geo_control)": 0.0,
            "pop_set_command": 0.0,
            "llc_set_command": 0.0,
            "llc_get_command": 0.0,
            "motor_0_tau_set": 0.0,
            "integrator_step": 0.0,
            "set_external_force_and_torque": 0.0,
            "ModelInit": 0.0,
            "ModelMotor": 0.0,
            "ModelBattery": 0.0,
            "ModelThrustTorqueSimple": 0.0,
            "ModelRotorSimpleAugmentation": 0.0,
            "ModelScale": 0.0,
            "ModelRigidBody": 0.0,
        }

    # ========================= Stepping ========================= #

    def push_command(self, actions: torch.Tensor, sim_time: float) -> None:
        """Queue high-level commands (collective thrust + body rates).

        Commands are stamped ``sim_time`` plus the per-env random latency
        (see :meth:`randomize_params`) and take effect once the queue clock
        passes ``timestamp + params.delay``.
        """
        timestamps = (
            torch.full((self.num_envs,), sim_time, device=self.device, dtype=self.dtype)
            + self.random_delay
        )
        self.cmd_queue.push(actions, timestamps)

    def run_substep(self, sim_time: float) -> None:
        """Advance LLC + dynamics by one integrator step.

        agisim mode: integrates the full state. Engine-coupled mode: leaves the
        analytic wrench in ``quad_state.force_torque`` for the caller to apply.
        """
        start_t = time.time()
        self.cmd_queue.step(current_time=sim_time, delay=self.delay)
        self.timer_dict["pop_set_command"] += time.time() - start_t

        self.llc.set_command(env_ids=slice(None))
        self.timer_dict["llc_set_command"] += time.time() - start_t

        start_t = time.time()
        self.llc.get_motor_commands()  # sets QuadState.motdes
        self.timer_dict["llc_get_command"] += time.time() - start_t

        start_t = time.time()
        if self.motor_tau == 0.0:
            self.quad_state.mot = self.quad_state.motdes
        self.timer_dict["motor_0_tau_set"] += time.time() - start_t

        start_t = time.time()
        self.integrator.step(self.quad_state, self.deriv_func)
        self.timer_dict["integrator_step"] += time.time() - start_t

        if not self.using_agisim:
            self._check_finite_wrench(sim_time)

    def deriv_func(self):
        """Compute the state derivative for integration.

        Recomputes the rotation matrix and body-frame velocity from the current
        intermediate state — critical during RK4 where att/vel change between
        substeps (k1-k4).
        """
        # 1. Update State Cache (R, vel_b)
        if self.using_agisim:
            self.quad_state.cuda_launch()

        # 2. Run Pipeline
        for model in self.model_pipeline:
            model.cuda_launch(self.quad_state)

    def reset(self, env_ids=None, *, pos=None, att=None, vel=None, ome=None):
        """Reset state and command queue for the given envs."""
        self.quad_state.reset_state(
            env_idx=env_ids,
            quad_pos=pos,
            quad_att=att,
            quad_vel=vel,
            quad_ome=ome,
        )
        self.cmd_queue.reset(env_ids=env_ids)

    def refresh_caches(self) -> None:
        """Recompute the derived state caches (R, vel_b) from att / vel."""
        self.quad_state.cache_state()

    # ========================= Domain randomization ========================= #

    def _draw_delay(self, n: int) -> torch.Tensor:
        """Draw n per-env latency offsets in [0, random_delay).

        Uses the dedicated generator when ``delay_seed > 0`` (re-seeded to the SAME
        value every episode so the band is sampled identically across all compared
        runs), else the global torch RNG (training)."""
        if self._delay_gen is not None:
            self._delay_gen.manual_seed(self._delay_seed)
            return torch.rand(n, generator=self._delay_gen) * self.randomization.random_delay
        return torch.rand(n, device=self.device, dtype=self.dtype) * self.randomization.random_delay

    def randomize_params(self, env_ids: torch.Tensor | None = None) -> tuple[bool, bool]:
        """Randomize physical params and command latency for the given envs.

        Returns (mass_changed, inertia_changed) so an engine-coupled host can
        mirror the changed body params into the engine (IsaacLab:
        :meth:`IsaacBridge.push_body_params`).
        """
        rd = self.randomization
        ps = self.param_server
        ps.apply_random_quad_mass(rd.random_mass, env_ids)
        ps.apply_random_quad_inertia(rd.random_inertia, env_ids)
        ps.apply_random_drag(rd.random_drag, env_ids)
        ps.apply_random_thrust(rd.random_thrust_map, env_ids)
        ps.apply_random_torque(rd.random_torque, env_ids)
        if rd.random_delay != 0.0:
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            self.random_delay[env_ids] = self._draw_delay(env_ids.numel())
        return rd.random_mass != 0.0, rd.random_inertia != 0.0

    def clear_random_delay(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear the per-episode command-latency offset for selected environments."""
        if env_ids is None:
            self.random_delay.zero_()
        else:
            self.random_delay[env_ids] = 0.0

    def sample_delay_only(self, env_ids: torch.Tensor | None = None) -> None:
        """Sample a fresh per-env command-latency offset (U(0, random_delay)) — WITHOUT
        perturbing physical params/init. So the canonical eval exercises the same [30,50] ms
        delay band as training while keeping mass/thrust nominal for a fair comparison."""
        if self.randomization.random_delay == 0.0:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.random_delay[env_ids] = self._draw_delay(env_ids.numel())

    def sample_init_perturbation(
        self, n: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample n init-state perturbations from the configured ranges.

        Returns (pos [m], delta attitude quaternion wxyz, vel [m/s],
        ome [rad/s]); all-zero config yields zero perturbations / identity
        quaternions.
        """

        def _uniform(dim: int) -> torch.Tensor:
            return 2.0 * torch.rand(n, dim, device=self.device, dtype=self.dtype) - 1.0

        pos_pert = _uniform(3) * self.init_pos_rd * self.init_pos_rd_scale
        rpy_rad = (
            _uniform(3) * self.init_att_rd * self.init_att_rd_scale * (math.pi / 180.0)
        )
        q_delta = quat_from_euler_xyz(rpy_rad[:, 0], rpy_rad[:, 1], rpy_rad[:, 2])
        vel_pert = _uniform(3) * self.init_vel_rd * self.init_vel_rd_scale
        ome_pert = (
            _uniform(3) * self.init_ome_rd * (math.pi / 180.0) * self.init_ome_rd_scale
        )
        return pos_pert, q_delta, vel_pert, ome_pert

    def perturb_init_state(self, env_ids: torch.Tensor | None = None) -> None:
        """Perturb pos/att/vel/ome in place with sampled init randomization."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        pos_pert, q_delta, vel_pert, ome_pert = self.sample_init_perturbation(
            env_ids.numel()
        )
        qs = self.quad_state
        qs.pos[env_ids] += pos_pert
        qs.att[env_ids] = quat_mul(q_delta, qs.att[env_ids])
        qs.vel[env_ids] += vel_pert
        qs.ome[env_ids] += ome_pert

    def _check_finite_wrench(self, sim_time: float) -> None:
        # Fail fast: a non-finite analytic wrench means the integrator diverged
        # upstream. Never feed it to the host engine (it would silently corrupt
        # the sim); a task must clamp its state so this cannot happen.
        ft = self.quad_state.force_torque
        if torch.isfinite(ft).all():
            return
        qs = self.quad_state
        bad = (~torch.isfinite(ft).all(dim=1)).nonzero().flatten()

        def _mx(t):
            tf = t[torch.isfinite(t)]
            return float(tf.abs().max()) if tf.numel() else float("nan")

        raise RuntimeError(
            f"[RPG] Non-finite analytic wrench: {int(bad.numel())} env(s) {bad[:6].tolist()} "
            f"at sim_time={sim_time:.4f}s. Finite maxima -> |vel|={_mx(qs.state[:, 7:10]):.1f} "
            f"|ome|={_mx(qs.state[:, 10:13]):.1f} |mot|={_mx(qs.state[:, 19:23]):.1f} "
            f"|force|={_mx(qs.force):.1f} |torque|={_mx(qs.torque):.3f}. The RPG integrator diverged; "
            f"clamp the task state to prevent this."
        )

    # ========================= Component builders ========================= #

    def _build_low_level_controller(self, cmd_tensor: torch.Tensor | None = None):
        controller_type = self.params.controller
        controller_params = self.params.controller_params
        print(f"[RPG_Simulator] Creating low-level controller: {controller_type}")
        if controller_type == "LowLevelControllerSimple":
            return LowLevelControllerSimple(
                params=controller_params,
                quad_params=self.param_server,
                quad_state=self.quad_state,
                cmd_tensor=cmd_tensor,
                device=self.device,
                dtype=self.dtype,
            )
        elif controller_type == "LowLevelControllerBetaFlight":
            return LowLevelControllerBetaFlight(
                params=controller_params,
                quad_params=self.param_server,
                quad_state=self.quad_state,
                cmd_tensor=cmd_tensor,
                ctrl_dt=self.params.integrator_dt,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            raise ValueError(f"Unknown low-level controller type: {controller_type}")

    def _build_model_pipeline(self):
        pipeline = []
        print("[RPG_Simulator] Creating model pipeline:")
        for model_name, model_params in self.params.model_pipeline.items():
            print(f"[RPG_Simulator] \tCreating model: {model_name}")
            if model_name == "ModelInit":
                model = ModelInit(
                    quad_state=self.quad_state,
                    num_env=self.num_envs,
                    device=self.device,
                    dtype=self.dtype,
                )
            elif model_name == "ModelMotor":
                model = ModelMotor(
                    motor_params=model_params,
                    quad_params=self.param_server,
                    quad_state=self.quad_state,
                    num_env=self.num_envs,
                    device=self.device,
                    dtype=self.dtype,
                )
                self.motor_tau = (
                    model_params.time_constant
                )  # override motor time constant
            elif model_name == "ModelBattery":
                quad_params = self.params.quad_params
                if model_params.cell_series != quad_params.cell_series:
                    raise ValueError(
                        f"ModelBattery cell_series={model_params.cell_series} does"
                        f" not match quad_params.battery_type="
                        f"{quad_params.battery_type!r} ({quad_params.cell_series}s)"
                    )
                model = ModelBattery(
                    battery_params=model_params,
                    quad_state=self.quad_state,
                    num_env=self.num_envs,
                    device=self.device,
                    dtype=self.dtype,
                )
            elif model_name == "ModelThrustTorqueSimple":
                model = ModelThrustTorqueSimple(
                    quad_params=self.param_server,
                    quad_state=self.quad_state,
                    num_env=self.num_envs,
                    device=self.device,
                    dtype=self.dtype,
                )
            elif model_name == "ModelRotorSimpleAugmentation":
                model = ModelRotorSimpleAugmentation(
                    params=model_params,
                    quad_params=self.param_server,
                    quad_state=self.quad_state,
                    num_env=self.num_envs,
                    device=self.device,
                    dtype=self.dtype,
                )
            elif model_name == "ModelRigidBody":
                if not self.using_agisim:
                    continue
                model = ModelRigidBody(
                    quad_params=self.param_server,
                    quad_state=self.quad_state,
                    num_env=self.num_envs,
                    device=self.device,
                    dtype=self.dtype,
                )
            elif model_name == "ModelScale":
                model = ModelScale(
                    quad_params=self.param_server,
                    quad_state=self.quad_state,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                raise ValueError(f"Unknown model type: {model_name}")
            pipeline.append(model)
        assert len(pipeline) > 0, "Model pipeline is empty!"
        return pipeline

    def _build_integrator(self):
        """Engine-coupled mode: always Euler (RK4 is unsupported with the
        engine coupling). agisim mode: pick from ``params.integrator``."""
        integrator_dt = self.params.integrator_dt
        if not self.using_agisim:
            requested = self.params.integrator
            if requested != "IntegratorEuler":
                print(
                    f"[RPG_Simulator] Engine-coupled backend only supports Euler — "
                    f"ignoring rpg_sim_params.integrator={requested!r}."
                )
            cls = IntegratorEuler
        else:
            integrator_type = self.params.integrator
            if integrator_type == "IntegratorRK4":
                cls = IntegratorRK4
            elif integrator_type == "IntegratorEuler":
                cls = IntegratorEuler
            else:
                raise ValueError(f"Unknown integrator type: {integrator_type}")

        backend = "agisim" if self.using_agisim else "engine"
        print(
            f"[RPG_Simulator] Creating {cls.__name__} with dt={integrator_dt} "
            f"(backend={backend})"
        )
        return cls(
            dt=integrator_dt,
            quad_state=self.quad_state,
            device=self.device,
            dtype=self.dtype,
        )
