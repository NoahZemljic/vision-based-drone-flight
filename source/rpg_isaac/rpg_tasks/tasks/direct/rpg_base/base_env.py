from __future__ import annotations

import time

import torch
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from rpg_simulator.simulator import QuadrotorSimulator
from rpg_tasks.tasks.direct.rpg_base.base_env_cfg import RPGBaseEnvCfg, resolve_backend, resolve_drone
from rpg_tasks.tasks.direct.rpg_base.isaac_bridge import IsaacBridge
from rpg_vision.onboard_camera_cfg import RPG_NEWTON_RENDERER_TYPE, OnboardCameraCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.sensors import Camera, ContactSensor


class RPGBaseEnv(DirectRLEnv):
    cfg: RPGBaseEnvCfg

    def __init__(
        self,
        cfg: RPGBaseEnvCfg,
        init_state=None,
        render_mode: str | None = None,
        **kwargs,
    ):
        camera = cfg.onboard_camera
        if camera is not None and not isinstance(camera, OnboardCameraCfg):
            raise RuntimeError(
                "cfg.onboard_camera is still an unresolved camera preset. Load RPG tasks through "
                "isaaclab_tasks.utils.resolve_task_config() before launch_simulation(); no RPG-specific "
                "camera preparation call is required."
            )
        if camera is not None and camera.renderer_cfg.renderer_type == RPG_NEWTON_RENDERER_TYPE:
            # Registers the RPG raytracer with the renderer factory before the
            # Camera is constructed. Deferred to here because it imports newton
            # (~1.7 s), which RTX runs never need.
            import rpg_vision.rpg_newton_renderer  # noqa: F401, PLC0415

        resolve_drone(cfg)
        self.backend = resolve_backend(cfg.backend)
        self.using_agisim = self.backend == "agisim"
        print(
            f"\033[1;32m[RPG_Simulator] integration backend: {self.backend} "
            f"(scene host: {type(cfg.sim.physics).__name__})\033[0m"
        )
        if (
            cfg.scene.env_spacing == 0.0
            and not self.using_agisim
            and isinstance(cfg.sim.physics, (PhysxCfg, OvPhysxCfg))
        ):
            print(
                f"\033[1;31m[WARNING] env_spacing=0.0 on a PhysX-integrated host: all envs clone at the "
                f"origin, so the GPU broadphase enumerates every drone-drone pair (N*(N-1)/2) before "
                f"collision-group filtering and overflows the contact/pair buffers at scale "
                f"(num_envs={cfg.scene.num_envs}). Set scene.env_spacing > 0 for PhysX training.\033[0m"
            )

        # lazy fabric-write state (kit-physx agisim render sync):
        # None=uninit, False=disabled after failure, tuple=(attrs, offsets)
        self._fabric_write_ctx = None

        super().__init__(cfg, render_mode, **kwargs)
        self._warn_frozen_kitless_viewer()

        if self._onboard_camera is not None:
            cam = cfg.onboard_camera
            print(
                f"\033[1;32m[RPG_Simulator] onboard camera: "
                f"{cam.width}x{cam.height} {cam.spawn.projection_type} "
                f"mount_pos_b_m={cam.offset.pos} renderer={cam.renderer_cfg.renderer_type}\033[0m"
            )
        # Each renderer writes its own AOV name; policies read observations["rgb"].
        if self._onboard_camera is not None:
            camera_output = self._onboard_camera.data.output
            colour_aov = cfg.onboard_camera.rgb_aov
            if "rgb" not in camera_output and colour_aov in camera_output:
                camera_output["rgb"] = camera_output[colour_aov]

        # 1. Setup Robot
        body_ids = self.robot.find_bodies("body")[0]
        self.quad_body_id = body_ids if self.using_agisim else body_ids[0]
        self.num_env = self.cfg.scene.num_envs
        self.delay = self.cfg.delay
        self.ctrl_dt = self.cfg.ctrl_dt
        self.integrator_dt = self.cfg.integrator_dt

        self.dtype = torch.float32 if self.cfg.quad_state_dtype == "float32" else torch.float64
        print(f"[RPG_Simulator] Using dtype: {self.dtype}")

        # 2. Setup the RPG simulator (state + LLC + model pipeline + integrator + domain randomization of its params/state).
        # Pure torch/warp — everything that crosses the simulator <-> Isaac boundary goes through self.bridge below.
        self.rpg_sim_params = self.cfg.rpg_sim_params
        self.quadsim = QuadrotorSimulator(
            self.rpg_sim_params,
            num_envs=self.robot.num_instances,
            device=self.device,
            dtype=self.dtype,
            using_agisim=self.using_agisim,
            init_state=init_state,
            randomization=cfg.quad_randomization_cfg,
        )
        # env-side phases time into the same profile dict
        self.timer_dict = self.quadsim.timer_dict

        # 3. Bridge: binds quad_state to the articulation (root-state pulls/pushes, wrench transfer, randomized body params).
        # agisim only ever pushes poses, for viz/sensors.
        self.bridge = IsaacBridge(
            robot=self.robot,
            quad_state=self.quadsim.quad_state,
            body_id=self.quad_body_id,
            param_server=self.quadsim.param_server,
        )
        if init_state is not None and not self.using_agisim:
            self.bridge.push_root_state()

        # 4. Domain randomization policy: quadsim owns the how (coeffs, sampling); the env only decides who/when (train envs on reset).
        self.test_env = cfg.test_env
        self.eval_env_idx = int(getattr(cfg, "eval_env_idx", 0))
        self._eval_env_mask = torch.zeros(self.num_env, dtype=torch.bool, device=self.device)
        if 0 <= self.eval_env_idx < self.num_env:
            self._eval_env_mask[self.eval_env_idx] = True

    @property
    def quad_state(self):
        """The rpg quad state (read/write view owned by :attr:`quadsim`)."""
        return self.quadsim.quad_state

    # ========================= IsaacLab DirectRLEnv Functions ========================= #
    def _setup_scene(self):
        # Scene-declared assets are constructed and cloned by InteractiveScene.
        # This path is required for IsaacLab 3 heterogeneous clone plans.
        scene_managed_robot = "robot" in self.scene.articulations
        self.robot = self.scene.articulations["robot"] if scene_managed_robot else Articulation(self.cfg.robot)
        if self.using_agisim and not self.cfg.deployment_mode:
            self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
            self.scene.sensors["contact_sensor"] = self.contact_sensor
        # Onboard camera spawned here so its parent prim (Robot/body) exists on env_0; clone_environments below replicates it to all envs.
        if self.cfg.onboard_camera is not None:
            self._onboard_camera = Camera(self.cfg.onboard_camera)
            self.scene.sensors["onboard_camera"] = self._onboard_camera
        else:
            self._onboard_camera = None
        # clone and replicate
        if not scene_managed_robot:
            self.scene.clone_environments(copy_from_source=False)

        # Disable collisions across environments: PhysX-based hosts put all envs in one collision world (Newton isolates envs as separate worlds,
        # so no filtering is needed there and the PhysX schemas don't apply).
        # Shared colliders (e.g. a global ground) stay enabled via the hook.
        if "physx" in self.scene.physics_backend:
            self.scene.filter_collisions(self._global_collider_paths())
        # add articulation to scene
        if not scene_managed_robot:
            self.scene.articulations["robot"] = self.robot

        # add default light — tasks can override in their own _setup_scene
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _global_collider_paths(self) -> list[str]:
        """Global collider prims every env keeps colliding with (e.g. a shared ground).

        Consumed by the inter-env collision filtering in :meth:`_setup_scene`.
        Collision groups reference paths, so the prims may be spawned later in a
        task's ``_setup_scene`` — they only need to exist when the sim starts.
        """
        return []

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions are collective thrust and body rates
        start_t = time.time()

        current_time = self._sim_step_counter * self.quadsim.integrator.dt
        self.quadsim.push_command(actions, current_time)

        self.timer_dict["pre_physics(geo_control)"] += time.time() - start_t

    def _apply_action(self) -> None:
        # LLC and integrator running at 1000 Hz
        self.quadsim.run_substep(self._sim_step_counter * self.quadsim.integrator.dt)

        if self.using_agisim:
            return

        # analytic wrench -> engine (composed lazily on read in write_data_to_sim)
        start_t = time.time()
        self.bridge.write_wrench()
        self.timer_dict["set_external_force_and_torque"] += time.time() - start_t

    def _get_observations(self) -> dict:
        pass

    def _get_rewards(self) -> torch.Tensor:
        pass

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # No early termination conditions - let episodes run full length
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Check if episodes have reached maximum length
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(
        self,
        env_ids: torch.Tensor | None,
        reset_pos=None,
        reset_att=None,
        reset_vel=None,
        reset_ome=None,
    ):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self.quadsim.reset(
            env_ids,
            pos=reset_pos,
            att=reset_att,
            vel=reset_vel,
            ome=reset_ome,
        )
        if not self.using_agisim:
            self.bridge.push_root_state(env_ids)

        # Canonical eval / test envs: physical DR stays OFF (deterministic init), but the
        # command-latency band MUST be sampled so eval always sees the same real-world
        # [30,50] ms latency as training (never cleared).
        if self.test_env:
            self.quadsim.sample_delay_only(env_ids)
            return

        train_env_ids = env_ids[~self._eval_env_mask[env_ids]]
        if train_env_ids.numel() == 0:
            return
        # rpg-side randomization: mass/inertia/drag/thrust/torque + delay
        mass_rd, inertia_rd = self.quadsim.randomize_params(train_env_ids)
        if not self.using_agisim and (mass_rd or inertia_rd):
            # engine integrates the dynamics -> mirror the changed body params
            self.bridge.push_body_params(train_env_ids, mass=mass_rd, inertia=inertia_rd)
        self.apply_random_init_state(env_ids=train_env_ids)

    @staticmethod
    def _dynamics_timing_dict() -> dict[str, float]:
        return {
            "[step] pre_decimation_t": 0.0,
            "[step] decimation_apply_action_t": 0.0,
            "[step] decimation_write_data_t": 0.0,
            "[step] decimation_sim_step_t": 0.0,
            "[step] decimation_render_t": 0.0,
            "[step] decimation_scene_update_t": 0.0,
            "[step] decimation_gpu_completion_t": 0.0,
        }

    def _step_dynamics(
        self,
        action: torch.Tensor,
        timings: dict[str, float] | None = None,
        *,
        camera_only: bool = False,
    ) -> None:
        """Advance control, dynamics, rendering, and required sensor buffers."""
        start_t = time.perf_counter() if timings is not None else 0.0
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)
        self._pre_physics_step(action)

        if camera_only:
            pose_sensors = self._onboard_camera is not None
            sync_scene = pose_sensors
        else:
            pose_sensors = any(k != "contact_sensor" for k in self.scene.sensors)
            sync_scene = self.sim.is_rendering or pose_sensors
        if timings is not None:
            timings["[step] pre_decimation_t"] += time.perf_counter() - start_t

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1

            start_t = time.perf_counter() if timings is not None else 0.0
            self._apply_action()
            if timings is not None:
                timings["[step] decimation_apply_action_t"] += time.perf_counter() - start_t

            start_t = time.perf_counter() if timings is not None else 0.0
            if not self.using_agisim:
                self.scene.write_data_to_sim()
            if timings is not None:
                timings["[step] decimation_write_data_t"] += time.perf_counter() - start_t

            is_render_substep = self._sim_step_counter % self.cfg.sim.render_interval == 0
            start_t = time.perf_counter() if timings is not None else 0.0
            if self.using_agisim:
                if sync_scene and is_render_substep:
                    if (
                        camera_only
                        and getattr(self.sim.physics_manager, "_fabric", None) is not None
                        and self._fabric_write_ctx is not False
                    ):
                        self._write_fabric_transforms()
                    else:
                        self.bridge.push_root_state()
                        self._sync_render_state()
            else:
                self.sim.step(render=False)
            if timings is not None:
                timings["[step] decimation_sim_step_t"] += time.perf_counter() - start_t

            start_t = time.perf_counter() if timings is not None else 0.0
            if is_render_substep and sync_scene:
                self.sim.render()
            if timings is not None:
                timings["[step] decimation_render_t"] += time.perf_counter() - start_t

            start_t = time.perf_counter() if timings is not None else 0.0
            if not self.using_agisim:
                self.scene.update(dt=self.physics_dt)
            elif is_render_substep and pose_sensors:
                sensor_dt = self.physics_dt * self.cfg.sim.render_interval
                if camera_only:
                    self._onboard_camera.update(dt=sensor_dt)
                else:
                    self.scene.update(dt=sensor_dt)
            if timings is not None:
                timings["[step] decimation_scene_update_t"] += time.perf_counter() - start_t

            if timings is not None and is_render_substep and str(self.device).startswith("cuda"):
                start_t = time.perf_counter()
                torch.cuda.synchronize(self.device)
                timings["[step] decimation_gpu_completion_t"] += time.perf_counter() - start_t

            if self.using_agisim:
                self.quadsim.refresh_caches()
            else:
                self.bridge.pull_root_state()

    def deployment_step(self, action: torch.Tensor, *, profile: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
        """Advance a vision policy without RL rewards, terminations, resets, or observations."""
        if self._onboard_camera is None:
            raise RuntimeError("deployment_step requires an onboard camera")
        timings = self._dynamics_timing_dict() if profile else None
        self._step_dynamics(action, timings, camera_only=True)
        start_t = time.perf_counter() if timings is not None else 0.0
        rgb = self._onboard_camera.data.output["rgb"][..., :3]
        if timings is not None:
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            timings["[step] camera_output_t"] = time.perf_counter() - start_t
        return rgb, timings or {}

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        self.extras = {
            **self._dynamics_timing_dict(),
            "[step] get_reward_t": 0.0,
            "[step] get_done_t": 0.0,
            "[step] reset_t": 0.0,
            "[step] get_obs_t": 0.0,
        }
        self._step_dynamics(action, self.extras)

        start_t = time.time()

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs

        self.extras["[step] get_done_t"] = time.time() - start_t
        start_t = time.time()

        self.reward_buf = self._get_rewards()

        self.extras["[step] get_reward_t"] = time.time() - start_t
        start_t = time.time()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                self.sim.render()
        self.extras["[step] reset_t"] = time.time() - start_t
        start_t = time.time()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])
        self.extras["[step] get_obs_t"] = time.time() - start_t

        # return observations, rewards, resets and extras
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    # ========================= Domain Randomization Functions ================ #

    def apply_random_init_state(
        self,
        env_ids: torch.Tensor | None = None,
    ):
        """Perturb the reset state in quadsim; tasks override to customize."""
        self.quadsim.perturb_init_state(env_ids)
        if not self.using_agisim:
            self.bridge.push_root_state(env_ids)

    def on_train_log(self, iteration: int, logger) -> None:
        return

    # ======================== Visualizer Helper Functions ==================== #

    def _warn_frozen_kitless_viewer(self):
        """Warn about IsaacLab's frozen kit-less viewers at num_envs=1 on PhysX hosts.

        When the scene-data provider's body order matches Newton's (always true
        with a single env), ``NewtonManager.update_visualization_state`` rebinds
        its output to the PhysX buffer instead of copying into ``_state_0.body_q``
        — the array the newton/rerun/viser viewers draw — so everything renders
        frozen at spawn. A task-local Newton renderer adapter can force a copy;
        otherwise use multiple environments or the Kit viewer.
        """
        if self.num_envs > 1 or not isinstance(self.cfg.sim.physics, PhysxCfg):
            return
        if not any(v != "kit" for v in self.sim.resolve_visualizer_types()):
            return
        print(
            "\033[1;91m[RPG_Simulator] WARNING: PhysX scene host + non-kit visualizer "
            "with num_envs=1 hits an IsaacLab bug — viewers render everything frozen "
            "at spawn poses. Use the RPG racing Newton renderer adapter, "
            "num_envs>=2, or viz=kit.\033[0m"
        )

    def _sync_render_state(self):
        """Push externally-written poses to the renderers on the agisim backend.

        ``physics_manager.forward()`` is the engines' publish-without-stepping
        hook: on Newton it runs eval_fk into ``state_0.body_q`` (kit-less
        viewer source); on kit PhysX it runs the articulation FK (refreshing
        the tensor view the kit-less viewers read) plus physx-fabric
        ``force_update``. The latter does NOT publish articulation transforms
        (only a real physics step does), so the RTX/kit renderer additionally
        needs the drone world matrices written into fabric directly.
        """
        pm = self.sim.physics_manager
        pm.forward()
        if getattr(pm, "_fabric", None) is None or self._fabric_write_ctx is False:
            return
        try:
            self._write_fabric_transforms()
        except Exception:
            self._fabric_write_ctx = False  # permanent: kit render stays stale
            print("[RPG_Simulator] WARNING: fabric transform write failed; kit/RTX view will not track agisim poses")

    def _write_fabric_transforms(self):
        """Write the drone body world matrices straight into fabric (kit PhysX).

        Per-body rest offsets relative to the root link are captured on first
        use from the spawn-state fabric matrices.
        """
        from usdrt import Gf

        if self._fabric_write_ctx is None:
            import omni.usd
            import usdrt

            stage_id = omni.usd.get_context().get_stage_id()
            rt_stage = usdrt.Usd.Stage.Attach(stage_id)
            attrs = []  # (num_envs, num_bodies) fabric worldMatrix attrs
            offsets = []  # per-body rest transform relative to the root link
            root_name = (
                self.robot.body_names[self.quad_body_id[0]]
                if isinstance(self.quad_body_id, list)
                else self.robot.body_names[self.quad_body_id]
            )
            for e in range(self.num_envs):
                env_attrs = []
                root_prim = rt_stage.GetPrimAtPath(f"/World/envs/env_{e}/Robot/{root_name}")
                m_root_inv = root_prim.GetAttribute("omni:fabric:worldMatrix").Get().GetInverse()
                for name in self.robot.body_names:
                    prim = rt_stage.GetPrimAtPath(f"/World/envs/env_{e}/Robot/{name}")
                    attr = prim.GetAttribute("omni:fabric:worldMatrix")
                    if not attr.IsValid():
                        raise RuntimeError(f"no fabric worldMatrix for {name}")
                    env_attrs.append(attr)
                    if e == 0:
                        offsets.append(attr.Get() * m_root_inv)
                attrs.append(env_attrs)
            self._fabric_write_ctx = (attrs, offsets)

        attrs, offsets = self._fabric_write_ctx
        pos = self.quad_state.pos.cpu()
        att = self.quad_state.att.cpu()  # wxyz
        for e in range(self.num_envs):
            q = att[e].tolist()
            m_root = Gf.Matrix4d(1.0)
            m_root.SetRotateOnly(Gf.Quatd(q[0], Gf.Vec3d(q[1], q[2], q[3])))
            p = pos[e].tolist()
            m_root.SetTranslateOnly(Gf.Vec3d(p[0], p[1], p[2]))
            for k, attr in enumerate(attrs[e]):
                attr.Set(offsets[k] * m_root)
