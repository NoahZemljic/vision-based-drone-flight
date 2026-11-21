# Course container for this checkout

This setup follows the course handout.
It extends `rpg_isaac:vbdf-student` with this repository's source and rebuilds
`detection_msgs`, `flightpilot`, and `drone_detect_deployment` against the
Agilicious binaries supplied by the instructors.

## Start and enter

Run these commands from the repository root on an x86_64 Linux host with Docker
and Compose. If the base image is missing, obtain the course tarball from the
instructors and load it first:

```bash
docker load < /path/to/rpg_isaac_vbdf-student.tar.gz
```

```bash
./docker/course/run.sh up
./docker/course/run.sh check
./docker/course/run.sh shell
```

`up` builds the image and leaves a container running. Repeat `shell` in each
terminal; ROS, the workspace, and `set_vars.sh` are sourced automatically.
The separate Compose project is `drone-seek-course`; the image is
`drone-seek:course`. The existing `vbdf-student` container is unaffected.

Agilicious and CPU policy inference do not require an NVIDIA GPU. For Isaac
rendering/training, enable the GPU (requires NVIDIA Container Toolkit):

```bash
export COURSE_GPU=1
./docker/course/run.sh up
./docker/course/run.sh exec nvidia-smi
```

The default CPU configuration uses Mesa software OpenGL for RViz. The GPU
overlay enables hardware OpenGL instead.

Run `up` from the local desktop terminal for GUI use. The launcher forwards the
current X11/XWayland authentication cookie without granting unrestricted X access.
After logging out and back in, run `up` again to refresh it. A headless host can
run `check`, but needs a display to open the course GUI.

## Code, checkpoints, and results

- `/workspace/isaaclab/source/rpg_isaac` contains the image's working source.
- `/workspace/flight_ws` points to its complete ROS workspace.
- `/opt/course_ws` contains the rebuilt ROS overlay, sourced after the handout.
- `/worktree` is this host repository, mounted for checkpoints and saved results.

Edit code on the host, then rerun `./docker/course/run.sh up` to rebuild and
recreate the container. Reopen shells afterwards. Code is copied into the image;
host edits are **not automatically active**. Whole-workspace bind mounts would
hide the untracked binaries and assets that this checkout does not contain.
Save new checkpoints, datasets, and rollouts under `/worktree` to retain them
across recreation. Files created there by the container are owned by root.

## Agilicious simulation

In separate container shells, replace `1` with your team number:

```bash
# Terminal 1: actor (course launch convention)
ros2 launch agiros agisim.launch.py quad_name:=actor_1 \
  simulation_config:=kolibri_simulation.yaml is_server:=true
```

```bash
# Terminal 2: camera
ros2 launch agiros agisim.launch.py quad_name:=camera_1 \
  simulation_config:=kolibri_simulation.yaml is_server:=false
```

For each GUI: **Connect → Arm → Start**. Use **Go To Pose** to move the actor.
Use **Land → Off** to finish. Only one simulator should publish `/clock` in a
ROS domain. For a single-camera test, launch the camera with `is_server:=true`.

**Supplied-image limitation:** the installed course AgiSim binary publishes
state but exposes no `/clock` publisher, even with `is_server:=true`. Keep
`use_sim_time=false`, as the supplied policy launches do. The launch commands
above retain the course convention for images that implement the clock server.

```bash
# Terminal 3: inspect the simulation
ros2 topic list
ros2 topic hz /camera_1/agiros_pilot/state
```

The default `ROS_DOMAIN_ID=71` separates this setup from the older container's
domain. Override it before `up` when communicating with a course renderer or
another machine. All participants need the same domain and CycloneDDS. ROS 2
does not use `roscore`.

## Hover-policy walkthrough

Start the camera simulation and take off using the GUI. This launch requires
a **15-input hover checkpoint** with its adjacent `model_vecnormalize.pkl`:

```bash
ros2 launch flightpilot sim_hoverpilot_isaac.launch.py \
  quad_name:=camera_1 \
  checkpoint:=/worktree/path/to/15-input-hover/model.zip \
  rollout_dir:=/worktree/outputs/agilicious-hover
```

In a third shell:

```bash
# Enable policy commands.
ros2 topic pub --once /camera_1/start_policy std_msgs/msg/Bool '{data: true}'
# Stop policy commands and save the rollout.
ros2 topic pub --once /camera_1/start_policy std_msgs/msg/Bool '{data: false}'
```

The final report's H0/H1 checkpoints expect 15 inputs, while H2 expects **47**.
`run_hover_pilot_isaac.py` constructs 15, so H2 requires its command-history
observations to be ported before deployment. Matching dimensions alone do not
validate action mapping, normalization or closed-loop behavior.

## Tracking and detection

The repository includes a virtual actor detector for local testing without an
Isaac renderer. It generates its own moving actor; it does **not** track the
separate `actor_1` AgiSim instance:

```bash
ros2 launch flightpilot virtual_actor_detector.launch.py quad_name:=camera_1
```

Launch a compatible tracking policy in another shell:

```bash
ros2 launch flightpilot sim_trackingpilot_isaac.launch.py \
  quad_name:=camera_1 checkpoint:=/worktree/path/to/model.zip \
  rollout_dir:=/worktree/outputs/agilicious-tracking
```

Use the same `start_policy` commands above. The current deployment node builds
51 observations (15 state + 4 command + 8 × 4 bounding-box history). The reported
`V3-final` policy uses 85 observations, so its observation construction must be
ported to the deployment node before it can run there. The final state policy
`S5-RS` uses 66 observations and also needs a corresponding ROS observation builder. Container setup alone
does not make differently shaped policies compatible.

For Task 1, inspect the detector's required model/image arguments with:

```bash
ros2 launch drone_detect_deployment yolov5.launch.py --show-args
ros2 run rqt_image_view rqt_image_view
```

## Checks and lifecycle

`check` runs the actual AgiSim binary in a separate ROS domain (172), requires
100 finite quad-state messages with advancing timestamps within 30 seconds,
reports clock availability, and shuts down its test process. It does not open
a GUI or assess policy flight quality.

```bash
./docker/course/run.sh logs
./docker/course/run.sh down
```

`down` removes the container; the image, caches, and host results remain.
The launcher can bypass a broken user Buildx symlink using the installed system
plugin, and falls back to Docker's legacy builder if Buildx is unavailable.

## Final-report checks

The course ROS workflow remains active. For the final report's Isaac policies, use a separate check with `COURSE_GPU=1`:

```bash
./docker/course/run.sh check-report
```

This runs H0, S5 50M and V3 50M for 50 steps with their saved overrides and normalization, writing only temporary results. It supplements the original AgiSim `check`; it is not a replacement for the course workflow or the full report evaluation. Neither simulator check was executed during this cleanup because the host lacks the course runtime and an available Docker daemon.

Inside the container, run the full available report evaluation from `/worktree`:

```bash
cd /worktree
"$ISAAC_PYTHON" run/vbdf/run_fleet_eval.py --available-only --write-csv
```

The original S5/V3 100M bundles remain missing. New results go to `outputs/eval_fleet/`, leaving the submitted evidence intact. See the root README for the report results.
