# Learning to Follow a Drone with Reinforcement Learning

Team 5 · _Autonomous Vision-Based Drone Flight_ seminar, September 2026 · [Robotics and Perception Group](https://rpg.ifi.uzh.ch/), University of Zurich

[![Isaac Lab](https://img.shields.io/badge/NVIDIA-Isaac%20Lab-76B900?style=flat&logo=nvidia&logoColor=white)](https://isaac-sim.github.io/IsaacLab/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3%20PPO-blue)](https://stable-baselines3.readthedocs.io/)
[![ROS 2](https://img.shields.io/badge/ROS-2-22314E?logo=ros)](https://docs.ros.org/)

[[Report](<documents/Reinforcement Learning for Drone Tracking.pdf>)] [[Video](https://youtu.be/sBNacU_Ex4M)]

[![Demo video](https://img.youtube.com/vi/sBNacU_Ex4M/maxresdefault.jpg)](https://youtu.be/sBNacU_Ex4M)

This project trains a quadrotor to follow a moving drone at a distance of 1.5 m while keeping it in view of a fixed onboard camera. The controller is a neural network policy trained with Proximal Policy Optimization (PPO) in NVIDIA Isaac Lab, using thousands of parallel simulations on cloud GPUs. The policy outputs collective thrust and body-rate commands, which a low-level controller converts into motor commands.

This repository contains:

- Isaac Lab environments for hovering, state-based tracking, and bounding-box tracking
- Training and evaluation scripts
- Trained checkpoints for 15 policy configurations, each with the command used to train it
- Evaluation results for all 17 configurations in the report, and the code to regenerate its figures
- A Docker setup based on the course image for running the policies

## The task

The project addresses Task 2 (Drone Control) of the seminar, which is set in an aerial-filming scenario. An actor drone flies along a trajectory that is unknown to the controller. A second drone carries a camera and must follow the actor: it should hold the target distance, keep the actor in the camera image, and remain inside a bounded flight volume.

The course defines the task as follows:

> Given the state (position, attitude, linear and angular velocity) of the camera drone and the bounding box of the actor drone in normalized image coordinates, develop a controller using reinforcement learning that tracks the actor at a specified distance (1.5 m).

## Method

We trained the policies with PPO (Stable-Baselines3) in **NVIDIA Isaac Lab** on rented cloud GPUs from [Vast.ai](https://vast.ai/). Isaac Lab simulates the quadrotors on the GPU, so a single machine runs 4,096 tracking environments in parallel, each with randomized initial conditions, actor trajectory, and command delay. A training run collects 50 million transitions.

Following the course's recommendation, the controller was developed in three stages: **hover** at a fixed point (H0–H2), **state-based tracking** with the actor's true relative position and velocity (S0–S5), and **bounding-box tracking** with synthetic detections projected through the calibrated camera model (V0–V3). In every stage the policy outputs collective thrust and body rates. To cope with the 30–50 ms command delay, the tracking policies also observe their recently issued commands, and a low-pass filter and a body-rate penalty keep the flight smooth.

The [report](<documents/Reinforcement Learning for Drone Tracking.pdf>) describes the observations, reward design, and training hyperparameters in detail.

## Results

Each policy was evaluated for 30 s in 32 parallel simulations with three seeds. The table reports the mean across seeds.

| Policy            | Training | Distance error | Angle to actor | Roll-rate std. | Crashes per 32 sims |
| ----------------- | -------- | -------------: | -------------: | -------------: | ------------------: |
| State-based (S5)  | 50 M     |    **0.108 m** |          14.9° |     0.48 rad/s |                 6.0 |
| State-based (S5)  | 100 M    |    **0.080 m** |          14.6° |     0.48 rad/s |                 7.7 |
| Bounding-box (V3) | 50 M     |    **0.205 m** |          15.2° |     0.54 rad/s |                 8.0 |
| Bounding-box (V3) | 100 M    |        0.212 m |          14.7° |     0.66 rad/s |                14.3 |

The main findings are:

- **Both policies track the actor.** The state-based policy holds the target distance to within about 11 cm on average, the bounding-box policy to within about 20 cm.
- **Longer training does not improve reliability.** Doubling the training budget reduced the state-based distance error by 26%, but both policies crashed more often.

The full analysis is in the [report](<documents/Reinforcement Learning for Drone Tracking.pdf>), and [`deliverable/eval_fleet/summary.md`](deliverable/eval_fleet/summary.md) lists the metrics for every configuration.

## Getting started

### Prerequisites

- x86-64 Linux with an NVIDIA GPU
- Docker with Compose and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- The course image `rpg_isaac:vbdf-student`, provided by the course instructors. It contains Isaac Lab, the compiled simulator kernels, and the drone assets.

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/NoahZemljic/vision-based-drone-flight.git
   cd vision-based-drone-flight
   ```

2. **Load the course image:**

   ```bash
   docker load < /path/to/rpg_isaac_vbdf-student.tar.gz
   ```

3. **Build and start the container with GPU support:**

   ```bash
   export COURSE_GPU=1
   ./docker/course/run.sh up
   ./docker/course/run.sh shell
   ```

   The repository is mounted at `/worktree` inside the container. The [container guide](docker/course/GUIDE.md) covers GUI forwarding, rebuilding after code changes, and the Agilicious simulator.

### Run the trained policies

Inside the container:

```bash
cd /worktree

# List the evaluation commands for all available checkpoints without running them
python3 run/vbdf/run_fleet_eval.py --available-only --dry-run

# Evaluate the final state-based (S5) and bounding-box (V3) policies
"$ISAAC_PYTHON" run/vbdf/run_fleet_eval.py --only S5-RS V3-final
```

The evaluation uses the same seeds and settings as the report and writes its results to `outputs/eval_fleet/`.

### Train a policy

Each checkpoint folder contains the command used to train it, for example for the final state-based policy:

```bash
cat deliverable/models/state/S5-RS/command.txt
```

These commands use the paths inside the course container. The launchers in [`run/vbdf/`](run/vbdf/) (`train_hover.sh`, `train_sb_tracking.sh`, `train_bb_tracking.sh`) provide a starting point for new runs.

### Regenerate the report figures (no GPU required)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python deliverable/eval_fleet/rebuild_summary.py   # rebuild the results table
python deliverable/plots/make_plots.py             # redraw the figures
python tools/verify_report_models.py               # check metrics and checkpoints against the report
```

The checkpoints of the two 100-million-transition runs are not included; only their evaluation results are.

> **ROS 2 deployment:** the course pilots in [`flight_ws/`](flight_ws/) expect the original observation layout of the course framework. Running H2, S5, or V3 over ROS 2 requires updating them to match these policies' inputs.

## Repository structure

```
├── deliverable/   Checkpoints (models/), evaluation results (eval_fleet/), report figures (plots/)
├── docker/        Course container (course/) and ROS 2 containers (ros2/)
├── documents/     Project report
├── flight_ws/     ROS 2 packages: policy execution, detector integration, visualization
├── run/vbdf/      Training and playback launchers, seeded evaluation, synthetic detections
├── scripts/sb3/   Stable-Baselines3 training and playback
├── source/        Isaac Lab extension: environments, quadrotor dynamics, camera model
└── tools/         Report verification and tests
```
