#!/bin/bash
# Train the VBDF actor-tracking task Rpg-Vbdf-Tracking-v0
# (SB3 PPO, large-scale recipe: 4096 envs x 100 steps, 4 minibatches, tanh
# policy, obs normalisation; ~14 min on one GPU). Arguments are name=value, e.g.:
#   ./train_tracking.sh run_name=tracking_solution
#   ./train_tracking.sh num_envs=100 agent.n_timesteps=100_000_000 env.<field>=<value>
set -e

SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SOURCE_DIR/../_lib.sh"
rpg_translate_args "$@"

# cd so train.py's CWD-relative log path resolves under run/vbdf/logs.
cd "$SOURCE_DIR"
rpg_exec "$PYTHON" "$SOURCE_DIR/../../scripts/sb3/train.py" \
    --task Rpg-Vbdf-Tracking-v0 \
    --num_envs 4096 \
    --viz none \
    --log_project_name rpg_vbdf \
    "${TRANSLATED[@]}"
