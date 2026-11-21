#!/bin/bash
# Train the state-based tracking task Rpg-Vbdf-SB-Tracking-v0.
# Arguments are name=value. Report settings are saved in deliverable/models/*/*/command.txt.
set -e

SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SOURCE_DIR/../_lib.sh"
rpg_translate_args "$@"

cd "$SOURCE_DIR"
rpg_exec "$PYTHON" "$SOURCE_DIR/../../scripts/sb3/train.py" \
    --task Rpg-Vbdf-SB-Tracking-v0 \
    --num_envs 4096 \
    --viz none \
    --log_project_name rpg_vbdf \
    "${TRANSLATED[@]}"
