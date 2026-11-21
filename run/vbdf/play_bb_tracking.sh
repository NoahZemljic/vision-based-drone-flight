#!/bin/bash
# Play the trained Rpg-Vbdf-BB-Tracking-v0 policy (BB-i08) through its Viz variant and
# record a 3rd-person rollout video. Arguments are name=value, e.g.:
#   ./play_bb_tracking.sh num_envs=4 video_length=400
set -e

SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SOURCE_DIR/../_lib.sh"
rpg_translate_args "$@"

CHECKPOINT="${CHECKPOINT:-$SOURCE_DIR/logs/sb3/Rpg-Vbdf-BB-Tracking-v0/2026-09-14_17-59-42_BB-i08/model.zip}"

cd "$SOURCE_DIR"
rpg_exec "$PYTHON" "$SOURCE_DIR/../../scripts/sb3/play.py" \
    --task Rpg-Vbdf-BB-Tracking-Viz-v0 \
    --checkpoint "$CHECKPOINT" \
    --num_envs 4 \
    --viz newton \
    --video \
    env.test_env=true \
    "${TRANSLATED[@]}"
