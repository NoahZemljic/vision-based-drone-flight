#!/bin/bash
# Play a trained Rpg-Vbdf-Hover-v0 policy (sb3); the latest run's model is auto-loaded
# from logs/sb3/Rpg-Vbdf-Hover-v0. Arguments are name=value, e.g.:
#   ./play_hover.sh viz=kit                          # full Kit viewer instead of the GL default
#   ./play_hover.sh num_envs=16 checkpoint=<path/to/model.zip>
set -e

SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SOURCE_DIR/../_lib.sh"
rpg_translate_args "$@"

cd "$SOURCE_DIR"
rpg_exec "$PYTHON" "$SOURCE_DIR/../../scripts/sb3/play.py" \
    --task Rpg-Vbdf-Hover-v0 \
    --num_envs 4 \
    --viz newton \
    env.test_env=true \
    "${TRANSLATED[@]}"
