#!/bin/bash
# Fine-tune the course's YOLOv5 fork (vendored in ./drone_detect_yolov5) on the
# synthetic dataset, inside the container with Isaac Sim's python. Extra
# train.py flags pass through, config via env vars:
#   ./train_yolov5.sh                                        # dataset/ next to this script
#   DATASET=/path EPOCHS=3 IMG=640 DEVICE=0 ./train_yolov5.sh
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
DATASET="${DATASET:-$HERE/dataset}"
# yolov5 auto-logs to any wandb login it finds; opt in with WANDB_MODE=online
export WANDB_MODE="${WANDB_MODE:-disabled}"
[ -d "$DATASET/train" ] || { echo "error: no $DATASET/train — render + label a dataset first" >&2; exit 1; }
# epoch-end val runs on validation/ when generated, else scores the train split
VAL_SPLIT="${VAL_SPLIT:-$([ -d "$DATASET/validation" ] && echo validation || echo train)}"
# yolov5 resolves relative dataset paths against its own datasets dir; write an
# absolute-path data yaml next to the dataset instead.
printf 'path: %s\ntrain: train\nval: %s\nnames:\n  0: drone\n' "$DATASET" "$VAL_SPLIT" > "$DATASET/drone.yaml"
cd "$HERE/drone_detect_yolov5"
"$PY" train.py --data "$DATASET/drone.yaml" --img "${IMG:-1280}" --device "${DEVICE:-0}" \
    --batch "${BATCH:-16}" --weights yolov5n.pt --cfg models/yolov5n.yaml --single-cls \
    --project "$HERE/runs" --name yolov5n_synth --exist-ok --epochs "${EPOCHS:-60}" "$@"
