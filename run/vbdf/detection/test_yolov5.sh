#!/bin/bash
# Score a trained detector on one dataset split (yolov5 val.py). SPLIT=train
# (default) evaluates on the training set; switch to the validation set with
# SPLIT=val once dataset/validation exists. Extra val.py flags pass through:
#   ./test_yolov5.sh
#   SPLIT=val WEIGHTS=runs/yolov5n_synth/weights/best.pt ./test_yolov5.sh
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
DATASET="${DATASET:-$HERE/dataset}"
WEIGHTS="${WEIGHTS:-$HERE/runs/yolov5n_synth/weights/best.pt}"
SPLIT="${SPLIT:-train}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
[ -f "$WEIGHTS" ] || { echo "error: no weights at $WEIGHTS — run train_yolov5.sh first" >&2; exit 1; }
VAL_SPLIT="$([ -d "$DATASET/validation" ] && echo validation || echo train)"
printf 'path: %s\ntrain: train\nval: %s\nnames:\n  0: drone\n' "$DATASET" "$VAL_SPLIT" > "$DATASET/drone.yaml"
cd "$HERE/drone_detect_yolov5"
"$PY" val.py --data "$DATASET/drone.yaml" --weights "$WEIGHTS" --task "$SPLIT" \
    --img "${IMG:-1280}" --device "${DEVICE:-0}" --single-cls \
    --project "$HERE/runs" --name "eval_$SPLIT" --exist-ok "$@"
