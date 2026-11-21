#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_path="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
"$python_path" -m pip install -e "$repo_dir/source/rpg_isaac"
