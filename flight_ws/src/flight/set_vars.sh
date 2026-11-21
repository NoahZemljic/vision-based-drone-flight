#!/bin/sh
# Environment variables for flightmare training with ROS2 agilicious.
# Source this before building flightlib or running training scripts.
#
# Usage:
#   cd <path-to>/flight_jazzy/src/flight
#   source set_vars.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

export AGI_PATH="${SCRIPT_DIR}/agilicious"
export FLIGHTMARE_PATH="${SCRIPT_DIR}/flightmare"
export PYTHONPATH="${PYTHONPATH}:${FLIGHTMARE_PATH}/flightpy"
export PYTHONPATH="${PYTHONPATH}:${FLIGHTMARE_PATH}/flightpy/stable-baselines3"
export PYTHONPATH="${PYTHONPATH}:${FLIGHTMARE_PATH}/flightpy/stable-baselines3-contrib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${AGI_PATH}/agilib/externals/acados-src/lib"