#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export COURSE_XAUTHORITY="${COURSE_XAUTHORITY:-/tmp/drone-seek-course-${UID}.xauth}"
compose=(docker compose -f compose.yaml)
if [[ ${COURSE_GPU:-0} == 1 ]]; then
    compose+=(-f compose.gpu.yaml)
fi

prepare_display() {
    touch "$COURSE_XAUTHORITY"
    chmod 600 "$COURSE_XAUTHORITY"
    if [[ -n ${DISPLAY:-} ]] && command -v xauth >/dev/null; then
        # FamilyWild allows the host cookie to work with the container hostname.
        xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$COURSE_XAUTHORITY" nmerge -
    fi
}

build_image() {
    if [[ -x /usr/libexec/docker/cli-plugins/docker-buildx ]]; then
        # Some Docker Desktop removals leave a broken ~/.docker plugin symlink
        # that shadows the working system plugin.
        /usr/libexec/docker/cli-plugins/docker-buildx build --load \
            --build-arg "BASE_IMAGE=${COURSE_BASE_IMAGE:-rpg_isaac:vbdf-student}" \
            -f Dockerfile -t drone-seek:course ../..
    elif docker buildx version >/dev/null 2>&1; then
        "${compose[@]}" build course
    else
        echo "Docker Buildx is unavailable; using the installed legacy builder."
        DOCKER_BUILDKIT=0 docker build --build-arg "BASE_IMAGE=${COURSE_BASE_IMAGE:-rpg_isaac:vbdf-student}" \
            -f Dockerfile -t drone-seek:course ../..
    fi
}

case "${1:-help}" in
    build) build_image ;;
    up) prepare_display; build_image; "${compose[@]}" up -d --no-build course ;;
    shell) "${compose[@]}" exec course bash ;;
    exec) shift; "${compose[@]}" exec -T course /usr/local/bin/drone-course-entrypoint "$@" ;;
    check) "${compose[@]}" exec -T course /usr/local/bin/drone-course-entrypoint /usr/bin/python3 /usr/local/lib/drone-course-smoke.py ;;
    check-report) "${compose[@]}" exec -T course /usr/local/bin/drone-course-entrypoint /workspace/isaaclab/_isaac_sim/python.sh /usr/local/lib/drone-report-smoke.py ;;
    down) "${compose[@]}" down ;;
    logs) "${compose[@]}" logs --tail 100 course ;;
    *) echo "Usage: $0 {up|build|shell|exec COMMAND...|check|check-report|logs|down}"
       echo "Set COURSE_GPU=1 for NVIDIA; ROS_DOMAIN_ID defaults to 71." ;;
esac
