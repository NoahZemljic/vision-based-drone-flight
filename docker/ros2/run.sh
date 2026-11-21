#!/usr/bin/env bash
# ============================================================
# run.sh — Manage rpg_isaac flight container (PC)
#
# Usage:
#   ./run.sh [--suffix NAME] start        # start container and attach
#   ./run.sh [--suffix NAME] up           # start container without attaching
#   ./run.sh [--suffix NAME] build        # build Docker images only
#   ./run.sh [--suffix NAME] shell [user] # open extra shell in running container
#   ./run.sh [--suffix NAME] x11          # check host/container X11 wiring
#   ./run.sh [--suffix NAME] stop         # stop and remove container
#
# The suffix selects the base image, derived image, container, and volume
# namespace. It defaults to ``v30`` but can name any parallel IsaacLab build.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yaml"

# Host paths. Layout: rpg_isaac/docker/ros2/run.sh → rpg_isaac is 2 levels up.
# flight_ws/ is a colcon workspace nested under rpg_isaac (brought out to
# /workspace/flight_ws in the container via bind mount).
export RPG_ISAAC_HOST_PATH="$(cd "$SCRIPT_DIR/../.." && pwd)"
# IsaacLab repo root (rpg_isaac is nested at <isaaclab>/source/rpg_isaac).
export ISAACLAB_HOST_PATH="$(cd "$RPG_ISAAC_HOST_PATH/../.." && pwd)"
FLIGHT_WS_HOST="$RPG_ISAAC_HOST_PATH/flight_ws"

SERVICE="flight"
CONTAINER_USER="${CONTAINER_USER:-user}"
CONTAINER_UID="${CONTAINER_UID:-$(id -u)}"
CONTAINER_GID="${CONTAINER_GID:-$(id -g)}"

# Accept ``--suffix NAME`` anywhere on the command line, mirroring the naming
# convention used by IsaacLab's ``docker/container.py --suffix NAME``.
RPG_ISAAC_SUFFIX="${RPG_ISAAC_SUFFIX:-v30}"
POSITIONAL_ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --suffix)
            if [ "$#" -lt 2 ]; then
                echo "--suffix requires a value" >&2
                exit 2
            fi
            RPG_ISAAC_SUFFIX="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL_ARGS[@]}"

if [[ ! "$RPG_ISAAC_SUFFIX" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
    echo "Invalid suffix '$RPG_ISAAC_SUFFIX': use lowercase letters, digits, '-' or '_'" >&2
    exit 2
fi

PROJECT="${PROJECT:-rpg-isaac-${RPG_ISAAC_SUFFIX}}"
BASE_IMAGE="${BASE_IMAGE:-isaac-lab-base-${RPG_ISAAC_SUFFIX}}"
FLIGHT_IMAGE="${FLIGHT_IMAGE:-rpg_isaac:ros2-${RPG_ISAAC_SUFFIX}}"
CONTAINER_HOSTNAME="${CONTAINER_HOSTNAME:-rpg-isaac-${RPG_ISAAC_SUFFIX}}"

export CONTAINER_USER CONTAINER_UID CONTAINER_GID PROJECT BASE_IMAGE FLIGHT_IMAGE CONTAINER_HOSTNAME RPG_ISAAC_SUFFIX

X11_AUTH_TARGET=/tmp/rpg-isaac.xauth
RPG_X11_AUTHORITY="$SCRIPT_DIR/container_home/.${PROJECT}.xauth"
export RPG_X11_AUTHORITY

# ── GPU overlay ─────────────────────────────────────────────
# RPG_GPU: auto (default, detect) | 1 (require) | 0 (CPU-only). The reservation
# is additive because a compose override cannot clear reservations.devices.
RPG_GPU="${RPG_GPU:-auto}"
COMPOSE_ARGS=(-f "$COMPOSE_FILE")

has_nvidia_runtime() {
    command -v nvidia-container-runtime-hook &>/dev/null \
        || command -v nvidia-ctk &>/dev/null \
        || docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia
}

case "$RPG_GPU" in
    auto) if has_nvidia_runtime; then GPU_MODE=gpu; else GPU_MODE=cpu; fi ;;
    1 | true | gpu) GPU_MODE=gpu ;;
    0 | false | cpu) GPU_MODE=cpu ;;
    *)
        echo "Invalid RPG_GPU='$RPG_GPU' (use auto, 1 or 0)" >&2
        exit 2
        ;;
esac

if [ "$GPU_MODE" = gpu ]; then
    COMPOSE_ARGS+=(-f "$SCRIPT_DIR/docker-compose.gpu.yaml")
elif [ "$RPG_GPU" = auto ]; then
    echo "[run.sh] No NVIDIA container runtime found — starting CPU-only." >&2
    echo "         device=cpu training and the kit-less Task 1 renderer work; Kit (viz=kit) does not." >&2
    echo "         RPG_GPU=1 forces the GPU reservation anyway." >&2
fi

dc() {
    docker compose --project-name "$PROJECT" "${COMPOSE_ARGS[@]}" "$@"
}

is_running() {
    docker ps --format '{{.Names}}' | grep -q "^${1}$"
}

find_x11_display() {
    if [ -n "${DISPLAY:-}" ]; then
        printf '%s\n' "$DISPLAY"
        return
    fi

    local host_user display socket socket_uid
    host_user="$(id -un)"
    display="$(who | awk -v user="$host_user" '$1 == user && $2 ~ /^:[0-9]+$/ { print $2; exit }')"
    if [ -n "$display" ]; then
        printf '%s\n' "$display"
        return
    fi

    # Last-resort local-session detection. Prefer the lowest X socket owned by
    # the invoking user (normally :1 rather than an NX helper such as :1001).
    while IFS= read -r socket; do
        socket_uid="$(stat -c '%u' "$socket" 2>/dev/null || true)"
        if [ "$socket_uid" = "$CONTAINER_UID" ]; then
            printf ':%s\n' "${socket##*/X}"
            return
        fi
    done < <(find /tmp/.X11-unix -maxdepth 1 -type s -name 'X*' 2>/dev/null | sort -V)
}

find_x11_authority() {
    local display="$1" candidate
    local -a candidates=()

    [ -n "${XAUTHORITY:-}" ] && candidates+=("$XAUTHORITY")
    candidates+=("$HOME/.Xauthority" "/run/user/$CONTAINER_UID/gdm/Xauthority")
    if [ -d "$HOME/.nx" ]; then
        while IFS= read -r candidate; do
            candidates+=("$candidate")
        done < <(find "$HOME/.nx" -type f -name authority 2>/dev/null | sort)
    fi

    for candidate in "${candidates[@]}"; do
        if [ -r "$candidate" ] && [ -n "$(xauth -f "$candidate" nlist "$display" 2>/dev/null)" ]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
}

configure_x11() {
    mkdir -p "$SCRIPT_DIR/container_home"
    : > "$RPG_X11_AUTHORITY"
    chmod 600 "$RPG_X11_AUTHORITY"

    if [ "${RPG_X11:-auto}" = "0" ]; then
        RPG_X11_DISPLAY=""
        export RPG_X11_DISPLAY
        echo "[run.sh] X11 disabled (RPG_X11=0); use viz=none for Kit-camera workloads."
        return
    fi
    if ! command -v xauth >/dev/null; then
        echo "[run.sh] X11 unavailable: install the host package 'xauth', or use viz=none." >&2
        RPG_X11_DISPLAY=""
        export RPG_X11_DISPLAY
        return
    fi

    local display source cookie
    display="$(find_x11_display)"
    if [ -z "$display" ]; then
        echo "[run.sh] X11 unavailable: no active host display found; use viz=none." >&2
        RPG_X11_DISPLAY=""
        export RPG_X11_DISPLAY
        return
    fi
    source="$(find_x11_authority "$display")"
    if [ -z "$source" ]; then
        echo "[run.sh] X11 unavailable: no authorization cookie found for DISPLAY=$display; use viz=none." >&2
        RPG_X11_DISPLAY=""
        export RPG_X11_DISPLAY
        return
    fi

    cookie="$(xauth -f "$source" nlist "$display" 2>/dev/null)"
    printf '%s\n' "$cookie" | sed -e 's/^..../ffff/' | xauth -f "$RPG_X11_AUTHORITY" nmerge -
    RPG_X11_DISPLAY="$display"
    export RPG_X11_DISPLAY

    if command -v xdpyinfo >/dev/null && ! DISPLAY="$display" XAUTHORITY="$RPG_X11_AUTHORITY" xdpyinfo >/dev/null 2>&1; then
        echo "[run.sh] X11 validation failed for DISPLAY=$display; use viz=none." >&2
        RPG_X11_DISPLAY=""
        export RPG_X11_DISPLAY
        return
    fi
    # The container runs as root, so the cookie above is not enough for a client
    # that reconnects with its own XAUTHORITY. Local-only, and not persistent.
    if command -v xhost >/dev/null; then
        DISPLAY="$display" XAUTHORITY="$source" xhost +SI:localuser:root >/dev/null 2>&1 \
            || echo "[run.sh] Could not grant root local X access; GUI apps may fail." >&2
    fi
    echo "[run.sh] X11 ready: DISPLAY=$display"
}

# ── Build image chain if missing ────────────────────────────
ensure_images() {
    local need_build=false

    if ! docker image inspect "$BASE_IMAGE" &>/dev/null; then
        echo "[run.sh] Base image '$BASE_IMAGE' is missing."
        echo "         Build it first via IsaacLab's docker tooling:"
        echo "             cd $ISAACLAB_HOST_PATH && ./docker/container.py build base --suffix $RPG_ISAAC_SUFFIX"
        exit 1
    fi

    if ! docker image inspect "$FLIGHT_IMAGE" &>/dev/null; then
        echo "[run.sh] Image '$FLIGHT_IMAGE' not found — building..."
        need_build=true
    fi

    if [ "$need_build" = true ]; then
        cmd_build
    fi
}

cmd_build() {
    echo "[run.sh] Building Docker image chain on top of '$BASE_IMAGE'..."
    echo ""

    if ! docker image inspect "$BASE_IMAGE" &>/dev/null; then
        echo "── '$BASE_IMAGE' not found. Build it with IsaacLab's tooling first."
        echo "   cd $ISAACLAB_HOST_PATH && ./docker/container.py build base --suffix $RPG_ISAAC_SUFFIX"
        exit 1
    fi

    if ! docker image inspect "$FLIGHT_IMAGE" &>/dev/null; then
        echo "── Building $FLIGHT_IMAGE ───────────────────────────"
        docker build \
            -t "$FLIGHT_IMAGE" \
            -f "$SCRIPT_DIR/Dockerfile.ros2" \
            --build-arg BASE_IMAGE="$BASE_IMAGE" \
            "$SCRIPT_DIR"
        echo ""
    else
        echo "── $FLIGHT_IMAGE already exists (skipping) ──────────"
    fi

    echo "[run.sh] Image chain ready."
}

sync_dotfiles() {
    local image="$FLIGHT_IMAGE"
    local chome="$SCRIPT_DIR/container_home"
    local digest_file="$chome/.image_digest"

    local current_digest cached_digest
    current_digest=$(docker image inspect "$image" --format '{{.Id}}')
    cached_digest=$(cat "$digest_file" 2>/dev/null || echo "")

    if [ "$current_digest" != "$cached_digest" ]; then
        echo "[run.sh] Syncing dotfiles from: $image"
        for f in .bashrc .bash_history .vimrc .gitconfig; do
            if [[ "$f" == *history* ]] && [ -s "$chome/$f" ]; then
                echo "         ~/$f (kept — existing history preserved)"
                continue
            fi
            docker run --rm --entrypoint sh "$image" \
                -c "cat /home/$CONTAINER_USER/$f 2>/dev/null || true" > "$chome/$f"
            echo "         ~/$f"
        done
        echo "$current_digest" > "$digest_file"
    fi
}

check_flight_submodule() {
    local flight_dir="$FLIGHT_WS_HOST/src/flight"
    if [ ! -d "$flight_dir" ] || [ -z "$(ls -A "$flight_dir" 2>/dev/null)" ]; then
        echo "[run.sh] Note: flight submodule not present at:"
        echo "           $flight_dir"
        echo "         Container will start with an empty colcon workspace."
        echo "         To add it later:"
        echo "           cd $RPG_ISAAC_HOST_PATH && \\"
        echo "             git submodule add git@github.com:uzh-rpg/flight.git flight_ws/src/flight && \\"
        echo "             git submodule update --init --recursive"
        echo ""
    fi
}

cmd_up() {
    local container="${PROJECT}_ros2"

    configure_x11

    # Build images if they don't exist
    ensure_images

    # Ensure directories exist (flight_ws layout + dotfile cache).
    mkdir -p "$FLIGHT_WS_HOST/src" \
             "$SCRIPT_DIR/container_home/bash_hist"

    # Heads-up if flight/ is missing, but proceed anyway.
    check_flight_submodule

    # Sync dotfiles if image changed
    sync_dotfiles

    echo "[run.sh] Starting '$SERVICE' (container: $container)..."
    echo "         rpg_isaac host path: $RPG_ISAAC_HOST_PATH"
    echo "         flight_ws host path: $FLIGHT_WS_HOST"
    echo "         Use './run.sh shell' to open more terminals."
    echo ""

    # Compose recreates an existing container only when the detected display or
    # generated Xauthority bind mount changed.
    dc up --detach --remove-orphans "$SERVICE"

    # Editable-install rpg_isaac against the mounted source. site-packages lives
    # in the container filesystem (not a volume), so a recreated container needs
    # this again; skipped in seconds when already installed.
    echo "[run.sh] Ensuring rpg_isaac editable install..."
    docker exec "$container" bash -c \
        "PIP=/workspace/isaaclab/_isaac_sim/python.sh; \
         \$PIP -m pip show rpg_isaac &>/dev/null || \
         \$PIP -m pip install -e /workspace/isaaclab/source/rpg_isaac/source/rpg_isaac"
}

cmd_start() {
    local container="${PROJECT}_ros2"

    cmd_up

    exec docker exec -it \
        -e DISPLAY="$RPG_X11_DISPLAY" \
        -e XAUTHORITY="$X11_AUTH_TARGET" \
        "$container" bash
}

cmd_shell() {
    local as_user="${1:-}"
    local container="${PROJECT}_ros2"

    if ! is_running "$container"; then
        echo "[run.sh] '$container' is not running."
        echo "         Start it first with: ./run.sh start"
        exit 1
    fi

    configure_x11
    local mounted_auth
    mounted_auth="$(docker inspect "$container" --format \
        "{{range .Mounts}}{{if eq .Destination \"$X11_AUTH_TARGET\"}}{{.Source}}{{end}}{{end}}")"
    if [ "$mounted_auth" != "$RPG_X11_AUTHORITY" ]; then
        echo "[run.sh] Container has stale X11 mounts. Refresh it with: ./run.sh --suffix $RPG_ISAAC_SUFFIX start" >&2
        exit 1
    fi

    if [ -n "$as_user" ]; then
        exec docker exec -it \
            -e DISPLAY="$RPG_X11_DISPLAY" \
            -e XAUTHORITY="$X11_AUTH_TARGET" \
            -u "$as_user" "$container" bash
    else
        exec docker exec -it \
            -e DISPLAY="$RPG_X11_DISPLAY" \
            -e XAUTHORITY="$X11_AUTH_TARGET" \
            "$container" bash
    fi
}

cmd_x11() {
    local container="${PROJECT}_ros2"
    configure_x11
    if ! is_running "$container"; then
        echo "[run.sh] Host X11 is configured; '$container' is not running."
        return
    fi

    local container_display mounted_auth
    container_display="$(docker inspect "$container" --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^DISPLAY=//p')"
    mounted_auth="$(docker inspect "$container" --format \
        "{{range .Mounts}}{{if eq .Destination \"$X11_AUTH_TARGET\"}}{{.Source}}{{end}}{{end}}")"
    if [ "$container_display" = "$RPG_X11_DISPLAY" ] && [ "$mounted_auth" = "$RPG_X11_AUTHORITY" ]; then
        if docker exec \
            -e DISPLAY="$RPG_X11_DISPLAY" \
            -e XAUTHORITY="$X11_AUTH_TARGET" \
            "$container" python3 -c \
            'import ctypes; x11 = ctypes.CDLL("libX11.so.6"); x11.XOpenDisplay.restype = ctypes.c_void_p; assert x11.XOpenDisplay(None)' \
            >/dev/null 2>&1; then
            echo "[run.sh] Container X11 connection is working."
        else
            echo "[run.sh] Container X11 wiring is current, but XOpenDisplay failed." >&2
            return 1
        fi
    else
        echo "[run.sh] Container X11 wiring is stale. Refresh it with: ./run.sh --suffix $RPG_ISAAC_SUFFIX start" >&2
        return 1
    fi
}

cmd_stop() {
    local container="${PROJECT}_ros2"

    if ! is_running "$container"; then
        echo "[run.sh] '$container' is not running."
        exit 0
    fi

    echo "[run.sh] Stopping '$container'..."
    docker stop "$container"
    docker rm "$container"
    echo "[run.sh] Done."
}

cmd_help() {
    sed -n '2,10p' "$0" | sed 's/^# \?//'
}

COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    start) cmd_start ;;
    up)    cmd_up ;;
    build) cmd_build ;;
    shell) cmd_shell "$@" ;;
    x11) cmd_x11 ;;
    stop)  cmd_stop ;;
    help|--help|-h) cmd_help ;;
    *) echo "Unknown command: '$COMMAND'"; echo ""; cmd_help; exit 1 ;;
esac
