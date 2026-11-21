#!/bin/bash
# Shared helpers for run/ scripts. All script arguments use name=value style:
#   scalars:  num_envs=512 viz=newton run_name=test seed=1 checkpoint=model_1499.pt
#   booleans: video=true evaluate=true
#   hydra:    env.<field>=<value> agent.<field>=<value> presets=<name>
# Dotted names and preset selectors pass through to hydra untouched; booleans map to
# --name store_true flags; everything else maps to --name value.

PYTHON="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"

_RPG_BOOL_ARGS=" video headless enable_cameras resume real-time keep_all_info use_pretrained_checkpoint deterministic verbose cpu xr info disable_fabric export_io_descriptors evaluate profile_phases profile_camera_internals rtpt_performance rtpt_disable_denoiser rtpt_dlss_performance rtpt_disable_caches rtpt_fast_dome stock_camera_pose rtpt_cache rtpt_no_aa rtpt_single_bounce rtpt_skip_most_lights disable_shadows no_publish_images replay_realtime "

# argparse BooleanOptionalAction flags: these default to on, so false must emit
# --no-<name> rather than simply omitting the flag.
_RPG_TRISTATE_ARGS=" condition_on_actions randomize_initial_collection "

# Kit args baked into every run. Multi-GPU rendering is never useful for
# these workloads and error-spams on heterogeneous GPU pairs (shared
# render-graph buffer import fails); pick the GPU with device=cuda:N instead.
_RPG_DEFAULT_KIT_ARGS="--/renderer/multiGpu/enabled=false"

# rpg_translate_args <tokens...> — fills the TRANSLATED array.
rpg_translate_args() {
    TRANSLATED=()
    local tok name value kit_args="$_RPG_DEFAULT_KIT_ARGS"
    for tok in "$@"; do
        if [[ "$tok" == -* ]]; then
            echo "error: '$tok' — arguments use name=value style (e.g. num_envs=16 viz=newton video=true)" >&2
            exit 2
        fi
        if [[ "$tok" != *=* ]]; then
            echo "error: '$tok' is not name=value" >&2
            exit 2
        fi
        name="${tok%%=*}"
        value="${tok#*=}"
        if [[ "$name" == *.* || "$name" == "physics" || "$name" == "presets" ]]; then
            TRANSLATED+=("$tok")
        elif [[ "$name" == "kit_args" ]]; then
            kit_args+=" $value"  # extras appended after the baked-in defaults
        elif [[ "$_RPG_TRISTATE_ARGS" == *" $name "* ]]; then
            case "$value" in
                true | 1) TRANSLATED+=("--$name") ;;
                false | 0) TRANSLATED+=("--no-$name") ;;
                *)
                    echo "error: $name expects true or false" >&2
                    exit 2
                    ;;
            esac
        elif [[ "$_RPG_BOOL_ARGS" == *" $name "* ]]; then
            case "$value" in
                true | 1) TRANSLATED+=("--$name") ;;
                false | 0) ;;
                *)
                    echo "error: $name expects true or false" >&2
                    exit 2
                    ;;
            esac
        else
            TRANSLATED+=("--$name" "$value")
        fi
    done
    # single-token --opt=value form: kit args start with --/ which argparse
    # would otherwise mistake for a new option
    TRANSLATED+=("--kit_args=$kit_args")
}

# rpg_parse_run_checkpoint <tokens...> — pulls out run= and checkpoint= into
# RUN / CHECKPOINT; everything else lands in the REMAINING array.
rpg_parse_run_checkpoint() {
    RUN=""
    CHECKPOINT=""
    REMAINING=()
    local tok
    for tok in "$@"; do
        case "$tok" in
            run=*) RUN="${tok#run=}" ;;
            checkpoint=*) CHECKPOINT="${tok#checkpoint=}" ;;
            *) REMAINING+=("$tok") ;;
        esac
    done
}

# rpg_resolve_checkpoint <log_dir> [preferred_checkpoint] — resolves RUN
# (latest when empty) and CHECKPOINT (preferred file, or latest model_*.pt)
# into CHECKPOINT_PATH.
rpg_resolve_checkpoint() {
    local log_dir="$1"
    local preferred_checkpoint="${2:-}"
    if [[ ! -d "$log_dir" ]]; then
        echo "Log directory not found: $log_dir" >&2
        exit 1
    fi
    if [[ -z "$RUN" ]]; then
        RUN="$(ls -1 "$log_dir" | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}' | sort | tail -n 1)"
        if [[ -z "$RUN" ]]; then
            echo "No runs found under $log_dir" >&2
            exit 1
        fi
    fi
    local run_dir="$log_dir/$RUN"
    if [[ ! -d "$run_dir" ]]; then
        echo "Run directory not found: $run_dir" >&2
        exit 1
    fi
    if [[ -z "$CHECKPOINT" ]]; then
        if [[ -n "$preferred_checkpoint" && -f "$run_dir/$preferred_checkpoint" ]]; then
            CHECKPOINT="$preferred_checkpoint"
        else
            CHECKPOINT="$(ls -1 "$run_dir" | grep -E '^model_[0-9]+\.pt$' | sort -t_ -k2 -n | tail -n 1)"
        fi
        if [[ -z "$CHECKPOINT" ]]; then
            echo "No usable checkpoint found in $run_dir" >&2
            exit 1
        fi
    fi
    CHECKPOINT_PATH="$run_dir/$CHECKPOINT"
    if [[ ! -f "$CHECKPOINT_PATH" ]]; then
        echo "Checkpoint file not found: $CHECKPOINT_PATH" >&2
        exit 1
    fi
    echo "[INFO] Playing run=$RUN checkpoint=$CHECKPOINT"
}

# rpg_exec <cmd...> — print the command instead of running it when RPG_DRY_RUN=1.
rpg_exec() {
    # A GUI viewer without a reachable X display silently renders headless
    # (pyglet EGL fallback: no window, no error) — fail loudly instead.
    local _prev="" _viz="" _a
    for _a in "$@"; do
        [[ "$_prev" == "--viz" ]] && _viz="$_a"
        _prev="$_a"
    done
    if [[ -n "$_viz" && "$_viz" != "none" && -z "${DISPLAY:-}" ]]; then
        echo "error: viz=$_viz needs an X display but \$DISPLAY is empty in this shell." >&2
        echo "       Enter the container via 'docker/course/run.sh shell' (wires up X11)," >&2
        echo "       or run headless with viz=none." >&2
        exit 2
    fi
    if [[ "${RPG_DRY_RUN:-0}" == "1" ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        exec "$@"
    fi
}
