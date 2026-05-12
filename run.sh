#!/bin/bash
set -o pipefail

LOG_PATH=${LOG_PATH:-last_run.log}
N_TIMES=1
RSEED=""
EXTRACT_FIRST_ONLY=0
GPU_DIAGNOSTICS=${DUMPLING_GPU_DIAGNOSTICS:-0}
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-times)
            if [[ $# -lt 2 ]]; then
                echo "error: --n-times requires an integer argument" >&2
                exit 1
            fi
            N_TIMES="$2"
            shift 2
            ;;
        --rseed|--splitter-seed)
            if [[ $# -lt 2 ]]; then
                echo "error: $1 requires an integer argument" >&2
                exit 1
            fi
            RSEED="$2"
            shift 2
            ;;
        --extract-first-only)
            EXTRACT_FIRST_ONLY=1
            shift
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! [[ "$N_TIMES" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --n-times must be a positive integer" >&2
    exit 1
fi

log_gpu_snapshot() {
    local label="$1"
    if [[ "$GPU_DIAGNOSTICS" != "1" ]]; then
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    {
        echo "== GPU snapshot: $label =="
        date --iso-8601=seconds 2>/dev/null || date
        nvidia-smi
    } | tee -a "$LOG_PATH"
}

run_once() {
    local run_index="$1"
    local seed_arg=()
    local effective_seed=""
    local run_args=("${FORWARD_ARGS[@]}")

    if [[ -n "$RSEED" ]]; then
        effective_seed=$((RSEED + run_index - 1))
    else
        effective_seed=$(( (RANDOM << 16) ^ RANDOM ^ run_index ))
    fi

    seed_arg=(--rseed "$effective_seed")

    if [[ "$EXTRACT_FIRST_ONLY" -eq 1 && "$run_index" -eq 1 ]]; then
        run_args+=(--extract)
    fi

    if [[ "$N_TIMES" -gt 1 ]]; then
        echo "== Run $run_index/$N_TIMES | splitter_seed=$effective_seed ==" | tee -a "$LOG_PATH"
    else
        echo "== Run $run_index/$N_TIMES | splitter_seed=$effective_seed ==" > "$LOG_PATH"
    fi

    log_gpu_snapshot "before_run_${run_index}"

    DUMPLING_BATCH_RUN_INDEX="$run_index" \
    DUMPLING_BATCH_N_TIMES="$N_TIMES" \
    python run.py "${run_args[@]}" "${seed_arg[@]}" 2>&1 | tee -a "$LOG_PATH"
    local run_status=${PIPESTATUS[0]}

    log_gpu_snapshot "after_run_${run_index}"
    return "$run_status"
}

for ((i = 1; i <= N_TIMES; i++)); do
    run_once "$i" || exit $?
done
