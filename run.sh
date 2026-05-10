#!/bin/bash
set -o pipefail

LOG_PATH=${LOG_PATH:-run.log}
N_TIMES=1
RSEED=""
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

run_once() {
    local run_index="$1"
    local seed_arg=()
    local effective_seed=""

    if [[ -n "$RSEED" ]]; then
        effective_seed=$((RSEED + run_index - 1))
    else
        effective_seed=$(( (RANDOM << 16) ^ RANDOM ^ run_index ))
    fi

    seed_arg=(--rseed "$effective_seed")

    if [[ "$N_TIMES" -gt 1 ]]; then
        echo "== Run $run_index/$N_TIMES | splitter_seed=$effective_seed ==" | tee -a "$LOG_PATH"
    else
        echo "== Run $run_index/$N_TIMES | splitter_seed=$effective_seed ==" > "$LOG_PATH"
    fi

    DUMPLING_BATCH_RUN_INDEX="$run_index" \
    DUMPLING_BATCH_N_TIMES="$N_TIMES" \
    python run.py "${FORWARD_ARGS[@]}" "${seed_arg[@]}" 2>&1 | tee -a "$LOG_PATH"
}

for ((i = 1; i <= N_TIMES; i++)); do
    run_once "$i" || exit $?
done
