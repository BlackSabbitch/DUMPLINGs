#!/usr/bin/env bash
# Usage:
#   sbatch --export=ALL,CONFIG_PATH=configs/real_series/a1_full.json,N_TIMES=10,BASE_RSEED=42 scripts/slurm_run_series.sh
# Optional environment overrides:
#   REPO_ROOT=$HOME/DUMPLINGs
#   VENV_DIR=$REPO_ROOT/.venv
#   CONFIG_PATH=config.json
#   N_TIMES=10
#   BASE_RSEED=42
#   RUN_EXTRACT=0
#   EXTRA_RUN_ARGS=""

#SBATCH --job-name=dumplings-series
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$HOME/DUMPLINGs}
VENV_DIR=${VENV_DIR:-$REPO_ROOT/.venv}
CONFIG_PATH=${CONFIG_PATH:-config.json}
N_TIMES=${N_TIMES:-10}
BASE_RSEED=${BASE_RSEED:-42}
RUN_EXTRACT=${RUN_EXTRACT:-0}
EXTRA_RUN_ARGS=${EXTRA_RUN_ARGS:-}

cd "$REPO_ROOT"

echo "== Slurm real-series job =="
echo "host: $(hostname)"
echo "repo: $REPO_ROOT"
echo "venv: $VENV_DIR"
echo "config: $CONFIG_PATH"
echo "n_times: $N_TIMES"
echo "base_rseed: $BASE_RSEED"
echo "run_extract: $RUN_EXTRACT"
echo "extra_run_args: ${EXTRA_RUN_ARGS:-<none>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtual environment not found: $VENV_DIR" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

CMD=(./run.sh --config "$CONFIG_PATH" --n-times "$N_TIMES" --rseed "$BASE_RSEED")

if [[ "$RUN_EXTRACT" == "1" ]]; then
  CMD+=(--extract-first-only)
fi

if [[ -n "$EXTRA_RUN_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=($EXTRA_RUN_ARGS)
  CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

echo "== Running series =="
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
