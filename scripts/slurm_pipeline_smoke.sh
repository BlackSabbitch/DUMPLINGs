#!/usr/bin/env bash
# Usage:
#   sbatch scripts/slurm_pipeline_smoke.sh
# Optional environment overrides:
#   REPO_ROOT=$HOME/DUMPLINGs
#   VENV_DIR=$HOME/venvs/dumplings
#   SMOKE_EPOCHS=2
#   SMOKE_BATCH_SIZE=2
#   SMOKE_NUM_WORKERS=0
#   SMOKE_EXPERIMENT_NAME=DUMPLING_colab_smoke
#   MODEL_FAMILY=A1
#   PROTEIN_CONTEXT_MODE=none
#   LIGAND_CONTEXT_MODE=none
#   RUN_PIPELINE_SMOKE=1
#   RUN_BOOTSTRAP_EXTRACT=1
#   BOOTSTRAP_N_TIMES=1
#   REPEAT_N_TIMES=3
#   BASE_RSEED=42
#   EXTRA_BOOTSTRAP_ARGS=""
#   EXTRA_REPEAT_ARGS=""

#SBATCH --job-name=dumplings-smoke
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$HOME/DUMPLINGs}
VENV_DIR=${VENV_DIR:-$REPO_ROOT/.venv}
SMOKE_EPOCHS=${SMOKE_EPOCHS:-2}
SMOKE_BATCH_SIZE=${SMOKE_BATCH_SIZE:-2}
SMOKE_NUM_WORKERS=${SMOKE_NUM_WORKERS:-0}
SMOKE_EXPERIMENT_NAME=${SMOKE_EXPERIMENT_NAME:-DUMPLING_slurm_smoke}
MODEL_FAMILY=${MODEL_FAMILY:-A1}
PROTEIN_CONTEXT_MODE=${PROTEIN_CONTEXT_MODE:-none}
LIGAND_CONTEXT_MODE=${LIGAND_CONTEXT_MODE:-none}
RUN_PIPELINE_SMOKE=${RUN_PIPELINE_SMOKE:-0}
RUN_BOOTSTRAP_EXTRACT=${RUN_BOOTSTRAP_EXTRACT:-1}
BOOTSTRAP_N_TIMES=${BOOTSTRAP_N_TIMES:-1}
REPEAT_N_TIMES=${REPEAT_N_TIMES:-3}
BASE_RSEED=${BASE_RSEED:-42}
EXTRA_BOOTSTRAP_ARGS=${EXTRA_BOOTSTRAP_ARGS:-}
EXTRA_REPEAT_ARGS=${EXTRA_REPEAT_ARGS:-}
ARCHIVE_NAME=${ARCHIVE_NAME:-pdbbind_v2016.tar.gz}

cd "$REPO_ROOT"

echo "== Slurm smoke job =="
echo "host: $(hostname)"
echo "repo: $REPO_ROOT"
echo "venv: $VENV_DIR"
echo "pwd: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtual environment not found: $VENV_DIR" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

python scripts/runtime_env_smoke.py --repo-root "$REPO_ROOT" --require-gpu

SMOKE_CONFIG="$REPO_ROOT/tmp/smoke_config.json"
python scripts/make_smoke_config.py \
  --base-config "$REPO_ROOT/config.json" \
  --output "$SMOKE_CONFIG" \
  --epochs "$SMOKE_EPOCHS" \
  --batch-size "$SMOKE_BATCH_SIZE" \
  --num-workers "$SMOKE_NUM_WORKERS" \
  --experiment-name "$SMOKE_EXPERIMENT_NAME" \
  --model-family "$MODEL_FAMILY" \
  --protein-context-mode "$PROTEIN_CONTEXT_MODE" \
  --ligand-context-mode "$LIGAND_CONTEXT_MODE"

echo
echo "== Smoke config =="
cat "$SMOKE_CONFIG"
echo

if [[ "$RUN_PIPELINE_SMOKE" != "1" ]]; then
  echo "Environment smoke completed. Set RUN_PIPELINE_SMOKE=1 to launch a short training run."
  exit 0
fi

if [[ ! -f "$REPO_ROOT/$ARCHIVE_NAME" ]]; then
  echo "Archive not found at $REPO_ROOT/$ARCHIVE_NAME" >&2
  exit 1
fi

echo "== Launching short pipeline smoke =="
echo "base_rseed: $BASE_RSEED"
echo "bootstrap_extract: $RUN_BOOTSTRAP_EXTRACT"
echo "bootstrap_n_times: $BOOTSTRAP_N_TIMES"
echo "repeat_n_times: $REPEAT_N_TIMES"
echo

if [[ "$RUN_BOOTSTRAP_EXTRACT" == "1" ]]; then
  echo "== Bootstrap run with extraction =="
  LOG_PATH="$REPO_ROOT/slurm_smoke_bootstrap.log" ./run.sh \
    --config "$SMOKE_CONFIG" \
    --n-times "$BOOTSTRAP_N_TIMES" \
    --rseed "$BASE_RSEED" \
    --extract \
    $EXTRA_BOOTSTRAP_ARGS
else
  echo "== Bootstrap extraction step skipped =="
fi

echo
echo "== Repeated smoke batch =="
LOG_PATH="$REPO_ROOT/slurm_smoke_batch.log" ./run.sh \
  --config "$SMOKE_CONFIG" \
  --n-times "$REPEAT_N_TIMES" \
  --rseed "$BASE_RSEED" \
  $EXTRA_REPEAT_ARGS
