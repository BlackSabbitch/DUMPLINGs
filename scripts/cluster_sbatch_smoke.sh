#!/usr/bin/env bash
# Usage:
#   sbatch scripts/cluster_sbatch_smoke.sh
# Optional environment overrides:
#   REPO_ROOT=$HOME/DUMPLINGs
#   VENV_DIR=$HOME/venvs/dumplings
#   SMOKE_EPOCHS=2
#   SMOKE_BATCH_SIZE=2
#   SMOKE_NUM_WORKERS=0
#   MODEL_FAMILY=A1
#   PROTEIN_CONTEXT_MODE=esm_frozen_whole
#   LIGAND_CONTEXT_MODE=basic_rdkit
#   RUN_PIPELINE_SMOKE=1
#   EXTRA_RUN_ARGS="--extract"

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
MODEL_FAMILY=${MODEL_FAMILY:-A1}
PROTEIN_CONTEXT_MODE=${PROTEIN_CONTEXT_MODE:-esm_frozen_whole}
LIGAND_CONTEXT_MODE=${LIGAND_CONTEXT_MODE:-basic_rdkit}
RUN_PIPELINE_SMOKE=${RUN_PIPELINE_SMOKE:-0}
EXTRA_RUN_ARGS=${EXTRA_RUN_ARGS:-}
ARCHIVE_NAME=${ARCHIVE_NAME:-pdbbind_v2016.tar.gz}

cd "$REPO_ROOT"

echo "== Cluster smoke job =="
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

python scripts/cluster_env_smoke.py --repo-root "$REPO_ROOT" --require-gpu

SMOKE_CONFIG="$REPO_ROOT/tmp/cluster_smoke_config.json"
python scripts/cluster_make_smoke_config.py \
  --base-config "$REPO_ROOT/config.json" \
  --output "$SMOKE_CONFIG" \
  --epochs "$SMOKE_EPOCHS" \
  --batch-size "$SMOKE_BATCH_SIZE" \
  --num-workers "$SMOKE_NUM_WORKERS" \
  --experiment-name "DUMPLING_cluster_smoke" \
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
LOG_PATH="$REPO_ROOT/cluster_smoke_run.log" ./run.sh --config "$SMOKE_CONFIG" $EXTRA_RUN_ARGS
