#!/usr/bin/env bash
# Usage:
#   sbatch scripts/slurm_assistant_journal.sh
# Optional environment overrides:
#   REPO_ROOT=$HOME/DUMPLINGs
#   VENV_DIR=$REPO_ROOT/.venv
#   ASSISTANT_MODE=live
#   ASSISTANT_LIMIT=0
#   ASSISTANT_FORCE_REFRESH=0
#   ASSISTANT_RUNS_DIR=runs
#   ASSISTANT_MODEL=qwen2.5:7b
#   ASSISTANT_TIMEOUT_SEC=1800
#   ASSISTANT_TEMPERATURE=0.2
#   ASSISTANT_OLLAMA_BASE_URL=http://127.0.0.1:11434/api/generate
#   ASSISTANT_OLLAMA_KEEP_ALIVE=30m
#   ASSISTANT_PULL_MODEL=1
#   ASSISTANT_READY_RETRIES=30

#SBATCH --job-name=dumplings-assistant
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$HOME/DUMPLINGs}
VENV_DIR=${VENV_DIR:-$REPO_ROOT/.venv}
ASSISTANT_MODE=${ASSISTANT_MODE:-live}
ASSISTANT_LIMIT=${ASSISTANT_LIMIT:-0}
ASSISTANT_FORCE_REFRESH=${ASSISTANT_FORCE_REFRESH:-0}
ASSISTANT_RUNS_DIR=${ASSISTANT_RUNS_DIR:-runs}
ASSISTANT_MODEL=${ASSISTANT_MODEL:-qwen2.5:7b}
ASSISTANT_TIMEOUT_SEC=${ASSISTANT_TIMEOUT_SEC:-1800}
ASSISTANT_TEMPERATURE=${ASSISTANT_TEMPERATURE:-0.2}
ASSISTANT_OLLAMA_BASE_URL=${ASSISTANT_OLLAMA_BASE_URL:-http://127.0.0.1:11434/api/generate}
ASSISTANT_OLLAMA_KEEP_ALIVE=${ASSISTANT_OLLAMA_KEEP_ALIVE:-30m}
ASSISTANT_PULL_MODEL=${ASSISTANT_PULL_MODEL:-1}
ASSISTANT_READY_RETRIES=${ASSISTANT_READY_RETRIES:-30}

cd "$REPO_ROOT"

echo "== Slurm assistant journal job =="
echo "host: $(hostname)"
echo "repo: $REPO_ROOT"
echo "venv: $VENV_DIR"
echo "mode: $ASSISTANT_MODE"
echo "runs_dir: $ASSISTANT_RUNS_DIR"
echo "model: $ASSISTANT_MODEL"
echo "timeout_sec: $ASSISTANT_TIMEOUT_SEC"
echo "force_refresh: $ASSISTANT_FORCE_REFRESH"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtual environment not found: $VENV_DIR" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not available on PATH." >&2
  exit 1
fi

echo "== Starting Ollama service =="
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_KEEP_ALIVE="$ASSISTANT_OLLAMA_KEEP_ALIVE" ollama serve > "$REPO_ROOT/assistant_ollama_serve.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" >/dev/null 2>&1 || true' EXIT

echo "== Waiting for Ollama API =="
READY=0
for ((i = 1; i <= ASSISTANT_READY_RETRIES; i++)); do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "$READY" != "1" ]]; then
  echo "Ollama API did not become ready after ${ASSISTANT_READY_RETRIES} retries." >&2
  exit 1
fi

if [[ "$ASSISTANT_PULL_MODEL" == "1" ]]; then
  echo "== Ensuring model is available =="
  ollama pull "$ASSISTANT_MODEL"
fi

LIMIT_ARGS=()
if [[ "$ASSISTANT_LIMIT" != "0" ]]; then
  LIMIT_ARGS=(--limit "$ASSISTANT_LIMIT")
fi

FORCE_ARGS=()
if [[ "$ASSISTANT_FORCE_REFRESH" == "1" ]]; then
  FORCE_ARGS=(--force-refresh)
fi

export ASSISTANT_LLM_PROVIDER=ollama
export ASSISTANT_LLM_MODEL="$ASSISTANT_MODEL"
export ASSISTANT_LLM_BASE_URL="$ASSISTANT_OLLAMA_BASE_URL"
export ASSISTANT_LLM_TIMEOUT_SEC="$ASSISTANT_TIMEOUT_SEC"
export ASSISTANT_LLM_TEMPERATURE="$ASSISTANT_TEMPERATURE"

echo
echo "== Running assistant pipeline =="
bash assistant/run_llm_journal.sh "--${ASSISTANT_MODE}" --runs-dir "$ASSISTANT_RUNS_DIR" "${LIMIT_ARGS[@]}" "${FORCE_ARGS[@]}"

echo
echo "== Assistant outputs =="
echo "run journal:    $REPO_ROOT/runs/experiment_journal_llm.md"
echo "series journal: $REPO_ROOT/runs/experiment_series_journal_llm.md"
