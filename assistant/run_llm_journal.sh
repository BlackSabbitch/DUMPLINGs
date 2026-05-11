#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="dry-run"
RUNS_DIR="runs"
LIMIT=""
FORCE_REFRESH=0
ENV_FILE="${SCRIPT_DIR}/.env"

usage() {
  cat <<'EOF'
Usage:
  bash assistant/run_llm_journal.sh [--dry-run|--live] [--runs-dir PATH] [--limit N] [--force-refresh] [--env-file PATH]

Examples:
  bash assistant/run_llm_journal.sh --dry-run
  bash assistant/run_llm_journal.sh --live --limit 1
  bash assistant/run_llm_journal.sh --live --limit 1 --force-refresh

Notes:
  --dry-run: build run-level and series-level context, prompt previews, and both LLM journals with mock notes
  --live:    build run-level and series-level context, prompt previews, and both LLM journals using the configured LLM backend
  --env-file: optional shell env file to source before running; defaults to assistant/.env if present

Required env vars for --live:
  ASSISTANT_LLM_PROVIDER=ollama
  ASSISTANT_LLM_MODEL=...
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --live)
      MODE="live"
      shift
      ;;
    --runs-dir)
      RUNS_DIR="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --force-refresh)
      FORCE_REFRESH=1
      shift
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "${RUNS_DIR}" ]]; then
  echo "runs dir does not exist: ${RUNS_DIR}" >&2
  exit 1
fi

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${ENV_FILE}"
  set +a
fi

if [[ "${MODE}" == "live" ]]; then
  : "${ASSISTANT_LLM_PROVIDER:?ASSISTANT_LLM_PROVIDER must be set for --live}"
  if [[ "${ASSISTANT_LLM_PROVIDER}" != "ollama" ]]; then
    echo "Unsupported ASSISTANT_LLM_PROVIDER: ${ASSISTANT_LLM_PROVIDER}. Only ollama is supported." >&2
    exit 1
  fi
  : "${ASSISTANT_LLM_MODEL:?ASSISTANT_LLM_MODEL must be set for --live}"
fi

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

FORCE_ARGS=()
if [[ "${FORCE_REFRESH}" == "1" ]]; then
  FORCE_ARGS=(--force-refresh)
fi

echo "[assistant] Building context packets..."
python assistant/build_run_contexts.py --runs-dir "${RUNS_DIR}"

echo "[assistant] Building series context packets..."
python assistant/build_series_contexts.py

echo "[assistant] Building prompt previews..."
python assistant/build_llm_prompts.py "${LIMIT_ARGS[@]}"

echo "[assistant] Building series prompt previews..."
python assistant/build_series_llm_prompts.py "${LIMIT_ARGS[@]}"

echo "[assistant] Building LLM journal (${MODE})..."
python assistant/build_llm_journal.py --mode "${MODE}" "${LIMIT_ARGS[@]}" "${FORCE_ARGS[@]}"

echo "[assistant] Building series LLM journal (${MODE})..."
python assistant/build_series_llm_journal.py --mode "${MODE}" "${LIMIT_ARGS[@]}" "${FORCE_ARGS[@]}"

echo "[assistant] Done."
echo "[assistant] Journal: ${REPO_ROOT}/runs/experiment_journal_llm.md"
echo "[assistant] Series Journal: ${REPO_ROOT}/runs/experiment_series_journal_llm.md"
