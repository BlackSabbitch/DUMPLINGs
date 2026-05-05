#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 LOCAL_RUNS DRIVE_RUNS [LOCAL_ESM DRIVE_ESM]" >&2
  exit 1
fi

LOCAL_RUNS=$1
DRIVE_RUNS=$2
LOCAL_ESM=${3:-}
DRIVE_ESM=${4:-}

mkdir -p "$DRIVE_RUNS"
rsync -a "$LOCAL_RUNS/" "$DRIVE_RUNS/"

if [[ -n "$LOCAL_ESM" && -n "$DRIVE_ESM" && -d "$LOCAL_ESM" ]]; then
  mkdir -p "$DRIVE_ESM"
  rsync -a "$LOCAL_ESM/" "$DRIVE_ESM/"
fi
