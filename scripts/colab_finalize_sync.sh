#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 LOCAL_RUNS DRIVE_RUNS [LOCAL_FEATURES DRIVE_FEATURES]" >&2
  exit 1
fi

LOCAL_RUNS=$1
DRIVE_RUNS=$2
LOCAL_FEATURES=${3:-}
DRIVE_FEATURES=${4:-}

mkdir -p "$DRIVE_RUNS"
rsync -a "$LOCAL_RUNS/" "$DRIVE_RUNS/"

if [[ -n "$LOCAL_FEATURES" && -n "$DRIVE_FEATURES" && -d "$LOCAL_FEATURES" ]]; then
  mkdir -p "$DRIVE_FEATURES"
  rsync -a "$LOCAL_FEATURES/" "$DRIVE_FEATURES/"
fi
