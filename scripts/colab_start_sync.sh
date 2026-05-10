#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 LOCAL_RUNS DRIVE_RUNS [LOCAL_PROTEIN DRIVE_PROTEIN [LOCAL_LIGAND DRIVE_LIGAND [INTERVAL_SECONDS]]]" >&2
  exit 1
fi

LOCAL_RUNS=$1
DRIVE_RUNS=$2
LOCAL_PROTEIN_FEATURES=${3:-}
DRIVE_PROTEIN_FEATURES=${4:-}
LOCAL_LIGAND_FEATURES=${5:-}
DRIVE_LIGAND_FEATURES=${6:-}
INTERVAL_SECONDS=${7:-120}

sync_dir_pair() {
  local local_dir="$1"
  local drive_dir="$2"
  if [[ -n "$local_dir" && -n "$drive_dir" && -d "$local_dir" ]]; then
    mkdir -p "$drive_dir"
    rsync -a "$local_dir/" "$drive_dir/"
  fi
}

mkdir -p "$DRIVE_RUNS"

while true; do
  rsync -a "$LOCAL_RUNS/" "$DRIVE_RUNS/"
  sync_dir_pair "$LOCAL_PROTEIN_FEATURES" "$DRIVE_PROTEIN_FEATURES"
  sync_dir_pair "$LOCAL_LIGAND_FEATURES" "$DRIVE_LIGAND_FEATURES"
  sleep "$INTERVAL_SECONDS"
done
