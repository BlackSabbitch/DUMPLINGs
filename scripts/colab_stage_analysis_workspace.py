#!/usr/bin/env python3
"""Stage a lightweight analysis workspace into Colab `/content`.

This helper is intentionally smaller than the training-oriented staging script.
It is meant for:

- factual rebuilds from copied run folders,
- assistant journal generation,
- interactive series analysis notebooks.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ANALYSIS_REPO_ITEMS = [
    "assistant",
    "configs",
    "scripts",
    "README.md",
    "requirements.txt",
]

TOP_LEVEL_RUN_FILES = {
    "experiment_registry.csv",
    "experiment_journal.md",
    "experiment_series_journal.md",
    "experiment_journal_llm.md",
    "experiment_series_journal_llm.md",
}


def copy_entry(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def stage_repo(src_root: Path, dst_root: Path) -> None:
    for name in ANALYSIS_REPO_ITEMS:
        src = src_root / name
        dst = dst_root / name
        if not src.exists():
            continue
        copy_entry(src, dst)


def should_copy_run(path: Path, include_prefixes: list[str]) -> bool:
    if not include_prefixes:
        return True
    return any(path.name.startswith(prefix) for prefix in include_prefixes)


def stage_runs(src_root: Path, dst_root: Path, include_prefixes: list[str]) -> None:
    runs_src = src_root / "runs"
    runs_dst = dst_root / "runs"
    runs_dst.mkdir(parents=True, exist_ok=True)

    if not runs_src.exists():
        print(f"No runs directory found at {runs_src}; skipping run copy.")
        return

    copied = 0
    for child in sorted(runs_src.iterdir()):
        if child.name in TOP_LEVEL_RUN_FILES:
            shutil.copy2(child, runs_dst / child.name)
            continue
        if not child.is_dir():
            continue
        if not should_copy_run(child, include_prefixes):
            continue
        copy_entry(child, runs_dst / child.name)
        copied += 1

    print(f"Copied {copied} run folders into {runs_dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage a lightweight DUMPLINGs analysis workspace into Colab."
    )
    parser.add_argument("--src", required=True, help="Drive-side repo root")
    parser.add_argument("--dst", required=True, help="Ephemeral Colab workspace")
    parser.add_argument("--keep-dst", action="store_true", help="Keep an existing dst instead of recreating it")
    parser.add_argument(
        "--include-run-prefix",
        action="append",
        default=[],
        help="Optional run-folder prefix filter. Repeat for multiple prefixes.",
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if dst_root.exists() and not args.keep_dst:
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    stage_repo(src_root, dst_root)
    stage_runs(src_root, dst_root, args.include_run_prefix)

    print(f"Analysis workspace staged into {dst_root}")


if __name__ == "__main__":
    main()
