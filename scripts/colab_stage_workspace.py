#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


REPO_ITEMS = [
    "run.py",
    "trainer.py",
    "evaluator.py",
    "extractor.py",
    "splitter.py",
    "tokenizer.py",
    "utils.py",
    "logger.py",
    "loss_functions.py",
    "config.json",
    "requirements.txt",
    "run.sh",
    "README.md",
    "bad_complexes.toml",
    "models",
    "parsers",
    "scripts",
]


def copy_entry(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def stage_repo(src_root: Path, dst_root: Path) -> None:
    for name in REPO_ITEMS:
        src = src_root / name
        dst = dst_root / name
        if not src.exists():
            continue
        copy_entry(src, dst)


def stage_protein_context_features(src_root: Path, dst_root: Path) -> None:
    features_src = src_root / "protein_context_features"
    features_dst = dst_root / "protein_context_features"
    features_dst.mkdir(parents=True, exist_ok=True)

    if not features_src.exists():
        print("No protein context features found on Drive; empty protein_context_features/ will be used.")
        return

    copied = 0
    for name in os.listdir(features_src):
        src = features_src / name
        dst = features_dst / name
        copy_entry(src, dst)
        copied += 1
    print(f"Copied protein context features from {features_src} ({copied} top-level entries)")


def stage_ligand_context_features(src_root: Path, dst_root: Path) -> None:
    features_src = src_root / "ligand_context_features"
    features_dst = dst_root / "ligand_context_features"
    features_dst.mkdir(parents=True, exist_ok=True)

    if not features_src.exists():
        print("No ligand context features found on Drive; empty ligand_context_features/ will be used.")
        return

    copied = 0
    for name in os.listdir(features_src):
        src = features_src / name
        dst = features_dst / name
        copy_entry(src, dst)
        copied += 1
    print(f"Copied ligand context features from {features_src} ({copied} top-level entries)")


def stage_archive(src_root: Path, dst_root: Path, archive_name: str) -> None:
    archive_src = src_root / archive_name
    archive_dst = dst_root / archive_name
    if not archive_src.exists():
        print(f"Archive {archive_src} not found; skipping.")
        return
    shutil.copy2(archive_src, archive_dst)
    print(f"Copied archive to {archive_dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage the DUMPLINGs workspace from Drive into a Colab /content workspace."
    )
    parser.add_argument("--src", required=True, help="Drive-side repo root, e.g. /content/drive/.../DUMPLINGs")
    parser.add_argument("--dst", required=True, help="Ephemeral Colab workspace, e.g. /content/DUMPLINGs")
    parser.add_argument("--drive-runs", default=None, help="Optional Drive-side runs directory to create if missing")
    parser.add_argument("--archive-name", default="pdbbind_v2016.tar.gz")
    parser.add_argument("--skip-protein-context-features", action="store_true")
    parser.add_argument("--skip-ligand-context-features", action="store_true")
    parser.add_argument("--skip-archive", action="store_true")
    parser.add_argument("--keep-dst", action="store_true", help="Keep an existing dst instead of recreating it")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if args.drive_runs:
        Path(args.drive_runs).mkdir(parents=True, exist_ok=True)

    if dst_root.exists() and not args.keep_dst:
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)

    stage_repo(src_root, dst_root)
    if not args.skip_protein_context_features:
        stage_protein_context_features(src_root, dst_root)
    if not args.skip_ligand_context_features:
        stage_ligand_context_features(src_root, dst_root)
    if not args.skip_archive:
        stage_archive(src_root, dst_root, args.archive_name)

    print(f"Workspace staged into {dst_root}")


if __name__ == "__main__":
    main()
