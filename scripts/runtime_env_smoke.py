#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import sys
from pathlib import Path


REQUIRED_IMPORTS = [
    ("torch", "PyTorch"),
    ("torch_geometric", "PyTorch Geometric"),
    ("rdkit", "RDKit"),
    ("esm", "fair-esm"),
    ("pandas", "pandas"),
]


def human_bool(value: bool) -> str:
    return "yes" if value else "no"


def import_status(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive runtime probe
        return False, f"{type(exc).__name__}: {exc}"
    version = getattr(module, "__version__", None)
    return True, version or "imported"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for the DUMPLINGs runtime environment."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the repository root to inspect.",
    )
    parser.add_argument(
        "--archive-name",
        default="pdbbind_v2016.tar.gz",
        help="Expected PDBBind archive filename.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if CUDA is not available to PyTorch.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    archive_path = repo_root / args.archive_name

    print("== DUMPLINGs runtime environment smoke test ==")
    print(f"repo_root: {repo_root}")
    print(f"python: {sys.executable}")
    print(f"python_version: {platform.python_version()}")
    print(f"hostname: {platform.node()}")
    print(f"platform: {platform.platform()}")
    print(f"cwd: {Path.cwd()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"archive_present: {human_bool(archive_path.exists())} ({archive_path})")
    print()

    all_ok = True
    print("== Import checks ==")
    for module_name, label in REQUIRED_IMPORTS:
        ok, detail = import_status(module_name)
        marker = "OK" if ok else "FAIL"
        print(f"[{marker}] {label:<20} {detail}")
        all_ok = all_ok and ok
    print()

    try:
        import torch
    except Exception:
        torch = None

    if torch is not None:
        print("== Torch / CUDA checks ==")
        cuda_available = bool(torch.cuda.is_available())
        print(f"torch.cuda.is_available: {cuda_available}")
        print(f"torch.version.cuda: {getattr(torch.version, 'cuda', None)}")
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        print(f"torch.cuda.device_count: {device_count}")
        for idx in range(device_count):
            props = torch.cuda.get_device_properties(idx)
            print(
                f"gpu[{idx}]: name={props.name} "
                f"total_memory_gb={props.total_memory / (1024**3):.2f}"
            )
        if args.require_gpu and not cuda_available:
            all_ok = False
        print()

    print("== Filesystem checks ==")
    for rel in ["datasets", "protein_context_features", "ligand_context_features", "runs"]:
        path = repo_root / rel
        exists = path.exists()
        writable = os.access(path, os.W_OK) if exists else os.access(path.parent, os.W_OK)
        print(f"{rel:<24} exists={human_bool(exists)} writable={human_bool(writable)}")
    print(f"git_present: {human_bool((repo_root / '.git').exists())}")
    print(f"run_sh_present: {human_bool((repo_root / 'run.sh').exists())}")
    print(f"nvidia-smi_present: {human_bool(shutil.which('nvidia-smi') is not None)}")
    print()

    print("== Result ==")
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
