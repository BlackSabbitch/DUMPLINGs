#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_SKIP_PREFIXES = (
    "torch",
    "torch-geometric",
    "torch-scatter",
    "torch-sparse",
    "torch-cluster",
    "pyg-lib",
)


def parse_requirements(requirements_path: Path, skip_prefixes: tuple[str, ...]) -> list[str]:
    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")

    reqs: list[str] = []
    with requirements_path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(skip_prefixes):
                continue
            reqs.append(line)
    return reqs


def install_requirements(reqs: list[str]) -> None:
    if not reqs:
        print("No non-PyG requirements to install.")
        return

    print("Installing filtered requirements:")
    for req in reqs:
        print("  -", req)

    subprocess.run(["pip", "install", *reqs], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Install regular project requirements from requirements.txt, while skipping "
            "Torch/PyG packages that are managed separately in Colab."
        )
    )
    parser.add_argument(
        "--requirements",
        default="requirements.txt",
        help="Path to the requirements file inside the staged workspace.",
    )
    parser.add_argument(
        "--skip-prefix",
        action="append",
        default=[],
        help=(
            "Additional package-name prefix to skip. "
            "Can be passed multiple times."
        ),
    )
    args = parser.parse_args()

    skip_prefixes = DEFAULT_SKIP_PREFIXES + tuple(args.skip_prefix)
    requirements_path = Path(args.requirements)

    print(f"Requirements file: {requirements_path}")
    print("Skipping prefixes:")
    for prefix in skip_prefixes:
        print("  -", prefix)

    reqs = parse_requirements(requirements_path, skip_prefixes)
    install_requirements(reqs)


if __name__ == "__main__":
    main()
