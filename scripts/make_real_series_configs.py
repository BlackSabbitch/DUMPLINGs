#!/usr/bin/env python3
"""Generate a first real-series config suite from the main project config.

This helper is intentionally conservative. It does not invent a new config
format; it derives several concrete experiment configs from `config.json`
so cluster launchers can run repeated seed series without hand-editing the
same knobs over and over.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


SERIES_SPECS = [
    {
        "filename": "a1_dimenet_only.json",
        "experiment_name": "DUMPLING_A1_dimenet_only",
        "model_family": "A1",
        "protein_context_mode": "none",
        "ligand_context_mode": "none",
        "launch_extra_args": "",
    },
    {
        "filename": "a1_esm_only.json",
        "experiment_name": "DUMPLING_A1_esm_only",
        "model_family": "A1",
        "protein_context_mode": "esm_only",
        "ligand_context_mode": "none",
        "launch_extra_args": "",
    },
    {
        "filename": "a1_dimenet_esm.json",
        "experiment_name": "DUMPLING_A1_dimenet_esm",
        "model_family": "A1",
        "protein_context_mode": "esm_frozen_whole",
        "ligand_context_mode": "none",
        "launch_extra_args": "",
    },
    {
        "filename": "a1_full.json",
        "experiment_name": "DUMPLING_A1_full",
        "model_family": "A1",
        "protein_context_mode": "esm_frozen_whole",
        "ligand_context_mode": "basic_rdkit",
        "launch_extra_args": "",
    },
    {
        "filename": "a2_full.json",
        "experiment_name": "DUMPLING_A2_full",
        "model_family": "A2",
        "protein_context_mode": "esm_frozen_whole",
        "ligand_context_mode": "basic_rdkit",
        "launch_extra_args": "",
    },
    {
        "filename": "a3_with_bias.json",
        "experiment_name": "DUMPLING_A3_with_bias",
        "model_family": "A3",
        "protein_context_mode": "esm_frozen_whole",
        "ligand_context_mode": "basic_rdkit",
        "launch_extra_args": "--a3-mixer-bias",
    },
    {
        "filename": "a3_without_bias.json",
        "experiment_name": "DUMPLING_A3_without_bias",
        "model_family": "A3",
        "protein_context_mode": "esm_frozen_whole",
        "ligand_context_mode": "basic_rdkit",
        "launch_extra_args": "--no-a3-mixer-bias",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the first real-series DUMPLINGs experiment configs."
    )
    parser.add_argument(
        "--base-config",
        default="config.json",
        help="Base config to clone and modify.",
    )
    parser.add_argument(
        "--output-dir",
        default="configs/real_series",
        help="Directory where the derived config files will be written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for dataset.batch_size across the whole suite.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional override for dataset.num_workers across the whole suite.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for training.epochs across the whole suite.",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=None,
        help="Optional override for training.plot_every_n_epochs across the whole suite.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_path = Path(args.base_config)
    output_dir = Path(args.output_dir)

    config = json.loads(base_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []

    for spec in SERIES_SPECS:
        cfg = deepcopy(config)
        cfg["experiment_name"] = spec["experiment_name"]
        cfg["model"]["selected"] = spec["model_family"]
        cfg["model"]["protein_context"]["selected"] = spec["protein_context_mode"]
        cfg["model"]["ligand_context"]["selected"] = spec["ligand_context_mode"]

        if args.batch_size is not None:
            cfg["dataset"]["batch_size"] = int(args.batch_size)
        if args.num_workers is not None:
            cfg["dataset"]["num_workers"] = int(args.num_workers)
        if args.epochs is not None:
            cfg["training"]["epochs"] = int(args.epochs)
        if args.plot_every is not None:
            cfg["training"]["plot_every_n_epochs"] = int(args.plot_every)

        output_path = output_dir / spec["filename"]
        output_path.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
        manifest.append(
            {
                "filename": spec["filename"],
                "experiment_name": spec["experiment_name"],
                "model_family": spec["model_family"],
                "protein_context_mode": spec["protein_context_mode"],
                "ligand_context_mode": spec["ligand_context_mode"],
                "launch_extra_args": spec["launch_extra_args"],
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote real-series configs to: {output_dir.resolve()}")
    print(f"Configs written: {len(manifest)}")
    for item in manifest:
        print(
            f"- {item['filename']}: {item['experiment_name']} | "
            f"model={item['model_family']} | protein_context={item['protein_context_mode']} | "
            f"ligand_context={item['ligand_context_mode']} | "
            f"extra_args={item['launch_extra_args'] or '<none>'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
