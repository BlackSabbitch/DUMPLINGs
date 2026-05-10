#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a reduced smoke-test config from the main DUMPLINGs config."
    )
    parser.add_argument("--base-config", default="config.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--experiment-name", default="DUMPLING_cluster_smoke")
    parser.add_argument(
        "--model-family",
        default=None,
        choices=["A1", "A2", "A3"],
        help="Optional override for model.selected",
    )
    parser.add_argument(
        "--protein-context-mode",
        default=None,
        help="Optional override for model.protein_context.selected",
    )
    parser.add_argument(
        "--ligand-context-mode",
        default=None,
        help="Optional override for model.ligand_context.selected",
    )
    args = parser.parse_args()

    base_path = Path(args.base_config)
    output_path = Path(args.output)
    config = json.loads(base_path.read_text())

    config["experiment_name"] = args.experiment_name
    config["debug_mode"] = True
    config["dataset"]["batch_size"] = int(args.batch_size)
    config["dataset"]["num_workers"] = int(args.num_workers)
    config["dataset"]["save_train_test_val_datasets"] = False
    config["training"]["epochs"] = int(args.epochs)
    config["training"]["plot_every_n_epochs"] = max(int(args.epochs), 1)
    config["training"]["save_only_best_epoch"] = True
    config["training"]["memory_log_every_n_batches"] = 0
    config["training"]["slow_batch_warn_seconds"] = 5.0

    early_stopping = config["training"].get("early_stopping", {})
    if early_stopping:
        early_stopping["patience"] = max(int(args.epochs), 1)

    if args.model_family is not None:
        config["model"]["selected"] = args.model_family
    if args.protein_context_mode is not None:
        config["model"]["protein_context"]["selected"] = args.protein_context_mode
    if args.ligand_context_mode is not None:
        config["model"]["ligand_context"]["selected"] = args.ligand_context_mode

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=4))
    print(f"Wrote smoke config to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
