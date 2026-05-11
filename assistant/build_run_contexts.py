#!/usr/bin/env python3
"""Stage 1 of the manual assistant pipeline: build per-run context packets.

This module scans a factual `runs/` tree and distills each run folder into one
compact JSON packet. The output is intentionally machine-oriented: it is not a
user-facing report, but a stable intermediate artifact for prompt assembly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rebuild_experiment_index import (
    build_registry_row,
    discover_run_dirs,
    find_first,
    load_json,
)


LOG_PATTERNS = (
    "Run started at:",
    "Experiment signature:",
    "Run settings ->",
    "Model selected ->",
    "Model branch summary ->",
    "Global graph settings ->",
    "Global encoder settings ->",
    "Local graph settings ->",
    "Local encoder settings ->",
    "Protein context settings ->",
    "Ligand context settings ->",
    "Improved:",
    "Training completed.",
    "FINAL TEST ->",
    "Experiment completed successfully",
    "[WARNING]",
    "[ERROR]",
)

HEADER_PATTERNS = (
    "Run started at:",
    "Experiment signature:",
    "Base Datasets folder:",
    "Base protein context features folder:",
    "Base ligand context features folder:",
    "Run results folder:",
    "Log file:",
    "Run settings ->",
    "Model selected ->",
    "Model branch summary ->",
    "Global graph settings ->",
    "Global encoder settings ->",
    "Local graph settings ->",
    "Local encoder settings ->",
    "Protein context settings ->",
    "Ligand context settings ->",
    "Optimizer settings ->",
    "Primary metric:",
    "Early stopping enabled",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for stage 1 context extraction."""
    parser = argparse.ArgumentParser(
        description="Build compact per-run context packets for future LLM journal generation."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing run folders. Defaults to ./runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assistant/experiment_journal_llm_context.json"),
        help="Where to write the extracted context packets.",
    )
    parser.add_argument(
        "--max-log-lines",
        type=int,
        default=24,
        help="Maximum number of selected log lines per run.",
    )
    return parser.parse_args()


def load_text(path: Path | None) -> str:
    """Read UTF-8 text when present; otherwise return an empty string."""
    if path is None or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def load_json_optional(path: Path | None) -> dict:
    """Load JSON or return an empty dict for missing/invalid files."""
    return load_json(path) or {} if path is not None else {}


def extract_history_summary(history: dict | None) -> dict:
    """Collapse dense metric arrays into a small numerical summary."""
    if not history:
        return {}

    summary: dict[str, object] = {}
    for key in (
        "train_loss",
        "val_loss",
        "train_pearson",
        "val_pearson",
        "train_rmse",
        "val_rmse",
        "train_ci",
        "val_ci",
    ):
        values = history.get(key)
        if isinstance(values, list) and values:
            block = {
                "n": len(values),
                "first": values[0],
                "last": values[-1],
            }
            if "loss" in key or "rmse" in key:
                block["best_min"] = min(values)
            if "pearson" in key or key.endswith("_ci"):
                block["best_max"] = max(values)
            summary[key] = block
    return summary


def build_front_loaded_epoch_indices(length: int, num_points: int, extra_indices: list[int] | None = None) -> list[int]:
    if length <= 0:
        return []
    if length == 1:
        return [0]

    extra_indices = extra_indices or []
    indices = {0, length - 1}
    if length > 1:
        indices.add(1)

    for i in range(num_points):
        frac = i / max(num_points - 1, 1)
        idx = int(round((frac ** 2) * (length - 1)))
        indices.add(min(max(idx, 0), length - 1))

    for idx in extra_indices:
        if 0 <= idx < length:
            indices.add(idx)

    return sorted(indices)


def extract_history_sampled_points(
    history: dict | None,
    *,
    best_epoch: str = "",
    num_points: int = 7,
) -> dict:
    """Keep a front-loaded subset of the history as prompt-friendly checkpoints."""
    if not history:
        return {}

    best_epoch_index: list[int] = []
    try:
        if best_epoch not in {"", None, "n/a"}:
            best_epoch_index = [max(int(best_epoch) - 1, 0)]
    except (TypeError, ValueError):
        best_epoch_index = []

    snapshot: dict[str, object] = {
        "_meta": {
            "kind": "sampled_checkpoints",
            "policy": "front_loaded_with_first_second_last_and_best_epoch",
            "num_requested_points": num_points,
        }
    }
    for key in (
        "train_loss",
        "val_loss",
        "train_pearson",
        "val_pearson",
        "train_rmse",
        "val_rmse",
        "train_ci",
        "val_ci",
    ):
        values = history.get(key)
        if isinstance(values, list) and values:
            indices = build_front_loaded_epoch_indices(len(values), num_points, extra_indices=best_epoch_index)
            snapshot[key] = [
                {"epoch": idx + 1, "value": values[idx]}
                for idx in indices
            ]
    return snapshot


def extract_log_excerpt(log_text: str, max_lines: int) -> list[str]:
    """Select the most informative log lines for LLM consumption."""
    if not log_text:
        return []

    selected: list[str] = []
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(pattern in line for pattern in LOG_PATTERNS):
            selected.append(line)
        if len(selected) >= max_lines:
            break
    return selected


def extract_log_header_excerpt(log_text: str, max_lines: int) -> list[str]:
    """Capture setup/header log lines separately from later training events."""
    if not log_text:
        return []

    selected: list[str] = []
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(pattern in line for pattern in HEADER_PATTERNS):
            selected.append(line)
        if len(selected) >= max_lines:
            break
    return selected


def extract_assistant_summary_excerpt(summary_text: str) -> list[str]:
    """Extract factual bullets from the per-run assistant summary."""
    if not summary_text:
        return []

    lines: list[str] = []
    for raw_line in summary_text.splitlines():
        line = raw_line.rstrip()
        if not line or line.startswith("# Assistant Summary") or line.startswith("## Caveat"):
            continue
        if line.startswith("## ") or line.startswith("- "):
            lines.append(line)
    return lines


def collect_artifact_paths(run_dir: Path) -> dict[str, str]:
    """Collect absolute paths to the artifacts that matter for run review."""
    candidates = {
        "run_dir": run_dir,
        "config": run_dir / "config.json",
        "summary": find_first(run_dir, "assistant_summary*.md"),
        "history": find_first(run_dir, "history*.json"),
        "test_results": find_first(run_dir, "test_results*.json"),
        "scatter_diagnostics": find_first(run_dir, "best_validation_scatter_diagnostics*.json"),
        "report": run_dir / "model_performance_report.png",
        "run_log": (run_dir / "run.log") if (run_dir / "run.log").exists() else (find_first(run_dir, "log*.txt") or run_dir / "log.txt"),
        "run_manifest": run_dir / "run_manifest.json",
    }
    paths: dict[str, str] = {}
    for label, path in candidates.items():
        if path is None:
            continue
        if path.exists():
            paths[label] = str(path.resolve())
    return paths


def extract_config_excerpt(config: dict | None) -> dict:
    """Project a large config down to the fields relevant for journal analysis."""
    config = config or {}
    training = config.get("training", {})
    dataset = config.get("dataset", {})
    model = config.get("model", {})
    global_graph = model.get("global_graph", {})
    global_encoder = model.get("global_encoder", {})
    local_graph = model.get("local_graph", {})
    local_encoder = model.get("local_encoder", {})

    return {
        "experiment_name": config.get("experiment_name", ""),
        "source_subset": dataset.get("source_subset", ""),
        "core_as_test": dataset.get("core_as_test", ""),
        "epochs": training.get("epochs", ""),
        "batch_size": training.get("batch_size", ""),
        "num_workers": training.get("num_workers", ""),
        "primary_metric": training.get("early_stopping", {}).get("primary_monitor", ""),
        "model_selected": model.get("selected", ""),
        "global_graph_selected": global_graph.get("selected", ""),
        "global_graph_config": global_graph.get(global_graph.get("selected", ""), {}) if isinstance(global_graph, dict) else {},
        "global_encoder_selected": global_encoder.get("selected", ""),
        "global_encoder_config": global_encoder.get(global_encoder.get("selected", ""), {}) if isinstance(global_encoder, dict) else {},
        "local_graph_selected": local_graph.get("selected", ""),
        "local_graph_config": local_graph.get(local_graph.get("selected", ""), {}) if isinstance(local_graph, dict) else {},
        "local_encoder_selected": local_encoder.get("selected", ""),
        "local_encoder_config": local_encoder.get(local_encoder.get("selected", ""), {}) if isinstance(local_encoder, dict) else {},
        "protein_context_selected": model.get("protein_context", {}).get("selected", ""),
        "protein_context_config": model.get("protein_context", {}).get(model.get("protein_context", {}).get("selected", ""), {}),
        "ligand_context_selected": model.get("ligand_context", {}).get("selected", ""),
        "ligand_context_config": model.get("ligand_context", {}).get(model.get("ligand_context", {}).get("selected", ""), {}),
    }


def build_run_context(run_dir: Path, max_log_lines: int) -> dict:
    """Build one complete stage-1 context packet for a single run folder."""
    row, _ = build_registry_row(run_dir)
    config = load_json(run_dir / "config.json") or {}
    history_path = find_first(run_dir, "history*.json")
    test_path = find_first(run_dir, "test_results*.json")
    scatter_path = find_first(run_dir, "best_validation_scatter_diagnostics*.json")
    summary_path = find_first(run_dir, "assistant_summary*.md")
    log_path = (run_dir / "run.log") if (run_dir / "run.log").exists() else (find_first(run_dir, "log*.txt") or run_dir / "log.txt")

    history = load_json_optional(history_path)
    test_results = load_json_optional(test_path)
    scatter = load_json_optional(scatter_path)

    return {
        "experiment_signature": row.get("experiment_signature", ""),
        "run_dir": str(run_dir.resolve()),
        "registry_snapshot": row,
        "artifact_paths": collect_artifact_paths(run_dir),
        "config_excerpt": extract_config_excerpt(config),
        "setup": {
            "experiment_name": row.get("experiment_name", ""),
            "model_family": row.get("model_family", ""),
            "execution_env": row.get("execution_env", ""),
            "hostname": row.get("hostname", ""),
            "source_subset": row.get("source_subset", ""),
            "splitter": row.get("splitter", ""),
            "splitter_seed": row.get("splitter_seed", ""),
            "core_as_test": row.get("core_as_test", ""),
            "primary_metric": row.get("primary_metric", ""),
            "protein_context": str(config.get("model", {}).get("protein_context", {}).get("selected", "")),
            "ligand_context": str(config.get("model", {}).get("ligand_context", {}).get("selected", "")),
        },
        "metrics": {
            "best_epoch": row.get("best_epoch", ""),
            "epochs_completed": row.get("epochs_completed", ""),
            "test_rmse": row.get("test_rmse", ""),
            "test_pearson": row.get("test_pearson", ""),
            "test_ci": row.get("test_ci", ""),
        },
        "history_summary": extract_history_summary(history),
        "history_sampled_points": extract_history_sampled_points(
            history,
            best_epoch=row.get("best_epoch", ""),
            num_points=7,
        ),
        "test_results": test_results,
        "scatter_diagnostics": scatter,
        "log_header_excerpt": extract_log_header_excerpt(load_text(log_path), 24),
        "log_excerpt": extract_log_excerpt(load_text(log_path), max_log_lines),
        "assistant_summary_excerpt": extract_assistant_summary_excerpt(load_text(summary_path)),
    }


def main() -> None:
    """Run stage 1 over the selected `runs/` tree and write context JSON."""
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    output_path = args.output.resolve()

    if not runs_dir.exists():
        raise SystemExit(f"runs dir does not exist: {runs_dir}")

    packets = []
    for run_dir in discover_run_dirs(runs_dir):
        packets.append(build_run_context(run_dir, args.max_log_lines))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "runs_dir": str(runs_dir),
                "num_runs": len(packets),
                "packets": packets,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Wrote assistant context packets to: {output_path}")
    print(f"Collected runs: {len(packets)}")


if __name__ == "__main__":
    main()
