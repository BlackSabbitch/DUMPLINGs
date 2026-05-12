#!/usr/bin/env python3
"""Stage 1s of the manual assistant pipeline: build per-series context packets.

This module groups the stage-1 run packets by `experiment_name` and produces a
series-level JSON context artifact. It is the factual substrate for the future
series journal assistant.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI options for series context extraction."""
    parser = argparse.ArgumentParser(
        description="Build compact per-series context packets from run-level assistant context."
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=Path("assistant/experiment_journal_llm_context.json"),
        help="Run-level context packet JSON produced by build_run_contexts.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assistant/experiment_series_llm_context.json"),
        help="Where to write the grouped series context packets.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load a JSON file used by the series context builder."""
    return json.loads(path.read_text(encoding="utf-8"))


def coerce_float(value: object) -> float | None:
    """Best-effort numeric conversion for factual metric aggregation."""
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_numeric(values: list[object], *, decimals: int = 4) -> dict[str, str]:
    """Summarize one numeric series with mean/std/min/max."""
    clean = [value for value in (coerce_float(v) for v in values) if value is not None]
    if not clean:
        return {}
    std = statistics.pstdev(clean) if len(clean) > 1 else 0.0
    return {
        "count": str(len(clean)),
        "mean": f"{statistics.fmean(clean):.{decimals}f}",
        "std": f"{std:.{decimals}f}",
        "min": f"{min(clean):.{decimals}f}",
        "max": f"{max(clean):.{decimals}f}",
    }


def consensus(packets: list[dict], key: str, *, from_block: str = "setup") -> str:
    """Return the unanimous value for a setup field, or `mixed`."""
    values = {
        str(packet.get(from_block, {}).get(key, ""))
        for packet in packets
        if str(packet.get(from_block, {}).get(key, ""))
    }
    if not values:
        return ""
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def build_series_packet(experiment_name: str, packets: list[dict]) -> dict:
    """Collapse multiple run packets into one factual series packet."""
    packets = sorted(
        packets,
        key=lambda packet: (
            str(packet.get("registry_snapshot", {}).get("started_at", "")),
            str(packet.get("experiment_signature", "")),
        ),
    )
    rows = [packet.get("registry_snapshot", {}) for packet in packets]
    seeds = sorted({str(row.get("splitter_seed", "")) for row in rows if str(row.get("splitter_seed", ""))})
    batch_positions = [
        f"{row.get('batch_run_index', '')}/{row.get('batch_n_times', '')}"
        for row in rows
        if row.get("batch_run_index", "") and row.get("batch_n_times", "")
    ]
    success_count = sum(1 for row in rows if row.get("status", "") == "success")

    return {
        "series_name": experiment_name,
        "setup": {
            "experiment_name": experiment_name,
            "model_family": consensus(packets, "model_family"),
            "source_subset": consensus(packets, "source_subset"),
            "splitter": consensus(packets, "splitter"),
            "core_as_test": consensus(packets, "core_as_test"),
            "primary_metric": consensus(packets, "primary_metric"),
            "protein_context": consensus(packets, "protein_context"),
            "ligand_context": consensus(packets, "ligand_context"),
            "execution_env": consensus(packets, "execution_env"),
            "hostname": consensus(packets, "hostname"),
        },
        "series_summary": {
            "started_at": str(rows[0].get("started_at", "")) if rows else "",
            "finished_at": str(rows[-1].get("finished_at", "")) if rows else "",
            "total_runs": len(rows),
            "success_count": success_count,
            "seeds": seeds,
            "batch_positions": batch_positions,
        },
        "aggregate_metrics": {
            "duration_sec": summarize_numeric([row.get("duration_sec", "") for row in rows], decimals=1),
            "test_rmse": summarize_numeric([row.get("test_rmse", "") for row in rows], decimals=4),
            "test_pearson": summarize_numeric([row.get("test_pearson", "") for row in rows], decimals=4),
            "test_ci": summarize_numeric([row.get("test_ci", "") for row in rows], decimals=4),
        },
        "members": [
            {
                "experiment_signature": packet.get("experiment_signature", ""),
                "run_dir": packet.get("run_dir", ""),
                "registry_snapshot": packet.get("registry_snapshot", {}),
                "metrics": packet.get("metrics", {}),
                "config_excerpt": packet.get("config_excerpt", {}),
                "artifact_paths": packet.get("artifact_paths", {}),
            }
            for packet in packets
        ],
    }


def main() -> None:
    """Run the series grouping stage and write grouped JSON packets."""
    args = parse_args()
    context_path = args.context.resolve()
    output_path = args.output.resolve()

    if not context_path.exists():
        raise SystemExit(f"context file does not exist: {context_path}")

    context = load_json(context_path)
    packets = context.get("packets", [])
    grouped: dict[str, list[dict]] = {}
    for packet in packets:
        experiment_name = str(packet.get("registry_snapshot", {}).get("experiment_name", "")) or str(packet.get("experiment_signature", ""))
        grouped.setdefault(experiment_name, []).append(packet)

    series_packets = [
        build_series_packet(experiment_name, members)
        for experiment_name, members in sorted(grouped.items())
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "run_context_path": str(context_path),
                "num_series": len(series_packets),
                "series_packets": series_packets,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Wrote assistant series context packets to: {output_path}")
    print(f"Collected series: {len(series_packets)}")


if __name__ == "__main__":
    main()
