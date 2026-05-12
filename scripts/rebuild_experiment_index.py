#!/usr/bin/env python3
"""Rebuild the factual experiment index from run folders.

This script treats `runs/<experiment_signature>/...` folders as the primary
portable artifacts. It rescans those folders and rewrites:

- `experiment_registry.csv`
- `experiment_journal.md`

That makes copied runs easy to merge across machines without manually merging
top-level index files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from datetime import datetime
from pathlib import Path


REGISTRY_FIELD_ORDER = [
    "started_at",
    "finished_at",
    "duration_sec",
    "status",
    "experiment_name",
    "experiment_signature",
    "exp_dir",
    "config_path",
    "git_commit",
    "execution_env",
    "hostname",
    "artifact_root",
    "model_family",
    "source_subset",
    "splitter",
    "splitter_seed",
    "core_as_test",
    "a3_mixer_bias",
    "batch_run_index",
    "batch_n_times",
    "device",
    "primary_metric",
    "best_epoch",
    "epochs_completed",
    "test_rmse",
    "test_pearson",
    "test_ci",
]


SUMMARY_LINE_RE = re.compile(r"^- ([A-Za-z0-9_]+): `(.*)`$")
SIGNATURE_TS_RE = re.compile(r"_(\d{8}_\d{6})(?:_(\d{6}))?$")


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the factual index rebuild."""
    parser = argparse.ArgumentParser(
        description="Rebuild experiment_registry.csv and experiment_journal.md by scanning run folders."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing run folders. Defaults to ./runs",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Where to write the rebuilt CSV. Defaults to <runs-dir>/experiment_registry.csv",
    )
    parser.add_argument(
        "--journal-path",
        type=Path,
        default=None,
        help="Where to write the rebuilt markdown journal. Defaults to <runs-dir>/experiment_journal.md",
    )
    parser.add_argument(
        "--series-journal-path",
        type=Path,
        default=None,
        help="Where to write the rebuilt series markdown journal. Defaults to <runs-dir>/experiment_series_journal.md",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    """Load JSON defensively and return `None` on missing/corrupt files."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_first(path: Path, pattern: str) -> Path | None:
    """Return the first child matching a glob pattern, if any."""
    matches = sorted(path.glob(pattern))
    return matches[0] if matches else None


def parse_summary_snapshot(summary_path: Path | None) -> dict[str, str]:
    """Extract factual key/value pairs from `assistant_summary*.md`."""
    if summary_path is None or not summary_path.exists():
        return {}

    snapshot: dict[str, str] = {}

    with summary_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            match = SUMMARY_LINE_RE.match(line.strip())
            if match:
                snapshot[match.group(1)] = match.group(2)
    return snapshot


def get_model_family(config: dict | None) -> str:
    if not config:
        return ""
    return str(config.get("model", {}).get("selected", ""))


def infer_legacy_model_family(config: dict | None, experiment_name: str, signature: str) -> str:
    explicit = get_model_family(config)
    if explicit:
        return explicit

    for candidate in (experiment_name, signature):
        match = re.search(r"\b(A[0-9]+[A-Za-z]*)\b", candidate)
        if match:
            return match.group(1)

    graph_encoder_selected = str((config or {}).get("model", {}).get("graph_encoder", {}).get("selected", ""))
    if graph_encoder_selected == "duo":
        return "legacy_duo"

    return ""


def get_splitter_info(config: dict | None) -> tuple[str, str]:
    if not config:
        return "", ""
    splitter_cfg = config.get("splitter", {})
    selected = str(splitter_cfg.get("selected", ""))
    available = splitter_cfg.get("available", {})
    selected_cfg = available.get(selected, {}) if isinstance(available, dict) else {}
    seed = selected_cfg.get("seed", "")
    return selected, str(seed)


def get_a3_mixer_bias(config: dict | None, test_metrics: dict | None, summary_snapshot: dict[str, str]) -> str:
    if summary_snapshot.get("mixer_has_bias") in {"True", "False"}:
        return "True" if summary_snapshot.get("mixer_has_bias") == "True" else "False"
    readout = (test_metrics or {}).get("readout_diagnostics", {})
    if "mixer_has_bias" in readout:
        return str(readout.get("mixer_has_bias"))
    model_cfg = (config or {}).get("model", {}).get("a3", {})
    if "mixer_bias" in model_cfg:
        return str(model_cfg.get("mixer_bias"))
    return ""


def parse_signature_timestamp(signature: str) -> str:
    match = SIGNATURE_TS_RE.search(signature)
    if not match:
        return ""
    base = match.group(1)
    micros = match.group(2) or "000000"
    try:
        dt = datetime.strptime(f"{base}_{micros}", "%Y%m%d_%H%M%S_%f")
        return dt.isoformat(timespec="seconds")
    except ValueError:
        return ""


def normalize_iso_like(value: str) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value).isoformat(timespec="seconds")
    except ValueError:
        return value


def build_factual_notes(
    row: dict[str, str],
    config: dict | None,
) -> list[str]:
    """Build the compact factual note used in the rebuilt markdown journal."""
    notes: list[str] = []

    model_family = row.get("model_family", "")
    if model_family:
        notes.append(f"model=`{model_family}`")

    protein_context = str((config or {}).get("model", {}).get("protein_context", {}).get("selected", ""))
    ligand_context = str((config or {}).get("model", {}).get("ligand_context", {}).get("selected", ""))
    if protein_context:
        notes.append(f"protein_context=`{protein_context}`")
    if ligand_context:
        notes.append(f"ligand_context=`{ligand_context}`")

    if row.get("test_rmse"):
        notes.append(f"test_RMSE=`{row.get('test_rmse', '')}`")
    if row.get("test_pearson"):
        notes.append(f"test_Pearson_R=`{row.get('test_pearson', '')}`")
    if row.get("test_ci"):
        notes.append(f"test_CI=`{row.get('test_ci', '')}`")

    best_epoch = row.get("best_epoch", "")
    epochs_completed = row.get("epochs_completed", "")
    if best_epoch and epochs_completed:
        notes.append(f"best_epoch=`{best_epoch}`")
        notes.append(f"epochs_completed=`{epochs_completed}`")
    elif epochs_completed:
        notes.append(f"epochs_completed=`{epochs_completed}`")

    return notes


def build_registry_row(run_dir: Path) -> tuple[dict[str, str], list[str]]:
    """Recover one portable registry row from the artifacts in a run folder."""
    manifest = load_json(run_dir / "run_manifest.json") or {}
    manifest_row = manifest.get("registry_row", {}) if isinstance(manifest, dict) else {}
    history = manifest.get("history_metrics") if isinstance(manifest, dict) else None

    config = load_json(run_dir / "config.json")
    history = history or load_json(find_first(run_dir, "history*.json") or Path(""))
    test_metrics = load_json(find_first(run_dir, "test_results*.json") or Path(""))
    summary_path = find_first(run_dir, "assistant_summary*.md")
    summary_snapshot = parse_summary_snapshot(summary_path)

    signature = (
        str(manifest_row.get("experiment_signature", ""))
        or summary_snapshot.get("experiment_signature", "")
        or run_dir.name
    )
    experiment_name = (
        str(manifest_row.get("experiment_name", ""))
        or summary_snapshot.get("experiment_name", "")
        or str((config or {}).get("experiment_name", ""))
    )
    started_at = (
        str(manifest_row.get("started_at", ""))
        or normalize_iso_like(summary_snapshot.get("started_at", ""))
        or parse_signature_timestamp(signature)
    )
    finished_at = (
        str(manifest_row.get("finished_at", ""))
        or normalize_iso_like(summary_snapshot.get("finished_at", ""))
        or started_at
    )
    duration_sec = str(manifest_row.get("duration_sec", ""))
    if not duration_sec and started_at and finished_at:
        try:
            duration_sec = str(round(
                (datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds(),
                1,
            ))
        except ValueError:
            duration_sec = ""

    splitter, splitter_seed = get_splitter_info(config)
    best_epoch = (
        str(manifest_row.get("best_epoch", ""))
        or summary_snapshot.get("best_epoch", "")
    )
    epochs_completed = (
        str(manifest_row.get("epochs_completed", ""))
        or summary_snapshot.get("epochs_completed", "")
    )
    if not epochs_completed and history:
        values = history.get("train_loss", [])
        if isinstance(values, list):
            epochs_completed = str(len(values))

    model_family = (
        str(manifest_row.get("model_family", ""))
        or summary_snapshot.get("model_family", "")
        or infer_legacy_model_family(config, experiment_name, signature)
    )

    row = {
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": duration_sec,
        "status": str(manifest_row.get("status", "")) or summary_snapshot.get("status", "") or ("success" if test_metrics else ""),
        "experiment_name": experiment_name,
        "experiment_signature": signature,
        "exp_dir": str(manifest_row.get("exp_dir", "")) or str(run_dir.resolve()),
        "config_path": str(manifest_row.get("config_path", "")) or str((run_dir / "config.json").resolve()),
        "git_commit": str(manifest_row.get("git_commit", "")) or summary_snapshot.get("git_commit", ""),
        "execution_env": str(manifest_row.get("execution_env", "")) or summary_snapshot.get("execution_env", ""),
        "hostname": str(manifest_row.get("hostname", "")) or summary_snapshot.get("hostname", ""),
        "artifact_root": str(manifest_row.get("artifact_root", "")) or summary_snapshot.get("artifact_root", "") or str(run_dir.resolve().parent),
        "model_family": model_family,
        "source_subset": str(manifest_row.get("source_subset", "")) or str((config or {}).get("dataset", {}).get("source_subset", "")),
        "splitter": str(manifest_row.get("splitter", "")) or splitter,
        "splitter_seed": str(manifest_row.get("splitter_seed", "")) or splitter_seed,
        "core_as_test": str(manifest_row.get("core_as_test", "")) or str((config or {}).get("dataset", {}).get("core_as_test", "")),
        "a3_mixer_bias": str(manifest_row.get("a3_mixer_bias", "")) or get_a3_mixer_bias(config, test_metrics, summary_snapshot),
        "batch_run_index": str(manifest_row.get("batch_run_index", "")),
        "batch_n_times": str(manifest_row.get("batch_n_times", "")),
        "device": str(manifest_row.get("device", "")),
        "primary_metric": str(manifest_row.get("primary_metric", "")) or str((config or {}).get("training", {}).get("early_stopping", {}).get("primary_monitor", "")),
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "test_rmse": str(manifest_row.get("test_rmse", "")) or str((test_metrics or {}).get("RMSE", "")),
        "test_pearson": str(manifest_row.get("test_pearson", "")) or str((test_metrics or {}).get("Pearson_R", "")),
        "test_ci": str(manifest_row.get("test_ci", "")) or str((test_metrics or {}).get("CI", "")),
    }
    return row, build_factual_notes(row, config)


def discover_run_dirs(runs_dir: Path) -> list[Path]:
    """Discover run folders that contain at least a `config.json` anchor."""
    run_dirs: list[Path] = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "config.json").exists():
            run_dirs.append(child)
    return run_dirs


def row_sort_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("started_at", ""), row.get("experiment_signature", ""))


def coerce_float(value: str) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def format_stat(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def consensus_value(rows: list[dict[str, str]], key: str) -> str:
    values = {row.get(key, "") for row in rows if row.get(key, "")}
    if not values:
        return ""
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def summarize_numeric_field(rows: list[dict[str, str]], key: str, *, decimals: int = 4) -> dict[str, str]:
    values = [coerce_float(row.get(key, "")) for row in rows]
    series = [value for value in values if value is not None]
    if not series:
        return {}
    std = statistics.pstdev(series) if len(series) > 1 else 0.0
    return {
        "count": str(len(series)),
        "mean": format_stat(statistics.fmean(series), decimals),
        "std": format_stat(std, decimals),
        "min": format_stat(min(series), decimals),
        "max": format_stat(max(series), decimals),
    }


def write_registry(registry_path: Path, rows: list[dict[str, str]]) -> None:
    """Rewrite `experiment_registry.csv` from scratch."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELD_ORDER)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REGISTRY_FIELD_ORDER})


def build_series_records(rows_with_notes: list[tuple[dict[str, str], list[str], Path]]) -> list[dict]:
    grouped: dict[str, list[tuple[dict[str, str], list[str], Path]]] = {}
    for item in rows_with_notes:
        row = item[0]
        key = row.get("experiment_name", "") or row.get("experiment_signature", "")
        grouped.setdefault(key, []).append(item)

    series_records: list[dict] = []
    for experiment_name, members in sorted(grouped.items()):
        members.sort(key=lambda item: row_sort_key(item[0]))
        rows = [row for row, _, _ in members]
        success_count = sum(1 for row in rows if row.get("status", "") == "success")
        seeds = sorted({row.get("splitter_seed", "") for row in rows if row.get("splitter_seed", "")})
        batch_positions = [
            f"{row.get('batch_run_index', '')}/{row.get('batch_n_times', '')}"
            for row in rows
            if row.get("batch_run_index", "") and row.get("batch_n_times", "")
        ]
        series_records.append(
            {
                "experiment_name": experiment_name,
                "model_family": consensus_value(rows, "model_family"),
                "source_subset": consensus_value(rows, "source_subset"),
                "splitter": consensus_value(rows, "splitter"),
                "core_as_test": consensus_value(rows, "core_as_test"),
                "primary_metric": consensus_value(rows, "primary_metric"),
                "started_at": rows[0].get("started_at", ""),
                "finished_at": rows[-1].get("finished_at", ""),
                "total_runs": len(rows),
                "success_count": success_count,
                "seeds": seeds,
                "batch_positions": batch_positions,
                "duration_sec_stats": summarize_numeric_field(rows, "duration_sec", decimals=1),
                "test_rmse_stats": summarize_numeric_field(rows, "test_rmse", decimals=4),
                "test_pearson_stats": summarize_numeric_field(rows, "test_pearson", decimals=4),
                "test_ci_stats": summarize_numeric_field(rows, "test_ci", decimals=4),
                "members": [
                    {
                        "experiment_signature": row.get("experiment_signature", ""),
                        "started_at": row.get("started_at", ""),
                        "finished_at": row.get("finished_at", ""),
                        "status": row.get("status", ""),
                        "splitter_seed": row.get("splitter_seed", ""),
                        "batch_run_index": row.get("batch_run_index", ""),
                        "batch_n_times": row.get("batch_n_times", ""),
                        "duration_sec": row.get("duration_sec", ""),
                        "best_epoch": row.get("best_epoch", ""),
                        "epochs_completed": row.get("epochs_completed", ""),
                        "test_rmse": row.get("test_rmse", ""),
                        "test_pearson": row.get("test_pearson", ""),
                        "test_ci": row.get("test_ci", ""),
                        "run_dir": str(run_dir.resolve()),
                    }
                    for row, _, run_dir in members
                ],
            }
        )
    return series_records


def build_artifact_links(run_dir: Path, journal_dir: Path) -> str:
    """Build markdown links to the key artifacts inside one run folder."""
    candidates = [
        ("folder", Path(".")),
        ("config", Path("config.json")),
        ("summary", find_first(run_dir, "assistant_summary*.md")),
        ("history", find_first(run_dir, "history*.json")),
        ("test", find_first(run_dir, "test_results*.json")),
        ("report", run_dir / "model_performance_report.png"),
        ("log", (run_dir / "run.log") if (run_dir / "run.log").exists() else (find_first(run_dir, "log*.txt") or run_dir / "log.txt")),
        ("error", run_dir / "run_err.log"),
    ]

    links: list[str] = []
    for label, rel_path in candidates:
        if rel_path is None:
            continue
        if rel_path == Path("."):
            folder_rel = Path(".") / run_dir.name
            links.append(f"[{label}]({folder_rel.as_posix()}/)")
            continue
        path_obj = rel_path if isinstance(rel_path, Path) else Path(rel_path)
        if not path_obj.is_absolute():
            path_obj = run_dir / path_obj
        if not path_obj.exists():
            continue
        relative_target = path_obj.resolve().relative_to(journal_dir.resolve())
        links.append(f"[{label}]({relative_target.as_posix()})")

    return " | ".join(links)


def build_report_preview(run_dir: Path, journal_dir: Path) -> list[str]:
    """Embed a lightweight markdown image preview when a report PNG exists."""
    report_path = run_dir / "model_performance_report.png"
    if not report_path.exists():
        return []

    relative_target = report_path.resolve().relative_to(journal_dir.resolve()).as_posix()
    return [
        "- report preview:",
        f"  ![]({relative_target})",
    ]


def build_journal_entry(
    row: dict[str, str],
    heuristic_notes: list[str],
    run_dir: Path,
    journal_dir: Path,
) -> list[str]:
    """Render one rebuilt journal entry from a factual registry row."""
    header = f"## {row.get('finished_at', '') or row.get('started_at', '')} | {row.get('experiment_signature', '')}"
    lines = [header, ""]
    lines.append(
        f"- status: `{row.get('status', '')}` | model: `{row.get('model_family', '')}` | "
        f"env: `{row.get('execution_env', '')}` | seed: `{row.get('splitter_seed', '')}` | "
        f"duration_sec: `{row.get('duration_sec', '')}`"
    )
    lines.append(
        f"- location: `{row.get('exp_dir', '')}` on `{row.get('hostname', '')}`"
    )
    if row.get("test_rmse") or row.get("test_pearson") or row.get("test_ci"):
        lines.append(
            f"- final metrics: RMSE=`{row.get('test_rmse', '')}`, "
            f"Pearson_R=`{row.get('test_pearson', '')}`, "
            f"CI=`{row.get('test_ci', '')}`"
        )
    artifact_links = build_artifact_links(run_dir, journal_dir)
    if artifact_links:
        lines.append(f"- artifacts: {artifact_links}")
    lines.extend(build_report_preview(run_dir, journal_dir))
    if heuristic_notes:
        lines.append(f"- assistant note: {' '.join(heuristic_notes)}")
    else:
        lines.append("- assistant note: no extracted assistant summary note.")
    lines.extend([
        "",
        "> Rebuilt from run-folder artifacts. This journal is a convenience index, not primary evidence.",
        "",
    ])
    return lines


def write_journal(journal_path: Path, rows_with_notes: list[tuple[dict[str, str], list[str], Path]]) -> None:
    """Rewrite `experiment_journal.md` from scratch."""
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("w", encoding="utf-8") as f:
        f.write("# Experiment Journal\n\n")
        f.write(
            "This file is rebuilt from the discovered run folders. "
            "Treat it as a readable index over the raw experiment artifacts.\n\n"
        )
        for row, heuristic_notes, run_dir in rows_with_notes:
            f.write("\n".join(build_journal_entry(row, heuristic_notes, run_dir, journal_path.parent)))
            f.write("\n")


def build_series_journal_entry(series: dict) -> list[str]:
    header = (
        f"## {series.get('experiment_name', '')} | "
        f"model=`{series.get('model_family', '')}` | runs=`{series.get('total_runs', '')}`"
    )
    lines = [header, ""]
    lines.append(
        f"- setup: subset=`{series.get('source_subset', '')}` | "
        f"splitter=`{series.get('splitter', '')}` | core_as_test=`{series.get('core_as_test', '')}` | "
        f"primary_metric=`{series.get('primary_metric', '')}`"
    )
    lines.append(
        f"- outcomes: success=`{series.get('success_count', 0)}` / observed=`{series.get('total_runs', 0)}` | "
        f"seeds=`{', '.join(series.get('seeds', [])) or 'n/a'}`"
    )
    if series.get("batch_positions"):
        lines.append(f"- batch positions: `{', '.join(series.get('batch_positions', []))}`")
    lines.append(
        f"- window: started=`{series.get('started_at', '')}` | finished=`{series.get('finished_at', '')}`"
    )

    duration_stats = series.get("duration_sec_stats", {})
    if duration_stats:
        lines.append(
            f"- duration_sec: mean=`{duration_stats.get('mean', '')}` | std=`{duration_stats.get('std', '')}` | "
            f"min=`{duration_stats.get('min', '')}` | max=`{duration_stats.get('max', '')}`"
        )

    pearson_stats = series.get("test_pearson_stats", {})
    if pearson_stats:
        lines.append(
            f"- test Pearson_R: mean=`{pearson_stats.get('mean', '')}` | std=`{pearson_stats.get('std', '')}` | "
            f"min=`{pearson_stats.get('min', '')}` | max=`{pearson_stats.get('max', '')}`"
        )
    rmse_stats = series.get("test_rmse_stats", {})
    if rmse_stats:
        lines.append(
            f"- test RMSE: mean=`{rmse_stats.get('mean', '')}` | std=`{rmse_stats.get('std', '')}` | "
            f"min=`{rmse_stats.get('min', '')}` | max=`{rmse_stats.get('max', '')}`"
        )
    ci_stats = series.get("test_ci_stats", {})
    if ci_stats:
        lines.append(
            f"- test CI: mean=`{ci_stats.get('mean', '')}` | std=`{ci_stats.get('std', '')}` | "
            f"min=`{ci_stats.get('min', '')}` | max=`{ci_stats.get('max', '')}`"
        )

    member_signatures = [member.get("experiment_signature", "") for member in series.get("members", []) if member.get("experiment_signature", "")]
    if member_signatures:
        lines.append(f"- members: `{', '.join(member_signatures)}`")
    lines.extend([
        "",
        "> Rebuilt from grouped run-folder artifacts. This is a factual series view, not an interpretive conclusion.",
        "",
    ])
    return lines


def write_series_journal(series_journal_path: Path, series_records: list[dict]) -> None:
    series_journal_path.parent.mkdir(parents=True, exist_ok=True)
    with series_journal_path.open("w", encoding="utf-8") as f:
        f.write("# Experiment Series Journal\n\n")
        f.write(
            "This file is rebuilt from grouped run folders. "
            "It is a factual series-level view over related experiments.\n\n"
        )
        for series in series_records:
            f.write("\n".join(build_series_journal_entry(series)))
            f.write("\n")


def main() -> None:
    """Run the full factual rebuild over the chosen `runs/` directory."""
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    registry_path = (args.registry_path or (runs_dir / "experiment_registry.csv")).resolve()
    journal_path = (args.journal_path or (runs_dir / "experiment_journal.md")).resolve()
    series_journal_path = (args.series_journal_path or (runs_dir / "experiment_series_journal.md")).resolve()

    if not runs_dir.exists():
        raise SystemExit(f"runs dir does not exist: {runs_dir}")

    rows_with_notes: list[tuple[dict[str, str], list[str], Path]] = []
    seen_signatures: set[str] = set()

    for run_dir in discover_run_dirs(runs_dir):
        row, heuristic_notes = build_registry_row(run_dir)
        signature = row.get("experiment_signature", "")
        if not signature or signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        rows_with_notes.append((row, heuristic_notes, run_dir))

    rows_with_notes.sort(key=lambda item: row_sort_key(item[0]))
    rows = [row for row, _, _ in rows_with_notes]
    series_records = build_series_records(rows_with_notes)

    write_registry(registry_path, rows)
    write_journal(journal_path, rows_with_notes)
    write_series_journal(series_journal_path, series_records)

    print(f"Rebuilt registry: {registry_path}")
    print(f"Rebuilt journal:  {journal_path}")
    print(f"Rebuilt series:   {series_journal_path}")
    print(f"Indexed runs:     {len(rows)}")


if __name__ == "__main__":
    main()
