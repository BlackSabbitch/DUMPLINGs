#!/usr/bin/env python3
"""Stage 3s of the manual assistant pipeline: build the series LLM journal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assistant.llm_backend import (  # noqa: E402
    build_prompt_hash,
    call_llm,
    load_cached_response,
    load_llm_config,
    save_cached_response,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for stage 3s series journal generation."""
    parser = argparse.ArgumentParser(
        description="Build experiment_series_journal_llm.md from series prompt previews."
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=Path("assistant/experiment_series_llm_context.json"),
        help="Series context packet JSON produced by build_series_contexts.py",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("assistant/llm_series_prompt_preview.json"),
        help="Series prompt preview JSON produced by build_series_llm_prompts.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/experiment_series_journal_llm.md"),
        help="Where to write the series LLM journal.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of entries. 0 means all.",
    )
    parser.add_argument(
        "--mode",
        choices=("dry-run", "live"),
        default="dry-run",
        help="dry-run keeps placeholder notes; live uses the local Ollama backend with cache.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("assistant/cache"),
        help="Directory for cached LLM responses.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore any cached response and call the backend again in live mode.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load one JSON artifact used by the series journal builder."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_mock_note(packet: dict, prompt_preview: dict) -> list[str]:
    """Build the placeholder note used in series dry-run mode."""
    setup = packet.get("setup", {})
    summary = packet.get("series_summary", {})
    agg = packet.get("aggregate_metrics", {})
    return [
        "- mock series LLM note: this is a dry-run placeholder; no model call was made.",
        f"  - series_name: `{packet.get('series_name', '')}`",
        f"  - model_family: `{setup.get('model_family', '')}`",
        f"  - total_runs: `{summary.get('total_runs', '')}`",
        f"  - seeds: `{', '.join(summary.get('seeds', []))}`",
        f"  - mean_test_Pearson_R: `{agg.get('test_pearson', {}).get('mean', '')}`",
        f"  - prompt_chars: `{len(prompt_preview.get('user_prompt', ''))}`",
    ]


def build_live_note_lines(response_text: str) -> list[str]:
    """Split a model response into markdown lines for series journal insertion."""
    stripped = response_text.strip()
    if not stripped:
        return ["- live series LLM note was empty."]
    return stripped.splitlines()


def resolve_note_lines(
    *,
    mode: str,
    packet: dict,
    prompt_preview: dict,
    cache_dir: Path,
    force_refresh: bool,
) -> tuple[list[str], str]:
    """Resolve one series note either from placeholders, cache, or Ollama."""
    if mode == "dry-run":
        return build_mock_note(packet, prompt_preview), "dry-run placeholder"

    series_name = packet.get("series_name", "")
    cache_key = f"series::{series_name}"
    system_prompt = prompt_preview.get("system_prompt", "")
    user_prompt = prompt_preview.get("user_prompt", "")
    config = load_llm_config()

    prompt_hash = build_prompt_hash(system_prompt, user_prompt, model=config.model)

    cached = None if force_refresh else load_cached_response(
        cache_dir,
        experiment_signature=cache_key,
        prompt_hash=prompt_hash,
    )
    if cached:
        return build_live_note_lines(cached.get("response_text", "")), f"cached response {cached.get('created_at', '')}"

    _, response_text = call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
    save_cached_response(
        cache_dir,
        experiment_signature=cache_key,
        prompt_hash=prompt_hash,
        model=config.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_text=response_text,
    )
    return build_live_note_lines(response_text), f"live ollama:{config.model}"


def build_entry(packet: dict, *, note_lines: list[str], note_source: str, mode: str) -> list[str]:
    """Render one `experiment_series_journal_llm.md` entry."""
    setup = packet.get("setup", {})
    summary = packet.get("series_summary", {})
    agg = packet.get("aggregate_metrics", {})

    lines = [
        f"## {packet.get('series_name', '')} | model=`{setup.get('model_family', '')}` | runs=`{summary.get('total_runs', '')}`",
        "",
        f"- setup: subset=`{setup.get('source_subset', '')}` | splitter=`{setup.get('splitter', '')}` | "
        f"core_as_test=`{setup.get('core_as_test', '')}` | primary_metric=`{setup.get('primary_metric', '')}`",
        f"- outcomes: success=`{summary.get('success_count', 0)}` / total=`{summary.get('total_runs', 0)}` | "
        f"failure=`{summary.get('failure_count', 0)}` | seeds=`{', '.join(summary.get('seeds', [])) or 'n/a'}`",
        f"- window: started=`{summary.get('started_at', '')}` | finished=`{summary.get('finished_at', '')}`",
    ]
    if summary.get("batch_positions"):
        lines.append(f"- batch positions: `{', '.join(summary.get('batch_positions', []))}`")

    for key, label in (("duration_sec", "duration_sec"), ("test_pearson", "test Pearson_R"), ("test_rmse", "test RMSE"), ("test_ci", "test CI")):
        stats = agg.get(key, {})
        if stats:
            lines.append(
                f"- {label}: mean=`{stats.get('mean', '')}` | std=`{stats.get('std', '')}` | "
                f"min=`{stats.get('min', '')}` | max=`{stats.get('max', '')}`"
            )

    member_signatures = [member.get("experiment_signature", "") for member in packet.get("members", []) if member.get("experiment_signature", "")]
    if member_signatures:
        lines.append(f"- members: `{', '.join(member_signatures)}`")

    lines.append("- assistant series note:")
    lines.extend(note_lines)
    lines.extend(["", f"> LLM series journal entry source: {note_source}. Mode={mode}.", ""])
    return lines


def main() -> None:
    """Run stage 3s and write the final series LLM journal artifact."""
    args = parse_args()
    context_path = args.context.resolve()
    prompts_path = args.prompts.resolve()
    output_path = args.output.resolve()
    cache_dir = args.cache_dir.resolve()

    if not context_path.exists():
        raise SystemExit(f"context file does not exist: {context_path}")
    if not prompts_path.exists():
        raise SystemExit(f"series prompt preview file does not exist: {prompts_path}")

    context = load_json(context_path)
    prompts = load_json(prompts_path)

    packets = context.get("series_packets", [])
    previews = prompts.get("prompts", [])
    preview_by_name = {preview.get("series_name", ""): preview for preview in previews}

    if args.limit > 0:
        packets = packets[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Experiment Series Journal LLM\n\n")
        if args.mode == "dry-run":
            f.write(
                "This file is a dry-run scaffold for the future series LLM journal. "
                "Entry structure is real; assistant notes are placeholders because no live model call was made.\n\n"
            )
        else:
            f.write(
                "This file is the manual LLM-backed experiment series journal. "
                "It groups related runs and adds a separate assistant note at the series level.\n\n"
            )
        for packet in packets:
            series_name = packet.get("series_name", "")
            preview = preview_by_name.get(series_name, {"series_name": series_name, "system_prompt": "", "user_prompt": ""})
            note_lines, note_source = resolve_note_lines(
                mode=args.mode,
                packet=packet,
                prompt_preview=preview,
                cache_dir=cache_dir,
                force_refresh=args.force_refresh,
            )
            f.write("\n".join(build_entry(packet, note_lines=note_lines, note_source=note_source, mode=args.mode)))
            f.write("\n")

    print(f"Wrote series LLM journal:  {output_path}")
    print(f"Series entries written:    {len(packets)}")


if __name__ == "__main__":
    main()
