#!/usr/bin/env python3
"""Stage 3 of the manual assistant pipeline: build the LLM journal artifact.

This module takes prompt previews plus context packets and writes
`runs/experiment_journal_llm.md`. In dry-run mode it uses placeholder notes; in
live mode it queries the local Ollama backend and caches the response per run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assistant.llm_backend import (
    build_prompt_hash,
    call_llm,
    load_cached_response,
    load_llm_config,
    save_cached_response,
)
from scripts.rebuild_experiment_index import build_artifact_links, build_report_preview


def parse_args() -> argparse.Namespace:
    """Parse CLI options for stage 3 journal generation."""
    parser = argparse.ArgumentParser(
        description="Build a dry-run experiment_journal_llm.md from prompt previews without calling an LLM."
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=Path("assistant/experiment_journal_llm_context.json"),
        help="Context packet JSON produced by build_run_contexts.py",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("assistant/llm_prompt_preview.json"),
        help="Prompt preview JSON produced by build_llm_prompts.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/experiment_journal_llm.md"),
        help="Where to write the dry-run LLM journal.",
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
        help="dry-run keeps placeholder notes; live uses a real LLM backend with cache.",
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
    """Load one JSON artifact used by the journal builder."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_mock_note(packet: dict, prompt_preview: dict) -> list[str]:
    """Build the placeholder note used in dry-run mode."""
    setup = packet.get("setup", {})
    metrics = packet.get("metrics", {})
    lines = [
        "- mock LLM note: this is a dry-run placeholder; no model call was made.",
        f"  - run_signature: `{packet.get('experiment_signature', '')}`",
        f"  - model_family: `{setup.get('model_family', '')}`",
        f"  - protein_context: `{setup.get('protein_context', '')}`",
        f"  - ligand_context: `{setup.get('ligand_context', '')}`",
        f"  - test_Pearson_R: `{metrics.get('test_pearson', '')}`",
        f"  - prompt_chars: `{len(prompt_preview.get('user_prompt', ''))}`",
    ]
    return lines


def build_live_note_lines(response_text: str) -> list[str]:
    """Split a model response into markdown lines for journal insertion."""
    stripped = response_text.strip()
    if not stripped:
        return ["- live LLM note was empty."]
    return stripped.splitlines()


def resolve_note_lines(
    *,
    mode: str,
    packet: dict,
    prompt_preview: dict,
    cache_dir: Path,
    force_refresh: bool,
) -> tuple[list[str], str]:
    """Resolve one note either from dry-run placeholders, cache, or Ollama."""
    if mode == "dry-run":
        return build_mock_note(packet, prompt_preview), "dry-run placeholder"

    signature = packet.get("experiment_signature", "")
    system_prompt = prompt_preview.get("system_prompt", "")
    user_prompt = prompt_preview.get("user_prompt", "")
    config = load_llm_config()

    prompt_hash = build_prompt_hash(
        system_prompt,
        user_prompt,
        model=config.model,
    )

    cached = None if force_refresh else load_cached_response(
        cache_dir,
        experiment_signature=signature,
        prompt_hash=prompt_hash,
    )
    if cached:
        return build_live_note_lines(cached.get("response_text", "")), f"cached response {cached.get('created_at', '')}"

    _, response_text = call_llm(system_prompt=system_prompt, user_prompt=user_prompt)

    save_cached_response(
        cache_dir,
        experiment_signature=signature,
        prompt_hash=prompt_hash,
        model=config.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_text=response_text,
    )
    return build_live_note_lines(response_text), f"live ollama:{config.model}"


def build_entry(
    packet: dict,
    prompt_preview: dict,
    journal_dir: Path,
    *,
    note_lines: list[str],
    note_source: str,
    mode: str,
) -> list[str]:
    """Render one `experiment_journal_llm.md` entry."""
    row = packet.get("registry_snapshot", {})
    run_dir = Path(packet.get("run_dir", ""))

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

    if run_dir.exists():
        artifact_links = build_artifact_links(run_dir, journal_dir)
        if artifact_links:
            lines.append(f"- artifacts: {artifact_links}")
        lines.extend(build_report_preview(run_dir, journal_dir))

    lines.append("- assistant note:")
    lines.extend(note_lines)
    lines.extend([
        "",
        f"> LLM journal entry source: {note_source}. Mode={mode}.",
        "",
    ])
    return lines


def main() -> None:
    """Run stage 3 and write the final LLM-backed journal artifact."""
    args = parse_args()
    context_path = args.context.resolve()
    prompts_path = args.prompts.resolve()
    output_path = args.output.resolve()
    cache_dir = args.cache_dir.resolve()

    if not context_path.exists():
        raise SystemExit(f"context file does not exist: {context_path}")
    if not prompts_path.exists():
        raise SystemExit(f"prompt preview file does not exist: {prompts_path}")

    context = load_json(context_path)
    prompts = load_json(prompts_path)

    packets = context.get("packets", [])
    previews = prompts.get("prompts", [])
    preview_by_signature = {
        preview.get("experiment_signature", ""): preview
        for preview in previews
    }

    if args.limit > 0:
        packets = packets[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Experiment Journal LLM\n\n")
        if args.mode == "dry-run":
            f.write(
                "This file is a dry-run scaffold for the future LLM journal. "
                "Entry structure is real; assistant notes are placeholders because no live model call was made.\n\n"
            )
        else:
            f.write(
                "This file is the manual LLM-backed experiment journal. "
                "Entry structure mirrors the factual journal, but assistant notes are produced by the separate assistant layer.\n\n"
            )
        for packet in packets:
            signature = packet.get("experiment_signature", "")
            preview = preview_by_signature.get(
                signature,
                {"experiment_signature": signature, "system_prompt": "", "user_prompt": ""},
            )
            note_lines, note_source = resolve_note_lines(
                mode=args.mode,
                packet=packet,
                prompt_preview=preview,
                cache_dir=cache_dir,
                force_refresh=args.force_refresh,
            )
            f.write(
                "\n".join(
                    build_entry(
                        packet,
                        preview,
                        output_path.parent,
                        note_lines=note_lines,
                        note_source=note_source,
                        mode=args.mode,
                    )
                )
            )
            f.write("\n")

    print(f"Wrote LLM journal:         {output_path}")
    print(f"Entries written:          {len(packets)}")


if __name__ == "__main__":
    main()
