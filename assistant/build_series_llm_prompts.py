#!/usr/bin/env python3
"""Stage 2s of the manual assistant pipeline: assemble reviewable series prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = """You are a careful research assistant reading a series of related experiment artifacts from a protein-ligand affinity project.

Your task is to write a compact journal note for one experiment series.

Rules:
- Base your note only on the provided artifacts.
- Treat the series as the unit of analysis, not any single run in isolation.
- Do not compare this series with outside series unless such comparison context is explicitly provided.
- Separate observations from hypotheses.
- Use cautious language.
- If data is missing, say so plainly.
- Keep the note structured and concise.
"""


def parse_args() -> argparse.Namespace:
    """Parse CLI options for series prompt assembly."""
    parser = argparse.ArgumentParser(
        description="Build dry-run LLM prompt previews for experiment series."
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=Path("assistant/experiment_series_llm_context.json"),
        help="Series context packet JSON produced by build_series_contexts.py",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("assistant/llm_series_prompt_preview.json"),
        help="Where to write structured series prompt previews.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("assistant/llm_series_prompt_preview.md"),
        help="Where to write a readable markdown preview of the series prompts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of series to include. 0 means all.",
    )
    parser.add_argument(
        "--system-context",
        type=Path,
        default=Path("assistant/system_context.md"),
        help="Stable project-context file.",
    )
    parser.add_argument(
        "--research-stage",
        type=Path,
        default=Path("assistant/research_stage.md"),
        help="Lightweight evolving research-stage memory file.",
    )
    parser.add_argument(
        "--recent-series-journal",
        type=Path,
        default=Path("runs/experiment_series_journal.md"),
        help="Optional factual series journal used as supplemental recent context.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load the stage-1s series context packet file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_text_if_exists(path: Path) -> str:
    """Read a text file if present, otherwise return an empty string."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def normalize_context_text(text: str) -> str:
    """Trim leading blank lines and one top-level heading from context files."""
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].lstrip().startswith("# "):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).strip()


def clip_json_block(obj: object, max_chars: int = 6000) -> str:
    """Serialize JSON for prompt inclusion with a conservative character cap."""
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15] + "\n... [truncated]"


def clip_text(text: str, max_chars: int) -> str:
    """Truncate long plain-text blocks for prompt previews."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15] + "\n... [truncated]"


def build_user_prompt(packet: dict, *, recent_series_journal_text: str = "") -> str:
    """Assemble the per-series user prompt from a series context packet."""
    sections = [
        "# Series Identity",
        clip_json_block({"series_name": packet.get("series_name", "")}, max_chars=600),
        "",
        "# Series Setup",
        clip_json_block(packet.get("setup", {}), max_chars=1800),
        "",
        "# Series Summary",
        clip_json_block(packet.get("series_summary", {}), max_chars=1800),
        "",
        "# Aggregate Metrics",
        clip_json_block(packet.get("aggregate_metrics", {}), max_chars=2200),
        "",
        "# Members",
        clip_json_block(packet.get("members", []), max_chars=7000),
        "",
    ]

    if recent_series_journal_text:
        sections.extend(
            [
                "# Recent Factual Series Journal Context",
                clip_text(recent_series_journal_text, max_chars=5000),
                "",
            ]
        )

    sections.extend(
        [
            "# Requested Output",
            "\n".join(
                [
                    "Write a compact markdown note for this experiment series.",
                    "Use these sections:",
                    "## Series Setup",
                    "## Aggregate Snapshot",
                    "## Across-Run Variation",
                    "## Cautious Notes",
                    "",
                    "Constraints:",
                    "- Keep the note concise.",
                    "- Focus on the series as a grouped object.",
                    "- Do not compare this series with outside series.",
                    "- Do not call the series good, bad, strong, weak, solid, or poor.",
                    "- If something is uncertain, say that directly.",
                    "- Prefer observations over conclusions.",
                ]
            ),
        ]
    )
    return "\n".join(sections).strip() + "\n"


def build_prompt_preview(
    packet: dict,
    *,
    system_context_text: str,
    research_stage_text: str,
    recent_series_journal_text: str,
) -> dict:
    """Build one series prompt preview with layered system and user context."""
    return {
        "series_name": packet.get("series_name", ""),
        "system_prompt": "\n\n".join(
            block.strip()
            for block in (
                DEFAULT_SYSTEM_PROMPT,
                "# Stable Project Context\n" + normalize_context_text(system_context_text) if system_context_text.strip() else "",
                "# Current Research Stage\n" + normalize_context_text(research_stage_text) if research_stage_text.strip() else "",
            )
            if block.strip()
        ),
        "user_prompt": build_user_prompt(packet, recent_series_journal_text=recent_series_journal_text),
    }


def write_json(path: Path, payload: dict) -> None:
    """Write structured series prompt previews to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, previews: list[dict]) -> None:
    """Write human-readable series prompt previews for manual inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LLM Series Prompt Preview",
        "",
        "This file is a dry-run prompt preview built from assistant series context packets.",
        "It is for prompt review only; no model call is performed here.",
        "",
    ]
    for preview in previews:
        lines.extend(
            [
                f"## {preview.get('series_name', '')}",
                "",
                "### System Prompt",
                "",
                "```text",
                preview.get("system_prompt", "").rstrip(),
                "```",
                "",
                "### User Prompt",
                "",
                "```text",
                preview.get("user_prompt", "").rstrip(),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run stage 2s and emit both JSON and markdown series prompt previews."""
    args = parse_args()
    context_path = args.context.resolve()
    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()
    system_context_path = args.system_context.resolve()
    research_stage_path = args.research_stage.resolve()
    recent_series_journal_path = args.recent_series_journal.resolve()

    if not context_path.exists():
        raise SystemExit(f"context file does not exist: {context_path}")

    data = load_json(context_path)
    packets = data.get("series_packets", [])
    if args.limit > 0:
        packets = packets[: args.limit]

    system_context_text = load_text_if_exists(system_context_path)
    research_stage_text = load_text_if_exists(research_stage_path)
    recent_series_journal_text = load_text_if_exists(recent_series_journal_path)

    previews = [
        build_prompt_preview(
            packet,
            system_context_text=system_context_text,
            research_stage_text=research_stage_text,
            recent_series_journal_text=recent_series_journal_text,
        )
        for packet in packets
    ]

    write_json(
        output_json,
        {
            "context_path": str(context_path),
            "system_context_path": str(system_context_path),
            "research_stage_path": str(research_stage_path),
            "recent_series_journal_path": str(recent_series_journal_path) if recent_series_journal_text else "",
            "num_prompts": len(previews),
            "prompts": previews,
        },
    )
    write_markdown(output_md, previews)

    print(f"Wrote series prompt preview JSON: {output_json}")
    print(f"Wrote series prompt preview MD:   {output_md}")
    print(f"Prepared series prompts:          {len(previews)}")


if __name__ == "__main__":
    main()
