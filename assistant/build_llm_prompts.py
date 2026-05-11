#!/usr/bin/env python3
"""Stage 2 of the manual assistant pipeline: assemble reviewable prompts.

This module reads the JSON packets produced by `build_run_contexts.py` and
combines them with stable context, lightweight stage memory, and recent factual
journal context. The result is a per-run prompt preview for human review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = """You are a careful research assistant reading experiment artifacts from a protein-ligand affinity project.

Your task is to write a compact journal note for one run.

Rules:
- Base your note only on the provided artifacts.
- Do not claim a run is good or bad in absolute terms.
- Do not compare runs unless comparison data is explicitly provided.
- Separate observations from hypotheses.
- Use cautious language.
- If data is missing, say so plainly.
- Keep the note structured and concise.
"""


def parse_args() -> argparse.Namespace:
    """Parse CLI options for stage 2 prompt assembly."""
    parser = argparse.ArgumentParser(
        description="Build dry-run LLM prompt previews from assistant context packets."
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=Path("assistant/experiment_journal_llm_context.json"),
        help="Path to the context packet JSON produced by build_run_contexts.py",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("assistant/llm_prompt_preview.json"),
        help="Where to write structured prompt previews.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("assistant/llm_prompt_preview.md"),
        help="Where to write a readable markdown preview of the prompts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of runs to include. 0 means all.",
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
        "--recent-journal",
        type=Path,
        default=Path("runs/experiment_journal.md"),
        help="Optional factual journal used as supplemental recent-stage context.",
    )
    return parser.parse_args()


def load_context(path: Path) -> dict:
    """Load the stage-1 context packet file."""
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


def clip_json_block(obj: object, max_chars: int = 5000) -> str:
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


def build_user_prompt(packet: dict, *, recent_journal_text: str = "") -> str:
    """Assemble the per-run user prompt from a stage-1 context packet."""
    sections = [
        "# Run Identity",
        clip_json_block(
            {
                "experiment_signature": packet.get("experiment_signature", ""),
                "run_dir": packet.get("run_dir", ""),
            },
            max_chars=800,
        ),
        "",
        "# Registry Snapshot",
        clip_json_block(packet.get("registry_snapshot", {}), max_chars=2200),
        "",
        "# Setup",
        clip_json_block(packet.get("setup", {}), max_chars=1800),
        "",
        "# Config Excerpt",
        clip_json_block(packet.get("config_excerpt", {}), max_chars=2800),
        "",
        "# Metrics",
        clip_json_block(packet.get("metrics", {}), max_chars=1200),
        "",
        "# History Summary",
        clip_json_block(packet.get("history_summary", {}), max_chars=2400),
        "",
        "# History Sampled Checkpoints",
        clip_json_block(packet.get("history_sampled_points", {}), max_chars=3200),
        "",
        "# Test Results",
        clip_json_block(packet.get("test_results", {}), max_chars=2400),
        "",
        "# Scatter Diagnostics",
        clip_json_block(packet.get("scatter_diagnostics", {}), max_chars=2400),
        "",
        "# Log Header Excerpt",
        clip_json_block(packet.get("log_header_excerpt", []), max_chars=2000),
        "",
        "# Log Excerpt",
        clip_json_block(packet.get("log_excerpt", []), max_chars=3000),
        "",
        "# Assistant Summary Excerpt",
        clip_json_block(packet.get("assistant_summary_excerpt", []), max_chars=2000),
        "",
    ]

    if recent_journal_text:
        sections.extend(
            [
                "# Recent Factual Journal Context",
                clip_text(recent_journal_text, max_chars=5000),
                "",
            ]
        )

    sections.extend(
        [
        "# Artifact Paths",
        clip_json_block(packet.get("artifact_paths", {}), max_chars=1800),
        "",
        "# Requested Output",
        "\n".join(
            [
                "Write a compact markdown note for this run.",
                "Use these sections:",
                "## Setup",
                "## Observed Metrics",
                "## Training Trace",
                "## Cautious Notes",
                "",
                "Constraints:",
                "- Keep the note concise.",
                "- Do not call the run good, bad, strong, weak, solid, or poor.",
                "- Do not compare with other runs.",
                "- If something is uncertain, say that directly.",
                "- Prefer observations over conclusions.",
            ]
        ),
    ])
    return "\n".join(sections).strip() + "\n"


def build_prompt_preview(
    packet: dict,
    *,
    system_context_text: str,
    research_stage_text: str,
    recent_journal_text: str,
) -> dict:
    """Build one prompt preview with layered system and user context."""
    return {
        "experiment_signature": packet.get("experiment_signature", ""),
        "system_prompt": "\n\n".join(
            block.strip()
            for block in (
                DEFAULT_SYSTEM_PROMPT,
                "# Stable Project Context\n" + normalize_context_text(system_context_text) if system_context_text.strip() else "",
                "# Current Research Stage\n" + normalize_context_text(research_stage_text) if research_stage_text.strip() else "",
            )
            if block.strip()
        ),
        "user_prompt": build_user_prompt(packet, recent_journal_text=recent_journal_text),
    }


def write_json(path: Path, payload: dict) -> None:
    """Write structured prompt previews to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, previews: list[dict]) -> None:
    """Write human-readable prompt previews for manual inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LLM Prompt Preview",
        "",
        "This file is a dry-run prompt preview built from assistant context packets.",
        "It is for prompt review only; no model call is performed here.",
        "",
    ]
    for preview in previews:
        lines.extend(
            [
                f"## {preview.get('experiment_signature', '')}",
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
    """Run stage 2 and emit both JSON and markdown prompt previews."""
    args = parse_args()
    context_path = args.context.resolve()
    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()
    system_context_path = args.system_context.resolve()
    research_stage_path = args.research_stage.resolve()
    recent_journal_path = args.recent_journal.resolve()

    if not context_path.exists():
        raise SystemExit(f"context file does not exist: {context_path}")

    data = load_context(context_path)
    packets = data.get("packets", [])
    if args.limit > 0:
        packets = packets[: args.limit]

    system_context_text = load_text_if_exists(system_context_path)
    research_stage_text = load_text_if_exists(research_stage_path)
    recent_journal_text = load_text_if_exists(recent_journal_path)

    previews = [
        build_prompt_preview(
            packet,
            system_context_text=system_context_text,
            research_stage_text=research_stage_text,
            recent_journal_text=recent_journal_text,
        )
        for packet in packets
    ]

    write_json(
        output_json,
        {
            "context_path": str(context_path),
            "system_context_path": str(system_context_path),
            "research_stage_path": str(research_stage_path),
            "recent_journal_path": str(recent_journal_path) if recent_journal_text else "",
            "num_prompts": len(previews),
            "prompts": previews,
        },
    )
    write_markdown(output_md, previews)

    print(f"Wrote prompt preview JSON: {output_json}")
    print(f"Wrote prompt preview MD:   {output_md}")
    print(f"Prepared prompts:          {len(previews)}")


if __name__ == "__main__":
    main()
