## Current State

- the factual experiment index and factual experiment journal already work;
- the LLM layer is still manual and external to the training pipeline;
- context packets are built from the current `runs/` tree;
- prompt previews are reviewed manually before any real model call is introduced.

## Current Priorities

1. make the prompt material informative without becoming noisy;
2. preserve cautious interpretation style;
3. keep legacy runs parseable enough, without overinvesting in deep archaeology;
4. prepare a future `experiment_journal_llm.md` that mirrors the factual journal structure.

## Current Design Constraints

- `assistant_summary.md` and `experiment_journal.md` should stay factual and non-interpretive;
- LLM reasoning belongs in a separate manual assistant layer;
- project context should be layered:
  - stable project context,
  - evolving research-stage context,
  - per-run artifact context.

## Working Caution

The `runs/` directory may be cleaned or pruned over time, so `runs/experiment_journal.md`
is useful as supplemental stage context but should not be the only long-term memory source.
