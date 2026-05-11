# Assistant Layer

This directory contains the **manual assistant pipeline** for experiment
analysis. It is intentionally separate from training.

The key rule is simple:

- `run.py`, `run.sh`, and run folders remain the primary factual source,
- `assistant/` reads those artifacts afterwards and builds optional derived
  outputs,
- nothing in the training pipeline depends on `assistant/`.

So this folder is not part of the experiment runtime. It is a **post-hoc
analysis layer**.

## What Lives Here

There are three kinds of files in `assistant/`:

### 1. Hand-authored source files

These define the assistant pipeline itself.

- [build_run_contexts.py](build_run_contexts.py)  
  Stage 1. Scans `runs/` and builds compact per-run JSON packets.

- [build_llm_prompts.py](build_llm_prompts.py)  
  Stage 2. Turns those packets into reviewable prompt previews.

- [build_llm_journal.py](build_llm_journal.py)  
  Stage 3. Builds `runs/experiment_journal_llm.md` from prompt previews and,
  in live mode, from Ollama responses.

- [build_series_contexts.py](build_series_contexts.py)  
  Stage 1s. Groups run-level context packets into series-level packets.

- [build_series_llm_prompts.py](build_series_llm_prompts.py)  
  Stage 2s. Turns those series packets into reviewable series prompt previews.

- [build_series_llm_journal.py](build_series_llm_journal.py)  
  Stage 3s. Builds `runs/experiment_series_journal_llm.md` from series prompt
  previews and, in live mode, from Ollama responses.

- [llm_backend.py](llm_backend.py)  
  Thin local Ollama backend plus cache helpers. It does not know what a run
  is; it only knows how to call the model and cache results.

- [run_llm_journal.sh](run_llm_journal.sh)  
  Convenience entrypoint that runs stages 1 -> 2 -> 3 in order.

- [system_context.md](system_context.md)  
  Stable project context for prompt construction.

- [research_stage.md](research_stage.md)  
  Lightweight evolving memory about the current phase of the project.

- [.env.example](.env.example)  
  Example local config for the Ollama backend.

### 2. Generated intermediate artifacts

These are useful and inspectable, but they are **products of the pipeline**,
not hand-maintained source files.

- [experiment_journal_llm_context.json](experiment_journal_llm_context.json)  
  Output of stage 1. Machine-oriented context packets per run.

- [llm_prompt_preview.json](llm_prompt_preview.json)  
  Output of stage 2. Structured prompt previews.

- [llm_prompt_preview.md](llm_prompt_preview.md)  
  Output of stage 2. Human-readable prompt previews.

- [experiment_series_llm_context.json](experiment_series_llm_context.json)  
  Output of stage 1s. Machine-oriented context packets per series.

- [llm_series_prompt_preview.json](llm_series_prompt_preview.json)  
  Output of stage 2s. Structured prompt previews for experiment series.

- [llm_series_prompt_preview.md](llm_series_prompt_preview.md)  
  Output of stage 2s. Human-readable prompt previews for experiment series.

These files are not mere trash; they are useful for debugging and prompt
design. But they are still **intermediate products**, not canonical inputs.

### 3. Local runtime state

- [cache/](cache/)  
  Cached per-run model responses.

- `.env`  
  Local machine config. Not committed.

## Pipeline Shape

The assistant pipeline currently has a clean, staged flow:

1. `build_run_contexts.py`  
   factual run folders -> compact JSON packets

2. `build_series_contexts.py`  
   run packets -> grouped series packets

3. `build_llm_prompts.py`  
   packets + context files -> prompt previews

4. `build_series_llm_prompts.py`  
   series packets + context files -> series prompt previews

5. `build_llm_journal.py`  
   prompt previews + backend/cache -> `runs/experiment_journal_llm.md`

6. `build_series_llm_journal.py`  
   series prompt previews + backend/cache -> `runs/experiment_series_journal_llm.md`

This is why the code is split into separate scripts rather than one large
module: each file owns one pipeline stage.

## Why It Is Scripts, Not Classes

Right now this layer is mostly:

- file I/O,
- prompt assembly,
- one backend call,
- one final markdown writer.

That kind of work is naturally **stage-oriented**, not object-oriented.

So the current design choice is:

- small modules with narrow responsibility,
- plain functions,
- explicit files passed between stages.

That is not accidental fragmentation. It is a conscious choice for a pipeline
that is still evolving.

If this layer later grows into:

- multiple backends,
- multiple journal formats,
- richer retrieval/memory systems,
- larger cache orchestration,

then it may become worth introducing internal packages or classes. Right now,
that would mostly add ceremony.

## Which Files Are "Real" and Which Are Scaffolding

### Stable and worth keeping

- `build_run_contexts.py`
- `build_llm_prompts.py`
- `build_llm_journal.py`
- `build_series_contexts.py`
- `build_series_llm_prompts.py`
- `build_series_llm_journal.py`
- `llm_backend.py`
- `run_llm_journal.sh`
- `system_context.md`
- `research_stage.md`

These are the real assistant system.

### Generated but useful

- `experiment_journal_llm_context.json`
- `llm_prompt_preview.json`
- `llm_prompt_preview.md`
- `experiment_series_llm_context.json`
- `llm_series_prompt_preview.json`
- `llm_series_prompt_preview.md`
- `cache/*.json`

These are partly "scaffolding", but useful scaffolding:

- context packets let us inspect what the model actually receives,
- prompt previews let us debug prompt design,
- cache keeps live runs from being painfully repetitive.

So I would **not hide them inside each other yet**. They are valuable as
separate inspection points.

## Recommended Mental Model

If you want one simple map:

- `build_run_contexts.py` = extractor
- `build_series_contexts.py` = series grouper
- `build_llm_prompts.py` = run-level prompt packer
- `build_series_llm_prompts.py` = series-level prompt packer
- `llm_backend.py` = model adapter
- `build_llm_journal.py` = run-level final journal writer
- `build_series_llm_journal.py` = series-level final journal writer
- `run_llm_journal.sh` = orchestrator

Everything else is either:

- configuration,
- context memory,
- generated intermediate state,
- or cached output.

## Running the Whole Flow

Dry run:

```bash
bash assistant/run_llm_journal.sh --dry-run
```

Live run with Ollama:

```bash
sudo snap install ollama
ollama serve
ollama pull qwen2.5:1.5b
cp assistant/.env.example assistant/.env
bash assistant/run_llm_journal.sh --live --limit 1
```

Useful checks:

```bash
ollama list
ollama ps
curl http://127.0.0.1:11434/api/tags
```

On smaller laptops, use the exact model tag reported by `ollama list`, and
prefer smaller models such as `qwen2.5:1.5b`.

## Running on a Cluster

For cluster use, the preferred path is not an interactive VPN session but a
dedicated Slurm job:

- [scripts/slurm_assistant_journal.sh](../scripts/slurm_assistant_journal.sh)

Typical example:

```bash
sbatch --export=ALL,\
ASSISTANT_MODE=live,\
ASSISTANT_MODEL=qwen2.5:7b,\
ASSISTANT_LIMIT=1,\
ASSISTANT_FORCE_REFRESH=0,\
ASSISTANT_TIMEOUT_SEC=1800 \
scripts/slurm_assistant_journal.sh
```

Recommended first pass:

1. probe one run and one series with `ASSISTANT_LIMIT=1`,
2. if that looks comfortable, rerun with `ASSISTANT_LIMIT=0`.

Suggested cluster model ladder:

1. `qwen2.5:7b`
2. then, if comfortable, `qwen2.5:14b`
3. if that turns out too heavy or too slow, fall back to `qwen2.5:3b`

This is a much better fit than the laptop profile:

- the assistant can think longer,
- stronger local models become realistic,
- and the whole assistant pass can run unattended after `sbatch`.

## Current Limits

This layer is already operational, but it is still early:

- local small models can be slow,
- live notes can still hallucinate or overinterpret,
- prompt previews and context packets still double as debugging artifacts.

That is fine. The important part is that the pipeline now works end to end.

## Short Version

`assistant/` is not a random pile of functions.

It is a staged post-hoc analysis pipeline with two parallel tracks:

- run-level:
  - extract run context,
  - assemble run prompts,
  - build `experiment_journal_llm.md`
- series-level:
  - group runs into series packets,
  - assemble series prompts,
  - build `experiment_series_journal_llm.md`

The generated files are separate on purpose, because right now they are useful
inspection points rather than clutter.
