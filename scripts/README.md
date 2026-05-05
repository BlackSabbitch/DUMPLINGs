# Scripts

This directory holds operational wrappers around the core experiment runner.

## Why these files live here

The repo currently has two layers:

1. **Core experiment entrypoints**
   - [`run.py`](../run.py)
   - [`run.sh`](../run.sh)

   These are the project-level runners. They define how an experiment is
   executed once the workspace is already prepared.

2. **Environment/orchestration helpers**
   - staging code from Drive into `/content`,
   - installing Colab-specific PyG wheels,
   - syncing `runs/` back to Drive,
   - syncing `esm_cache/` back to Drive.

Those helpers are notebook/runtime concerns rather than experiment logic, so
they live in `scripts/`.

## Mapping from `main.ipynb`

The current Colab notebook already has a working flow. These scripts mirror its
cells instead of inventing a new workflow:

- `colab_stage_workspace.py`
  - mirrors the "copy selected repo files from Drive into /content" cell
- `colab_install_pyg.py`
  - mirrors the "install PyG wheels matching the active torch build" cell
- `colab_start_sync.sh`
  - mirrors the background `rsync` loop for `runs/`
- `colab_finalize_sync.sh`
  - mirrors the final one-shot sync of `runs/` and `esm_cache/`

## Why `run.py` stays in the repo root

`run.py` is not just a helper for Colab. It is the primary experiment runner of
the project and is referenced by:

- `run.sh`,
- README examples,
- direct CLI usage,
- future non-Colab execution paths.

Keeping it in the root makes the project easier to discover and easier to run
outside Colab. The `scripts/` directory is for wrappers around that runner, not
for the runner itself.
