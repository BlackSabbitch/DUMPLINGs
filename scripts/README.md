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
   - syncing `protein_context_features/` back to Drive,
   - syncing `ligand_context_features/` back to Drive,
   - rebuilding portable experiment indices from copied run folders,
   - running smoke checks on local, Colab, or Slurm environments.

Those helpers are notebook/runtime concerns rather than experiment logic, so
they live in `scripts/`.

## Mapping from `colab_launch_main.ipynb`

The current Colab notebook already has a working flow. These scripts mirror its
cells instead of inventing a new workflow:

- `colab_stage_workspace.py`
  - mirrors the "copy selected repo files from Drive into /content" cell
- `colab_install_pyg.py`
  - mirrors the "install PyG wheels matching the active torch build" cell
- `colab_start_sync.sh`
  - mirrors the background `rsync` loop for `runs/`,
    `protein_context_features/`, and `ligand_context_features/`
- `colab_finalize_sync.sh`
  - mirrors the final one-shot sync of `runs/`,
    `protein_context_features/`, and `ligand_context_features/`

## Smoke Helpers

The smoke helpers are intentionally lightweight. They do not replace
`run.py`; they only make it easier to sanity-check a new Slurm environment
before launching a long experiment.

- `runtime_env_smoke.py`
  - checks imports, CUDA visibility, filesystem readiness, and archive presence
- `make_smoke_config.py`
  - derives a tiny smoke-test config from the main `config.json`
  - can override:
    - model family (`A1` / `A2` / `A3`)
    - protein-context mode
    - ligand-context mode
    - epochs / batch size / workers
- `slurm_pipeline_smoke.sh`
  - sample `sbatch` wrapper that runs the environment smoke test and can
    optionally launch a very short pipeline run once the archive is present
  - defaults to a very cheap smoke profile with `MODEL_FAMILY=A1`
- `rebuild_experiment_index.py`
  - rescans a `runs/` directory and rewrites:
    - `experiment_registry.csv`
    - `experiment_journal.md`
  - useful after copying fresh run folders from Colab, another workstation,
    or a cluster scratch directory
  - treats the run folders themselves as the portable source of truth rather
    than trying to merge top-level journals by hand

Example:

```bash
python scripts/rebuild_experiment_index.py --runs-dir runs
```

Typical workflow:

1. copy only new `runs/<experiment_signature>/...` folders from the other
   machine,
2. skip or overwrite the copied top-level `experiment_registry.csv` /
   `experiment_journal.md`,
3. rebuild those two files locally from the imported run folders.

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
