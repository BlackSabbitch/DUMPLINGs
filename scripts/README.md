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
- `make_real_series_configs.py`
  - derives a first real-series config suite from the main `config.json`
  - useful when you want one stable config per real experimental condition
- `slurm_pipeline_smoke.sh`
  - sample `sbatch` wrapper that runs the environment smoke test and can
    optionally launch a very short pipeline run once the archive is present
  - defaults to a very cheap smoke profile with `MODEL_FAMILY=A1`
  - can now reproduce the two-step smoke pattern used in Colab:
    1. one bootstrap run with `--extract`
    2. one repeated batch run with `--n-times N`
- `slurm_run_series.sh`
  - sample `sbatch` wrapper for a real repeated experiment series
  - launches `./run.sh --config ... --n-times ... --rseed ...`
  - supports optional `RUN_EXTRACT=1`, which applies extraction only to the
    first run in the repeated series
  - the normal post-bootstrap case is still `RUN_EXTRACT=0`
- `slurm_assistant_journal.sh`
  - sample `sbatch` wrapper for the manual assistant layer
  - starts a local Ollama service inside the job, ensures a chosen model is
    available, and then runs:
    - `assistant/run_llm_journal.sh --live`
  - writes both:
    - `runs/experiment_journal_llm.md`
    - `runs/experiment_series_journal_llm.md`
- `rebuild_experiment_index.py`
  - rescans a `runs/` directory and rewrites:
    - `experiment_registry.csv`
    - `experiment_journal.md`
    - `experiment_series_journal.md`
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

## Recommended Cluster Smoke Flow

For the first real Slurm sanity check, the recommended path is:

1. submit the environment-only smoke first:

```bash
sbatch --export=ALL,RUN_PIPELINE_SMOKE=0 scripts/slurm_pipeline_smoke.sh
```

2. once the environment looks healthy, submit the short pipeline smoke:

```bash
sbatch --export=ALL,\
RUN_PIPELINE_SMOKE=1,\
RUN_BOOTSTRAP_EXTRACT=1,\
BOOTSTRAP_N_TIMES=1,\
REPEAT_N_TIMES=3,\
BASE_RSEED=42,\
SMOKE_EXPERIMENT_NAME=DUMPLING_colab_smoke,\
MODEL_FAMILY=A1,\
PROTEIN_CONTEXT_MODE=none,\
LIGAND_CONTEXT_MODE=none,\
SMOKE_EPOCHS=2,\
SMOKE_BATCH_SIZE=2,\
SMOKE_NUM_WORKERS=0 \
scripts/slurm_pipeline_smoke.sh
```

This reproduces the current Colab smoke shape closely:

- bootstrap extraction run:
  - `./run.sh --config tmp/...json --n-times 1 --rseed 42 --extract`
- repeated short batch:
  - `./run.sh --config tmp/...json --n-times 3 --rseed 42`

### When to use this again

Use the full bootstrap smoke again only when something material changed about
the raw-data side of the workspace, for example:

- a fresh cluster account or fresh filesystem location,
- missing `data/` or `datasets/`,
- a new source archive or a different source subset,
- a parser/cache-signature change that should invalidate old cached datasets.

Otherwise, once extraction and dataset caches already exist, later smoke or
tuning jobs should normally use:

```bash
RUN_BOOTSTRAP_EXTRACT=0
```

That keeps the comparison focused on runtime knobs rather than paying the
one-time extraction cost again.

## Practical Cluster Run Playbook

This is the shortest version of the intended workflow.

### 1. First contact with a new cluster workspace

1. run environment-only smoke,
2. run full bootstrap smoke once,
3. confirm that `runs/`, `data/`, and `datasets/` are now populated.

### 2. Smoke or tuning after caches already exist

Do not rerun extraction. Submit the short repeated batch only:

```bash
sbatch --export=ALL,\
RUN_PIPELINE_SMOKE=1,\
RUN_BOOTSTRAP_EXTRACT=0,\
REPEAT_N_TIMES=3,\
BASE_RSEED=42,\
SMOKE_EXPERIMENT_NAME=DUMPLING_colab_smoke_bs2_nw0,\
MODEL_FAMILY=A1,\
PROTEIN_CONTEXT_MODE=none,\
LIGAND_CONTEXT_MODE=none,\
SMOKE_EPOCHS=2,\
SMOKE_BATCH_SIZE=2,\
SMOKE_NUM_WORKERS=0 \
scripts/slurm_pipeline_smoke.sh
```

### 3. First real experiment series

For a real series, you usually do **not** use `make_smoke_config.py`.
Instead:

1. choose or edit the real config you want to study,
2. give that series a stable `experiment_name`,
3. launch repeated runs with `--n-times` and a base seed.

Typical pattern:

```bash
./run.sh --config config.json --n-times 5 --rseed 42
```

On Slurm, the same principle applies: one config per intended series, many
repeated runs from that config.

If you want a ready-made first suite of real configs, generate it once:

```bash
python3 scripts/make_real_series_configs.py \
  --base-config config.json \
  --output-dir configs/real_series \
  --batch-size 2 \
  --num-workers 0
```

Then launch a real repeated series on Slurm:

```bash
sbatch --export=ALL,\
CONFIG_PATH=configs/real_series/a1_full.json,\
N_TIMES=10,\
BASE_RSEED=42,\
RUN_EXTRACT=0 \
scripts/slurm_run_series.sh
```

### 4. Do you need a new config file for every series?

Usually, no.

Use one config file per meaningful experimental condition. Create a new config
only when you are actually changing something substantive, such as:

- model family,
- protein or ligand context mode,
- loss/optimizer settings,
- data split policy,
- training length.

Do **not** clone a config just because you want more seeds. Repeated runs of
the same condition should usually share the same config and differ only by
`--rseed` / `--n-times`.

### 5. What to rebuild after a series finishes

If new run folders were created, rebuild the factual top-level views:

```bash
python scripts/rebuild_experiment_index.py --runs-dir runs
```

That is the normal post-series maintenance step. You do not need to rebuild
anything before every run.

## Recommended Cluster Assistant Flow

Once you already have a small set of runs on the cluster, the assistant layer
can also be executed as a Slurm job instead of through an interactive VPN
session.

Example:

```bash
sbatch --export=ALL,\
ASSISTANT_MODE=live,\
ASSISTANT_MODEL=qwen2.5:7b,\
ASSISTANT_LIMIT=1,\
ASSISTANT_FORCE_REFRESH=0,\
ASSISTANT_TIMEOUT_SEC=1800 \
scripts/slurm_assistant_journal.sh
```

Recommended workflow:

1. start with `ASSISTANT_LIMIT=1` to probe one run and one series,
2. if runtime and memory look comfortable, rerun with `ASSISTANT_LIMIT=0`.

Suggested model ladder:

1. `qwen2.5:7b`
2. if that is comfortable, try `qwen2.5:14b`
3. if it becomes too slow or memory-heavy, fall back to `qwen2.5:3b`

That job will:

1. start a local Ollama service on the allocated node,
2. pull the selected model if needed,
3. rebuild run-level and series-level assistant context,
4. write both LLM journals back into `runs/`.

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
