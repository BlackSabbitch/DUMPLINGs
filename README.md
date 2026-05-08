# DUMPLINGs

**Different Unsorted Models of Protein-Ligand Interaction using Graphs**

`DUMPLINGs` is a research-oriented sandbox for protein-ligand binding affinity
prediction on **PDBBind 2016**. The current working baseline combines:

- a fused **ligand-pocket 3D interaction graph**,
- a **DimeNet++** geometric encoder,
- an optional frozen **ESM protein-context branch**,
- an optional lightweight **ligand-context branch** based on cached RDKit
  physicochemical descriptors.

The repository was split out of the broader `ELIQSIR-QDL` codebase so the
interaction-graph line of work could evolve quickly without carrying the full
hybrid classical/quantum stack.

## Project Status

The active experiment family is the `A1` line, centered on
[`models/a1.py`](models/a1.py).

The current code supports:

- cached dataset extraction and parsing from the PDBBind 2016 archive,
- fused ligand-pocket graph construction,
- frozen-sequence protein enrichment via ESM,
- cached ligand enrichment via compact RDKit descriptors,
- experiment tracking, checkpointing, early stopping, and ASCII dashboards,
- richer evaluation diagnostics, including validation-scatter agreement
  summaries and PCA-axis overlays.

This project is intentionally optimized for experimentation rather than
production packaging.

## High-Level Pipeline

The default pipeline is:

1. read PDBBind index files,
2. filter complexes by resolution and valid affinity,
3. extract raw molecular files from the archive when needed,
4. parse and cache a reusable dataset under `datasets/`,
5. derive train / validation / test splits,
6. precompute protein-context embeddings if enabled,
7. precompute ligand-context descriptor vectors if enabled,
8. train the model,
9. evaluate the best checkpoint and save plots, metrics, and diagnostics.

## Current Model

The active model class is [`A1DimeNet`](models/a1.py).

It uses:

- `torch_geometric.nn.DimeNetPlusPlus` for geometry-aware message passing,
- atomic numbers from `complex_graph.x[:, 0]`,
- 3D coordinates from `complex_graph.pos`,
- late fusion with optional global protein and ligand context vectors.

### Geometry Branch

The geometric branch consumes the fused `complex_graph` produced by
[`parsers/interaction_graph_parser.py`](parsers/interaction_graph_parser.py).
DimeNet++ reconstructs radial and angular neighborhoods internally from
positions, so the parser intentionally stores a richer graph than DimeNet++
strictly requires. That keeps the dataset reusable for future EGNN or
edge-aware baselines.

### Protein Context Branch

Protein context is implemented in
[`models/protein_context.py`](models/protein_context.py).

The current production-ready mode is:

- `esm_frozen_whole`: frozen full-sequence ESM embedding with cached
  sequence-level vectors.

Cache files are stored under `protein_context_features/`. Each unique
protein-sequence/configuration pair is cached as an individual `.pt` file.

### Ligand Context Branch

Ligand context is implemented in
[`models/ligand_context.py`](models/ligand_context.py).

The current first-wave mode is:

- `basic_rdkit`: cached global RDKit descriptors
  (`MW`, `logP`, `TPSA`, `HBD`, `HBA`, `Lipinski violations`,
  `Wiener index`).

Cache files are stored under `ligand_context_features/`. Each unique
ligand/configuration pair is cached as an individual `.pt` file.

## Data Representation

The default config uses graph-encoder mode `duo` with parser signature `NE`:

- `N` means no separate slot-specific protein encoder is built for the main
  model input,
- `E` means one fused ligand-pocket interaction graph is parsed.

Each dataframe row can contain:

- `pdb_id`: PDBBind complex identifier,
- `pkd`: binding-affinity target,
- `res`: structure resolution,
- `protein`: optional protein sequence or representation,
- `ligand`: optional ligand representation,
- `ligand_smiles`: canonical ligand SMILES when ligand context is enabled,
- `pocket`: optional pocket representation,
- `complex_graph`: fused ligand-pocket graph with node features, coordinates,
  and edges.

For the current A1 baseline, `complex_graph` is the primary geometric input.

## Dataset Flow

Dataset preparation is orchestrated by
[`PDBBindOrchestrator`](extractor.py) in [`extractor.py`](extractor.py).

The intended flow is:

```text
pdbbind_v2016.tar.gz
        |
        v
data/v2016/... raw extracted files
        |
        v
datasets/pdbbds_ref_*.pickle
        |
        +--> train / val from refined minus core
        |
        +--> test_core derived by PDB ID from refined
```

Only the expensive parsed source dataset is cached globally under `datasets/`.
Per-run split snapshots are stored under `runs/<experiment>/datasets/` only if
`save_train_test_val_datasets` is enabled.

## Cache Model

### Dataset Cache

Dataset cache names are derived from a stable signature that includes:

- parser classes and simple parser parameters,
- bad-complex registry contents,
- enabled context modes,
- parser-presence flags relevant to additional cached fields such as
  `protein` and `ligand_smiles`.

This lets the project safely rebuild cached datasets whenever the real parsing
contract changes.

### Protein Context Cache

Protein context cache keys are derived from:

- ESM model name,
- representation layer,
- pooling strategy,
- raw protein sequence.

### Ligand Context Cache

Ligand context cache keys are derived from:

- ligand-context descriptor-set name,
- canonicalized ligand SMILES.

## Train / Validation / Test Strategies

Splitting is handled by [`splitter.py`](splitter.py).

Supported validation strategies:

- `random`
- `scaffold`
- `scaffold_balanced`
- `cold_protein`

Supported test strategies:

- `core_as_test = true`: use PDBBind core as the final test set,
- `core_as_test = false`: randomly carve out a held-out test split from the
  configured source subset.

## Main Files

| File | Purpose |
|---|---|
| `run.py` | End-to-end experiment orchestration: parsing, splitting, context precompute, training, testing. |
| `extractor.py` | PDBBind archive/index handling and parsed dataset caching. |
| `trainer.py` | Training loop, optimizer/scheduler setup, checkpointing, early stopping, memory diagnostics. |
| `evaluator.py` | Loader evaluation, metrics, plots, scatter diagnostics, and report generation. |
| `tokenizer.py` | Converts dataframe rows into PyTorch/PyG inputs. |
| `models/a1.py` | Current DimeNet++ baseline with optional protein and ligand context fusion. |
| `models/protein_context.py` | Frozen ESM sequence encoder and cache utilities. |
| `models/ligand_context.py` | Frozen RDKit ligand descriptor encoder and cache utilities. |
| `parsers/interaction_graph_parser.py` | Fused ligand-pocket graph parser. |
| `parsers/cnn_parser.py` | Protein-sequence and ligand-SMILES extraction helpers. |
| `splitter.py` | Train/validation/test splitting utilities. |
| `logger.py` | Structured logging and ASCII dashboard helpers. |
| `config.json` | Default experiment configuration. |

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install the regular dependencies:

```bash
pip install -r requirements.txt
```

PyTorch Geometric compiled extensions often need manual installation from the
wheel index that matches your installed PyTorch build. Example for a CPU
PyTorch 2.11 environment:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
pip install torch-sparse  -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
```

Adjust the wheel URL to match your actual PyTorch and CUDA/CPU build.

## Data

Place the PDBBind 2016 archive in the repository root:

```text
pdbbind_v2016.tar.gz
```

The pipeline populates:

```text
data/v2016/
datasets/
protein_context_features/
ligand_context_features/
runs/
```

Generated datasets and cache directories are intentionally treated as runtime
artifacts rather than source-controlled assets.

## Running

Run the default experiment:

```bash
./run.sh
```

Use a custom config:

```bash
./run.sh --config config.json
```

Force extraction before rebuilding datasets:

```bash
./run.sh --extract
```

Run from prebuilt train / validation / test dataframes:

```bash
./run.sh --train_path runs/<exp>/datasets/train.pickle \
         --val_path runs/<exp>/datasets/val.pickle \
         --test_path runs/<exp>/datasets/test_core.pickle
```

Choose the test-set strategy:

```bash
./run.sh --core-as-test
./run.sh --no-core-as-test
```

`core_as_test` is configured in `config.json` and may be overridden from the
CLI. When enabled, the model trains on `source_subset - core` and tests on the
PDBBind core subset. When disabled, the configured source subset is split into
train / validation / test according to `test_frac`.

## Experiment Outputs

Each run writes to:

```text
runs/<experiment_name>_<timestamp>/
```

Typical outputs include:

- `config.json`: resolved config snapshot,
- `log.txt`: experiment log,
- `err_log.txt`: traceback if execution fails,
- `best_model.pt`: best checkpoint by the configured primary validation metric,
- `history.json`: train/validation history,
- `test_results.json`: final test metrics,
- `best_validation_scatter_diagnostics.json`: extended agreement diagnostics
  for the best validation predictions,
- `model_performance_report.png`: evaluation plots,
- `datasets/*.pickle`: optional per-run split snapshots.

## Diagnostics and Monitoring

The project includes:

- structured stage-aware logging (`[LEVEL][STAGE] ...`),
- optional memory snapshots during training,
- optional slow-batch warnings,
- ASCII dashboards for train/validation curves,
- validation scatter diagnostics including slope, intercept, bias,
  concordance, orthogonal RMSE to the diagonal, and PCA-axis overlays.

## Notes

- Targets are normalized with train-set statistics before training.
- Validation RMSE is reported in denormalized `pKd` units.
- The trainer still contains some inherited classical/quantum optimizer
  scaffolding, but the current DimeNet++ baseline uses only the classical path
  unless a true quantum branch is introduced.
- On constrained Colab runtimes, `num_workers = 0` is often the most stable
  setting for this project because the dataloader can otherwise pressure host
  RAM and stall batch delivery.

## Roadmap

- Add a lightweight smoke-test config for one batch and one forward/backward pass.
- Expand ligand context with optional ablations such as `SASA` and electronic
  descriptors once the compact baseline is stable.
- Add additional geometric baselines that consume richer parser output.
- Tighten regression-test coverage around dataset cache signatures and runtime
  diagnostics.
- Port stable interaction-graph pieces back into the broader research stack
  once the baseline line is experimentally mature.
