# DUMPLINGs

**Different Unsorted Models of Protein-Ligand Interaction using Graphs**

`DUMPLINGs` is a compact research sandbox for protein-ligand binding affinity
prediction on PDBBind 2016. The current baseline builds a fused ligand-pocket
3D interaction representation and trains a DimeNet++ model to predict `pKd`.

This repository was split out of the broader `ELIQSIR-QDL` framework so the
3D interaction-graph baseline can be developed, debugged, and evaluated without
the extra weight of the full hybrid classical/quantum pipeline.

## Project Overview

The current pipeline:

1. reads PDBBind index files,
2. filters complexes by resolution and valid affinity,
3. parses molecular files into reusable representations,
4. builds a cached `refined` dataset,
5. derives the PDBBind `core` test set from `refined`,
6. splits non-core refined complexes into train/validation,
7. trains a DimeNet++ baseline,
8. evaluates on the core set.

The project is research-oriented. It favors clear experiment tracking and
reusable molecular parsing over production packaging.

## Current Model

The active model is `DumplingA1` in `models/baseline.py`.

It uses:

- `torch_geometric.nn.DimeNetPlusPlus`,
- atomic numbers from `complex_graph.x[:, 0]`,
- 3D coordinates from `complex_graph.pos`,
- PyG batch assignments from the data loader.

DimeNet++ constructs its own geometric neighborhoods internally from positions
and cutoff radius. The parser still stores `edge_index` and extra node features
because the parsed dataset is intentionally model-agnostic: future GNN, EGNN,
or message-passing baselines can consume richer features without rebuilding the
raw molecular dataset.

## Data Representation

The default config uses `duo` mode with `NE`:

- `N` means no separate protein encoder,
- `E` means an interaction parser builds one fused ligand-pocket graph.

Each row can contain:

- `pdb_id` - PDBBind complex identifier,
- `pkd` - binding affinity target,
- `res` - structure resolution,
- `protein`, `ligand`, `pocket` - optional slot representations,
- `complex_graph` - fused ligand-pocket graph with atom features, coordinates,
  and edges.

For the DimeNet++ baseline, `complex_graph` is the main input.

## Dataset Flow

`PDBBindOrchestrator` in `extractor.py` handles extraction, parsing, caching,
and metadata.

The intended flow is:

```text
pdbbind_v2016.tar.gz
        |
        v
data/v2016/... raw molecular files
        |
        v
datasets/pdbbds_ref_*.pickle
        |
        +--> train / val from refined minus core
        |
        +--> test_core derived by PDB ID from refined
```

Only the expensive parsed `refined` dataset is cached globally under
`datasets/`. The core set is a deterministic slice of refined and is saved per
experiment under `runs/<experiment>/datasets/test_core.pickle` when
`save_train_test_val_datasets` is enabled.

## Cache Safety

Dataset cache names are based on a stable parser signature:

- parser class name,
- simple parser parameters such as `dist_threshold`, `ca_only`, `is_ligand`.

Runtime objects such as Biopython parser instances are excluded from the hash,
so repeated runs reuse the same cache file when the real parser configuration
has not changed.

Dataset writes are atomic: the dataframe is first written to a temporary file
and then moved into place. If a process is killed during serialization, the
official cache file is not replaced by a partial pickle. Cached datasets are
also validated on load; unreadable cache files are rebuilt.

## Main Files

| File | Purpose |
|---|---|
| `run.py` | Full experiment orchestration: parsing, splitting, training, testing. |
| `extractor.py` | PDBBind archive/index handling and parsed dataset caching. |
| `parsers/interaction_graph_parser.py` | Fused ligand-pocket graph parser. |
| `tokenizer.py` | Converts dataframe rows into PyTorch/PyG objects. |
| `models/baseline.py` | Current DimeNet++ baseline model. |
| `trainer.py` | Training loop, early stopping, checkpointing. |
| `evaluator.py` | RMSE, Pearson R, concordance index, plots. |
| `splitter.py` | Random, scaffold, scaffold-balanced, and cold-protein splits. |
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

PyTorch Geometric compiled extensions often need to be installed manually from
the wheel index that matches the installed PyTorch and CPU/CUDA build. Example
for a CPU PyTorch 2.11 environment:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
pip install torch-sparse  -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
```

Adjust the wheel URL for your actual `torch` version and CUDA/CPU build.

## Data

Place the PDBBind archive in the repository root:

```text
pdbbind_v2016.tar.gz
```

The extractor expects the PDBBind 2016 archive layout and will populate:

```text
data/v2016/
datasets/
runs/
```

The archive and generated datasets are intentionally not source-code assets.

## Running

Run the default experiment:

```bash
./run.sh
```

Use a custom config:

```bash
./run.sh --config config.json
```

Force extraction before building datasets:

```bash
./run.sh --extract
```

Run with prebuilt train/test/validation dataframes:

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

`core_as_test` is configured in `config.json` and can be overridden from the
CLI. When enabled, the PDBBind core subset is used as the final test set and
the model trains/validates on `source_subset - core`. When disabled, the
configured `source_subset` is randomly split into train/validation/test.
`test_frac` and validation fractions live in `config.json`, not in shell flags.

`run.sh` uses `pipefail`, so Python failures are not hidden by `tee`.

## Experiment Outputs

Each run writes to:

```text
runs/<experiment_name>_<timestamp>/
```

Typical outputs:

- `config.json` - resolved config with dataset paths and normalization stats,
- `log.txt` - experiment log,
- `err_log.txt` - traceback if Python raises an exception,
- `best_model.pt` - best checkpoint by primary validation metric,
- `history.json` - training and validation history,
- `test_results.json` - final test metrics,
- `model_performance_report.png` - plots,
- `datasets/*.pickle` - optional per-run train/val/test snapshots.

## Notes

- By default, `core` is treated as the final test set.
- `source_subset - core` is split into train and validation.
- Alternatively, `--no-core-as-test` splits the configured source subset into
  train/validation/test using `dataset.test_frac`.
- Targets are normalized using train-set statistics before training.
- Validation RMSE is reported in denormalized `pKd` units.
- The trainer still contains some hybrid/quantum scaffolding inherited from
  `ELIQSIR-QDL`; it is inert for the current DimeNet++ baseline.

## Roadmap

- Add a smoke test for one `DataLoader` batch and one forward/backward pass.
- Add a small config for fast CPU debugging.
- Add denormalized final test RMSE for consistency with validation reporting.
- Add an EGNN or edge-aware GNN baseline that consumes the full parser output.
- Port stable DimeNet/interaction-graph pieces back into `ELIQSIR-QDL` once the
  baseline is experimentally useful.
