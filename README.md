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

## Table of Contents

- [Project Status](#project-status)
- [High-Level Pipeline](#high-level-pipeline)
- [Current Models](#current-models)
  - [Geometry Branch](#geometry-branch)
  - [Protein Context Branch](#protein-context-branch)
  - [Ligand Context Branch](#ligand-context-branch)
- [Data Representation](#data-representation)
- [Dataset Flow](#dataset-flow)
- [Cache Model](#cache-model)
  - [Dataset Cache](#dataset-cache)
  - [Protein Context Cache](#protein-context-cache)
  - [Ligand Context Cache](#ligand-context-cache)
- [Train / Validation / Test Strategies](#train--validation--test-strategies)
- [Main Files](#main-files)
- [Setup](#setup)
- [Data](#data)
- [Evaluation Artifacts](#evaluation-artifacts)
  - [`test_results.json`](#test_resultsjson)
  - [`best_validation_scatter_diagnostics.json`](#best_validation_scatter_diagnosticsjson)
- [Experimental Appendix](#experimental-appendix)
  - [Why the Architecture Was Split into Stages](#why-the-architecture-was-split-into-stages)
  - [A1](#a1)
  - [A2](#a2)
  - [A3](#a3)
  - [Readout Diagnostics in A3](#readout-diagnostics-in-a3)
  - [Immediate Next Candidate: A3a Pair Scoring](#immediate-next-candidate-a3a-pair-scoring)
  - [Beyond Pairs: Motifs and "Interaction Responsibility"](#beyond-pairs-motifs-and-interaction-responsibility)
  - [Attention and Pair Scoring](#attention-and-pair-scoring)
  - [FiLM as a Separate Axis](#film-as-a-separate-axis)
  - [MolE as a Future Ligand-Context Candidate](#mole-as-a-future-ligand-context-candidate)
  - [Quantum Layer Speculation](#quantum-layer-speculation)
- [Running](#running)
- [Experiment Outputs](#experiment-outputs)
- [Diagnostics and Monitoring](#diagnostics-and-monitoring)
- [Notes](#notes)
- [Roadmap](#roadmap)

## Project Status

The active experiment family is now the `A1 -> A2 -> A3` ladder:

- [`A1DimeNet`](models/a1.py): one coarse global geometry branch plus optional
  protein/ligand context
- [`A2DimeNet`](models/a2.py): A1 + an explicit local geometric branch over a
  tighter ligand-pocket interaction zone
- [`A3DimeNet`](models/a3.py): A2 branch encoders + a linear combination of
  branch-level scalar outputs

The current code supports:

- cached dataset extraction and parsing from the PDBBind 2016 archive,
- fused ligand-pocket graph construction,
- frozen-sequence protein enrichment via ESM,
- cached ligand enrichment via compact RDKit descriptors,
- explicit coarse-vs-local architectural experiments,
- experiment tracking, checkpointing, early stopping, and ASCII dashboards,
- richer evaluation diagnostics, including validation-scatter agreement
  summaries and PCA-axis overlays,
- A3-specific readout diagnostics exported into `test_results.json`.

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

## Current Models

The active model family is selected through `model.selected` in `config.json`.
At the moment the repo ships with:

- [`A1DimeNet`](models/a1.py): one global geometry branch plus optional
  protein/ligand context fusion
- [`A2DimeNet`](models/a2.py): A1-style global fusion plus an optional tighter
  local DimeNet++ branch over a ligand-pocket subgraph
- [`A3DimeNet`](models/a3.py): A2 branch encoders plus an explicit linear
  mixture of branch-level scalar outputs

All three models use:

- `torch_geometric.nn.DimeNetPlusPlus` for geometry-aware message passing,
- atomic numbers from `complex_graph.x[:, 0]`,
- 3D coordinates from `complex_graph.pos`,
- late fusion with optional global protein and ligand context vectors.

The main architectural distinction is where the model is allowed to place
expressive power:

- `A1` asks whether one coarse whole-complex representation is already enough,
- `A2` asks whether a dedicated local correction branch helps,
- `A3` asks whether the final prediction can be decomposed into an explicit
  coarse term plus an explicit local correction term.

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

The default config separates graph construction from graph encoding:

- `model.global_graph.selected = "interaction"` builds one fused
  ligand-pocket graph,
- `model.global_encoder.selected = "DimeNet"` encodes that graph with the
  main DimeNet++ branch.

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

For the current A1/A2/A3 family, `complex_graph` is the primary geometric
input.

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
| `models/a1.py` | A1 coarse global branch baseline with optional protein and ligand context fusion. |
| `models/a2.py` | A2 coarse+local concatenation model. |
| `models/a3.py` | A3 branch-wise scalar readout with linear coarse/local mixing. |
| `models/graph_components.py` | Shared encoder config helpers and DimeNet++ backbone construction. |
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

## Evaluation Artifacts

The most important machine-readable evaluation files are:

- `runs/<experiment>/test_results.json`
- `runs/<experiment>/best_validation_scatter_diagnostics.json`

They serve different purposes.

### `test_results.json`

This is the main final-evaluation file. It is produced after:

1. loading the best checkpoint selected on validation,
2. running the held-out test set,
3. denormalizing predictions,
4. saving the final test metrics.

At minimum it contains:

- `RMSE`
- `Pearson_R`
- `CI`

For model families that expose extra diagnostics, the file can also contain
additional blocks. In particular `A3` currently writes a
`readout_diagnostics` block with coarse/local contribution statistics.

### `best_validation_scatter_diagnostics.json`

This file does **not** summarize the held-out test set. It summarizes the
geometry of the **validation scatter plot** at the best validation checkpoint.

It is useful for questions like:

- is the cloud well aligned with the diagonal `y = x`?
- does the model show systematic positive or negative bias?
- is the output range compressed or stretched?

Important fields include:

- `bias`: mean signed residual `y_pred - y_true`
- `mae`, `rmse`: standard pointwise error summaries on the validation scatter
- `pearson_r`: correlation on the validation scatter
- `ccc`: concordance correlation coefficient, which is stricter than Pearson
  because it rewards agreement with the diagonal rather than with an arbitrary
  affine line
- `ols_slope`, `ols_intercept`: ordinary least squares fit of predicted vs true
- `pca_slope`, `pca_intercept`: principal-axis fit of the scatter cloud
- `delta_angle_deg`: angular deviation of the principal axis from the ideal
  diagonal
- `orthogonal_rmse_to_diagonal`: cloud distance to the ideal diagonal measured
  orthogonally rather than vertically

## Experimental Appendix

This section is intentionally more detailed and more speculative than the rest
of the README. It records the current research narrative, the architectural
steps already implemented, and the hypotheses under active discussion.

### Why the Architecture Was Split into Stages

The underlying scientific intuition is that protein-ligand binding is not only
a global shape-matching problem. A model can often make a decent **coarse**
affinity estimate from the whole fused ligand-pocket geometry, but the final
prediction may depend on a more delicate **local correction** driven by a small
interaction zone.

This motivates a staged model family:

1. `A1`: learn the best coarse estimate from the whole fused graph and optional
   context branches
2. `A2`: add a dedicated local geometric branch
3. `A3`: force the final prediction into an explicit coarse-plus-local form

The coarse/local split is inspired less by exact physical formalism than by a
useful approximation mindset: a whole-complex estimate plus a correction term
whose magnitude and structure we can inspect.

### A1

`A1` is the deliberately conservative baseline:

- one global DimeNet++ branch over the fused interaction graph
- optional frozen ESM whole-sequence context
- optional cached RDKit ligand descriptors
- one compact late-fusion MLP head

The main question of `A1` is:

> how far can a single coarse representation already go?

### A2

`A2` introduces an explicit local branch:

- a ligand-centered radius-based local subgraph
- a second DimeNet++ encoder over that tighter zone
- concatenation of coarse global and local representations
- a small final MLP

This step asks:

> does an explicitly modeled local interaction zone provide extra signal?

In practice `A2` also forced the codebase to become more explicit about
coarse-vs-local responsibilities, local graph extraction, and bad-complex
handling. One especially important debugging episode involved pathological local
geometry around `2iw4`, which motivated the current bad-complex registry.

### A3

`A3` keeps the `A2` branch encoders but replaces the final concatenation head
with a branch-wise scalar decomposition:

- `y_global = global_head(h_global)`
- `y_local = local_head(h_local)`
- `y = alpha * y_global + beta * y_local`

The current mixer is intentionally bias-free. The purpose is interpretability:
if the local term is meant to behave like a correction, it is useful to inspect
its effective contribution directly rather than letting a hidden bias absorb
part of that role.

`A3` therefore asks:

> is the local branch really functioning as a correction, or was `A2` only
> exploiting a more expressive final MLP?

### Readout Diagnostics in A3

`A3` exports additional diagnostics into `test_results.json`, including:

- the mixer coefficients (`alpha`, `beta`, and reserved `gamma`)
- statistics of raw branch outputs (`y_global`, `y_local`)
- statistics of effective contributions (`alpha * y_global`,
  `beta * y_local`)
- the mean absolute ratio of local-to-global contribution magnitudes

This is important because the raw mixer weights alone are not enough: if the
branch outputs have different scales, the coefficients by themselves can be
misleading. The contribution statistics are therefore treated as the more
meaningful quantities.

### Immediate Next Candidate: A3a Pair Scoring

If `A3` confirms that the local term is both real and not degenerate, the next
planned direction is a **pair-scoring local correction** step, informally
referred to as `A3a`.

The main hypothesis is not merely that "locality matters", but that the local
correction may be **sparse**:

> a relatively small subset of ligand-pocket interactions may carry a large
> fraction of the correction signal

The first practical version would likely work in two phases:

1. enumerate local ligand-pocket contact pairs inside the current local cutoff
2. score each pair with a small MLP using pairwise interaction features

Possible pair features include:

- ligand atom embedding
- pocket atom embedding
- pair distance or distance basis expansion
- atom-type-derived interaction hints such as donor/acceptor/aromatic/charge

The first version of `A3a` is expected to be **soft**, meaning:

- the current local graph is left intact,
- pair scores are used for weighting and interpretation,
- pruning is postponed until the score distribution is understood.

Only if those scores show meaningful concentration would a later sparse version
be considered.

### Beyond Pairs: Motifs and "Interaction Responsibility"

Although the first practical unit is likely a pair, the long-term hypothesis is
slightly richer. Sometimes one chemically meaningful local signal may not be a
single pair at all, but a small **motif**:

- two pocket atoms "pinching" one ligand atom,
- one charged site organizing two nearby ligand groups,
- a small triangle or claw-like interaction pattern

The current working idea is therefore:

- score pairs first,
- then inspect whether highly scored pairs cluster into motifs,
- only later consider explicit higher-order modeling if the pair-level view
  proves informative.

### Attention and Pair Scoring

The current view is that classical node-level attention is probably *not* the
most scientifically aligned next step for this repository. What matters more is
not just "which atom is important", but:

> which local ligand-pocket interactions are responsible for the correction?

That makes pair scoring or edge-level importance more attractive than generic
node-attention pooling.

Attention-like ideas are still relevant, but in this line they are likely to
appear as:

- pair-weighting mechanisms,
- soft interaction scoring,
- later motif-aware aggregation

rather than as a wholesale replacement of the local geometric encoder.

### FiLM as a Separate Axis

Another future direction is **FiLM-style modulation** of geometric features by
protein context.

This would be a different idea from `A2/A3/A3a`: instead of asking how a local
correction should be extracted, it asks whether frozen sequence context should
do more than simply sit beside the geometry vector. In a FiLM-style design, an
ESM-derived context embedding would generate feature-wise scaling and shifting
parameters that modulate geometric hidden states.

In other words:

- current context fusion: "append context and let the final head use it"
- FiLM-style context fusion: "use context to change how geometry features are
  interpreted upstream"

This direction is considered promising, but orthogonal to the present
coarse/local-correction ladder.

### MolE as a Future Ligand-Context Candidate

The current ligand-context branch intentionally uses a very lightweight and
interpretable RDKit descriptor set. A future line may replace or augment that
with a stronger pretrained molecular representation, such as **MolE**.

The reason this is not the immediate next step is experimental hygiene:
changing the ligand prior at the same time as changing the local-correction
mechanism would make attribution much harder.

### Quantum Layer Speculation

The repository was originally split from a broader classical/quantum sandbox,
so it is natural to ask where a future quantum-inspired or genuinely quantum
module could fit.

The current view is deliberately conservative:

- applying a quantum layer to the whole coarse graph is too diffuse
- applying it to a poorly understood local branch is premature

The first plausible place where it may become meaningful is *after* the local
correction has been localized further:

1. `A3` confirms the local correction is real
2. `A3a` confirms the correction is concentrated in a compact interaction zone
3. only then does it become reasonable to ask whether a more expensive
   specialized module should refine that small responsible region

In that sense, the pair-scoring direction is not only an interpretability
exercise but also a possible precursor to a more focused quantum refinement
module.

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
