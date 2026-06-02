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
- [Experiment Journal](#experiment-journal)
  - [Stage 0: Fused-Graph Baseline Setup](#stage-0-fused-graph-baseline-setup)
  - [Stage 1: A1 as the Coarse Baseline](#stage-1-a1-as-the-coarse-baseline)
  - [Stage 2: A2 and the Explicit Local Branch](#stage-2-a2-and-the-explicit-local-branch)
  - [Stage 3: Data Quality and the 2iw4 Lesson](#stage-3-data-quality-and-the-2iw4-lesson)
  - [Stage 4: Colab Workflow Hardening and Experiment Operations](#stage-4-colab-workflow-hardening-and-experiment-operations)
  - [Stage 5: A3 and the Linear Coarse-plus-Local Test](#stage-5-a3-and-the-linear-coarse-plus-local-test)
  - [Stage 6: What A3 Actually Taught Us](#stage-6-what-a3-actually-taught-us)
  - [Current Working Interpretation](#current-working-interpretation)
  - [Immediate Next Questions](#immediate-next-questions)
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

## Experiment Journal

This section is a working lab-book summary for the current line of
experiments. It is intentionally narrative. The goal is to preserve not only
what the code does, but what we learned while moving from one architectural
step to the next.

### Stage 0: Fused-Graph Baseline Setup

The first major structural decision of the current repository line was to move
away from multi-object pipelines and toward one **fused ligand-pocket graph**
as the main geometric object.

That choice had several motivations:

- it keeps the geometric input contract simple,
- it allows one DimeNet++ branch to see both ligand atoms and nearby pocket
  atoms together,
- it keeps the cached dataset reusable for later graph models,
- it reduces architectural noise when asking whether context branches or local
  corrections help.

At the same time, sequence and ligand metadata were retained as optional side
channels:

- protein context via frozen ESM sequence embeddings,
- ligand context via cached RDKit descriptor vectors.

So even before the A1/A2/A3 ladder was formalized, the project already had one
important separation of concerns:

- geometry comes from the fused graph,
- context comes from optional cached side branches.

### Stage 1: A1 as the Coarse Baseline

`A1` was the deliberate conservative baseline:

- one global DimeNet++ encoder over the fused interaction graph,
- optional protein context,
- optional ligand context,
- a small late-fusion prediction head.

The point of `A1` was not to be "the final model", but to answer a clean
question:

> how much affinity signal can already be extracted from one whole-complex
> geometric representation, possibly enriched by frozen side context?

This stage was important because it created the coarse reference point for
later work. Without that anchor, every later architectural change would be much
harder to interpret.

### Stage 2: A2 and the Explicit Local Branch

`A2` introduced the first major architectural hypothesis:

> perhaps the whole-complex estimate is not enough, and a dedicated local
> ligand-pocket correction branch helps.

This led to:

- a ligand-centered local subgraph,
- a second DimeNet++ encoder over that tighter zone,
- concatenation of global and local representations before the final MLP.

This was the first place where the repository became explicitly about
**coarse-plus-local structure**, even though the final readout was still a
generic nonlinear head.

The most important outcome of this stage was empirical:

- the local branch appeared useful,
- the model remained trainable and stable,
- the project now had a strong reason to take "local correction" seriously.

### Stage 3: Data Quality and the `2iw4` Lesson

One of the most practically important lessons of the A2 stage was that not all
instability is architectural. Some of it is simply data quality.

In particular, pathological behavior around `2iw4` pushed the project toward a
more explicit data-quality workflow:

- temporary numerical safeguards inside the local branch,
- a persistent `bad_complexes.toml` registry,
- explicit logging of excluded complexes and local-guard activations.

This stage mattered because it separated two very different failure modes:

- "the model idea is bad",
- "the dataset contains a toxic example for this parser/encoder combination".

The exclusion of `2iw4` and related cleanup substantially improved confidence
that later model comparisons were about architecture rather than hidden data
corruption.

The registry eventually stabilized around the current explicitly excluded
complexes, including `2iw4` and `4bps`. In the project narrative, `2iw4` is
remembered as the most diagnostic case, but the real lesson is broader: the
pipeline needed a formal mechanism for known-bad complexes rather than
one-off local hacks.

### Stage 4: Colab Workflow Hardening and Experiment Operations

In parallel with model work, the project also became much more operationally
structured.

Originally, a larger fraction of the workflow lived directly inside notebook
cells. Over time, repeated friction around Colab runs pushed the repository
toward a clearer separation between:

- experiment logic in `run.py`,
- notebook control flow in `colab_launch_main.ipynb`,
- reusable environment helpers under `scripts/`.

This produced a small but important tooling layer for:

- staging the workspace from Drive into `/content`,
- installing Colab-specific dependencies cleanly,
- background syncing of `runs/`,
- final syncing of `runs/`, `protein_context_features/`, and
  `ligand_context_features/`,
- lightweight smoke checks for non-Colab environments.

This part of the evolution was not scientifically glamorous, but it improved
the real experimental loop:

- fewer manual notebook edits,
- better cache reuse,
- cleaner reruns,
- less risk of losing artifacts at the end of long jobs.

### Stage 5: A3 and the Linear Coarse-plus-Local Test

Once `A2` suggested that the local branch mattered, the next question became
sharper:

> is the local branch really acting like a correction, or is A2 simply winning
> because the final concat-head is more expressive?

That question motivated `A3`.

`A3` keeps the same branch encoders as `A2`, but changes the readout:

- `y_global = global_head(h_global)`
- `y_local = local_head(h_local)`
- `y = alpha * y_global + beta * y_local`

This was intentionally more restrictive than `A2`.

The intention was not merely to chase performance, but to ask whether the model
could sustain an interpretable decomposition:

- one branch for a coarse estimate,
- one branch for a local correction,
- one linear mixer at the end.

To support that reading, `A3` also introduced explicit readout diagnostics into
`test_results.json`.

### Stage 6: What A3 Actually Taught Us

The first serious `A3` run produced a result that was highly informative even
though it did not beat `A2`.

Best validation checkpoint:

- epoch `19`
- `val RMSE = 1.3776`
- `val Pearson R = 0.7497`
- `val CI = 0.7752`

Final held-out test:

- `RMSE = 1.5104`
- `Pearson R = 0.7208`
- `CI = 0.7595`

Those numbers were respectable enough to show that `A3` is a real model rather
than a broken toy. But the readout diagnostics told the more important story.

The key observations were:

- `y_global` had large sample-dependent variation,
- `y_local` was almost constant across the test set,
- the effective local contribution was tiny compared with the global one.

In the recorded diagnostics:

- `global_branch_output.mean_abs ~= 6.50`
- `local_branch_output.mean_abs ~= 0.0837`
- `global_contribution.mean_abs ~= 0.619`
- `local_contribution.mean_abs ~= 0.0119`
- mean local/global absolute contribution ratio `~= 0.019`

This means that the local branch did **not** survive as a meaningful
sample-specific correction term in this formulation.

Instead, it nearly collapsed into a small constant offset.

### Current Working Interpretation

The first A3 result suggests a very specific reading:

1. local information may still be useful, because `A2` previously benefited
   from it,
2. but the current **scalar local readout + bias-free linear mixer** is too
   restrictive,
3. in practice the local branch almost behaved like a surrogate bias term.

This is a subtle but important distinction.

The result does **not** cleanly say:

> local information is useless

It says something closer to:

> this particular attempt to force locality into one scalar correction channel
> caused the local branch to collapse

That is why the first A3 run is treated as a useful negative result rather than
as a dead end.

A later rerun, initially intended to be an `A3 + bias` experiment, exposed a
separate operational lesson. The Colab launcher silently continued after a
failed `git pull`, so the staged workspace still contained the older bias-free
`A3` code. The final `test_results.json` therefore reported:

- `mixer_has_bias = false`
- `gamma = null`

So scientifically that run was **not** the intended `A3 + bias` test.

Still, the result was informative. The second no-bias A3 run achieved:

- `RMSE = 1.4945`
- `Pearson R = 0.7790`
- `CI = 0.7862`

and, unlike the first no-bias run, its local branch remained meaningfully
alive:

- `global_contribution.mean_abs ~= 0.902`
- `local_contribution.mean_abs ~= 0.347`
- aggregate local/global absolute contribution ratio `~= 0.385`

This sharpened the interpretation considerably. The first A3 collapse was not a
universal fate of the architecture. Instead, the project now has evidence that
the bias-free scalar decomposition can land in at least two regimes:

- one where the local branch nearly collapses,
- one where the local branch remains active and materially contributes.

That makes A3 look less "fundamentally broken" and more **trajectory-sensitive
or seed-sensitive**, while also emphasizing that experiment orchestration must
fail loudly when code sync goes wrong.

The dedicated `A3 + bias` series eventually clarified the picture further. The
important outcome was not a dramatic headline gain over the strongest `A1`
family baselines, but a much cleaner readout-level diagnosis.

Across the completed `A3 + bias` runs:

- the global branch remained the dominant source of predictive signal,
- the local branch contribution was highly variable across seeds,
- the mixer coefficients (`alpha`, `beta`, `gamma`) did not converge to one
  stable regime,
- and the most revealing trend was that **larger local relative contribution
  correlated with worse held-out performance**.

In the run-level diagnostics for the completed `A3 + bias` series:

- mean `global_contribution.mean_abs ~= 0.840`
- mean `local_contribution.mean_abs ~= 0.353`
- median `local_to_global_abs_contribution_ratio ~= 0.248`
- mean `local_to_global_abs_contribution_ratio ~= 0.443`

More importantly, the cross-run correlations were strongly directional:

- `local_contribution.mean_abs` vs `test_pearson` `~ -0.875`
- `local_contribution.mean_abs` vs `test_rmse` `~ +0.893`
- `local/global abs contribution ratio` vs `test_pearson` `~ -0.752`
- `local/global abs contribution ratio` vs `test_rmse` `~ +0.795`

So the best `A3 + bias` runs were not the ones where the local branch took
over. They were the ones where the local branch stayed present but relatively
small, behaving more like a modest correction than like a co-equal expert.

This is a strong update to the earlier interpretation. The question is no
longer simply:

> does a bias term rescue A3?

The fuller answer now looks closer to:

> a bias term can keep the scalar decomposition trainable, but the current
> local branch still does not look trustworthy enough to dominate the final
> prediction.

The next experiment family tested whether the problem was simply that the local
branch was too geometry-only. A lightweight `ext` variant was added for `A2`
and `A3`, enriching the local branch with optional chemistry-aware features
derived during graph construction:

- atom-level charge / aromaticity / hybridization style descriptors,
- donor / acceptor flags,
- residue-class and sidechain/backbone labels on the pocket side,
- and local contact-type participation summaries.

This enrichment path was implemented as a switchable parser-side extension, so
the underlying `A2` and `A3` architectures remained unchanged. In other words,
`A2ext` and `A3ext` are not new model families; they are the same families fed
an enriched local signal.

After the incomplete runs were finished and the `ext` families were compared on
the same 10-seed footing as the earlier `A1/A2/A3` series, the conclusion
became a little sharper.

The result was again informative and again negative in the scientifically
useful sense:

- `A2_full` remained the strongest overall series and the current performance
  reference,
- `A2ext_full` landed extremely close to `A2_full` and may be best described as
  neutral-to-slightly-positive, but not as a clear new record,
- `A3ext_with_bias` did not materially improve on `A3_with_bias`,
- `A3ext_without_bias` looked healthier than plain `A3_without_bias`, but still
  did not rise into the same class as `A2`.

So the first local-chemistry enrichment pass did not rescue the local branch or
surpass the current `A2_full` record. The most plausible reading is not that
local chemistry is irrelevant, but that this particular injection path is too
weak or too indirect: chemistry-aware node summaries did not translate into a
stronger local predictor under the current local DimeNet++ + late fusion
design.

The nuanced positive signal is that enrichment appears easier to read as a
stabilizer than as a breakthrough. In particular, `A3ext_without_bias` moved
closer to the bias-enabled `A3` variants, which suggests that chemistry-aware
local summaries may make the local branch less fragile even when they do not
make it decisively more predictive.

> **WARNING (HISTORICAL CONFIG BUG)**
> Earlier `A3_without_bias` and `A3s_without_bias` JSON configs did not
> explicitly set `model.a3.mixer_bias=false`. Some historical runs launched by
> direct `CONFIG_PATH=...` execution may therefore have behaved like
> bias-enabled runs. Treat those series as provisional until the no-bias
> variants are re-run under the corrected configs. The `A3ext_without_bias`
> config was already explicit and is not affected by this specific issue.

For `A3ext_with_bias`, the diagnostic picture also stayed cautious rather than
transformative. The old pattern "larger relative local contribution tends to
hurt" appears to weaken somewhat, but no new stable high-performing coefficient
regime emerges. That suggests the enrichment may make the local branch slightly
less pathological without making it decisively more useful.

### Immediate Next Questions

The next questions are therefore slightly different from the original ones:

1. can the **local branch itself** be improved, rather than only the scalar
   mixer around it?
2. would a different local encoder (for example a SchNet-style alternative)
   produce a local signal that is more useful and less destabilizing than the
   current local DimeNet++ path?
3. do we need true **pair/edge-aware** local interaction features, rather than
   node-level chemistry summaries, before local chemistry can really matter?
4. only after that, does it make sense to move toward pair-scoring or more
   explicit local-interaction experiments (`A3a`)?

This ordering still matters.

If the local branch in its current form tends to hurt performance when its
relative contribution grows, then piling more expressive local readout
machinery on top of the same branch risks optimizing the wrong object.

So the present interpretation of the roadmap is:

- `A2` showed that locality may matter,
- first `A3` showed that a too-restrictive scalar decomposition can suppress
  that locality,
- later `A3` reruns showed that the scalar decomposition is trajectory- and
  seed-sensitive rather than uniformly collapsed,
- the completed `A3 + bias` series showed that the current local branch is best
  used as a small correction, and tends to degrade quality when it becomes too
  influential,
- therefore the next meaningful experiments should target the **content and
  encoder of the local signal**, not just the final scalar mixing rule.

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
- `y = alpha * y_global + beta * y_local + gamma`

The mixer now supports an explicit bias term `gamma`. By default the current
code enables it, but for ablations it can be disabled externally without
editing `config.json`:

- CLI: `python run.py --no-a3-mixer-bias`
- env: `DUMPLING_A3_MIXER_BIAS=0 python run.py`

This keeps the main config cleaner while still allowing cluster or notebook
launchers to switch between bias-enabled and bias-free A3 runs.

`A3` therefore asks:

> is the local branch really functioning as a correction, or was `A2` only
> exploiting a more expressive final MLP?

### Readout Diagnostics in A3

`A3` exports additional diagnostics into `test_results.json`, including:

- the mixer coefficients (`alpha`, `beta`, `gamma`)
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

The default shell entrypoint is [`run.sh`](run.sh). It is now more than a thin
wrapper: it can forward normal `run.py` arguments, override the splitter seed,
and launch small repeated batches of the same experiment without cloning the
config file.

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

Override the configured splitter seed from the CLI:

```bash
./run.sh --rseed 42
```

Launch several repeated runs of the same experiment:

```bash
./run.sh --n-times 10
```

Launch a deterministic series of repeated runs:

```bash
./run.sh --n-times 10 --rseed 42
```

In that last case, the runs use splitter seeds:

- `42`
- `43`
- `44`
- ...

so one command can produce a clean repeated batch without proliferating config
files. The experiment name still comes from `config.json`, while the actual run
directories remain unique because each run gets its own timestamped signature.

Disable the per-run automatic assistant note:

```bash
./run.sh --no-auto-summary
```

For a very short cluster-, Colab-, or launcher-side smoke run, the repo also ships with:

- [`scripts/make_smoke_config.py`](scripts/make_smoke_config.py)
- [`scripts/make_real_series_configs.py`](scripts/make_real_series_configs.py)

For Colab, there are now three distinct notebook roles:

- [colab_launch_main.ipynb](colab_launch_main.ipynb)
  - the heavy training / sync notebook
- [colab_assistant_journal.ipynb](colab_assistant_journal.ipynb)
  - lightweight post-hoc assistant-journal notebook for copied `runs/`
- [series_analysis.ipynb](series_analysis.ipynb)
  - lightweight local factual series-analysis notebook for `runs/`

The assistant notebook is intentionally separate from the training notebook so
you can analyze completed results in Colab without staging the full runtime
stack again. The plotting notebook is local-first because this work is light.
- [`scripts/rebuild_experiment_index.py`](scripts/rebuild_experiment_index.py)
- [`scripts/runtime_env_smoke.py`](scripts/runtime_env_smoke.py)
- [`scripts/slurm_pipeline_smoke.sh`](scripts/slurm_pipeline_smoke.sh)
- [`scripts/slurm_run_series.sh`](scripts/slurm_run_series.sh)

The smoke-config helper can derive a reduced config from the main
`config.json` while overriding key knobs such as:

- model family (`A1`, `A2`, or `A3`)
- protein-context mode
- ligand-context mode
- epochs
- batch size
- number of workers

Example:

```bash
python scripts/make_smoke_config.py \
  --base-config config.json \
  --output tmp/a1_smoke.json \
  --experiment-name DUMPLING_registry_smoke \
  --model-family A1 \
  --protein-context-mode none \
  --ligand-context-mode none \
  --epochs 2 \
  --batch-size 2 \
  --num-workers 0
```

This is the quickest useful path for checking launcher behavior such as
`--rseed`, `--n-times`, and experiment-registry writes before moving to a full
cluster run.

For Slurm specifically, the most useful first cluster check is to mirror the
current Colab smoke pattern:

1. one bootstrap smoke run with `--extract`,
2. one repeated short batch with `--n-times 3` on the same reduced config.

That flow is wrapped by:

- [`scripts/slurm_pipeline_smoke.sh`](scripts/slurm_pipeline_smoke.sh)

Example:

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

That is preferred over feeding two manual `run.sh` invocations into Slurm by
hand, because the whole smoke path stays reproducible in one `sbatch` script.

## Practical Launch Playbook

If you just want the operational answer, use this:

### A. First smoke on a new machine or new cluster workspace

1. run the environment-only smoke,
2. run the full bootstrap smoke once,
3. let it create `data/`, `datasets/`, and the first `runs/...` folders.

Environment-only smoke:

```bash
sbatch --export=ALL,RUN_PIPELINE_SMOKE=0 scripts/slurm_pipeline_smoke.sh
```

Full bootstrap smoke:

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

### B. Later smoke or tuning runs after caches already exist

Do not pay the extraction cost again unless the raw-data side really changed.
Use:

```bash
RUN_BOOTSTRAP_EXTRACT=0
```

That is the normal setting for later runtime tuning such as `batch_size` or
`num_workers`.

### C. Real experiment series

For a real series, the intended pattern is simple:

1. pick one real config for one real condition,
2. keep one stable `experiment_name` for that condition,
3. launch repeated runs by varying only the base seed.

Example:

```bash
./run.sh --config config.json --n-times 5 --rseed 42
```

If you want a ready-made first suite of real configs, derive them once from
the main config:

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

If `RUN_EXTRACT=1` is used there, the wrapper applies extraction only to the
first run of the repeated series, not to every seed.

You normally do **not** make a new config file for every seed. One config
describes one experimental condition; repeated runs of that condition should
usually share it.

Create a new config only when the condition itself changes, for example:

- model family,
- protein context mode,
- ligand context mode,
- splitter policy,
- loss or optimizer settings,
- training length.

### D. What to do after runs finish

Rebuild the factual top-level indices:

```bash
python scripts/rebuild_experiment_index.py --runs-dir runs
```

This rewrites:

- `runs/experiment_registry.csv`
- `runs/experiment_journal.md`
- `runs/experiment_series_journal.md`

That rebuild is the normal post-series step. It is not something you need to
do between every seed inside one running batch.

When runs are produced on multiple machines, the recommended workflow is:

1. copy only the new `runs/<experiment_signature>/...` folders,
2. keep the terse indices rebuildable rather than hand-merged,
3. rebuild them locally when convenient:

```bash
python scripts/rebuild_experiment_index.py --runs-dir runs
```

That script rescans the discovered run folders and rewrites:

- `runs/experiment_registry.csv`
- `runs/experiment_journal.md`
- `runs/experiment_series_journal.md`

In other words, the factual top-level indices are treated as **rebuildable
views**, not as irreplaceable primary artifacts. The portable truth is the run
folder itself.

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

For A3-specific readout ablations, `run.py` also supports:

```bash
./run.sh --a3-mixer-bias
./run.sh --no-a3-mixer-bias
```

Those flags are only valid when `model.selected == "A3"`.

## Experiment Outputs

Each run writes to:

```text
runs/<experiment_name>_<timestamp>/
```

Typical outputs include:

- `runs/experiment_registry.csv`: global one-line-per-experiment registry with
  timestamps, runtime-location metadata, seed, model family, status, best
  epoch, completed epochs, and final test metrics when available,
- `config.json`: resolved config snapshot,
- `run.log`: experiment log for that specific run folder,
- `run_err.log`: traceback if execution fails,
- `best_model.pt`: best checkpoint by the configured primary validation metric,
- `history.json`: train/validation history,
- `test_results.json`: final test metrics,
- `assistant_summary.md`: automatic post-run assistant note with a compact
  factual snapshot of the run,
- `run_manifest.json`: machine-readable run snapshot used by future index
  rebuilds and portability workflows,
- `runs/experiment_journal.md`: accumulated automatic run journal for quick
  cross-run scanning without opening each experiment folder,
- `best_validation_scatter_diagnostics.json`: extended agreement diagnostics
  for the best validation predictions,
- `model_performance_report.png`: evaluation plots,
- `datasets/*.pickle`: optional per-run split snapshots.

The global registry is intentionally terse: one row per experiment. At the
moment it records fields such as:

- start / finish timestamps,
- run duration,
- success or failure status,
- experiment name and unique experiment signature,
- experiment directory,
- config path,
- git commit hash,
- execution environment (`local`, `colab`, `cluster`, or a manual override),
- hostname,
- artifact root,
- model family,
- source subset and split strategy,
- effective splitter seed,
- `core_as_test`,
- A3 mixer-bias state when relevant,
- batch position for `run.sh --n-times`,
- device,
- primary early-stopping metric,
- `best_epoch`,
- `epochs_completed`,
- final test RMSE / Pearson / CI when the run reached testing.

The automatic note layer is intentionally split in two:

- `runs/<exp>/assistant_summary.md`: the richer per-run card, where
  model-specific commentary such as A3 readout behavior can live comfortably,
- `runs/experiment_journal.md`: the short rolling journal, useful when you
  want to scan a batch of runs in one file before drilling into individual
  folders.

Both are assistant-authored convenience artifacts. They should help review,
not replace manual interpretation of configs, logs, and raw metrics.

At the launcher level, [`run.sh`](run.sh) also writes a separate
`last_run.log` in the repo root by default. That file is the outer shell log
for the most recent launcher invocation, while each experiment folder keeps its
own `run.log`.

## Assistant Layer

LLM-related tooling is intentionally separated from the training pipeline.

Rules of the current design:

- `run.py`, `run.sh`, and run folders do not know about the assistant layer;
- assistant tooling is launched manually when needed;
- assistant tools operate over a whole `runs/` directory;
- factual indexing stays in `scripts/rebuild_experiment_index.py`;
- richer LLM analysis will live under `assistant/`.

The current live backend for the assistant layer is a local Ollama server.
That keeps the analysis path manual, cheap to iterate on, and independent from
the training pipeline.

The long-term target is a second journal, tentatively
`runs/experiment_journal_llm.md`, that keeps the same entry structure as
`runs/experiment_journal.md` but expands the `assistant note` section.

The same split now exists at the series level:

- `runs/experiment_series_journal.md`: factual grouped view over related runs,
- `runs/experiment_series_journal_llm.md`: assistant-style grouped series notes.

### Assistant Roadmap

1. MVP-1: build compact per-run context packets from the existing `runs/`
   structure, without calling any model.
2. MVP-2: validate those packets manually and adjust what we keep from config,
   history, logs, summaries, and diagnostics.
3. MVP-3: add a manual LLM journal builder that reads the packets and writes
   `experiment_journal_llm.md`.
4. MVP-4: add caching for unchanged runs so repeated journal builds stay cheap.

Current status of that roadmap:

- MVP-1 is done: context packets are built from the factual `runs/` tree.
- MVP-2 is done: prompt previews are generated and manually inspectable.
- MVP-3 is done in operational form:
  - dry-run journal generation works,
  - live journal generation works through a local Ollama backend,
  - small local models can still be weak or slow, but the pipeline itself is
    functional end to end.
- MVP-4 is partially done: per-run response caching exists under
  `assistant/cache/`.
- a parallel series-level path now exists too:
  - factual grouping into `experiment_series_journal.md`,
  - assistant grouping into `experiment_series_journal_llm.md`.

### Assistant Experiment Result: Local LLM Journal Feasibility

The assistant-journal branch was taken seriously enough to treat it as an
experiment rather than a vague future wish. That experiment should now be
considered **operationally successful but scientifically negative**.

What was tested:

- factual run and series packets generated from the completed `runs/` tree,
- dry-run prompt previews,
- live Ollama-backed journal generation on lightweight local hardware where
  possible,
- a stronger Colab-hosted path for post-hoc assistant runs,
- model sizes up through:
  - `qwen2.5:7b`,
  - `qwen2.5:14b`,
  - `qwen2.5:32b`.

What worked:

- the staged assistant pipeline itself works end to end;
- Colab is a viable temporary host for larger Ollama models;
- live generation of
  - `runs/experiment_journal_llm.md`,
  - `runs/experiment_series_journal_llm.md`
  is operationally reproducible;
- caching and prompt-preview inspection are useful and worth keeping as
  infrastructure.

What failed:

- scaling the local model size did **not** make the journal trustworthy;
- the same failure modes persisted across 7B, 14B, and 32B models:
  - copying metrics from the wrong run,
  - mixing facts across different experiment series,
  - drifting from factual headers into invented or reinterpreted numbers,
  - adding unsupported narrative details such as wrong durations, wrong seeds,
    or even nonsensical units,
  - turning structured factual packets into polished summaries rather than
    actual analysis.

In other words, the assistant notes became smoother with larger models, but not
more reliable. The dominant problem was not simply "use a bigger local LLM".
The core issue is that a generic local model, given one experiment packet at a
time, tends to produce:

- paraphrase,
- shallow summary,
- or confident hallucination,

rather than the kind of grounded comparative reasoning that this research
workflow actually needs.

Current interpretation:

- the factual journals are already useful because they do not invent facts;
- the current local-LLM journal hypothesis is **not** a good path to reliable
  scientific analysis in this project;
- further progress would likely require a different problem formulation
  entirely, such as:
  - deterministic analytic feature extraction first,
  - then tightly scoped reasoning over those derived signals,
  - or a separate learned assistant trained on manually curated labels.

For now, this branch is kept as infrastructure and as a documented negative
result, not as an active bet for the main research loop.

### MVP-1

The first assistant step is implemented as:

- [assistant/build_run_contexts.py](assistant/build_run_contexts.py)

It scans a `runs/` directory and writes compact context packets to:

- `assistant/experiment_journal_llm_context.json`

Run it manually:

```bash
python assistant/build_run_contexts.py --runs-dir runs
```

This step does not call any LLM yet. It only prepares compact structured
inputs for future prompt design and journal generation.

### Live Assistant Backend

The manual assistant journal can now run against a local Ollama model.

Typical setup:

```bash
sudo snap install ollama
ollama serve
ollama pull qwen2.5:1.5b
cp assistant/.env.example assistant/.env
bash assistant/run_llm_journal.sh --live --limit 1
```

The assistant tooling remains fully optional:

- factual experiment tracking still lives in `runs/experiment_registry.csv`
  and the two factual journals
  - `runs/experiment_journal.md`
  - `runs/experiment_series_journal.md`,
- the assistant layer adds only a separate manual artifact,
  - `runs/experiment_journal_llm.md`
  - `runs/experiment_series_journal_llm.md`.

Use the exact model tag reported by `ollama list`. On small laptops, a 1.5B
class model is a much safer first live test than 3B/7B models.

On the cluster, a stronger assistant model becomes realistic. The preferred
entrypoint there is:

- [`scripts/slurm_assistant_journal.sh`](scripts/slurm_assistant_journal.sh)

Suggested cluster ladder:

1. try `qwen2.5:7b` with `ASSISTANT_LIMIT=1`,
2. if runtime and memory remain comfortable, try `qwen2.5:14b`,
3. if needed, fall back to `qwen2.5:3b`.

That path is especially attractive for the series-level assistant journals,
because the prompts are longer and benefit more from a stronger local model.

### When to run the assistant

The cleanest timing is:

1. let a run or series finish,
2. rebuild the factual indices,
3. then run the assistant layer on top of the finished artifacts.

In other words, the assistant is not part of training orchestration. It is a
post-hoc reader over completed run folders and rebuilt top-level views.

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
- Refine the factual rebuild workflow and keep the top-level experiment indices
  documented as rebuildable views over run folders.
- Keep the manual assistant journal flow as optional infrastructure, but treat
  the first local-LLM-journal experiment as a documented negative result:
  stronger local Ollama models up to `qwen2.5:32b` improved fluency but did
  not produce reliable run- or series-level scientific analysis. The useful
  pieces to preserve are the factual packets, prompt previews, caching, and
  Colab execution path, not the expectation that a larger generic local model
  will automatically become a trustworthy experiment analyst.
- Treat `A2_full` as the current performance reference. The completed
  `A2ext` / `A3ext` chemistry-enrichment series were a useful negative result:
  parser-side local chemistry summaries were operationally clean and
  scientifically interpretable, but they did not beat `A2_full`.
- If the local branch is revisited next, prioritize stronger local signal
  content over more scalar readout tricks: pair/edge-aware interaction
  features, directional local geometry, or a different local encoder are more
  promising than another round of the same node-summary enrichment.
- Add additional geometric baselines that consume richer parser output.
- Tighten regression-test coverage around dataset cache signatures and runtime
  diagnostics.
- Port stable interaction-graph pieces back into the broader research stack
  once the baseline line is experimentally mature.
