# LLM Prompt Preview

This file is a dry-run prompt preview built from assistant context packets.
It is for prompt review only; no model call is performed here.

## DUMPLING_A15_esm_DimeNet_20260507_195111

### System Prompt

```text
You are a careful research assistant reading experiment artifacts from a protein-ligand affinity project.

Your task is to write a compact journal note for one run.

Rules:
- Base your note only on the provided artifacts.
- Do not claim a run is good or bad in absolute terms.
- Do not compare runs unless comparison data is explicitly provided.
- Separate observations from hypotheses.
- Use cautious language.
- If data is missing, say so plainly.
- Keep the note structured and concise.

# Stable Project Context
This project studies protein-ligand binding affinity prediction on PDBBind 2016.

Core assumptions:

- the main geometric object is a fused ligand-pocket interaction graph;
- the current baseline family uses DimeNet++ as the geometry encoder;
- optional side branches provide frozen protein context and lightweight ligand context;
- experiments are exploratory and research-oriented rather than production-oriented.

## Model Family

- `A1`: one coarse global geometry branch plus optional protein/ligand context.
- `A2`: `A1` plus an explicit local geometric branch over a tighter ligand-pocket zone.
- `A3`: `A2` branch encoders plus an explicit linear combination of branch-level scalar outputs.

## Artifacts

Typical run artifacts include:

- `config.json`
- `run.log`
- `history.json`
- `test_results.json`
- `best_validation_scatter_diagnostics.json`
- `model_performance_report.png`
- `assistant_summary.md`
- `run_manifest.json`

## Interpretation Policy

- treat logs, metrics, and saved artifacts as primary evidence;
- do not make absolute good/bad judgments from metrics alone;
- do not compare runs unless comparison context is explicitly present;
- separate observations from hypotheses;
- when data is missing, state that clearly.

# Current Research Stage
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
```

### User Prompt

```text
# Run Identity
{
  "experiment_signature": "DUMPLING_A15_esm_DimeNet_20260507_195111",
  "run_dir": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111"
}

# Registry Snapshot
{
  "started_at": "2026-05-07T19:51:11",
  "finished_at": "2026-05-07T19:51:11",
  "duration_sec": "0.0",
  "status": "success",
  "experiment_name": "DUMPLING_A15_esm_DimeNet",
  "experiment_signature": "DUMPLING_A15_esm_DimeNet_20260507_195111",
  "exp_dir": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111",
  "config_path": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111/config.json",
  "git_commit": "",
  "execution_env": "",
  "hostname": "",
  "artifact_root": "/home/yaroslav/FCUL/DUMPLINGs/runs",
  "model_family": "legacy_duo",
  "source_subset": "refined",
  "splitter": "random",
  "splitter_seed": "42",
  "core_as_test": "True",
  "a3_mixer_bias": "",
  "batch_run_index": "",
  "batch_n_times": "",
  "device": "",
  "primary_metric": "val_pearson",
  "best_epoch": "",
  "epochs_completed": "43",
  "test_rmse": "1.353843036008509",
  "test_pearson": "0.7828758146629085",
  "test_ci": "0.7907588549015856"
}

# Setup
{
  "experiment_name": "DUMPLING_A15_esm_DimeNet",
  "model_family": "legacy_duo",
  "execution_env": "",
  "hostname": "",
  "source_subset": "refined",
  "splitter": "random",
  "splitter_seed": "42",
  "core_as_test": "True",
  "primary_metric": "val_pearson",
  "protein_context": "esm_frozen_whole",
  "ligand_context": ""
}

# Config Excerpt
{
  "experiment_name": "DUMPLING_A15_esm_DimeNet",
  "source_subset": "refined",
  "core_as_test": true,
  "epochs": 100,
  "batch_size": "",
  "num_workers": "",
  "primary_metric": "val_pearson",
  "model_selected": "",
  "global_graph_selected": "",
  "global_graph_config": {},
  "global_encoder_selected": "",
  "global_encoder_config": {},
  "local_graph_selected": "",
  "local_graph_config": {},
  "local_encoder_selected": "",
  "local_encoder_config": {},
  "protein_context_selected": "esm_frozen_whole",
  "protein_context_config": {},
  "ligand_context_selected": "",
  "ligand_context_config": {}
}

# Metrics
{
  "best_epoch": "",
  "epochs_completed": "43",
  "test_rmse": "1.353843036008509",
  "test_pearson": "0.7828758146629085",
  "test_ci": "0.7907588549015856"
}

# History Summary
{
  "train_loss": {
    "n": 43,
    "first": 0.36868263812528207,
    "last": 0.006254076220691354,
    "best_min": 0.006254076220691354
  },
  "val_loss": {
    "n": 43,
    "first": 0.3335710738553147,
    "last": 0.2201186834218588,
    "best_min": 0.21611507641780736
  },
  "val_pearson": {
    "n": 43,
    "first": 0.5804553461382019,
    "last": 0.7559672593645294,
    "best_max": 0.7604901456996054
  },
  "val_rmse": {
    "n": 43,
    "first": 1.7161249528283034,
    "last": 1.3685467051483333,
    "best_min": 1.3568219355626867
  },
  "val_ci": {
    "n": 43,
    "first": 0.7037186665572508,
    "last": 0.7759647007581257,
    "best_max": 0.7791966771242985
  }
}

# History Sampled Checkpoints
{
  "_meta": {
    "kind": "sampled_checkpoints",
    "policy": "front_loaded_with_first_second_last_and_best_epoch",
    "num_requested_points": 7
  },
  "train_loss": [
    {
      "epoch": 1,
      "value": 0.36868263812528207
    },
    {
      "epoch": 2,
      "value": 0.3116353205693396
    },
    {
      "epoch": 6,
      "value": 0.24581984430805914
    },
    {
      "epoch": 11,
      "value": 0.1903283187010522
    },
    {
      "epoch": 20,
      "value": 0.11756717683764135
    },
    {
      "epoch": 30,
      "value": 0.027926706595347597
    },
    {
      "epoch": 43,
      "value": 0.006254076220691354
    }
  ],
  "val_loss": [
    {
      "epoch": 1,
      "value": 0.3335710738553147
    },
    {
      "epoch": 2,
      "value": 0.30367898289217443
    },
    {
      "epoch": 6,
      "value": 0.26557708063170293
    },
    {
      "epoch": 11,
      "value": 0.24831410848365515
    },
    {
      "epoch": 20,
      "value": 0.2347420917250916
    },
    {
      "epoch": 30,
      "value": 0.2171225233338088
    },
    {
      "epoch": 43,
      "value": 0.2201186834218588
    }
  ],
  "val_pearson": [
    {
      "epoch": 1,
      "value": 0.5804553461382019
    },
    {
      "epoch": 2,
      "value": 0.6535291241849629
    },
    {
      "epoch": 6,
      "value": 0.7093091113911928
    },
    {
      "epoch": 11,
      "value": 0.7200279344132107
    },
    {
      "epoch": 20,
      "value": 0.7344249570781642
    },
    {
      "epoch": 30,
      "value": 0.7597552625020951
    },
    {
      "epoch": 43,
      "value": 0.7559672593645294
    }
  ],
  "val_rmse": [
    {
      "epoch": 1,
      "value": 1.7161249528283034
    },
    {
      "epoch": 2,
      "value": 1.60361030112254
    },
    {
      "epoch": 6,
      "value": 1.507229806963033
    },
    {
      "epoch": 11,
      "value": 1.468633008845395
    },
    {
      "epoch": 20,
      "value": 1.4188744202380998
    },
    {
      "epoch": 30,
      "value": 1.3568219355626867
    },
    {
      "epoch": 43,
      "value": 1.3685467051483333
    }
  ],
  "val_ci": [
    {
      "epoch": 1,
      "value": 0.7037186665572508
    },
    {
      "epoch": 2,
      "value": 0.7320552718458246
    },
    {
      "epoch": 6,
      "value": 0.7554871005005775
    },
    {
      "epoch": 11,
      "value": 0.7622856132864528
    },
    {
      "epoch": 20,
      "value": 0.768004696465657
    },
    {
      "epoch": 30,
      "value": 0.7768231944803904
    },
    {
      "epoch": 43,
      "value": 0.7759647007581257
    }
  ]
}

# Test Results
{
  "RMSE": 1.353843036008509,
  "Pearson_R": 0.7828758146629085,
  "CI": 0.7907588549015856
}

# Scatter Diagnostics
{}

# Log Header Excerpt
[
  "[INFO][EXPERIMENT] Log file: runs/DUMPLING_A15_esm_DimeNet_20260507_195111/log.txt",
  "[INFO][PROTEIN_CONTEXT] Protein context settings -> mode=esm_frozen_whole, model=esm2_t33_650M_UR50D, repr_layer=33, pooling=mean, cache_path=protein_context_features, max_length=1022",
  "[INFO][EXPERIMENT] Run settings -> source_subset=refined, core_as_test=True, splitter=random, batch_size=2, num_workers=2, splitter_seed=42, epochs=100",
  "[INFO][OPTIMIZER] Optimizer settings -> type=AdamW, lr=0.0001, weight_decay=0.01",
  "[INFO][METRICS] Primary metric: val_pearson with mode max",
  "[INFO][METRICS] Early stopping enabled with patience 10"
]

# Log Excerpt
[
  "[WARNING][REGISTRY] Excluded 1 complexes from subset refined using bad_complexes.toml: 4bps(training)",
  "[INFO][PROTEIN_CONTEXT] Protein context settings -> mode=esm_frozen_whole, model=esm2_t33_650M_UR50D, repr_layer=33, pooling=mean, cache_path=protein_context_features, max_length=1022",
  "[INFO][EXPERIMENT] Run settings -> source_subset=refined, core_as_test=True, splitter=random, batch_size=2, num_workers=2, splitter_seed=42, epochs=100",
  "[INFO][TRAINER] Improved: Primary metric 'val_pearson': 0.5805",
  "[INFO][TRAINER] Improved: Secondary metric 'val_rmse': 1.7161",
  "[INFO][TRAINER] Improved: Secondary metric 'val_ci': 0.7037",
  "[INFO][TRAINER] Improved: Primary metric 'val_pearson': 0.6535",
  "[INFO][TRAINER] Improved: Secondary metric 'val_rmse': 1.6036",
  "[INFO][TRAINER] Improved: Secondary metric 'val_ci': 0.7321",
  "[INFO][TRAINER] Improved: Primary metric 'val_pearson': 0.6703",
  "[INFO][TRAINER] Improved: Secondary metric 'val_rmse': 1.5449",
  "[INFO][TRAINER] Improved: Secondary metric 'val_ci': 0.7387",
  "[INFO][TRAINER] Improved: Primary metric 'val_pearson': 0.6793",
  "[INFO][TRAINER] Improved: Secondary metric 'val_ci': 0.7431",
  "[INFO][TRAINER] Improved: Primary metric 'val_pearson': 0.7093",
  "[INFO][TRAINER] Improved: Secondary metric 'val_rmse': 1.5072",
  "[INFO][TRAINER] Improved: Secondary metric 'val_ci': 0.7555",
  "[INFO][TRAINER] Improved: Secondary metric 'val_ci': 0.7585",
  "[INFO][TRAINER] Improved: Primary metric 'val_pearson': 0.7109",
  "[INFO][TRAINER] Improved: Secondary metric 'val_rmse': 1.4661",
  "[INFO][TRAINER] Improved: Secondary metric 'val_ci': 0.7593",
  "[INFO][TRAINER] Improved: Primary metric 'val_pearson': 0.7291",
  "[INFO][TRAINER] Improved: Secondary metric 'val_rmse': 1.4301",
  "[INFO][TRAINER] Improved: Secondary metric 'val_ci': 0.7644"
]

# Assistant Summary Excerpt
[]

# Recent Factual Journal Context
# Experiment Journal

This file is rebuilt from the discovered run folders. Treat it as a readable index over the raw experiment artifacts.

##  | DUMPLING_A1_DimeNet_milestone_16_epochs_CI074

- status: `` | model: `legacy_duo` | env: `` | seed: `42` | duration_sec: ``
- location: `/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A1_DimeNet_milestone_16_epochs_CI074` on ``
- artifacts: [folder](DUMPLING_A1_DimeNet_milestone_16_epochs_CI074/) | [config](DUMPLING_A1_DimeNet_milestone_16_epochs_CI074/config.json) | [history](DUMPLING_A1_DimeNet_milestone_16_epochs_CI074/history.json) | [log](DUMPLING_A1_DimeNet_milestone_16_epochs_CI074/log.txt)
- assistant note: model=`legacy_duo` epochs_completed=`16`

> Rebuilt from run-folder artifacts. This journal is a convenience index, not primary evidence.

## 2026-05-03T15:47:19 | DUMPLING_A1_DimeNet_20260503_154719

- status: `success` | model: `legacy_duo` | env: `` | seed: `42` | duration_sec: `0.0`
- location: `/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A1_DimeNet_20260503_154719` on ``
- final metrics: RMSE=`0.8140774957898527`, Pearson_R=`0.680922538999824`, CI=`0.7368521751608351`
- artifacts: [folder](DUMPLING_A1_DimeNet_20260503_154719/) | [config](DUMPLING_A1_DimeNet_20260503_154719/config.json) | [history](DUMPLING_A1_DimeNet_20260503_154719/history.json) | [test](DUMPLING_A1_DimeNet_20260503_154719/test_results.json) | [report](DUMPLING_A1_DimeNet_20260503_154719/model_performance_report.png) | [log](DUMPLING_A1_DimeNet_20260503_154719/log.txt)
- report preview:
  ![](DUMPLING_A1_DimeNet_20260503_154719/model_performance_report.png)
- assistant note: model=`legacy_duo` test_RMSE=`0.8140774957898527` test_Pearson_R=`0.680922538999824` test_CI=`0.7368521751608351` epochs_completed=`40`

> Rebuilt from run-folder artifacts. This journal is a convenience index, not primary evidence.

## 2026-05-03T21:08:15 | DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815

- status: `success` | model: `legacy_duo` | env: `` | seed: `42` | duration_sec: `0.0`
- location: `/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815` on ``
- final metrics: RMSE=`0.8091408350981468`, Pearson_R=`0.6866728747345389`, CI=`0.7497189869179441`
- artifacts: [folder](DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815/) | [config](DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815/config.json) | [history](DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815/history.json) | [test](DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815/test_results.json) | [report](DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815/model_performance_report.png) | [log](DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815/log.txt)
- report preview:
  ![](DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815/model_performance_report.png)
- assistant note: model=`legacy_duo` test_RMSE=`0.8091408350981468` test_Pearson_R=`0.6866728747345389` test_CI=`0.7497189869179441` epochs_completed=`40`

> Rebuilt from run-folder artifacts. This journal is a convenience index, not primary evidence.

## 2026-05-06T09:30:30 | DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030

- status: `success` | model: `legacy_duo` | env: `` | seed: `42` | duration_sec: `0.0`
- location: `/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030` on ``
- final metrics: RMSE=`0.9514149115496928`, Pearson_R=`0.5101147026194092`, CI=`0.6685600172195251`
- artifacts: [folder](DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030/) | [config](DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030/config.json) | [history](DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030/history.json) | [test](DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030/test_results.json) | [report](DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030/model_performance_report.png) | [log](DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030/log.txt)
- report preview:
  ![](DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030/model_performance_report.png)
- assistant note: model=`legacy_duo` protein_context=`esm_only` test_RMSE=`0.9514149115496928` test_Pearson_R=`0.5101147026194092` test_CI=`0.6685600172195251` epochs_completed=`41`

> Rebuilt from run-folder artifacts. This journal is a convenience index, not primary evidence.

## 2026-05-07T19:51:11 | DUMPLING_A15_esm_DimeNet_20260507_195111

- status: `success` | model: `legacy_duo` | env: `` | seed: `42` | duration_sec: `0.0`
- location: `/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111` on ``
- final metrics: RMSE=`1.353843036008509`, Pearson_R=`0.7828758146629085`, CI=`0.7907588549015856`
- artifacts: [folder](DUMPLING_A15_esm_DimeNet_20260507_195111/) | [config](DUMPLING_A15_esm_DimeNet_20260507_195111/config.json) | [history](DUMPLING_A15_esm_DimeNet_20260507_195111/history.json) | [test](DUMPLING_A15_esm_DimeNet_20260507_195111/test_results.json) | [report](DUMPLING_A15_esm_Dim
... [truncated]

# Artifact Paths
{
  "run_dir": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111",
  "config": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111/config.json",
  "history": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111/history.json",
  "test_results": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111/test_results.json",
  "report": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111/model_performance_report.png",
  "run_log": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111/log.txt"
}

# Requested Output
Write a compact markdown note for this run.
Use these sections:
## Setup
## Observed Metrics
## Training Trace
## Cautious Notes

Constraints:
- Keep the note concise.
- Do not call the run good, bad, strong, weak, solid, or poor.
- Do not compare with other runs.
- If something is uncertain, say that directly.
- Prefer observations over conclusions.
```

## DUMPLING_A15_esm_DimeNet_20260508_001049

### System Prompt

```text
You are a careful research assistant reading experiment artifacts from a protein-ligand affinity project.

Your task is to write a compact journal note for one run.

Rules:
- Base your note only on the provided artifacts.
- Do not claim a run is good or bad in absolute terms.
- Do not compare runs unless comparison data is explicitly provided.
- Separate observations from hypotheses.
- Use cautious language.
- If data is missing, say so plainly.
- Keep the note structured and concise.

# Stable Project Context
This project studies protein-ligand binding affinity prediction on PDBBind 2016.

Core assumptions:

- the main geometric object is a fused ligand-pocket interaction graph;
- the current baseline family uses DimeNet++ as the geometry encoder;
- optional side branches provide frozen protein context and lightweight ligand context;
- experiments are exploratory and research-oriented rather than production-oriented.

## Model Family

- `A1`: one coarse global geometry branch plus optional protein/ligand context.
- `A2`: `A1` plus an explicit local geometric branch over a tighter ligand-pocket zone.
- `A3`: `A2` branch encoders plus an explicit linear combination of branch-level scalar outputs.

## Artifacts

Typical run artifacts include:

- `config.json`
- `run.log`
- `history.json`
- `test_results.json`
- `best_validation_scatter_diagnostics.json`
- `model_performance_report.png`
- `assistant_summary.md`
- `run_manifest.json`

## Interpretation Policy

- treat logs, metrics, and saved artifacts as primary evidence;
- do not make absolute good/bad judgments from metrics alone;
- do not compare runs unless comparison context is explicitly present;
- separate observations from hypotheses;
- when data is missing, state that clearly.

# Current Research Stage
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
```

### User Prompt

```text
# Run Identity
{
  "experiment_signature": "DUMPLING_A15_esm_DimeNet_20260508_001049",
  "run_dir": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260508_001049"
}

# Registry Snapshot
{
  "started_at": "2026-05-08T00:10:49",
  "finished_at": "2026-05-08T00:10:49",
  "duration_sec": "0.0",
  "status": "success",
  "experiment_name": "DUMPLING_A15_esm_DimeNet",
  "experiment_signature": "DUMPLING_A15_esm_DimeNet_20260508_001049",
  "exp_dir": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260508_001049",
  "config_path": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260508_001049/config.json",
  "git_commit": "",
  "execution_env": "",
  "hostname": "",
  "artifact_root": "/home/yaroslav/FCUL/DUMPLINGs/runs",
  "model_family": "legacy_duo",
  "source_subset": "refined",
  "splitter": "random",
  "splitter_seed": "42",
  "core_as_test": "True",
  "a3_mixer_bias": "",
  "batch_run_index": "",
  "batch_n_times": "",
  "device": "",
  "primary_metric": "val_pearson",
  "best_epoch": "",
  "epochs_completed": "36",
  "test_rmse": "1.39894675085749",
  "test_pearson": "0.7676593466194417",
  "test_ci": "0.7793030875565016"
}

# Setup
{
  "experiment_name": "DUMPLING_A15_esm_DimeNet",
  "model_family": "legacy_duo",
  "execution_env": "",
  "hostname": "",
  "source_subset": "refined",
  "splitter": "random",
  "splitter_seed": "42",
  "core_as_test": "True",
  "primary_metric": "val_pearson",
  "protein_context": "esm_frozen_whole",
  "ligand_context": ""
}

# Config Excerpt
{
  "experiment_name": "DUMPLING_A15_esm_DimeNet",
  "source_subset": "refined",
  "core_as_test": true,
  "epochs": 100,
  "batch_size": "",
  "num_workers": "",
  "primary_metric": "val_pearson",
  "model_selected": "",
  "global_graph_selected": "",
  "global_graph_config": {},
  "global_encoder_selected": "",
  "global_encoder_config": {},
  "local_graph_selected": "",
  "local_graph_config": {},
  "local_encoder_selected": "",
  "local_encoder_config": {},
  "protein_context_selected": "esm_frozen_whole",
  "protein_context_config": {},
  "ligand_context_selected": "",
  "ligand_context_config": {}
}

# Metrics
{
  "best_epoch": "",
  "epochs_completed": "36",
  "test_rmse": "1.39894675085749",
  "test_pearson": "0.7676593466194417",
  "test_ci": "0.7793030875565016"
}

# History Summary
{
  "train_loss": {
    "n": 36,
    "first": 0.36741446999828425,
    "last": 0.00885636691430516,
    "best_min": 0.00885636691430516
  },
  "val_loss": {
    "n": 36,
    "first": 0.32274654864912344,
    "last": 0.22288592888860953,
    "best_min": 0.21859159478785614
  },
  "train_pearson": {
    "n": 36,
    "first": 0.5977322355075964,
    "last": 0.9923883855270463,
    "best_max": 0.9923883855270463
  },
  "val_pearson": {
    "n": 36,
    "first": 0.6038551447440362,
    "last": 0.7502873765088208,
    "best_max": 0.7573264363118919
  },
  "train_rmse": {
    "n": 36,
    "first": 1.614287011911643,
    "last": 0.24407861692666713,
    "best_min": 0.24407861692666713
  },
  "val_rmse": {
    "n": 36,
    "first": 1.6871268952570977,
    "last": 1.3806287485671018,
    "best_min": 1.368488823869856
  },
  "train_ci": {
    "n": 36,
    "first": 0.7149467329434331,
    "last": 0.9770798716396146,
    "best_max": 0.9770798716396146
  },
  "val_ci": {
    "n": 36,
    "first": 0.7142099648396321,
    "last": 0.7745570235517653,
    "best_max": 0.7795059873624675
  }
}

# History Sampled Checkpoints
{
  "_meta": {
    "kind": "sampled_checkpoints",
    "policy": "front_loaded_with_first_second_last_and_best_epoch",
    "num_requested_points": 7
  },
  "train_loss": [
    {
      "epoch": 1,
      "value": 0.36741446999828425
    },
    {
      "epoch": 2,
      "value": 0.3074448007784422
    },
    {
      "epoch": 5,
      "value": 0.25428547203720914
    },
    {
      "epoch": 10,
      "value": 0.20095960034129662
    },
    {
      "epoch": 17,
      "value": 0.12900709695248402
    },
    {
      "epoch": 25,
      "value": 0.0480093139425802
    },
    {
      "epoch": 36,
      "value": 0.00885636691430516
    }
  ],
  "val_loss": [
    {
      "epoch": 1,
      "value": 0.32274654864912344
    },
    {
      "epoch": 2,
      "value": 0.2952805074139875
    },
    {
      "epoch": 5,
      "value": 0.30401704351975856
    },
    {
      "epoch": 10,
      "value": 0.24319701342224656
    },
    {
      "epoch": 17,
      "value": 0.2851649157788377
    },
    {
      "epoch": 25,
      "value": 0.21961353974274841
    },
    {
      "epoch": 36,
      "value": 0.22288592888860953
    }
  ],
  "train_pearson": [
    {
      "epoch": 1,
      "value": 0.5977322355075964
    },
    {
      "epoch": 2,
      "value": 0.6654050802038177
    },
    {
      "epoch": 5,
      "value": 0.7022960457902456
    },
    {
      "epoch": 10,
      "value": 0.8114764881490205
    },
    {
      "epoch": 17,
      "value": 0.878338554486991
    },
    {
      "epoch": 25,
      "value": 0.9621143187252119
    },
    {
      "epoch": 36,
      "value": 0.9923883855270463
    }
  ],
  "val_pearson": [
    {
      "epoch": 1,
      "value": 0.6038551447440362
    },
    {
      "epoch": 2,
      "value": 0.6449885120587927
    },
    {
      "epoch": 5,
      "value": 0.6536782424146466
    },
    {
      "epoch": 10,
      "value": 0.7221430870691404
    },
    {
      "epoch": 17,
      "value": 0.7296156348185941
    },
    {
      "epoch": 25,
      "value": 0.7533204320781659
    },
    {
      "epoch": 36,
      "value": 0.7502873765088208
    }
  ],
  "train_rmse": [
    {
      "epoch": 1,
      "value": 1.614287011911643
    },
    {
      "epoch": 2,
      "value": 1.4812185962299975
    },
    {
      "epoch": 5,
      "value": 1.4366945359917458
    },
    {
      "epoch": 10,
      "value": 1.1634199328106254
    },
    {
      "epoch": 17,
      "value": 1.1443717208768625
    },
    {
      "epoch": 25,
      "value": 0.5529402623701872
    },
    {
      "epoch": 36,
      "value": 0.24407861692666713
    }
  ],
  "val_rmse": [
    {
      "epoch": 1,
      "value": 1.6871268952570977
    },
    {
      "epoch": 2,
      "value": 1.5928943455099693
    },
    {
      "epoch": 5,
      "value": 1.604069469123872
    },
    {
      "epoch": 10,
      "value": 1.4400175323461515
    },
    {
      "epoch": 17,
      "value": 1.5669559856089974
    },
    {
      "epoch": 25,
      "value": 1.368488823869856
    },
    {
      "epoch": 36,
      "value": 1.3806287485671018
    }
  ],
  "train_ci": [
    {
      "epoch": 1,
      "value": 0.7149467329434331
    },
    {
      "epoch": 2,
      "value": 0.7416792226404939

... [truncated]

# Test Results
{
  "RMSE": 1.39894675085749,
  "Pearson_R": 0.7676593466194417,
  "CI": 0.7793030875565016
}

# Scatter Diagnostics
{}

# Log Header Excerpt
[
  "[INFO][EXPERIMENT] Log file: runs/DUMPLING_A15_esm_DimeNet_20260508_001049/log.txt",
  "[INFO][PROTEIN_CONTEXT] Protein context settings -> mode=esm_frozen_whole, model=esm2_t33_650M_UR50D, repr_layer=33, pooling=mean, cache_path=protein_context_features, max_length=1022",
  "[INFO][EXPERIMENT] Run settings -> source_subset=refined, core_as_test=True, splitter=random, batch_size=2, num_workers=0, splitter_seed=42, epochs=100",
  "[INFO][OPTIMIZER] Optimizer settings -> type=AdamW, lr=0.0001, weight_decay=0.01",
  "[INFO][METRICS] Primary metric: val_pearson with mode max",
  "[INFO][METRICS] Early stopping enabled with patience 10"
]

# Log Excerpt
[
  "[WARNING][REGISTRY] Excluded 1 complexes from subset refined using bad_complexes.toml: 4bps(training)",
  "[INFO][PROTEIN_CONTEXT] Protein context settings -> mode=esm_frozen_whole, model=esm2_t33_650M_UR50D, repr_layer=33, pooling=mean, cache_path=protein_context_features, max_length=1022",
  "[INFO][EXPERIMENT] Run settings -> source_subset=refined, core_as_test=True, splitter=random, batch_size=2, num_workers=0, splitter_seed=42, epochs=100",
  "[WARNING][OPTIMIZER] Quantum: No quantum parameters found, optimizer disabled",
  "[INFO][TRAINER] Improved: Primary metric 'val_pearson': 0.6039",
  "[INFO][TRAINER] Improved: Secondary metric 'val_rmse': 1.6871",
  "[INFO][TRAINER] Improved: Secondary metric 'val_ci': 0.7142",
  "[INFO][TRAINER] Improved: Primary metric 'val_pearson': 0.6450",
  "[INFO][TRAINER] Improved: Secondary metric 'val_rmse': 1.5929",
  "[INFO][TRAINER] Improved: Secondary metric 'val_ci': 0.7289",
  "[INFO][TRAINER] Improved: Primary metric 'val_pearson': 0.6624",
  "[INFO][TRAINER] Improved: Secondary metric 'val_rmse': 1.5849",
  "[INFO][TRAINER] Improved: Secondary metric 'val_ci': 0.7380",
  "[INFO][TRAINER] Improved: Primary metric 'val_pearson': 0.6871",
  "[INFO][TRAINER] Improved: Secondary metric 'val_ci': 0.7488",
  "[INFO][TRAINER] Improved: Primary metric 'val_pearson': 0.7003",
  "[INFO][TRAINER] Improved: Secondary metric 'val_rmse': 1.4922",
  "[INFO][TRAINER] Improved: Secondary metric 'val_ci': 0.7536",
  "[INFO][TRAINER] Improved: Primary metric 'val_pearson': 0.7031",
  "[INFO][TRAINER] Improved: Secondary metric 'val_ci': 0.7557",
  "[INFO][TRAINER] Improved: Secondary metric 'val_rmse': 1.4852",
  "[INFO][TRAINER] Improved: Secondary metric 'val_ci': 0.7558",
  "[INFO][TRAINER] Improved: Primary metric 'val_pearson': 0.7221",
  "[INFO][TRAINER] Improved: Secondary metric 'val_rmse': 1.4400"
]

# Assistant Summary Excerpt
[]

# Recent Factual Journal Context
# Experiment Journal

This file is rebuilt from the discovered run folders. Treat it as a readable index over the raw experiment artifacts.

##  | DUMPLING_A1_DimeNet_milestone_16_epochs_CI074

- status: `` | model: `legacy_duo` | env: `` | seed: `42` | duration_sec: ``
- location: `/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A1_DimeNet_milestone_16_epochs_CI074` on ``
- artifacts: [folder](DUMPLING_A1_DimeNet_milestone_16_epochs_CI074/) | [config](DUMPLING_A1_DimeNet_milestone_16_epochs_CI074/config.json) | [history](DUMPLING_A1_DimeNet_milestone_16_epochs_CI074/history.json) | [log](DUMPLING_A1_DimeNet_milestone_16_epochs_CI074/log.txt)
- assistant note: model=`legacy_duo` epochs_completed=`16`

> Rebuilt from run-folder artifacts. This journal is a convenience index, not primary evidence.

## 2026-05-03T15:47:19 | DUMPLING_A1_DimeNet_20260503_154719

- status: `success` | model: `legacy_duo` | env: `` | seed: `42` | duration_sec: `0.0`
- location: `/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A1_DimeNet_20260503_154719` on ``
- final metrics: RMSE=`0.8140774957898527`, Pearson_R=`0.680922538999824`, CI=`0.7368521751608351`
- artifacts: [folder](DUMPLING_A1_DimeNet_20260503_154719/) | [config](DUMPLING_A1_DimeNet_20260503_154719/config.json) | [history](DUMPLING_A1_DimeNet_20260503_154719/history.json) | [test](DUMPLING_A1_DimeNet_20260503_154719/test_results.json) | [report](DUMPLING_A1_DimeNet_20260503_154719/model_performance_report.png) | [log](DUMPLING_A1_DimeNet_20260503_154719/log.txt)
- report preview:
  ![](DUMPLING_A1_DimeNet_20260503_154719/model_performance_report.png)
- assistant note: model=`legacy_duo` test_RMSE=`0.8140774957898527` test_Pearson_R=`0.680922538999824` test_CI=`0.7368521751608351` epochs_completed=`40`

> Rebuilt from run-folder artifacts. This journal is a convenience index, not primary evidence.

## 2026-05-03T21:08:15 | DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815

- status: `success` | model: `legacy_duo` | env: `` | seed: `42` | duration_sec: `0.0`
- location: `/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815` on ``
- final metrics: RMSE=`0.8091408350981468`, Pearson_R=`0.6866728747345389`, CI=`0.7497189869179441`
- artifacts: [folder](DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815/) | [config](DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815/config.json) | [history](DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815/history.json) | [test](DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815/test_results.json) | [report](DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815/model_performance_report.png) | [log](DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815/log.txt)
- report preview:
  ![](DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815/model_performance_report.png)
- assistant note: model=`legacy_duo` test_RMSE=`0.8091408350981468` test_Pearson_R=`0.6866728747345389` test_CI=`0.7497189869179441` epochs_completed=`40`

> Rebuilt from run-folder artifacts. This journal is a convenience index, not primary evidence.

## 2026-05-06T09:30:30 | DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030

- status: `success` | model: `legacy_duo` | env: `` | seed: `42` | duration_sec: `0.0`
- location: `/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030` on ``
- final metrics: RMSE=`0.9514149115496928`, Pearson_R=`0.5101147026194092`, CI=`0.6685600172195251`
- artifacts: [folder](DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030/) | [config](DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030/config.json) | [history](DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030/history.json) | [test](DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030/test_results.json) | [report](DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030/model_performance_report.png) | [log](DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030/log.txt)
- report preview:
  ![](DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030/model_performance_report.png)
- assistant note: model=`legacy_duo` protein_context=`esm_only` test_RMSE=`0.9514149115496928` test_Pearson_R=`0.5101147026194092` test_CI=`0.6685600172195251` epochs_completed=`41`

> Rebuilt from run-folder artifacts. This journal is a convenience index, not primary evidence.

## 2026-05-07T19:51:11 | DUMPLING_A15_esm_DimeNet_20260507_195111

- status: `success` | model: `legacy_duo` | env: `` | seed: `42` | duration_sec: `0.0`
- location: `/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111` on ``
- final metrics: RMSE=`1.353843036008509`, Pearson_R=`0.7828758146629085`, CI=`0.7907588549015856`
- artifacts: [folder](DUMPLING_A15_esm_DimeNet_20260507_195111/) | [config](DUMPLING_A15_esm_DimeNet_20260507_195111/config.json) | [history](DUMPLING_A15_esm_DimeNet_20260507_195111/history.json) | [test](DUMPLING_A15_esm_DimeNet_20260507_195111/test_results.json) | [report](DUMPLING_A15_esm_Dim
... [truncated]

# Artifact Paths
{
  "run_dir": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260508_001049",
  "config": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260508_001049/config.json",
  "history": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260508_001049/history.json",
  "test_results": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260508_001049/test_results.json",
  "report": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260508_001049/model_performance_report.png",
  "run_log": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260508_001049/log.txt"
}

# Requested Output
Write a compact markdown note for this run.
Use these sections:
## Setup
## Observed Metrics
## Training Trace
## Cautious Notes

Constraints:
- Keep the note concise.
- Do not call the run good, bad, strong, weak, solid, or poor.
- Do not compare with other runs.
- If something is uncertain, say that directly.
- Prefer observations over conclusions.
```
