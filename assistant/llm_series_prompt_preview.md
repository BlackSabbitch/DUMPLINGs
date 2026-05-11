# LLM Series Prompt Preview

This file is a dry-run prompt preview built from assistant series context packets.
It is for prompt review only; no model call is performed here.

## DUMPLING_A15_esm_DimeNet

### System Prompt

```text
You are a careful research assistant reading a series of related experiment artifacts from a protein-ligand affinity project.

Your task is to write a compact journal note for one experiment series.

Rules:
- Base your note only on the provided artifacts.
- Treat the series as the unit of analysis, not any single run in isolation.
- Do not compare this series with outside series unless such comparison context is explicitly provided.
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
# Series Identity
{
  "series_name": "DUMPLING_A15_esm_DimeNet"
}

# Series Setup
{
  "experiment_name": "DUMPLING_A15_esm_DimeNet",
  "model_family": "legacy_duo",
  "source_subset": "refined",
  "splitter": "random",
  "core_as_test": "True",
  "primary_metric": "val_pearson",
  "protein_context": "esm_frozen_whole",
  "ligand_context": "",
  "execution_env": "",
  "hostname": ""
}

# Series Summary
{
  "started_at": "2026-05-07T19:51:11",
  "finished_at": "2026-05-08T00:10:49",
  "total_runs": 2,
  "success_count": 2,
  "failure_count": 0,
  "seeds": [
    "42"
  ],
  "batch_positions": []
}

# Aggregate Metrics
{
  "duration_sec": {
    "count": "2",
    "mean": "0.0",
    "std": "0.0",
    "min": "0.0",
    "max": "0.0"
  },
  "test_rmse": {
    "count": "2",
    "mean": "1.3764",
    "std": "0.0226",
    "min": "1.3538",
    "max": "1.3989"
  },
  "test_pearson": {
    "count": "2",
    "mean": "0.7753",
    "std": "0.0076",
    "min": "0.7677",
    "max": "0.7829"
  },
  "test_ci": {
    "count": "2",
    "mean": "0.7850",
    "std": "0.0057",
    "min": "0.7793",
    "max": "0.7908"
  }
}

# Members
[
  {
    "experiment_signature": "DUMPLING_A15_esm_DimeNet_20260507_195111",
    "run_dir": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111",
    "registry_snapshot": {
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
    },
    "metrics": {
      "best_epoch": "",
      "epochs_completed": "43",
      "test_rmse": "1.353843036008509",
      "test_pearson": "0.7828758146629085",
      "test_ci": "0.7907588549015856"
    },
    "config_excerpt": {
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
    },
    "artifact_paths": {
      "run_dir": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111",
      "config": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111/config.json",
      "history": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111/history.json",
      "test_results": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111/test_results.json",
      "report": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111/model_performance_report.png",
      "run_log": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260507_195111/log.txt"
    }
  },
  {
    "experiment_signature": "DUMPLING_A15_esm_DimeNet_20260508_001049",
    "run_dir": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260508_001049",
    "registry_snapshot": {
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
    },
    "metrics": {
      "best_epoch": "",
      "epochs_completed": "36",
      "test_rmse": "1.39894675085749",
      "test_pearson": "0.7676593466194417",
      "test_ci": "0.7793030875565016"
    },
    "config_excerpt": {
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
    },
    "artifact_paths": {
      "run_dir": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260508_001049",
      "config": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260508_001049/config.json",
      "history": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260508_001049/history.json",
      "test_results": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260508_001049/test_results.json",
      "report": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260508_001049/model_performance_report.png",
      "run_log": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_20260508_001049/log.txt"
    }
  }
]

# Recent Factual Series Journal Context
# Experiment Series Journal

This file is rebuilt from grouped run folders. It is a factual series-level view over related experiments.

## DUMPLING_A15_esm_DimeNet | model=`legacy_duo` | runs=`2`

- setup: subset=`refined` | splitter=`random` | core_as_test=`True` | primary_metric=`val_pearson`
- outcomes: success=`2` / total=`2` | failure=`0` | seeds=`42`
- window: started=`2026-05-07T19:51:11` | finished=`2026-05-08T00:10:49`
- duration_sec: mean=`0.0` | std=`0.0` | min=`0.0` | max=`0.0`
- test Pearson_R: mean=`0.7753` | std=`0.0076` | min=`0.7677` | max=`0.7829`
- test RMSE: mean=`1.3764` | std=`0.0226` | min=`1.3538` | max=`1.3989`
- test CI: mean=`0.7850` | std=`0.0057` | min=`0.7793` | max=`0.7908`
- members: `DUMPLING_A15_esm_DimeNet_20260507_195111, DUMPLING_A15_esm_DimeNet_20260508_001049`

> Rebuilt from grouped run-folder artifacts. This is a factual series view, not an interpretive conclusion.

## DUMPLING_A15_esm_DimeNet_ligand_context_test | model=`legacy_duo` | runs=`1`

- setup: subset=`refined` | splitter=`random` | core_as_test=`True` | primary_metric=`val_pearson`
- outcomes: success=`1` / total=`1` | failure=`0` | seeds=`42`
- window: started=`2026-05-08T11:13:27` | finished=`2026-05-08T11:13:27`
- duration_sec: mean=`0.0` | std=`0.0` | min=`0.0` | max=`0.0`
- test Pearson_R: mean=`0.7753` | std=`0.0000` | min=`0.7753` | max=`0.7753`
- test RMSE: mean=`1.3771` | std=`0.0000` | min=`1.3771` | max=`1.3771`
- test CI: mean=`0.7862` | std=`0.0000` | min=`0.7862` | max=`0.7862`
- members: `DUMPLING_A15_esm_DimeNet_ligand_context_test_20260508_111327`

> Rebuilt from grouped run-folder artifacts. This is a factual series view, not an interpretive conclusion.

## DUMPLING_A1_DimeNet | model=`legacy_duo` | runs=`2`

- setup: subset=`refined` | splitter=`random` | core_as_test=`True` | primary_metric=`val_pearson`
- outcomes: success=`1` / total=`2` | failure=`1` | seeds=`42`
- window: started=`` | finished=`2026-05-03T15:47:19`
- duration_sec: mean=`0.0` | std=`0.0` | min=`0.0` | max=`0.0`
- test Pearson_R: mean=`0.6809` | std=`0.0000` | min=`0.6809` | max=`0.6809`
- test RMSE: mean=`0.8141` | std=`0.0000` | min=`0.8141` | max=`0.8141`
- test CI: mean=`0.7369` | std=`0.0000` | min=`0.7369` | max=`0.7369`
- members: `DUMPLING_A1_DimeNet_milestone_16_epochs_CI074, DUMPLING_A1_DimeNet_20260503_154719`

> Rebuilt from grouped run-folder artifacts. This is a factual series view, not an interpretive conclusion.

## DUMPLING_A1b_DimeNet_ESM_only_ESM | model=`legacy_duo` | runs=`1`

- setup: subset=`refined` | splitter=`random` | core_as_test=`True` | primary_metric=`val_pearson`
- outcomes: success=`1` / total=`1` | failure=`0` | seeds=`42`
- window: started=`2026-05-06T09:30:30` | finished=`2026-05-06T09:30:30`
- duration_sec: mean=`0.0` | std=`0.0` | min=`0.0` | max=`0.0`
- test Pearson_R: mean=`0.5101` | std=`0.0000` | min=`0.5101` | max=`0.5101`
- test RMSE: mean=`0.9514` | std=`0.0000` | min=`0.9514` | max=`0.9514`
- test CI: mean=`0.6686` | std=`0.0000` | min=`0.6686` | max=`0.6686`
- members: `DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030`

> Rebuilt from grouped run-folder artifacts. This is a factual series view, not an interpretive conclusion.

## DUMPLING_A1b_DimeNet_Initial_milestone_bs2 | model=`legacy_duo` | runs=`1`

- setup: subset=`refined` | splitter=`random` | core_as_test=`True` | primary_metric=`val_pearson`
- outcomes: success=`1` / total=`1` | failure=`0` | seeds=`42`
- window: started=`2026-05-03T21:08:15` | finished=`2026-05-03T21:08:15`
- duration_sec: mean=`0.0` | std=`0.0` | min=`0.0` | max=`0.0`
- test Pearson_R: mean=`0.6867` | std=`0.0000` | min=`0.6867` | max=`0.6867`
- test RMSE: mean=`0.8091` | std=`0.0000` | min=`0.8091` | max=`0.8091`
- test CI: mean=`0.7497` | std=`0.0000` | min=`0.7497` | max=`0.7497`
- members: `DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815`

> Rebuilt from grouped run-folder artifacts. This is a factual series view, not an interpretive conclusion.

## DUMPLING_A2_first_run | model=`mixed` | runs=`3`

- setup: subset=`refined` | splitter=`random` | core_as_test=`True` | primary_metric=`val_pearson`
- outcomes: success=`3` / total=`3` | failure=`0` | seeds=`42`
- window: started=`2026-05-09T13:59:23` | finished=`2026-05-10T04:18:20`
- duration_sec: mean=`0.0` | std=`0.0` | min=`0.0` | max=`0.0`
- test Pearson_R: mean=`0.7614` | std=`0.0288` | min=`0.7208` | max=`0.7845`
- test RMSE: mean=`1.4567` | std=`0.0651` | min=`1.3651` | max=`1.5104`
- test CI: mean=`0.7797` | std=`0.0146` | min=`0.7595` | max=`0.7934`
- members: `DUMPLING_A2_first_run_20260509_135923, DUMPLING_A2_first_run_20260509_191050, DUMPLING_A2_first_run_20260510_041820`

> Rebuilt from grouped run-folder artifacts. This is a factual series view, not an interpretive conclusion.

## DUMPLING_A2_sanity_run_of_A1_from_new_codebase | model=`A1` | runs=`1`

- setup: subset=`refined` | splitter=`random` | core_as_test=`True` | primary_metric=`val_pearson`
-
... [truncated]

# Requested Output
Write a compact markdown note for this experiment series.
Use these sections:
## Series Setup
## Aggregate Snapshot
## Across-Run Variation
## Cautious Notes

Constraints:
- Keep the note concise.
- Focus on the series as a grouped object.
- Do not compare this series with outside series.
- Do not call the series good, bad, strong, weak, solid, or poor.
- If something is uncertain, say that directly.
- Prefer observations over conclusions.
```

## DUMPLING_A15_esm_DimeNet_ligand_context_test

### System Prompt

```text
You are a careful research assistant reading a series of related experiment artifacts from a protein-ligand affinity project.

Your task is to write a compact journal note for one experiment series.

Rules:
- Base your note only on the provided artifacts.
- Treat the series as the unit of analysis, not any single run in isolation.
- Do not compare this series with outside series unless such comparison context is explicitly provided.
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
# Series Identity
{
  "series_name": "DUMPLING_A15_esm_DimeNet_ligand_context_test"
}

# Series Setup
{
  "experiment_name": "DUMPLING_A15_esm_DimeNet_ligand_context_test",
  "model_family": "legacy_duo",
  "source_subset": "refined",
  "splitter": "random",
  "core_as_test": "True",
  "primary_metric": "val_pearson",
  "protein_context": "esm_frozen_whole",
  "ligand_context": "basic_rdkit",
  "execution_env": "",
  "hostname": ""
}

# Series Summary
{
  "started_at": "2026-05-08T11:13:27",
  "finished_at": "2026-05-08T11:13:27",
  "total_runs": 1,
  "success_count": 1,
  "failure_count": 0,
  "seeds": [
    "42"
  ],
  "batch_positions": []
}

# Aggregate Metrics
{
  "duration_sec": {
    "count": "1",
    "mean": "0.0",
    "std": "0.0",
    "min": "0.0",
    "max": "0.0"
  },
  "test_rmse": {
    "count": "1",
    "mean": "1.3771",
    "std": "0.0000",
    "min": "1.3771",
    "max": "1.3771"
  },
  "test_pearson": {
    "count": "1",
    "mean": "0.7753",
    "std": "0.0000",
    "min": "0.7753",
    "max": "0.7753"
  },
  "test_ci": {
    "count": "1",
    "mean": "0.7862",
    "std": "0.0000",
    "min": "0.7862",
    "max": "0.7862"
  }
}

# Members
[
  {
    "experiment_signature": "DUMPLING_A15_esm_DimeNet_ligand_context_test_20260508_111327",
    "run_dir": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_ligand_context_test_20260508_111327",
    "registry_snapshot": {
      "started_at": "2026-05-08T11:13:27",
      "finished_at": "2026-05-08T11:13:27",
      "duration_sec": "0.0",
      "status": "success",
      "experiment_name": "DUMPLING_A15_esm_DimeNet_ligand_context_test",
      "experiment_signature": "DUMPLING_A15_esm_DimeNet_ligand_context_test_20260508_111327",
      "exp_dir": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_ligand_context_test_20260508_111327",
      "config_path": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_ligand_context_test_20260508_111327/config.json",
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
      "epochs_completed": "29",
      "test_rmse": "1.3771292078063415",
      "test_pearson": "0.7752929785769919",
      "test_ci": "0.7862010836845273"
    },
    "metrics": {
      "best_epoch": "",
      "epochs_completed": "29",
      "test_rmse": "1.3771292078063415",
      "test_pearson": "0.7752929785769919",
      "test_ci": "0.7862010836845273"
    },
    "config_excerpt": {
      "experiment_name": "DUMPLING_A15_esm_DimeNet_ligand_context_test",
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
      "ligand_context_selected": "basic_rdkit",
      "ligand_context_config": {}
    },
    "artifact_paths": {
      "run_dir": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_ligand_context_test_20260508_111327",
      "config": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_ligand_context_test_20260508_111327/config.json",
      "history": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_ligand_context_test_20260508_111327/history.json",
      "test_results": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_ligand_context_test_20260508_111327/test_results.json",
      "scatter_diagnostics": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_ligand_context_test_20260508_111327/best_validation_scatter_diagnostics.json",
      "report": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_ligand_context_test_20260508_111327/model_performance_report.png",
      "run_log": "/home/yaroslav/FCUL/DUMPLINGs/runs/DUMPLING_A15_esm_DimeNet_ligand_context_test_20260508_111327/log.txt"
    }
  }
]

# Recent Factual Series Journal Context
# Experiment Series Journal

This file is rebuilt from grouped run folders. It is a factual series-level view over related experiments.

## DUMPLING_A15_esm_DimeNet | model=`legacy_duo` | runs=`2`

- setup: subset=`refined` | splitter=`random` | core_as_test=`True` | primary_metric=`val_pearson`
- outcomes: success=`2` / total=`2` | failure=`0` | seeds=`42`
- window: started=`2026-05-07T19:51:11` | finished=`2026-05-08T00:10:49`
- duration_sec: mean=`0.0` | std=`0.0` | min=`0.0` | max=`0.0`
- test Pearson_R: mean=`0.7753` | std=`0.0076` | min=`0.7677` | max=`0.7829`
- test RMSE: mean=`1.3764` | std=`0.0226` | min=`1.3538` | max=`1.3989`
- test CI: mean=`0.7850` | std=`0.0057` | min=`0.7793` | max=`0.7908`
- members: `DUMPLING_A15_esm_DimeNet_20260507_195111, DUMPLING_A15_esm_DimeNet_20260508_001049`

> Rebuilt from grouped run-folder artifacts. This is a factual series view, not an interpretive conclusion.

## DUMPLING_A15_esm_DimeNet_ligand_context_test | model=`legacy_duo` | runs=`1`

- setup: subset=`refined` | splitter=`random` | core_as_test=`True` | primary_metric=`val_pearson`
- outcomes: success=`1` / total=`1` | failure=`0` | seeds=`42`
- window: started=`2026-05-08T11:13:27` | finished=`2026-05-08T11:13:27`
- duration_sec: mean=`0.0` | std=`0.0` | min=`0.0` | max=`0.0`
- test Pearson_R: mean=`0.7753` | std=`0.0000` | min=`0.7753` | max=`0.7753`
- test RMSE: mean=`1.3771` | std=`0.0000` | min=`1.3771` | max=`1.3771`
- test CI: mean=`0.7862` | std=`0.0000` | min=`0.7862` | max=`0.7862`
- members: `DUMPLING_A15_esm_DimeNet_ligand_context_test_20260508_111327`

> Rebuilt from grouped run-folder artifacts. This is a factual series view, not an interpretive conclusion.

## DUMPLING_A1_DimeNet | model=`legacy_duo` | runs=`2`

- setup: subset=`refined` | splitter=`random` | core_as_test=`True` | primary_metric=`val_pearson`
- outcomes: success=`1` / total=`2` | failure=`1` | seeds=`42`
- window: started=`` | finished=`2026-05-03T15:47:19`
- duration_sec: mean=`0.0` | std=`0.0` | min=`0.0` | max=`0.0`
- test Pearson_R: mean=`0.6809` | std=`0.0000` | min=`0.6809` | max=`0.6809`
- test RMSE: mean=`0.8141` | std=`0.0000` | min=`0.8141` | max=`0.8141`
- test CI: mean=`0.7369` | std=`0.0000` | min=`0.7369` | max=`0.7369`
- members: `DUMPLING_A1_DimeNet_milestone_16_epochs_CI074, DUMPLING_A1_DimeNet_20260503_154719`

> Rebuilt from grouped run-folder artifacts. This is a factual series view, not an interpretive conclusion.

## DUMPLING_A1b_DimeNet_ESM_only_ESM | model=`legacy_duo` | runs=`1`

- setup: subset=`refined` | splitter=`random` | core_as_test=`True` | primary_metric=`val_pearson`
- outcomes: success=`1` / total=`1` | failure=`0` | seeds=`42`
- window: started=`2026-05-06T09:30:30` | finished=`2026-05-06T09:30:30`
- duration_sec: mean=`0.0` | std=`0.0` | min=`0.0` | max=`0.0`
- test Pearson_R: mean=`0.5101` | std=`0.0000` | min=`0.5101` | max=`0.5101`
- test RMSE: mean=`0.9514` | std=`0.0000` | min=`0.9514` | max=`0.9514`
- test CI: mean=`0.6686` | std=`0.0000` | min=`0.6686` | max=`0.6686`
- members: `DUMPLING_A1b_DimeNet_ESM_only_ESM_20260506_093030`

> Rebuilt from grouped run-folder artifacts. This is a factual series view, not an interpretive conclusion.

## DUMPLING_A1b_DimeNet_Initial_milestone_bs2 | model=`legacy_duo` | runs=`1`

- setup: subset=`refined` | splitter=`random` | core_as_test=`True` | primary_metric=`val_pearson`
- outcomes: success=`1` / total=`1` | failure=`0` | seeds=`42`
- window: started=`2026-05-03T21:08:15` | finished=`2026-05-03T21:08:15`
- duration_sec: mean=`0.0` | std=`0.0` | min=`0.0` | max=`0.0`
- test Pearson_R: mean=`0.6867` | std=`0.0000` | min=`0.6867` | max=`0.6867`
- test RMSE: mean=`0.8091` | std=`0.0000` | min=`0.8091` | max=`0.8091`
- test CI: mean=`0.7497` | std=`0.0000` | min=`0.7497` | max=`0.7497`
- members: `DUMPLING_A1b_DimeNet_Initial_milestone_bs2_20260503_210815`

> Rebuilt from grouped run-folder artifacts. This is a factual series view, not an interpretive conclusion.

## DUMPLING_A2_first_run | model=`mixed` | runs=`3`

- setup: subset=`refined` | splitter=`random` | core_as_test=`True` | primary_metric=`val_pearson`
- outcomes: success=`3` / total=`3` | failure=`0` | seeds=`42`
- window: started=`2026-05-09T13:59:23` | finished=`2026-05-10T04:18:20`
- duration_sec: mean=`0.0` | std=`0.0` | min=`0.0` | max=`0.0`
- test Pearson_R: mean=`0.7614` | std=`0.0288` | min=`0.7208` | max=`0.7845`
- test RMSE: mean=`1.4567` | std=`0.0651` | min=`1.3651` | max=`1.5104`
- test CI: mean=`0.7797` | std=`0.0146` | min=`0.7595` | max=`0.7934`
- members: `DUMPLING_A2_first_run_20260509_135923, DUMPLING_A2_first_run_20260509_191050, DUMPLING_A2_first_run_20260510_041820`

> Rebuilt from grouped run-folder artifacts. This is a factual series view, not an interpretive conclusion.

## DUMPLING_A2_sanity_run_of_A1_from_new_codebase | model=`A1` | runs=`1`

- setup: subset=`refined` | splitter=`random` | core_as_test=`True` | primary_metric=`val_pearson`
-
... [truncated]

# Requested Output
Write a compact markdown note for this experiment series.
Use these sections:
## Series Setup
## Aggregate Snapshot
## Across-Run Variation
## Cautious Notes

Constraints:
- Keep the note concise.
- Focus on the series as a grouped object.
- Do not compare this series with outside series.
- Do not call the series good, bad, strong, weak, solid, or poor.
- If something is uncertain, say that directly.
- Prefer observations over conclusions.
```
