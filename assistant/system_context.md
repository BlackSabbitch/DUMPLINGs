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
