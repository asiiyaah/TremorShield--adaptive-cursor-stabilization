# TremorShield project structure

This archive reorganizes the existing project files into numbered phases.
Existing file contents have been copied byte-for-byte; files were moved, not
rewritten. The original Git metadata is retained at the project root.

## Phases

- `01_DATA_COLLECTION/`: mouse-event collection backend, frontend and deployment files.
- `02_DATA_CLEANING/`: reserved for the cleaning phase. The uploaded repository did not
  contain a dedicated data-cleaning implementation, so no cleaning code was invented.
- `03_SYNTHETIC_TREMOR_GENERATION/`: synthetic tremor generation pipeline and scripts.
- `04_DATA_VALIDATION/`: signal, spatial, RobustPDX-reference and statistical validation.
- `05_ML_MODELS/`: task-classification training, feature data, per-model results,
  comparison outputs and exploratory experiments.
- `06_APPLICATION/`: the HTML application.
- `99_ARCHIVE_REVIEW/`: legacy Parkinson's-disease-related files, stale outputs,
  empty logs and files that should be reviewed before reuse.

## Important note

Because the request was to leave file contents unchanged, Python imports, relative paths,
deployment references and README instructions were not rewritten. Some scripts may
therefore still refer to their old locations and may need path/import updates before
running from this new layout. The task-feature CSVs were moved out of the old
`models/robustpdx/` directory based on their use by task-classification scripts; their
contents were not changed.
