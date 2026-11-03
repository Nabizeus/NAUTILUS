# Ocean Transport Next-Step Tracer Prediction
See README in the chat for full details.

## MareNostrum 5 work (synced)

`./sync_from_mn5.sh` copies code, configs, job scripts, reports and small per-run results from MN5 (`~/NAUTILUS`) into:
- `ACC_NAUTILUS/ACC_FULL_COLUMN_v2`: current NAUTILUS surrogate (v2–v7: light models, GNN, coupling benchmarks); evaluations in `eval/`, reports in `docs/`
- `ACC_NAUTILUS/ACC_FULL_COLUMN`, `ACC_NAUTILUS/ACC_PARTIAL_COLUMN`: earlier versions (code only)
- `FTORCH`: Fortran/FTorch coupling pipeline for NEMO
- `NAUTIFLOWER`: Flower models (arXiv:2603.04430) on the NAUTILUS problem

Checkpoints, predictions, exported models and binary data are not included.
