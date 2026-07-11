# legacy/ — superseded tau10-lineage scripts (provenance only)

These scripts are **not part of the dm10 (tau18) reproduction** and are not imported by any active
path. They are kept only for provenance / history.

- `parallel_measure.py`, `combine_ens.py` — the tau10-lineage parallel measurement orchestrator and
  its aggregator. They hard-set `TAU_GABA=10` / `GNMDA=0.50` and resolve the old *tau10* checkpoints,
  so they **FATAL on the dm10 operating point**. The dm10 pipeline reproduces all 7 validations via
  `measure/run_all.sh` → `measure/val394_dm10_stage2.py` (the runner shim) instead; see the top-level
  `README.md` §5.

Nothing in the active dm10 reproduction (the top-level modules, `measure/`, `plots/`,
`mechanism_influence/`) imports anything here — verified by grep before quarantining.
