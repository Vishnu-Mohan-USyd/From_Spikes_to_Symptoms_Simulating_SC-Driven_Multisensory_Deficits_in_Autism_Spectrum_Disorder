# GPU Performance Speedup Summary

## Scope
This branch speeds up the original training and assay code while preserving the network's observable behavior closely enough for SBW/TBW evaluation and legacy checkpoint reuse.

Baseline reference used during this work:
- legacy code/worktree: `/home/vishnu/coding_proj/cl2/_perf_baseline_worktree`
- representative legacy checkpoint: `checkpoint/msi_model_surr_10_00.pt`

## Main code changes

### 1. Hot-path simulator cleanup
Files:
- `Training.py`

Changes:
- Made probe logging opt-in (`net.enable_probe = False` by default) so training does not carry probe overhead unless explicitly enabled.
- Moved spike debug counters onto device tensors to avoid repeated GPU→CPU synchronization.
- Kept the main substep simulator on GPU and reduced per-substep overhead in `update_all_layers_batch(...)`:
  - reused cached weights in frozen eval,
  - used `F.linear(...)` for synaptic projections,
  - used mask-based spike reset/update paths,
  - removed unnecessary host-visible bookkeeping from the hot loop.
- Converted the STDP weight update path to direct non-negative weights instead of `torch.nn.utils.parametrize`, and added a legacy checkpoint translation path so old `parametrizations.*.original` checkpoints still load cleanly.

### 2. Evaluation-path cleanup
Files:
- `Training.py`
- `SBW_test.py`
- `TBW_test.py`

Changes:
- Switched main assay/evaluation paths to `torch.inference_mode()`.
- Removed `torch.cuda.empty_cache()` calls from hot loops in SBW/TBW workflows.
- Preserved legacy checkpoint loading and the existing assay scripts.

### 3. Vectorized stimulus generation
Files:
- `Training.py`

Changes:
- Rewrote `generate_av_batch_tensor(...)` to generate the full `(batch, time, neuron)` analog input tensors in batched tensor form instead of looping over every batch item and time step and creating many tiny GPU tensors.
- Gaussian input construction is now broadcast over all active frames in one shot.
- This is the largest single throughput win for full training runs.

### 4. Better GPU occupancy in canonical training
Files:
- `Training.py`

Changes:
- Changed `run_training(...)` default `batch_size` from `256` to `1000`.
- Ensured the unsupervised phase actually uses the passed `batch_size` inside `train_unsupervised_batch(...)`.
- This reduces the number of Python/simulator passes per epoch from 4 mini-batches to 1 mini-batch for the canonical `1000`-sequence epoch.

## Measured speedups

### Microbenchmarks from the first optimization pass
- `update_all_layers_batch`, batch `32`:
  - baseline: `741.37 ms`
  - optimized: `200.28 ms`
  - speedup: `3.70x`

- `train_unsupervised_batch(64, batch_size=32)`:
  - baseline: `35.52 s`
  - optimized: `10.76 s`
  - speedup: `3.30x`

- deterministic TBW subset wall time:
  - baseline: `20.02 s`
  - optimized: `13.97 s`

- deterministic SBW trace wall time:
  - baseline: `4.42 s`
  - optimized: `3.78 s`

### Full-epoch training benchmark after the deeper pass
Canonical epoch workload here means `train_unsupervised_batch(1000, batch_size=...)` with the original `n=180`, `n_substeps=100`, and the original training mechanisms intact.

- before large-batch/vectorized-input pass (`batch_size=256`):
  - measured epoch-like runtime: `23.019 s`

- after vectorized input generation + `batch_size=1000`:
  - measured epoch-like runtime: `5.368 s`

### Real 80-epoch run
Command used:
```bash
cd /home/vishnu/coding_proj/cl2/_master_figs_worktree
MPLBACKEND=Agg PYTHONUNBUFFERED=1 /usr/bin/time -f 'wall_s=%e' python -u - <<'PY' 2>&1 | tee perf_artifacts/train80_batch1000_full.log
from Training import run_training
run_training(n_unsup_epochs=80)
PY
```

Observed result:
- total wall time: `431.76 s` (`7.20 min`)
- the 80-epoch training block stayed under the requested `10 min` target
- a post-training evaluation/reset bug still causes the wrapper to crash after saving the checkpoint
- saved checkpoint from that run: `checkpoint/msi_redone_agc_fix_.pt`

## Validation against baseline behavior

### Legacy checkpoint compatibility
- Old checkpoints containing only `parametrizations.*.original` keys load successfully on the direct-weight path.
- Example result for `checkpoint/msi_model_surr_10_00.pt`:
  - missing keys: `0`
  - unexpected keys: `0`

### Deterministic SBW check
Command used:
```bash
python sbw_gpu_trace_disparity.py --ckpt checkpoint/msi_model_surr_10_00.pt --disparity-deg 80 --duration 5 --intensity 1.0 --bg-lambda 0.0 --seed 0 --max-tries 1 --target any --out /tmp/sbw_opt_latest.npz
```
Baseline and optimized code produced the same qualitative readout for the fixed checkpoint/trial:
- `last = two_peak_deep`
- `full = single_peak`

### Deterministic TBW subset check
Using `tbw_offset_explain.compute_tbw_readouts(...)` on offsets `[-40, 0, 40]`:
- `fusion_full` max diff: `0.00568`
- `fusion_last` max diff: `0.0`
- `int_full` max diff: `31`
- `int_last` max diff: `6`

## Important caveat
The new vectorized `generate_av_batch_tensor(...)` is bit-identical to the legacy version when `noise_std = 0`.
With nonzero training noise, the exact analog stimulus tensors are not bitwise identical because CUDA RNG draw ordering changes when noise is generated in batched form rather than many tiny sequential draws.

That means:
- fixed-checkpoint evaluation behavior is preserved to tight tolerance,
- but stochastic training trajectories are not expected to be bitwise replay-identical.

## Outstanding issue
`run_training(...)` still needs one cleanup fix in the post-training evaluation path:
- current failure: inference-tensor reset-state interaction after training/evaluation handoff
- location at failure time: `Training.py:1705` during the post-training `reset_state()` path

This does **not** invalidate the measured training speedup or the saved checkpoint, but it should be fixed before treating the full wrapper as production-clean.
