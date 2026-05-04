# CHANGES — Optimized TBW/SBW/E-I Pipeline

## Overview

This repository implements a Superior Colliculus (SC)-driven multisensory binding
model used to investigate Autism Spectrum Disorder (ASD)-relevant deficits.
The pipeline characterises each trained network along three axes:

- **TBW** — Temporal Binding Window: the half-width (HW) of the P(fusion)
  curve as a function of audio–visual onset asynchrony.
- **SBW** — Spatial Binding Window: the half-width of the P(fusion) curve
  as a function of audio–visual spatial separation.
- **E/I** — Excitation-to-Inhibition ratio, measured from separated synaptic
  current traces over the evoked response window.

All measures are computed across 5 conditions (control + 4 perturbations) and
pooled over 10 trained model checkpoints (`checkpoint/msi_model_surr_10_{00..09}.pt`)
to give per-model means and bootstrap-style summaries.

## Pipeline architecture

- **Entry point:** `generate_all_fresh.py` — runs all 10 figures from scratch
  (10 models × 50 trials × 5 conditions) and writes both per-condition `.npz`
  caches in `cache/` and the final SVG/PNG figures in `Saved_Images/`.
- **TBW** — `fusion_method='temporal_fusion'` classifier inside
  `TBW_test.run_fusion_across_models`. Offsets span **−500 to +500 ms in 20 ms
  steps** (51 values). Per-model P(fusion) curves are stacked and a pooled
  psychometric fit produces the canonical HW; per-model HWs are also retained.
- **SBW** — enhancement metric: `AV_spikes − max(A_spikes, V_spikes) > 10`
  threshold, evaluated by `SBW_test.run_spatial_binding_across_models` with
  `method="enhancement"`, `enhancement_threshold=10.0`. Separations span
  **−80° to +80° in 5° steps** (33 values). Post-processing:
  1. `subtract_control_floor` removes the asymptote estimated from the control
     condition's far-separation tails.
  2. A **symmetric pedestal fit** (`fit_sbw_pedestal` in
     `replot_all_cosmetic.py`) is applied; HW is read off the 0.5 crossing.
- **E/I** — `run_ei_balance.py` calls the separated-currents probe to record
  AMPA + NMDA (excitatory) and FFInh + RecurInh + LatInh (inhibitory) traces,
  then averages over an **evoked window of 125 ms** (pulse 50 ms + 75 ms
  tail, following Haider 2013).

## Conditions

| Name            | Manipulation                                                                  |
|-----------------|-------------------------------------------------------------------------------|
| control         | none                                                                          |
| ff_inhibition   | `pv_nmda=0.8`, `targ_ratio=0.8` (AGC setpoint reduction)                      |
| adaptation      | `aM=0.001`, `bM=0.2`, `cM=-60`, `dM=0.01` (Izhikevich adaptation altered)     |
| nmda            | `gNMDA=0.02` (≈4× reduction from baseline)                                    |
| nmda_increase   | `gNMDA=0.2` (≈4× increase from baseline)                                      |

## Latest results

10 models × 50 trials, branch `investigate-ffinh-sbw-narrowing`:

**TBW half-width (ms)**

| Condition       | HW (ms) | Δ vs control |
|-----------------|---------|--------------|
| control         | 107     | —            |
| ff_inhibition   | 146     | +39          |
| adaptation      | 216     | +109         |
| nmda            | 94      | −13          |
| nmda_increase   | 108     | +1           |

**SBW half-width (°), floor 0.0683 subtracted**

| Condition       | HW (°) | Δ vs control |
|-----------------|--------|--------------|
| control         | 24.3   | —            |
| ff_inhibition   | 27.7   | +3.4         |
| adaptation      | 29.4   | +5.1         |
| nmda            | 13.1   | −11.2        |
| nmda_increase   | 30.0   | +5.7         |

**E/I ratio:** 1.042 ± 0.005

## Sensitivity analyses

- **10-model setpoint sweep (TBW HW):**
  - control: **212.4 ± 4.9 ms**
  - `targ_ratio = 0.8`: **291.0 ± 6.0 ms** (+78.6 ms, 10/10 models widened)
  - `targ_ratio = 0.5`: **300.8 ± 5.7 ms** (+88.5 ms, 10/10 models widened)
- **Numerical step-size sensitivity (M00 control, TBW only):**
  - Trace correlation between dt = 0.1 ms and dt = 0.05 ms: Pearson r = 0.974,
    NRMSE = 0.097.
  - TBW HW shifts from 215.8 ms (dt = 0.1, the training step size) to 256.5 ms
    (dt = 0.05). Networks were trained at dt = 0.1, so the canonical pipeline
    keeps dt = 0.1; the smaller step is a robustness check, not a target.
  - Output: `Saved_Images/TBW_stepsize_sensitivity.{svg,png}`.

## Key files

- `generate_all_fresh.py` — canonical entry point; produces all 10 TBW + SBW
  figures and their `.npz` caches in one run.
- `TBW_test.py` — `run_fusion_across_models`, `compute_tbw_temporal_fusion_persep`,
  `is_temporally_fused`, `_find_crossings`, `load_msi_model`.
- `SBW_test.py` — `run_spatial_binding_across_models`,
  `spatial_binding_diagnostics`.
- `replot_all_cosmetic.py` — `plot_tbw`, `plot_sbw`, `fit_sbw_pedestal`,
  `subtract_control_floor`, `_load_pooled`. Cosmetic style: data #939598 grey
  with black edge; perturbation fit #b469a3 mauve solid; control overlay
  #39b54a green dashed.
- `replot_tbw_from_cache.py` — re-renders the four canonical TBW SVGs/PNGs
  from per-model `.npy` cache without simulating.
- `run_ei_balance.py` — produces `Saved_Images/EI_balance.{svg,png}` from
  separated synaptic currents over the evoked window.
- `run_nmda_increase.py` — standalone TBW + SBW for the `nmda_increase`
  condition (kept for incremental re-runs).
- `Training.py` — `MultiBatchAudVisMSINetworkTime` definition and training
  loop; all test harnesses import from here.

## Known caveats

- **ff_inhibition does not narrow SBW under the enhancement metric.** Under
  the canonical pipeline (10 models × 50 trials, enhancement threshold = 10,
  pedestal fit), the `ff_inhibition` perturbation **widens both TBW and SBW**.
  Re-running with the alternative `is_fused` classifier at `intensity=0.5`
  yields only ~1° of SBW narrowing — within fit noise. The strong SBW
  narrowing reported in the previous paper used `intensity=1` together with
  fits whose R² fell below 0.8; that result is not robustly reproducible
  at clean intensities. The branch name `investigate-ffinh-sbw-narrowing`
  reflects an open investigation of this discrepancy.
- **E/I metric is the separated synaptic decomposition.** This replaces the
  earlier sign-based `clamp(I_M)` heuristic. The new decomposition is more
  biologically accurate (separating AMPA + NMDA from FFInh + RecurInh +
  LatInh) but can produce numerically different ratios than the old metric;
  comparisons against earlier figures should account for this change.
- **Cache filename mismatch.** `generate_all_fresh.py` writes SBW caches as
  `sbw_{cond}_t10.npz`. The `sbw_{cond}_final.npz` files (mod time 2026-03-30
  08:50–08:55) come from a separate runner and are not consumed by the
  canonical pipeline. Use `_t10.npz` for re-plotting.
- **dt = 0.1 ms is the training step size.** The dt = 0.05 ms sweep is a
  numerical-stability check; do not switch the canonical pipeline to dt = 0.05
  without re-training.
