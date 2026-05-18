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

## dt-correctness fix (2026-05-16)

### What changed
Two integrator bugs caused a 146 ms TBW HW shift when running inference at
dt = 0.05 ms (n_substeps = 200) compared to the training-time dt = 0.1 ms
(n_substeps = 100). The pre-fix pipeline produced the correct paper values
at dt = 0.1, but the qualitative behavior at any other dt was unreliable.
The fix removes both root causes; the canonical dt = 0.1 pipeline produces
the same TBW/SBW values as the paper (within 1.8% for TBW, 3.9% for SBW)
and dt = 0.05 now matches dt = 0.1 within ~14% (a residual transient effect,
acknowledged as a known limitation — see "Known caveats" below).

### Root causes (both proven empirically; see `debug_dt/` for evidence)

1. **NMDA → I\_M bare-add bug** at `Training.py:2049` (and the inhibitory
   analogue at 2138). The substep integration was:
   ```
   self.I_M.mul_(decay_factor)   # decay_factor = 1 - dt/tau_syn
   ...
   self.I_M.add_(I_nmda)         # raw add — missing the dt or (1 - decay) factor
   ```
   Steady-state I\_M from NMDA = `I_nmda / (dt/tau_syn) = I_nmda × τ_syn/dt`,
   so reducing dt amplified the effective NMDA conductance by `τ_syn/dt = 25`
   at dt = 0.1 → 50 at dt = 0.05. The trained checkpoints absorbed this
   stoichiometric factor into the learned gain — the network was doing the
   right computation, only the calibration depended on dt.

2. **Conduction delays stored in substep counts** instead of physical ms.
   Eight attributes (`conduction_delay_a2msi`, `conduction_delay_v2msi`, …)
   indexed ring buffers in substep units. Halving dt while doubling
   n_substeps halved every physical delay (e.g. 25 ms → 12.5 ms for the
   a2msi path), collapsing the temporal-binding integration window at finer
   dt. This was the dominant residual after fix (1) was applied
   (~77% of the remaining 86 ms HW gap at dt = 0.05).

### Implementation (`dt_correct_nmda` flag gates the entire fix)

- New attribute `self.dt_correct_nmda` (default `True` as of 2026-05-16) at
  `Training.py:1368`. Set to `False` for bit-identical legacy reproduction.
- Per-step source factor computed once per `update_all_layers_batch` call at
  line 1910: `nmda_source_scale = 1 - exp(-dt / tau_syn)` (Form 2,
  step-source exp-Euler).
- Gated NMDA→I\_M add at lines 2059-2063 (excitatory) and 2153-2157
  (inhibitory): when flag is `True`, replaces bare add with
  `self.I_M.add_(I_nmda * nmda_source_scale)`. Steady-state I\_M from NMDA
  is now `I_nmda` regardless of dt — dt-invariant by construction.
- New physical-ms delay attributes (8 of them) initialised in `__init__`
  immediately before `_reset_delay_buffers()`:
  `conduction_delay_a2msi_ms`, `conduction_delay_v2msi_ms`,
  `conduction_delay_inA_inh_ms`, `conduction_delay_inV_inh_ms`,
  `conduction_delay_a2msi_inh_ms`, `conduction_delay_v2msi_inh_ms`,
  `conduction_delay_msi_inh2exc_ms`, `conduction_delay_msi2out_ms`.
  Each = `conduction_delay_X * self.dt` at construction (so existing
  checkpoints' physical delays are preserved).
- Helper `_delay_substeps_from_ms(ms)` returns
  `max(1, int(round(ms / self.dt)))`. Called from `_reset_delay_buffers()`
  and from the substep loop's local-aliases block (lines 1962-1981).
  When flag is `True`, both buffer sizing AND the per-substep ring-buffer
  index modulus use ms-derived substep counts — buffers automatically
  re-size when dt is changed at evaluation time, provided the caller
  invokes `net._reset_delay_buffers()` after the change.
- When flag is `False`, behaviour at every site is bit-identical to the
  pre-2026-05-16 code (verified via fixed-seed regression on M00 dt = 0.1,
  HW = 149.06 ms ± 0).

### gNMDA recalibration (0.05 → 1.30)

The fix removes the implicit `τ_syn/dt = 25` amplification at dt = 0.1.
A coarse + fine sweep on M00 fixed-seed identified `gNMDA = 1.30` as the
calibration point that exactly preserves the dt = 0.1 control HW
(148.91 ms vs 149.06 ms legacy, Δ = −0.15 ms). Theory predicted ~1.25;
empirical 1.30 is within the calibration noise band of ±5 ms.

The NMDA perturbation conditions in `generate_all_fresh.py:CONDITIONS`
are rescaled by the same ×25 factor to preserve the perturbation magnitude
ratios:

| Condition       | Pre-fix gNMDA | Post-fix gNMDA |
|-----------------|---------------|----------------|
| control         | 0.05          | **1.30**       |
| nmda            | 0.02          | **0.50**       |
| nmda_increase   | 0.20          | **5.00**       |

`run_training` (`Training.py:3525`) is updated so newly trained checkpoints
bake in `gNMDA = 1.30` directly.

### Verification

Full pipeline `debug_dt/stage_e_full_pipeline.py` — 10 models × 50 trials
× 5 conditions × (TBW + SBW), dt = 0.1:

**TBW HW (ms)** — fix vs paper

| Condition       | Post-fix | Paper | % off |
|-----------------|----------|-------|-------|
| control         | 107.2    | 107   | 0.2   |
| ff_inhibition   | 145.7    | 146   | 0.2   |
| adaptation      | 215.2    | 216   | 0.4   |
| nmda            | 93.2     | 94    | 0.8   |
| nmda_increase   | 109.9    | 108   | 1.8   |

**SBW HW (°, floor 0.0681 subtracted)** — fix vs paper

| Condition       | Post-fix | Paper | % off |
|-----------------|----------|-------|-------|
| control         | 23.63    | 24.3  | 2.7   |
| ff_inhibition   | 27.78    | 27.7  | 0.3   |
| adaptation      | 29.38    | 29.4  | 0.1   |
| nmda            | 12.58    | 13.1  | 3.9   |
| nmda_increase   | 30.09    | 30.0  | 0.3   |

Rank ordering preserved on both axes; max deviation 1.8% (TBW) / 3.9% (SBW).

**dt-independence (M00 fixed-seed, freeze_g_FFinh, control):**

| Configuration                  | dt = 0.1 | dt = 0.05 | Δ HW    |
|--------------------------------|----------|-----------|---------|
| Pre-fix (legacy bug)           | 149.06   | 295.79    | +146.74 |
| Post-fix (Form 2 + delays-ms)  | 148.91   | 127.84    |  −21.07 |

dt = 0.05 residual (~14%) is the brief-event nmda_m transient effect —
50-ms input events vs τ\_nmda = 80 ms means the NMDA gating variable
never reaches steady state, so the integrator-form difference matters
slightly. Multiple alternative integrator substitutions for
`nmda_m` / `ampa_m` / `I_M` decays were tested by the debugger (see
`debug_dt/` task #17 series); all moved HW the wrong direction. Accepted
as a known limitation per the user's qualitative-preservation goal.

### Re-training note for existing checkpoints

The 10 checkpoints in `checkpoint/msi_model_surr_10_{00..09}.pt` were
saved with `gNMDA = 0.05` baked into `mutable_hparams` (the legacy
calibration). Running `generate_all_fresh.py` on those checkpoints
without modification produces the wrong control gNMDA. The recommended
workflow is to re-train via `run_training` (which now sets
`gNMDA = 1.30`); for ad-hoc inference on existing checkpoints, override
manually with `net.gNMDA = 1.30` before measurement. The Stage E
verification above used such an override.

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
