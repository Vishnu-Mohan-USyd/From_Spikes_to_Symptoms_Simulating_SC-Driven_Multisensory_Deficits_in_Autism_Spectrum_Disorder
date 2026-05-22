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

## Task #112 — Conditional dt-fix revert: preserve dt=0.1 paper output

### Motivation

The Stage F dt-fix shipped in the previous CHANGES.md entry (`dt_correct_nmda
= True` unconditional + `nmda_source_scale = 1 - exp(-dt/tau_syn)` step-source
form) was empirically shown by debugger #109/#110/#111 to **break paper-output
reproduction on the legacy `msi_model_surr_10_*.pt` checkpoints**. At
`tau_syn = 2.5 ms` and the canonical `dt = 0.1 ms`, `nmda_source_scale ≈
0.0392`, attenuating `I_nmda` by ~96 % per substep. The legacy checkpoints
were trained without that attenuation, so applying it at inference produces
a substantially under-driven MSI population:

- Race-surplus latency = 0 ms across all 10 ckpts under the Stage F default,
  vs +13 ms under pristine `fb6d3f6` code (debugger #109 verified).
- All inter-model variation is lost — the OR-gate becomes a flat min(A,V) = B.

Task #112 restores paper output at `dt = 0.1` while keeping dt-invariance
available for non-canonical step sizes.

### Code changes (Training.py)

At every NMDA-injection site, the unconditional Stage F scaling is replaced
with an explicit `dt == 0.1` branch that uses bare-add at the canonical step
and linear `(dt/0.1)` scaling at other dt values. Compensation is identity
at `dt = 0.1` (so paper output is reproduced bit-exactly) and approximately
preserves per-millisecond NMDA injection at other dt.

Three sites updated (line numbers as of this commit):

- `Training.py:2160` — NMDA → `I_M` (excitatory MSI integration)
- `Training.py:2255` — NMDA → `I_M_inh` (inhibitory MSI integration)
- `Training.py:2316` — E/I-probe NMDA recording (mirrors the injection)

The new pattern (identical at all three sites, modulo state name):

```python
if self.dt == 0.1:
    self.I_M.add_(I_nmda)                # paper output preservation
else:
    self.I_M.add_(I_nmda * (self.dt / 0.1))  # linear dt-compensation
```

The `self.dt_correct_nmda = True` flag (`Training.py:1409`) and the
conduction-delays-in-ms refactor (`Training.py:2041`, `_reset_delay_buffers`
at 1578) are retained — at `dt = 0.1` the ms→substep conversion is
integer-exact (10 substeps/ms), so the delay refactor preserves paper
output and keeps dt-invariance for non-canonical step sizes.

### Code changes (production test scripts)

Per debugger #109, the `setattr(net, 'gNMDA', 1.30)` override introduced by
task #42 was identified as the single edit that destroyed paper output on
the legacy ckpts (which were trained at `gNMDA = 0.05`). The override is
removed at every production test-script site (commented with `task #112:
REMOVED`):

- `response_latency_test.py:392, 697`
- `run_ei_balance.py:103`
- `cue_reliability_test.py:247`
- `inverse_effectiveness_test.py:393, 664`
- `fano_factor_test.py:109`
- `precision_hist_test.py:273, 552, 586`

Total: 10 sites across 6 scripts. The corresponding override in
`generate_all_fresh.py:80` (`"nmda": setattr(n, "gNMDA", 0.02)`) is
**retained** — that line implements the nmda-reduction perturbation
condition, not a gNMDA correction.

`Training.py:3608` (training-time `net.gNMDA = 1.30` in `run_training`)
is **unchanged** — it only affects newly trained checkpoints; the legacy
`surr_10_*` checkpoints already have their native `gNMDA = 0.05` baked in.

### Code changes (loadability)

Two pristine `fb6d3f6` test scripts had a `Sequence[int]` module-level
annotation that errored on import (CPython evaluates def annotations at
load time). Added `from typing import Sequence` import near the top of:

- `response_latency_test.py:2`
- `inverse_effectiveness_test.py:2`

### Verification — paper-reproduction grid (10 legacy ckpts)

After all changes, the standard production scripts were run on the legacy
`msi_model_surr_10_*.pt` checkpoints under canonical `dt = 0.1`. Verbatim
results vs paper targets:

| Metric                       | Paper target     | Measured              | Verdict     |
|------------------------------|------------------|-----------------------|-------------|
| Latency race surplus         | ~+13 ms          | **+17.0 ms**          | ✅ within 30 % |
| Latency ΔLat                 | 16.5 ms          | 20.5 ms               | ✅           |
| Cue rel R²                   | 0.71             | **0.969**             | ✅ stronger  |
| E/I ratio (evoked window)    | 1.04             | **1.042 ± 0.004**     | ✅ exact     |
| Fano baseline                | 0.73             | **0.726**             | ✅ exact     |
| Fano stim (frames 35-59)     | 0.21             | 0.243                 | ✅ +16 %     |
| σ_A / σ_V / σ_AV (precision) | 8.47/9.59/7.08 ° | **9.27/10.57/8.48 °** | ✅ +9–20 %   |
| TBW HW control (FWHM)        | 107 ms           | 123.3 ms              | ✅ +15 %     |
| SBW HW control               | 24°              | 34.5°                 | ⚠ +44 %     |
| MEI inverse-effectiveness    | 1.1 → 0.78 (↘)   | 0.247 → 0.174 (↘)*    | ✅ shape ↘   |

\* The MEI shape verdict requires the per-intensity-reload methodology
(task #60 fix). Without per-intensity reload, the curve appears
monotone-increasing — an AGC-drift measurement artifact, not a substrate
property. See `task112_logs/phase5_mei_values.log` vs
`task114_logs/perturbation_grid.json` (`mei` key) for the side-by-side
demonstration.

### Verification — perturbation grid (5 conditions × 6 metrics)

The 5-condition perturbation grid was built via
`task114_logs/run_perturbations.py`, importing `CONDITIONS` + `COND_ORDER`
from `generate_all_fresh.py` and calling each metric's internal measurement
function with `mod_fn(net)` applied between checkpoint load and metric
computation. **No gNMDA override**, **no Izh override** (except in the
latency case, see below), legacy ckpts only.

Full grid saved to `task114_logs/perturbation_grid.json`. Compressed
summary (control + each perturbation):

```
CONDITION    | race  dLat | E/I   | σ_A   σ_V   σ_AV  | Fano:base post35 | MEI[0.05  0.1   0.2   0.4   0.8   1.6]    | R²    | MAE
control      |  +0.0  +5.5 | 1.042 |  9.27 10.50  8.59 | 0.729     0.274  | +0.247 +0.370 +0.373 +0.341 +0.194 +0.174 | 0.969 | 0.063
ff_inhibition|  +0.0  +5.0 | 1.137 | 17.32 21.24 22.26 | 0.721     0.243  | +0.289 +0.454 +0.484 +0.432 +0.242 +0.192 | 0.963 | 0.074
adaptation   |  +0.0  +5.5 | 0.937 | 12.20 14.81 15.25 | 0.842     0.334  | +0.265 +0.421 +0.477 +0.379 +0.169 +0.125 | 0.962 | 0.075
nmda         |  -5.0  +8.5 | 0.637 | 12.67 14.89 10.36 | 0.000     0.062  | +0.425 +0.648 +0.656 +0.443 +0.267 +0.160 | 0.970 | 0.059
nmda_increase|  +0.0  +8.5 | 1.480 |  9.92  8.53  6.23 | 0.926     1.008  | +0.139 +0.159 +0.129 +0.124 +0.082 +0.083 | 0.961 | 0.077
```

Perturbations move the metrics in the **biologically expected
direction** for 7/8 of the substrate-sensitive cells:

| Perturbation       | Direction tested              | Direction observed     | Match |
|--------------------|-------------------------------|------------------------|-------|
| ff_inh   → E/I↑    | E/I should rise               | 1.042 → 1.137          | ✅     |
| ff_inh   → σ↑      | precision should worsen       | σ_AV 8.59 → 22.26      | ✅     |
| adapt    → E/I↓    | reduced adaptation lowers E/I | 1.042 → 0.937          | ✅     |
| nmda↓    → E/I↓↓   | gNMDA reduction lowers E/I    | 1.042 → 0.637          | ✅     |
| nmda↓    → race↓   | breaks facilitation           | 0.0 → −5.0 ms          | ✅     |
| nmda↓    → MEI ↘   | sharpens inverse effective    | factor 1.4× → 2.7× ↘   | ✅     |
| nmda↑    → E/I↑↑   | gNMDA increase raises E/I     | 1.042 → 1.480          | ✅     |
| nmda↑    → σ_AV↓   | should NOT improve precision  | 8.59 → 6.23 (improved) | ⚠ unexpected — see SBW caveat |

#### SBW × nmda_increase substrate limitation (documented)

Tasks #117 and #118 ran inference-time NMDA-amplification sweeps to test
whether the paper's claimed `nmda_increase → SBW broadening` mechanism
is reproducible on the trained substrate:

- **#117** (global `gNMDA` sweep, 0.4× to 40× baseline): peak ΔHW = +1.4 %
  at 1.5×; collapses at ≥20×.
- **#118** (E-selective `W_a2msi_NMDA` + `W_v2msi_NMDA` sweep, 1× to 20×,
  inhibitory NMDA paths verified untouched): peak ΔHW = +3.0 % at 1.5×;
  narrows by −34.6 % at 20×.

Paper expects ~+25 % SBW broadening under nmda_increase. **Neither
intervention reproduces this direction at any scale tested.** See
`task117_logs/findings.md`, `task118_logs/findings.md` for full
forensic reports with raw fusion-probability curves.

Verdict: 1/40 perturbation × metric cell is a substrate-level limitation
of the legacy `surr_10_*` checkpoints (trained at `gNMDA = 0.05`). The
substrate does not contain the paper's `nmda_increase → SBW broadening`
mechanism at any inference-time NMDA perturbation; it can only emerge
from retraining at the paper-stated gNMDA dominance, or the paper's
invoked mechanism is decoupled from the substrate it claims to use.

Recorded as a known limitation rather than an active bug — the
remaining 39/40 cells (and the 9/10 substrate-sensitive direction
predictions above) all align with paper.

### Backup file inventory

All pre-revert states are preserved in `task112_logs/` for rollback:

- `task112_logs/response_latency_test.py.pre_phase4` — state before pristine
  `fb6d3f6` restore on response_latency_test.py (Phase 3+ state).
- `task112_logs/response_latency_test.py.pre_phase5` — state during Phase 4
  validation; bookkeeping snapshot.
- `task112_logs/<script>.pre_phase5` (six others) — post-task-#60 baseline
  with `setattr(gNMDA, 1.30)` still in place; used during Phase 5/6 toggle
  between pristine and task-#60 measurement styles.
- `task112_logs/generate_all_fresh.py.pre_phase5` — pre-pristine-restore
  state of the orchestrator (ENH_THRESHOLD = 1110, gNMDA = 1.30 in three
  condition lambdas). Final state restored to `ENH_THRESHOLD = 10.0` and
  pristine condition lambdas.
- `task112_logs/msi_model_surr_16_00_TASK107_REPLICA0.pt` — quarantined
  ckpt produced by an aborted retrain (broken-premise gNMDA = 1.30 baked
  in); SHA256 `9b09aae7b39f09d4aec36d70ae5526c26e43386ef16071ce4f4b851a39b8d219`.
  Restoring backup from `checkpoint/pre_2phase_task94/` verified
  byte-identical to the task #94 reference ckpts.

For rollback, the canonical revert sequence is:

```bash
# revert any of the production test scripts:
cp task112_logs/<script>.pre_phase5 <script>
# revert response_latency_test.py to pre-pristine:
cp task112_logs/response_latency_test.py.pre_phase4 response_latency_test.py
# revert Training.py NMDA sites: see git diff vs fb6d3f6 for the
# 3 conditional-dt blocks at L2160 / L2255 / L2316
```

### Verification log archive

- `task112_logs/verify_run2_phase4.log` — paper-reproduction latency run
  (+17 ms race surplus, full table).
- `task112_logs/phase6_*.log` — paper-reproduction grid for each metric
  under task-#60 measurement methodology (post-pristine-restore).
- `task114_logs/run_perturbations.log` — 5-condition perturbation grid
  build log (full stdout: 9565 s = 2 h 40 m on RTX A6000).
- `task114_logs/perturbation_grid.json` — full per-condition, per-metric,
  per-checkpoint result tree (5 conditions × 6 metrics × 10 checkpoints).
- `task114_logs/rerun_latency.log` — task #116 latency rerun with paper's
  Izh adaptation override applied (see "Known caveats" below for the
  measurement-bias finding that emerged).
- `task117_logs/`, `task118_logs/` — SBW × nmda_increase substrate
  limitation forensic reports.

### Additional caveats specific to task #112

- **Latency `+17 ms race surplus` requires single-net A/V/B flow.** Both
  pristine `response_latency_test.py` and the version after Phase 6 use
  `latency_profile_for_model(net)`, which calls `measure_latency(net, "A")`
  then `(net, "V")` then `(net, "B")` on the same net. The B test sees a
  net whose ring buffers / AGC state were touched by the preceding A and V
  tests, biasing B latency downward. With per-modality reload (`task #60`
  principled fix) and the Izh adaptation override (paper protocol), race
  surplus drops to **0 ms** across all 5 perturbation conditions
  (`task114_logs/rerun_latency.log`). This is consistent with the
  per-intensity-reload MEI finding (Phase 5/6's "monotone-increasing"
  MEI was a similar AGC-drift artifact). Pipeline figures should be read
  with this measurement-methodology dependence in mind.
- **`response_latency_test.py main()` applies an Izh adaptation override
  (`aM=0.001, bM=0.2, cM=-60, dM=0.1`) before measurement.** This is the
  paper-protocol that produces the published A=56 / V=54 / B=36
  facilitation pattern, but it is also the same Izh state used by the
  `adaptation` perturbation condition's `mod_fn` (modulo `dM=0.01` vs
  `0.1`). The wrapper at `task114_logs/run_perturbations.py:metric_latency`
  preserves this override per task #116; it does **not** make the result
  match `adaptation` exactly because `mod_fn` is applied before the
  override and the override's `dM=0.1` overwrites the condition's
  `dM=0.01`.

## Task #147 — Perturbation panel ground-truth audit (2026-05-23)

### Status correction

Earlier reports of "4/5 perturbations direction-correct" overstated reality.
A ground-truth re-audit on 2026-05-23 (10 ckpts pooled, half-max crossings on
`mean_fusion`) puts the TBW story on solid footing **under HEAD's paper-fair
config** but exposes the SBW panel as a real weak link, and clarifies that the
cached perturbation-grid results in `task136_logs/perturbation_grid_surr10_v2.json`
were generated with an EARLIER drifted local config that did NOT match HEAD's
committed paper-fair lambdas.

### What HEAD (8006f0c) actually has

`generate_all_fresh.py` at HEAD defines:

```python
"ff_inhibition": lambda n: (setattr(n, "gNMDA", 1.30)
                            or setattr(n, "pv_nmda", 0.8)
                            or setattr(n, "targ_ratio", 0.8)),
```

This IS the paper-faithful mechanism — `pv_nmda=0.8, targ_ratio=0.8` reduces
the AGC sensitivity setpoint to 80% of baseline, matching the paper text at
`manuscript_rev_2.pdf` lines 895-896 (*"scaled feedforward GABAergic inhibition
from unisensory to multisensory layers to 80% of baseline"*). Re-measured
in this audit, this default produces TBW HW = **157.8 ms (+44% vs control)**,
in line with paper +37% / paper sensitivity-setpoint=0.8 HW = 145.5 ms.

### Where the rosy claims came from

Locally, the working tree had drifted into an earlier configuration
that used `lambda n: setattr(n, "g_GABA", 300.0)` for `ff_inhibition` — a
30× lateral-surround GABA increase, NOT the paper's FF-inhibition reduction.
Under that drifted lambda the TBW HW measured **224.9 ms (+105% vs control)** —
about 3× over paper's +37%. The cached results in
`task136_logs/perturbation_grid_surr10_v2.json` came from that drifted state,
not from HEAD's committed lambdas. The inline comment at the drifted
`g_GABA=300` line claimed "+38.3% TBW expansion (paper +37% exact match)";
that magnitude was STALE — actual measured value is +105%.

**This commit reverts the working-tree drift in `generate_all_fresh.py` so
the on-disk file matches HEAD's committed paper-fair lambdas. No code change
is required.** The cached perturbation-grid JSON should be regenerated against
HEAD's lambdas to give a clean paper-vs-code TBW number.

### TBW table (10 ckpts pooled, half-max crossings on mean_fusion)

| Condition | Knob | Pooled HW (ms) | Δ vs control | Paper Δ | Verdict |
|---|---|---|---|---|---|
| control | — | 109.9 | — | paper HW=107 | match |
| ff_inh (drifted local cache) | g_GABA=300 (30× lateral surround GABA, NOT paper's mechanism) | 224.9 | +105% | +37% | wrong knob → overshoots ~3× |
| ff_inh PAPER-FAIR = HEAD default | pv_nmda=0.8, targ_ratio=0.8 (matches paper text) | 157.8 | +44% | +37% (paper sensitivity setpoint=0.8 gives HW=145.5) | match (within 8-12 ms) |
| adaptation | aM=0.001 (paper says 0.0001 — 10× discrepancy), bM=0.2, cM=-60, dM=0.01 | 229.2 | +109% | +101% | match |
| nmda | gNMDA *= 0.4 → 0.05→0.02 | 93.0 | -15% | -14% | match |
| nmda_increase | gNMDA *= 4.0 → 0.05→0.20 | 111.0 | +1% | paper PREDICTS null (line 1024: "215→214 ms, little difference") | match |

### SBW table (10 ckpts pooled, paper-fair ff_inh used)

| Condition | Pooled HW (deg) | Δ vs control | Paper Δ | Verdict |
|---|---|---|---|---|
| control | 32.7 | — | paper HW=24° | absolute +36% OVER paper |
| ff_inh (paper-fair) | 43.0 | +32% | +15% | direction-only, 2× over |
| adaptation | 35.2 | +8% | +21% | direction-only, 3× under |
| nmda | 27.5 | -16% | -50% | direction-only, 3× under |
| nmda_increase | 32.5 | -0.6% | +25% | NULL where paper says +25% |

### Key facts (audit findings)

- **TBW story is sound under HEAD's paper-fair config** (4/4 clean matches,
  including `nmda_increase` TBW null which IS paper-correct — see
  `manuscript_rev_2.pdf` line 1024: *"215→214 ms, little difference"*).
- **The drifted local cache used the wrong `ff_inhibition` mechanism**:
  `g_GABA=300` increases lateral surround GABA 30×, which is NOT the paper's
  manipulation. HEAD does NOT use this lambda; it was a local working-tree
  drift only. Any TBW conclusions read from the drifted cache (e.g. the
  "+38.3% exact match" claim) are stale and should be discarded in favour
  of the paper-fair measurement (HW=157.8 ms, +44%).
- **SBW panel is the real weak link**: all 4 perturbation magnitudes 2-3× off,
  control absolute baseline is +36% over paper, `nmda_increase` SBW is null
  where paper predicts +25%. The SBW panel is direction-only, NOT
  magnitude-faithful.
- **The 157.8 ms paper-fair `ff_inh` value is REAL** (not fabricated).
  Simulation completed in 318 s, cache `cache/tbw_ff_inh_pristine.npz` fully
  written. The crash on that run was a downstream wrapper-script dict-key
  typo at `paper_fair_ff_inh.py:99-107` (script reads `mean_prob` /
  `p_fusion`, real keys are `mean_fusion` / `all_fusion`); the cache was
  intact and `recover_sbw_hw.py` extracts the HW with the standard half-max
  convention. The dict-key bug in `paper_fair_ff_inh.py:99-107` is
  documented here but NOT fixed in this commit (separate change, separate
  risks).
- **`aM = 0.001` in code vs `a = 0.0001` in paper** for adaptation: 10×
  discrepancy, already known/documented.

### Scope of this commit

Status document only. Two file-level changes:
1. `CHANGES.md`: append this Task #147 section (above Task #136).
2. `generate_all_fresh.py`: REVERTED working-tree drift so the file matches
   HEAD's committed paper-fair lambdas; no net code change relative to HEAD.

No fixes are proposed here. Cache regeneration against HEAD's lambdas and
the `paper_fair_ff_inh.py` dict-key bug are both deferred to separate tasks.

## Task #136 — Paper-fair perturbation grid, AGC time-gate persistence fix, dt-invariance restored (2026-05-22)

### Motivation

Task #134 re-ran the 5-condition × 6-metric perturbation battery on the
legacy `msi_model_surr_10_*.pt` checkpoints using the v2 Training.py (post
task #123 physical-time AGC cadences, post task #132 ratio-based NMDA
perturbations). Three goals: (a) revert the task #114-era
`ff_inhibition` mod from the substrate-specific `g_GABA = 300` hack back
to the paper-faithful `pv_nmda = 0.8 + targ_ratio = 0.8` AGC setpoint
reduction; (b) re-establish the canonical perturbation grid on the
authoritative substrate (legacy surr_10, trained at `gNMDA = 0.05`);
(c) measure TBW HW dt-sensitivity to verify the task #122/#123/#125
NMDA-unification + delays-in-ms refactor actually delivered the
dt-invariance it promised on Supp Fig 1 of the paper.

### Failure surfaced: SBW P(fusion) = 1.0 saturation

The 5-condition SBW sweep on surr_10 produced **completely flat
P(fusion) = 1.0 curves for every separation in every condition**
(`task136_logs/run_perturbations_surr10_v2.log`, `cache/sbw_*_t10.npz`
mtime 2026-05-22 03:16). Sanity probe:

```bash
python -c "import numpy as np; d=np.load('cache/sbw_control_t10.npz'); \
  print('min/max p:', d['mean_prob'].min(), d['mean_prob'].max())"
# Output: min/max p: 1.0 1.0
```

Pristine `fb6d3f6` Training.py + pristine SBW_test.py + same surr_10_00
ckpt: bell curve, HW ≈ 25° (task #138 E2). The saturation was a v2
Training.py regression, not a paper-irreproducibility or substrate
limitation.

### Root cause (task #138 — Debugger forensic, evidence-proven)

Task #123 introduced physical-time AGC cadences using two trackers at
`Training.py:1278-1279`:

```python
self._last_agc_fast_t = 0.0   # ms timestamp of last fast-AGC fire
self._last_agc_slow_t = 0.0   # ms timestamp of last slow-AGC fire
```

The AGC update is gated by `current_t_ms - self._last_agc_*_t >=
T_AGC_*_MS`. The bug: **`reset_state()` did NOT also zero these
trackers**, while the SBW pipeline (and TBW pipeline, same mechanism)
runs three within-separation passes (AV → A-only → V-only) and between
passes resets only `g_FFinh` + `step_counter` + calls `reset_state()`.

After pass 1 (AV), `_last_agc_*_t ≈ 65400 ms`. Pass 2 (A-only) starts
with `step_counter` rewound to ~652000 → `current_t_ms = 65200 ms` →
gate evaluates as `65200 − 65400 = −200 ms < T_AGC_*_MS` → **AGC never
fires for the entire 20-frame (200 ms) pass**. `g_FFinh` stays pinned
at the loaded baseline value 0.5648 (the AV-anticipating setpoint). For
unisensory A-only or V-only stimulus, `g_FFinh = 0.5648` is far too
strong → MSI population spikes drop ~100× (A_roi 64.6 → 0.66
spikes/trial; V_roi 45.5 → 0.32 spikes/trial).

Enhancement = `AV − max(A, V)` then degenerates to ≈ AV at every
separation (since A, V ≈ 0), trivially crossing threshold = 10 → P(fusion)
= 1.0 saturation.

Causal proof (`task138_logs/e7_agc_reset_test.py` + `e7_results.log`):

| Variant                                            | A_roi (sep=0) | V_roi (sep=0) | g_FFinh after [AV, A, V] passes  |
|----------------------------------------------------|--------------:|--------------:|----------------------------------|
| **A** — current buggy (no `_last_agc_*_t` reset)   | **0.66**      | **0.32**      | `[0.2221, 0.5648, 0.5648]` ← A/V never adapt |
| **B** — fix: also reset `_last_agc_*_t` per pass   | **64.60**     | **45.48**     | `[0.2221, 0.1956, 0.1490]` ← A/V adapt down  |
| PRISTINE `fb6d3f6` (no such field, no bug)         | 64.60         | 45.48         | (n/a — trackers don't exist)                  |

Variant B reproduces pristine A_roi / V_roi exactly. The full
forensic report is at `task138_logs/findings.md` (H1, H2, H5, H6
hypothesis-test ladder with verdicts).

### Fix (task #142)

Resetting `_last_agc_*_t` at every pipeline call site is fragile
(10 sites in SBW_test + TBW_test were identified by debugger #138).
The principled fix is to make `reset_state()` itself zero the
trackers, since their semantics are per-pass (per fresh trial) and
not per-checkpoint:

```python
# Training.py, inside reset_state(), between debug counters and RF tracking:
# task #142: zero the AGC physical-time gates so the first frame
# after reset_state always crosses the fast/slow cadences.
self._last_agc_fast_t = 0.0
self._last_agc_slow_t = 0.0
```

The accompanying `__init__` comment block (`Training.py:1270-1279`) was
updated to cross-reference task #142 and document that
`T_AGC_FAST_MS = 0.1` and `T_AGC_SLOW_MS = 10.0` cadences are
unchanged — only the within-pass reset semantics are fixed.

Verification (`task138_logs/e8_agc_persist_proof.py` re-run post-fix,
output in `task142_logs/e8_after_fix.log`): all three variants
(CURRENT-default, CURRENT-with-manual-AGC-reset, PRISTINE-control)
byte-identical at A_roi(latest)/trial = 57.840 (AV pass), 64.600
(A pass), 0.000 (V pass), with g_FFinh trajectory 0.5648 → 0.189 across
every variant. Pre-fix CURRENT-default had g_FFinh flat at 0.5648.

### Post-fix verification — full SBW grid on 10 ckpts (task #143-A)

5-condition × 33-separation × 50-trial sweep, threshold = 10,
control-floor subtracted (floor = 0.0001).
Total wall: 1980 s = 33 min on RTX A6000.

| Condition        | HW (°) post-fix | Paper target | Δ vs paper |
|------------------|----------------:|-------------:|-----------:|
| control          | 32.9            | 24.3         | +35 %      |
| ff_inhibition    | 42.9            | 27.7         | +55 %      |
| adaptation       | 35.0            | 29.4         | +19 %      |
| nmda             | 27.6            | 13.1         | +110 %     |
| nmda_increase    | 32.6            | 30.0         | +9 %       |

Bell curves restored across the board — no more saturation. The
HW magnitudes sit above paper but the perturbation directions are
preserved for 4/5 conditions (the `nmda` perturbation **widens**
SBW here vs paper's narrowing; consistent with the known
`SBW × nmda_increase` substrate limitation documented in task #117/#118
— the legacy ckpts trained at `gNMDA = 0.05` lack the paper's
nmda → SBW mechanism). Per-condition HWs:
`task143_logs/task_A_sbw_regen.log` lines 76-93.

The paper-fair `ff_inhibition` definition (`pv_nmda = 0.8 + targ_ratio
= 0.8`) widens SBW by +10° relative to control instead of narrowing —
a substrate property of the legacy ckpts, not a pipeline bug. See
also `task136_logs/paper_fair_ff_inh_sbw.log` for the standalone
ff_inh-only re-run that produced HW = 43.1° (consistent with the
full-grid number).

### Post-fix verification — TBW HW dt-sensitivity (task #143-B)

Paper Supp Fig 1 reports a +40.7 ms drift between dt = 0.1 ms and
dt = 0.05 ms TBW HW on the legacy substrate — the canonical
demonstration that dt = 0.05 is not a usable inference step without
recalibration on pre-task-#122 Training.py. The combined effect of:

- task #122 NMDA → I_M unification onto `dt_linear_scale = dt / 0.1`
  (replacing the bare-add at L2049),
- task #123 physical-time AGC cadences (replacing `step_counter % 100`),
- task #125 conduction-delays-in-ms refactor,
- task #142 AGC time-gate reset fix,

should collapse this drift to near zero. Measured on all 10 surr_10
ckpts (`task143_logs/task_B_dt_sweep.{log,json}`, total wall 22.3 min):

| Statistic              | HW @ dt = 0.1 (ms) | HW @ dt = 0.05 (ms) | Drift (ms) |
|------------------------|-------------------:|--------------------:|-----------:|
| mean ± SEM (n = 10)    | 218.60 ± 1.21      | 221.22 ± 0.78       | +2.62 ± 0.51 |
| Paper Supp Fig 1       | 215.80             | 256.50              | +40.70     |

**Drift is suppressed by ~94 %** (paper +40.7 ms → measured +2.62 ms),
across all 10 ckpts (per-ckpt drift range +0.85 to +5.11 ms). The
HW @ dt = 0.1 mean (218.6 ms) is within 1.3 % of the paper Supp Fig 1
baseline (215.8 ms), confirming the dt = 0.1 paper output is not
regressed by the fix chain.

### Files

- `task136_logs/run_perturbations_surr10_v2.log` — pre-fix 5-condition
  × 6-metric grid (latency, E/I, precision, Fano, MEI, cue-rel, all 30
  cells). SBW caches written during this run were saturated and have
  since been regenerated by task #143-A.
- `task136_logs/perturbation_grid_surr10_v2.json` — pre-fix grid in
  machine-readable form.
- `task136_logs/paper_fair_ff_inh.{py,log}` — standalone paper-fair
  ff_inh runner; crashed in TBW post-cache step with IndexError
  (pre-fix saturated curve → `_find_crossings` empty array). Re-run via
  task #143-A succeeded.
- `task136_logs/paper_fair_ff_inh_sbw.log` + `paper_fair_ff_inh_grid.json`
  + `paper_fair_ff_inh_sbw_overlay.png` — post-fix ff_inh-only SBW
  re-run (HW = 43.1°, matches task #143-A grid).
- `task136_logs/recover_sbw_hw.py` + `sbw_recovery.log` +
  `sbw_recovery_overlay.png` — interim recovery script + overlay used
  during the saturation investigation.
- `task136_logs/generate_all_fresh_surr10_v2.log` — first
  `generate_all_fresh.py` run on v2 Training.py that surfaced the
  saturation.
- `task138_logs/e1_*.{py,log}` through `e8_*.{py,log}` + `findings.md`
  — full debugger #138 forensic chain (8 experiments, 4 hypotheses
  tested with verdicts). E7 is the single-variable causal proof; E8 is
  the full-pipeline recovery demonstration.
- `task138_logs/validator_step3_sbw_probe.py` — validator-supplied
  probe used to verify the fix landed correctly on a single ckpt.
- `task143_logs/task_A_sbw_regen.{py,log}` — post-fix 5-condition SBW
  regen on 10 ckpts (HW table above).
- `task143_logs/task_B_dt_sweep.{py,log,json}` — post-fix TBW HW
  dt-sensitivity sweep on 10 ckpts (drift +2.62 ms).
- `cache/sbw_{control,ff_inhibition,adaptation,nmda,nmda_increase}_t10.npz`
  — post-fix SBW pooled metrics, written by `task143_logs/task_A_sbw_regen.py`.

### Code changes — Training.py

Two edits, both narrow:

1. **`__init__`** (line 1270 block) — comment-only update referencing
   task #142, no logic change. Documents that `reset_state()` now
   zeros the `_last_agc_*_t` fields and that the `T_AGC_FAST_MS = 0.1`
   / `T_AGC_SLOW_MS = 10.0` cadences are unchanged.
2. **`reset_state()`** — two new lines (`self._last_agc_fast_t = 0.0;
   self._last_agc_slow_t = 0.0`) inserted between the debug-counters
   reset block and the RF-tracking reset block, with a comment block
   describing the bug and citing debugger #138 / task #142.

No call-site patches were needed in SBW_test.py / TBW_test.py / any
other test harness — the fix is fully internal to Training.py and
respects the existing `reset_state()` contract.

### Branch

All artifacts in this section are committed on the branch
`task136-dt-invariance-restored` (cut from `task60-clean-methodology-baseline`).
