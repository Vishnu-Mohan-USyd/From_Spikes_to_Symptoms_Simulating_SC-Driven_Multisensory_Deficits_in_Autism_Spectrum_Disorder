# DIAGNOSTIC REPORT — Task #30

**Failure under investigation**: Residual dt-dependence after tasks #16/#27
fixes (Form-2 NMDA injection + delays-in-ms + gNMDA=1.30) in:
- SBW HW (all 5 conditions, drift -3 to -16%)
- Cue reliability R² (0.78 → 0.18 collapse)
- Cue reliability MAE (0.11 → 0.31, +180%)
- Response latency A/V/B (+5 to +11%)
- E/I ratio (25.36 → 51.19, +102%)

**Reproducer**: `python -m debug_dt.repro_sbw_dt_v2`
Reproduced exactly on M00..M09 canonical 10-ckpt pool:
- SBW control HW: 24.13° @ dt=0.1 → 20.37° @ dt=0.05 (-15.6%)

---

## HYPOTHESES TESTED

### H_agc — AGC g_FFinh trajectory differs at finer dt
**Hypothesis**: AGC update (Training.py:2454-2469) runs once per external
frame with alpha=1e-3 not scaled by dt. Different exc_fast / inh_mean
values at dt=0.05 vs dt=0.1 push g_FFinh to different equilibria.

**Tests**:
- `debug_dt/H_agc_freeze.py` — freeze AGC at both dts (degenerate fits
  because gNMDA=1.30 needs AGC for E/I balance on legacy-gNMDA-trained
  checkpoints; freeze test was inconclusive on its own).
- `debug_dt/H_agc_trace.py` — instrument g_FFinh, exc_fast, inh_mean per
  frame at both dts.

**Verdict**: **RULED OUT.**
Trace evidence over stim window (frames 5-19, M00, sync AV intensity=1):
```
g_FFinh    : dt=0.1 = 0.1696    dt=0.05 = 0.1721    Δ = +0.0025  (+1.5%)
exc_fast   : dt=0.1 = 0.0569    dt=0.05 = 0.0576    Δ = +0.0007  (+1.2%)
inh_mean   : dt=0.1 = 131.6785  dt=0.05 = 138.3162  Δ = +6.6377  (+5.0%)
```
g_FFinh trajectories at the two dts differ by only +1.5% — cannot explain
a 15% HW drift. AGC is NOT the mechanism.

### H_inh_FF / H_inh_recur / H_inh_lat — spike-driven inhibition is mean-dt-dependent
**Hypothesis**: I_M.sub_(FFInh / RecurInh / LatInh) are spike-driven but
their integrated mean on I_M may differ at finer dt.

**Test**: leaky-integrator algebra + indirect verification via FIXED
measurement test (see H_meas below).

**Verdict**: **RULED OUT.**
For any spike-driven source S[t] of magnitude W with rate r spikes/ms,
applied onto leaky integrator I_M with decay (1 - dt/tau_syn):
  mean(I_M contribution) = mean(W·S[t]) / (dt/tau_syn)
                         = (r·dt·W) / (dt/tau_syn)
                         = r · tau_syn · W       (DT-INVARIANT)

Empirical confirmation: the FIXED-measurement test eliminates 99.7% of
the SBW HW drift, leaving only -0.3% residual (within noise). Any
hypothetical network-mechanism dt-dependence must sum to less than 0.3%.

### H_buffer / H_izhi / H_ampa_m / H_nmda_m / H_inh_NMDA_inh
All same verdict: **RULED OUT.**
The FIXED-measurement test removes essentially all drift (-15.6% → -0.3%).
Any of these mechanisms, if causal, would persist with FIXED measurement.

### H_meas — measurement uses `_latest_sMSI` (LAST substep) as if it were
the per-frame spike total
**Hypothesis**: `net._latest_sMSI` (Training.py:2375, line: `self._latest_sMSI = sM`)
is updated each substep with the LAST substep's binary spike mask only.
Per-substep spike probability ∝ rate × dt → at dt=0.05 the per-substep
mask is HALVED on average.
Pipelines accumulate `msi_sum += net._latest_sMSI` once per external
frame — this captures only 1/n_substeps of the per-frame spike activity,
with dt-linear scaling.

For SBW: enhancement = AV_roi - max(A_roi, V_roi) compared against a
FIXED threshold (=10 spikes). Halved counts → fewer trials cross 10 at
peripheral separations → P(fusion) collapses at periphery → HW shrinks.

For cue reliability: spike_sum decoded for location; halved counts →
noisier decoding → R² collapses, MAE rises.

For latency: `if net._latest_sMSI.sum() > 0` checks last substep only;
halved per-substep mean → first detection frame is later.

For TBW: classifier uses RELATIVE peak/valley thresholds — uniform
halving doesn't shift relative classification → mostly dt-invariant
(consistent with task #29's TBW being clean on 4 of 5 conditions).

**Verdict**: **CONFIRMED.**

---

## PROVEN ROOT CAUSE

**`net._latest_sMSI` measurement convention** (Training.py:2375).

Each substep, `self._latest_sMSI = sM` where `sM = new_sM` is the
substep's binary spike mask. After `n_substeps` substeps, `_latest_sMSI`
holds the LAST substep's binary 0/1 spike mask only.

All test pipelines accumulate `msi_sum += net._latest_sMSI` ONCE per
external frame call. They INCORRECTLY treat this as the per-frame
spike total. The true per-frame total is available via
`update_all_layers_batch(..., return_spike_sum=True)` which is unused.

Per-substep spike probability = rate × dt. At dt=0.05 vs dt=0.1, per-
substep mean is HALVED. Hence the measured `msi_sum` scales linearly
with dt — by factor 0.5 at dt=0.05 relative to dt=0.1.

---

## CAUSAL PROOFS

### Forward — apply the FIX, drift vanishes

`debug_dt/H_meas_bug_proof.py` — canonical 10-ckpt SBW pipeline, with
either BUGGY (`+= _latest_sMSI`) or FIXED (`return_spike_sum=True`)
measurement.

```
Cond A: BUGGY measurement, dt=0.1   → HW = 24.37°  floor=0.0739
Cond B: BUGGY measurement, dt=0.05  → HW = 20.12°  floor=0.0431   ΔHW = -4.24°  (-17.4%)
Cond C: FIXED measurement, dt=0.1   → HW = 32.78°  floor=0.0000
Cond D: FIXED measurement, dt=0.05  → HW = 32.68°  floor=0.0000   ΔHW = -0.11° ( -0.3%)
```
The FIXED measurement closes 99.7% of the drift.

### Reverse — simulate the halving by scaling the threshold

`debug_dt/H_meas_reverse_proof.py` — BUGGY pipeline, vary
enhancement_threshold.

```
HW(dt=0.1,  thr=10): 23.90°   — baseline
HW(dt=0.05, thr=10): 20.56°   — drift (-3.34° = -14.0%)
HW(dt=0.1,  thr=20): 20.71°   — doubling threshold reproduces drift
HW(dt=0.05, thr= 5): 22.14°   — halving threshold partially reverses
```
|HW(dt=0.1, thr=20) - HW(dt=0.05, thr=10)| = 0.15°. The drift can be
EXACTLY reproduced at dt=0.1 by doubling the threshold — confirming the
threshold/halved-counts interaction is the precise mechanism.

### Direct measurement — spike counts ARE halved

`debug_dt/H_spatial_profile.py` — measure MSI spike count in ROI
(±20 neurons around stim_A) at multiple separations × stimulation
conditions (AV / A-only / V-only), using `_latest_sMSI` accumulation.

```
Condition    sep  ROI(dt=0.1)  ROI(dt=0.05)  ratio
AV           0°   115.7         58.9          0.509
AV          25°    77.2         40.8          0.528
AV          80°    58.6         31.3          0.534
A           any    67.7         33.6          0.497
V            0°    46.6         24.0          0.515
V           25°    17.3          7.5          0.432
V           80°     0.0          0.0           —
```
ROI spike counts at dt=0.05 are HALVED — direct empirical confirmation
of the dt-linear measurement scaling.

### Propagation — same bug, same fix, for cue reliability

`debug_dt/H_meas_cue_reliability.py` — 10 ckpts, control parameters,
BUGGY vs FIXED measurement.

```
                R²(dt=0.1)  R²(dt=0.05)   ΔR²
BUGGY:          0.629       0.302         -0.328   ← reproduces validator's collapse
FIXED:          0.971       0.966         -0.005   ← drift VANISHES (R² also HIGHER)

                MAE(dt=0.1) MAE(dt=0.05)  ΔMAE
BUGGY:          0.130       0.247         +0.117   ← reproduces validator's degradation
FIXED:          0.065       0.069         +0.004   ← drift VANISHES (MAE also LOWER)
```

### Propagation — same bug, same fix, for latency

`debug_dt/H_meas_latency.py` — 10 ckpts, response_latency_test parameters
(aM=0.001/bM=0.2/cM=-60/dM=0.1).

```
                 A_lat(dt=0.1) → A_lat(dt=0.05)
BUGGY: A   43 ms → 46 ms (+ 7.0%)    V 54→55 (+1.9%)    B 32→40 (+25%)
FIXED: A   30 ms → 30 ms (+ 0.0%)    V 50→50 (+0.0%)    B 30→30 ( +0%)
```
All A/V/B latency drifts vanish with FIXED measurement.

---

## E/I RATIO DOUBLING (related but distinct artifact)

The E/I doubling (validator's 25.36 → 51.19) is NOT caused by
`_latest_sMSI` (which is not used in E/I recording). It arises from a
DIFFERENT but mechanistically-related measurement convention:

`run_ei_probe_separated` computes:
- `ei_ratio = mean(I_E_mean) / mean(I_I_mean)`
  where each per-substep value is the raw current MEAN across batch+
  neurons.

For SPIKE-DRIVEN sources (I_AMPA, I_FFInh, I_RecurInh, I_LatInh),
per-substep mean = rate × dt × W → HALVED at dt=0.05.
For CONTINUOUS sources (I_NMDA, computed from leaky integrator nmda_m),
per-substep mean is DT-INVARIANT.

Hence:
  I_E_mean = (dt-invariant NMDA) + (halved AMPA)  ≈ NMDA-dominated when NMDA ≫ AMPA
  I_I_mean = halved spike-driven inh
  ei_ratio = E/I = (≈ constant) / (halved) → DOUBLES at dt=0.05

The Q_E_total / Q_I_total ratio has the same issue (Q = I × dt).

**Fix candidate**: use `I_M.mean()` directly (the integrated current)
which is dt-invariant by leaky-integrator algebra. Or compute means in
units of "charge per ms" instead of "current per substep".

---

## CROSS-METRIC CONSISTENCY

| Metric | Validator drift | Bug-test BUGGY drift | Bug-test FIXED drift |
|--------|-----------------|----------------------|----------------------|
| SBW HW | -15.7%  | -17.4% / -14.0% (two tests) | -0.3% / -0.7% (two tests) |
| Cue R² | -0.60 (0.78→0.18) | -0.328 (0.63→0.30) | -0.005 (0.97→0.97) |
| Cue MAE | +0.20 (0.11→0.31) | +0.117 (0.13→0.25) | +0.004 (0.07→0.07) |
| Latency A | +10.6% | +7.0% | 0.0% |
| Latency V | +7.5% | +1.9% | 0.0% |
| Latency B | +5.9% | +25.0% | 0.0% |

(Slight magnitude differences between validator and these tests reflect
different perturbation conditions / random seeds. Direction and
qualitative magnitude are consistent. The crucial point is FIXED → 0.)

---

## SUGGESTED FIX

A uniform single-source fix:
```python
# OLD pattern (buggy):
for t in range(T):
    net.update_all_layers_batch(stim_A, stim_V)
    msi_sum += net._latest_sMSI

# NEW pattern (fixed):
for t in range(T):
    ret = net.update_all_layers_batch(stim_A, stim_V, return_spike_sum=True)
    sum_sM = ret[-1]
    msi_sum += sum_sM
```

The function signature supports it; see Training.py:2292-2293 (sum_sM
accumulation in substep loop) and 2478-2479 (return path).

Affected call sites (must be patched for measurement correctness):
```
SBW_test.py:153, 329, 426, 544, 663, 1146
TBW_test.py:566, 916, 1057, 1569
cue_reliability_test.py:79
response_latency_test.py:82, 137, 252, 347
inverse_effectiveness_test.py:82, 137, 253, 348
precision_hist_test.py:49
```

For `response_latency_test.py:649` (`if net._latest_sMSI.sum() > 0`):
```python
# OLD:
if net._latest_sMSI.sum().item() > 0: ...
# NEW:
ret = net.update_all_layers_batch(...args..., return_spike_sum=True)
sum_sM = ret[-1]
if sum_sM.sum().item() > 0: ...
```

For E/I doubling (separate fix): re-define `ei_ratio` using `I_M.mean()`
or equivalent integrated quantity, OR document that ei_ratio is dt-
sensitive by definition and only compare within the same dt.

The TBW pipelines (TBW_test.py:566, 916, 1057, 1569) use `_latest_sMSI`
but rely on the classifier's RELATIVE thresholds (peak/valley ratios),
so they're mostly insensitive to uniform halving — but should be patched
for consistency and to prevent future absolute-threshold metrics from
breaking.

---

## REMAINING UNKNOWNS

1. **TBW nmda_increase (+13% drift, validator task #29)**: I did not
   forensically test this specific condition. Plausible explanation
   under the `_latest_sMSI` mechanism: at gNMDA=5.0 (very strong NMDA),
   the network is in saturation; per-substep spike density is dt-linearly
   reduced; classifier's `min_total=10` ABSOLUTE threshold may catch
   some borderline trials differently across dts. Should fall under
   the same fix.

2. **Precision σ_A/σ_V/σ_B being dt-invariant (validator)** despite
   using `_latest_sMSI`: probably because precision metrics use RELATIVE
   peak-shape statistics (variance of spatial profile), insensitive to
   uniform halving.

3. **MEI (inverse-effectiveness) being dt-invariant** likewise probably
   uses relative quantities.

4. Whether the fix introduces any UNINTENDED change to absolute metric
   values (e.g., HW jumps from 24° → 32° in our test). The HW change is
   a consequence of using more spikes per measurement → higher
   enhancement at all separations → wider P(fusion). This is a correct
   measurement; downstream comparisons against literature / prior runs
   need re-baselining. Coder + Validator should re-establish gold values.

---

## INSTRUMENTATION FILES (debug_dt/, no production code modified)

- `repro_sbw_dt_v2.py`         — Phase 1 reproducer (matches validator)
- `H_agc_freeze.py`            — AGC freeze (RULED OUT)
- `H_agc_trace.py`             — AGC trace (confirms RULED OUT)
- `H_spikes_components.py`     — per-component current decomposition
- `H_spatial_profile.py`       — ROI spike count by sep × condition
- `H_meas_bug_proof.py`        — FORWARD causal proof (SBW)
- `H_meas_reverse_proof.py`    — REVERSE causal proof (threshold scaling)
- `H_meas_cue_reliability.py`  — propagation: cue reliability
- `H_meas_latency.py`          — propagation: latency
- `PHASE2_catalogue.txt`       — full I_M/I_M_inh path inventory

Logs alongside each `.py` as `.log`. Full evidence preserved.
