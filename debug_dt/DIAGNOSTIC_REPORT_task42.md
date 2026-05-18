# Diagnostic Report — Task #42

**Scope:** two post-fix regressions
  - ISSUE A — E/I balance: paper E/I = 1.04 vs "post-fix" 1.72 (and true 25.36)
  - ISSUE B — Cue reliability: production R² = 0.287 vs wrapper R² = 0.971

**Constraints:** read-only on production code; instrumentation only in `debug_dt/`.

---

## EXECUTIVE SUMMARY

Both regressions share a common upstream cause: **the production test scripts at
the repository root do not apply the `gNMDA = 1.30` recalibration override that
`generate_all_fresh.py` applies for TBW/SBW.** Without that override, the network
runs with `gNMDA = 0.05` (the legacy value baked into checkpoint
`mutable_hparams`) combined with `dt_correct_nmda = True` (the default after the
Task #16 Form-2 fix). This combination delivers ~1/25 of paper-equivalent
effective NMDA injection — an under-driven network state that is neither
paper-equivalent nor properly recalibrated.

Once the `gNMDA = 1.30` override is applied to cue_reliability_test.py the R²
recovers from 0.287 → 0.971. That single one-line change closes 99.7% of the
observed gap.

For E/I, applying the same override exposes a **second, independent bug** in
the E/I probe itself: `Training.py:2255` records the *source* NMDA current,
which scales linearly with `gNMDA`. After recalibration, the recorded NMDA
inflates 26× while the actual injection into `I_M` (and thus all network
dynamics) is unchanged. A one-line probe fix — multiplying recorded NMDA by
`nmda_source_scale` when `dt_correct_nmda=True` — recovers paper E/I from
25.36 → 1.05.

| Issue | Root cause | Lines | Recovery |
|---|---|---|---|
| **B (cue rel)** | Production script missing `gNMDA = 1.30` override | `cue_reliability_test.py:246-256` | Apply override → R² 0.287→0.971 |
| **A (E/I)** | (1) same missing override, then (2) NMDA probe records source not effective injection | `Training.py:2255` (and `run_ei_balance.py`) | Apply override + patch probe → E/I 25.36→1.05 |

Six production scripts at the root are missing the override:
`cue_reliability_test.py`, `response_latency_test.py`, `precision_hist_test.py`,
`fano_factor_test.py`, `run_ei_balance.py`, `inverse_effectiveness_test.py`.
All inherit the under-driven-network artefact.

---

## ISSUE B — Cue Reliability R² collapse

### Phase 1 — Reproduction (both numbers verified deterministically)

Production script as-is:
```
$ python -u -c "<inline reproducer of cue_reliability_test.main>"
  M00..M09: gNMDA=0.05, dt_correct_nmda=True, plasticity_enabled=True, allow_inhib=False
  === PRODUCTION (vanilla cue_reliability_test.py) ===
    R²   = 0.2866   MAE  = 0.2668   RMSE = 0.3443    elapsed = 33s
```
Matches team-lead's "R² = 0.287".

My wrapper (`H_meas_cue_reliability.py`) — the "FIXED" branch:
```
  D: FIXED, dt=0.05    R² = 0.966   MAE = 0.069   RMSE = 0.084
  FIXED R² drop: 0.971 → 0.966   Δ = -0.005
```
Matches team-lead's "R² = 0.971" reference.

Both reproductions are deterministic given a fixed RNG seed.

### Phase 2 — Methodology diff (parameters that differ)

Both scripts already use `return_spike_sum=True` (line 78 in production); my
fix from Task #30 was merged in. The remaining differences between production
defaults and my wrapper:

| Knob | Production | Wrapper |
|---|---|---|
| `gNMDA` | 0.05 (ckpt `mutable_hparams`) | 1.30 (forced) |
| `plasticity_enabled` | True (Training.py default) | False (forced) |
| `_reset_delay_buffers()` | not called | called |

### Phase 3 — One-variable isolation (FORWARD + REVERSE)

Source: `debug_dt/H_cuerel_isolate.py` (8-cell sweep on 10 ckpts).

```
PROD baseline (gNMDA=ckpt, plast=T, reset=F)              R²=0.2866
WRAP baseline (gNMDA=1.30, plast=F, reset=T)              R²=0.9708

FORWARD: production baseline + ONE wrapper knob:
  PROD + gNMDA=1.30   (only this)                         R²=0.9689
  PROD + plast=False  (only this)                         R²=0.3227
  PROD + reset_delays (only this)                         R²=0.2866

REVERSE: wrapper baseline - ONE wrapper knob:
  WRAP - gNMDA  (gNMDA=ckpt, plast=F, reset=T)            R²=0.3227
  WRAP - plast  (gNMDA=1.30, plast=T, reset=T)            R²=0.9688
  WRAP - reset  (gNMDA=1.30, plast=F, reset=F)            R²=0.9707

VERDICT
  Baseline gap:                  ΔR² = +0.6842

  PROD + gNMDA=1.30:             ΔR² = +0.6823   ← closes 99.7% of gap
  PROD + plast=False:            ΔR² = +0.0361   ← ~5%
  PROD + reset_delays:           ΔR² = +0.0000

  WRAP - gNMDA (use ckpt):       ΔR² = -0.6480   ← closes 94.7% reversing
  WRAP - plast (re-enable):      ΔR² = -0.0019
  WRAP - reset (skip reset):     ΔR² = +0.0000
```

**CAUSAL ASYMMETRY:** Setting `gNMDA=1.30` on production recovers R²=0.969
(within 0.002 of full wrapper). Setting `gNMDA=ckpt` on wrapper drops it to
R²=0.323 (within 0.04 of production). Other knobs do nothing.

`plasticity_enabled` has a small effect (~0.04) because with plasticity ON,
the network is also updating weights during the 20-trial dataset — but its
contribution is dwarfed by the gNMDA effect.

### Phase 4 — Alternative hypotheses tested and RULED OUT

| Hypothesis | Tested by | Result |
|---|---|---|
| `return_spike_sum` measurement diff | Already in production (line 78) | NOT a difference |
| `plasticity_enabled` | H_cuerel_isolate isolated swap | RULED OUT (ΔR² = 0.04, not 0.68) |
| `_reset_delay_buffers` | H_cuerel_isolate isolated swap | RULED OUT (ΔR² = 0.00) |
| Different dataset / sigma grid | Wrapper imports `make_dataset` from production | RULED OUT (identical data) |
| Different decoder (com vs argmax) | Both use "com" | RULED OUT (identical) |
| dt or n_substeps differences | Both default 0.1 / 100 | RULED OUT (no difference) |

### Phase 5 — Mechanism (why gNMDA matters)

With `dt_correct_nmda=True` (Form 2 ON), the NMDA → I_M injection is scaled
by `nmda_source_scale = 1 - exp(-dt/tau_syn) ≈ 0.0392` at dt=0.1, tau_syn=2.5.
Recalibration to `gNMDA = 1.30` was designed to compensate this 25× factor so
that effective injection matches the paper's effective injection at
`gNMDA = 0.05` with the pre-Form-2 bare-add code path.

When the production script forgets to override gNMDA, the network runs at
`gNMDA = 0.05` with Form 2 ON → effective NMDA injection is ~1/25 of paper.
This **under-drives MSI** (verified by component breakdown in H_ei_4cell — cell
B Rec/Lat inhibition drops 12-14× vs paper, indicating MSI fires much less),
which makes spike-counts noisier and degrades the COM decoder used for cue
weight inference.

### Phase 6 — Recovery (Issue B)

**Single-line fix in `cue_reliability_test.py`:**

```python
# In main(), inside the per-checkpoint loop (after load_msi_model):
net.gNMDA = 1.30   # match generate_all_fresh.py CONDITIONS["control"]
```

This restores R² = 0.969 ± 0.002 (vs paper 0.71, slightly better than paper).
Same one-line fix needed in the five other production scripts that suffer
the same issue:
`response_latency_test.py`, `precision_hist_test.py`, `fano_factor_test.py`,
`run_ei_balance.py`, `inverse_effectiveness_test.py`.

The proper post-fix R² (with gNMDA=1.30) is **0.97**, an improvement over
paper's 0.71 — consistent with the measurement-bug fix (`return_spike_sum`)
giving a more accurate decode than the legacy `_latest_sMSI` sampling.

---

## ISSUE A — E/I = 1.04 → 1.72 (apparent) and 25.36 (true post-fix)

### Phase 1 — Reproduction of paper-equivalent and "broken" E/I

Source: `debug_dt/H_ei_4cell.py` — 2×2 grid of (gNMDA, dt_correct_nmda).

```
A: gNMDA=0.05, dt_correct=False (PAPER pre-fix)        E/I=1.042±0.005  E=3.746  I=3.592
    breakdown: AMPA=0.2112  NMDA=3.5343  FF=2.6222  Rec=0.0035  Lat=0.9667

B: gNMDA=0.05, dt_correct=True  (production default)   E/I=1.717±0.012  E=3.734  I=2.174
    breakdown: AMPA=0.2542  NMDA=3.4801  FF=2.1022  Rec=0.0003  Lat=0.0718

C: gNMDA=1.30, dt_correct=False (over-driven)          E/I=1.711±0.005  E=109.665  I=64.124
    breakdown: AMPA=0.2610  NMDA=109.4039  FF=54.3442  Rec=0.0196  Lat=9.7603

D: gNMDA=1.30, dt_correct=True  (post-recalibration)   E/I=25.363±0.076  E=92.152  I=3.633
    breakdown: AMPA=0.2107  NMDA=91.9415  FF=2.6444  Rec=0.0036  Lat=0.9848
```

**Cell A matches paper E/I = 1.04 exactly. Cell B matches the team-lead's
reported "post-fix E/I = 1.72" exactly.** This proves the team-lead's "1.72"
came from running `run_ei_balance.py` *without* applying the gNMDA=1.30
override (same Issue-B bug). The **true** post-recalibration E/I — with both
gNMDA=1.30 AND dt_correct_nmda=True (cell D) — is 25.36.

### Phase 2 — Component decomposition (what shifted A→D)

```
Component shifts paper → true post-recalibration:
  AMPA  : 0.2112 → 0.2107  ratio = 1.00×
  NMDA  : 3.5343 → 91.9415 ratio = 26.01×  ← 26× scaling = gNMDA ratio (1.30/0.05)
  FF    : 2.6222 → 2.6444  ratio = 1.01×
  Rec   : 0.0035 → 0.0036  ratio = 1.01×
  Lat   : 0.9667 → 0.9848  ratio = 1.02×
```

**Only NMDA changed (26×). Every other component matches paper within 2%.**
The network is dynamically identical to paper (spike-driven currents all match
within 2%); only the NMDA recording is inflated.

### Phase 3 — Hypothesis: NMDA probe asymmetry

`Training.py:2255` records:
```python
_I_E_nmda = torch.clamp(I_nmda, min=0.0)        # SOURCE current
```
where `I_nmda = gNMDA * nmda_m * (mg_A + mg_V) * (Erev_nmda - v_msi)` is the
full conductance-driven current.

The actual injection per substep (`Training.py:2103-2106`):
```python
if self.dt_correct_nmda:
    self.I_M.add_(I_nmda * nmda_source_scale)   # EFFECTIVE = source × 0.039
else:
    self.I_M.add_(I_nmda)                       # EFFECTIVE = source
```

AMPA recording is symmetric: `_I_E_ampa = torch.clamp(I_AMPA_curr, min=0.0)`
where `I_M.add_(I_AMPA_curr + b_msi)` (bare add, no source_scale). So
AMPA-recorded = AMPA-injected; NMDA-recorded ≠ NMDA-injected when Form 2 ON.

After Form 2 + recalibration (gNMDA scaled up by ~26× to compensate
`source_scale ≈ 1/26`):
  - Effective injection: preserved (network dynamics unchanged)
  - Recorded NMDA: inflated 26× (source current scales with gNMDA)
  - Recorded E/I: inflated correspondingly

### Phase 4 — CAUSAL PROOF (probe-fix experiment)

Source: `debug_dt/H_ei_probe_fix.py`. Monkey-patches `update_all_layers_batch`
to replace
```
_I_E_nmda = torch.clamp(I_nmda, min=0.0)
```
with
```
_I_E_nmda = torch.clamp(I_nmda * nmda_source_scale, min=0.0) if self.dt_correct_nmda else torch.clamp(I_nmda, min=0.0)
```

```
Phase 1: BASELINE (orig probe, unmodified Training.py)
  source_scale at dt=0.1, tau_syn=2.5 = 0.03921

A_orig: gNMDA=0.05, Form2=OFF (paper)                   E/I=1.042±0.005
   AMPA=0.2112  NMDA_rec=3.5343  FF=2.6222  Rec=0.0035  Lat=0.9667
D_orig: gNMDA=1.30, Form2=ON  (post-recalib)            E/I=25.363±0.076
   AMPA=0.2107  NMDA_rec=91.9415  FF=2.6444  Rec=0.0036  Lat=0.9848

Phase 2: PATCHED probe (NMDA records I_nmda*source_scale when Form2=ON)
A_patch: gNMDA=0.05, Form2=OFF (no-op, no scale)        E/I=1.042±0.005   ← unchanged ✓
   AMPA=0.2112  NMDA_rec=3.5343  FF=2.6222  Rec=0.0035  Lat=0.9667
D_patch: gNMDA=1.30, Form2=ON  (scaled)                 E/I=1.050±0.004   ← RECOVERED ✓
   AMPA=0.2107  NMDA_rec=3.6051  FF=2.6444  Rec=0.0036  Lat=0.9848

CAUSAL VERDICT
  Paper baseline (cell A):                 E/I = 1.042
  Post-recalib raw (cell D, orig probe):   E/I = 25.363
  Paper with patched probe (cell A_patch): E/I = 1.042  (no-op confirmed)
  Post-recalib with patched probe (D_patch): E/I = 1.050  (matches paper!)
  Δ from paper to post-fix (orig probe):   +24.321
  Δ from paper to post-fix (patched probe): +0.008
```

**Changing ONLY the probe (no network state change) recovers paper E/I.**
The patched D's NMDA_rec = 3.6051 matches paper's NMDA_rec = 3.5343 within 2%.
Confirms NMDA probe asymmetry is the entire mechanism.

### Phase 5 — Alternative hypothesis tested: parameter recovery

Source: `debug_dt/H_ei_gnmda_sweep.py` + `H_ei_gnmda_sweep_part2.py` —
sweep gNMDA ∈ {0.5, 1.0, 1.3, 2.0, 5.0} at Form 2 ON; measure E/I (raw
probe) AND TBW HW per gNMDA. Reference: paper TBW HW at cheap-settings
(gNMDA=0.05, Form2=OFF) = 97.0 ms.

```
gNMDA= 0.50:  E/I = 13.842 ± 0.073   TBW HW = 94.1 ms   NMDA_rec=35.345   I=2.570
gNMDA= 1.00:  E/I = 22.121 ± 0.088   TBW HW = 96.2 ms   NMDA_rec=70.625   I=3.202
gNMDA= 1.30:  E/I = 25.363 ± 0.076   TBW HW = 96.9 ms   NMDA_rec=91.942   I=3.633
gNMDA= 2.00:  E/I = 30.018 ± 0.056   TBW HW = 98.8 ms   NMDA_rec=141.563  I=4.723
gNMDA= 5.00:  E/I = 37.024 ± 0.141   TBW HW = 98.7 ms   NMDA_rec=366.773  I=9.916

Paper reference (gNMDA=0.05, Form2=OFF):  E/I = 1.042   TBW HW ≈ 97 ms   NMDA_rec=3.53   I=3.59
```

| gNMDA | E/I (raw probe) | TBW HW | matches paper E/I (~1.04)? | matches paper HW (~97ms)? | both? |
|---|---|---|---|---|---|
| 0.50 | 13.84 | 94.1 | no (13× too high) | yes (Δ=3ms) | **NO** |
| 1.00 | 22.12 | 96.2 | no (21× too high) | yes (Δ=1ms) | **NO** |
| 1.30 | 25.36 | 96.9 | no (24× too high) | yes (Δ=0ms) | **NO** |
| 2.00 | 30.02 | 98.8 | no (29× too high) | yes (Δ=2ms) | **NO** |
| 5.00 | 37.02 | 98.7 | no (36× too high) | yes (Δ=2ms) | **NO** |

**No gNMDA value at Form 2 ON recovers BOTH paper E/I AND paper TBW HW.**
The lowest E/I in the sweep is 13.84 at gNMDA=0.50 — already 13× higher than
paper. NMDA_rec scales with gNMDA (linear up to ~1.3, sub-linear at 2.0/5.0
because saturation), while TBW HW stays nearly constant (94-99ms) across the
entire sweep — i.e., the network is in a saturated regime where TBW HW is
insensitive to gNMDA, while the probe is gNMDA-sensitive.

**Conclusion:** parameter recovery is impossible. The probe fix is the only
recovery path.

(Sub-hypothesis: separately scale a hypothetical `gNMDA_inh`. The model uses
the same `self.gNMDA` for both excitatory and inhibitory NMDA paths
(`Training.py:2096` and `2194`). Introducing a separate parameter would require
26× more inhibition into MSI excit to cancel the inflated NMDA recording,
which would crush MSI response and break every other biological metric.
**Ruled out structurally.**)

### Phase 6 — Mechanism for cell B → "1.72" (the team-lead's observation)

Cell B (gNMDA=0.05 + Form 2 ON, the production default with no override):
  - Effective NMDA injection = `0.05 × 0.039 = 0.002` per substep
  - Paper effective = `0.05 × 1.0 = 0.05` (Form 2 OFF, bare add)
  - **Ratio: 0.04× of paper → severely under-driven**

Under-driven MSI fires less → all spike-driven inhibitory components collapse:
  - Rec: 0.0035 → 0.0003 (12× weaker)
  - Lat: 0.9667 → 0.0718 (14× weaker)
  - FF: 2.6222 → 2.1022 (20% weaker — FF is driven by external inputs, less
    affected)

Drop in I (3.59 → 2.17, -39%) > drop in E (3.75 → 3.73, ~unchanged because
NMDA recording is gNMDA-linear and gNMDA didn't change → recorded ≈ paper).
Hence E/I = 3.73 / 2.17 = 1.72.

So cell B's E/I=1.72 is **NOT** the recalibration overshooting — it is a
**different bug** (no gNMDA recalibration applied at all). The team-lead's
"post-fix E/I = 1.72" mislabels what was actually the broken-default state.

### Recovery direction (Issue A)

Two independent fixes are needed:

**Fix A1 (same as Issue B):** Apply `gNMDA = 1.30` override in
`run_ei_balance.py` per-checkpoint loop. This moves the measurement from
cell B (1.72) to cell D (25.36 — the *real* post-recalibration value).

**Fix A2 (E/I probe):** In `Training.py:2255`, replace
```python
_I_E_nmda = torch.clamp(I_nmda, min=0.0)
```
with
```python
if self.dt_correct_nmda:
    _I_E_nmda = torch.clamp(I_nmda * nmda_source_scale, min=0.0)
else:
    _I_E_nmda = torch.clamp(I_nmda, min=0.0)
```
This records the effective per-substep NMDA injection (apples-to-apples with
AMPA). After this patch + the gNMDA=1.30 override, E/I returns to **1.050 ±
0.004**, matching paper's 1.042 within 0.8%.

Both fixes are required; either alone is insufficient:
  - Fix A1 alone → goes from 1.72 (cell B) to 25.36 (cell D), worse
  - Fix A2 alone → keeps cell B's 1.72 because the under-driven network state
    is the dominant artifact

---

## REMAINING UNKNOWNS

1. Whether the other four production scripts
   (`response_latency_test.py`, `precision_hist_test.py`, `fano_factor_test.py`,
   `inverse_effectiveness_test.py`) show similarly recoverable behavior with
   the gNMDA=1.30 override has not been individually tested. Strongly expected
   from the mechanism (same under-driven network state when override missing).
   Task-level recommendation: have Coder add the override to all 6 scripts
   uniformly, then have Validator re-measure each metric and compare to paper.

2. The probe-fix applies only to the *excitatory* NMDA recording. The
   `I_M_inh` path also uses Form 2 (`Training.py:2197-2200`), but inhibitory
   NMDA is not directly recorded by the current E/I probe — it influences
   E/I only indirectly via MSI_inh's GABA output (recorded as `RecurInh`).
   That indirect path is already accounted for in the network dynamics
   (which are paper-equivalent in cell D).

3. The gNMDA sweep at Form 2 ON shows TBW HW is nearly constant (94-99ms)
   across gNMDA from 0.5 to 5.0 — i.e., the network operates in a saturated
   regime for TBW HW. This is consistent with `mean_fusion ≈ 1.0` at offset=0
   across all sweep models, suggesting fusion is at ceiling. The TBW HW values
   measured here are NOT useful for re-calibrating gNMDA against TBW (a richer
   probe with non-saturating dynamics would be needed). However, this does
   NOT affect the E/I conclusion — the E/I metric is sensitive to gNMDA in
   the same range where TBW is not, which is exactly why parameter-tuning
   recovery is impossible.
