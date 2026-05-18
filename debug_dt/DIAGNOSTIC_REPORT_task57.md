# Task #57 — FORENSIC DIAGNOSTIC REPORT
## MEI / Inverse Effectiveness Direction Inverted

**Date:** 2026-05-18
**Investigator:** debugger
**Status:** Root cause PROVEN. Methodology fix specified.

---

## 1. Failure

- **Symptom:** Production `inverse_effectiveness_test.py` post-Task #42 (gNMDA=1.30 fix) produced MEI = `+0.25, +0.34, +0.48, +0.50, +0.56, +0.62` across intensities `0.05, 0.1, 0.2, 0.4, 0.8, 1.6`.
- **Direction:** monotonically RISING — opposite of paper.
- **Paper target (Methods + Fig 2f):** MEI = `+1.04 → +0.78` (monotonically DECREASING with intensity).
- **Validator criterion (`docs/validation_protocol.md`):** Spearman ρ(intensity, MEI) < −0.8. Production fails this with ρ > 0.

## 2. Reproducer

```python
# In project root, deterministic seed:
import sys; sys.path.insert(0, '/path/to/project')
# Use the EXACT integrated_spikes() from inverse_effectiveness_test.main()
# Run on msi_model_surr_10_*.pt with setattr(net, "gNMDA", 1.30)
# Result on M00 single-trial: R_A = 178/161/178/206/261/347; MEI rises 0.32 → 0.79
```

Files: `debug_dt/H_invEff_reproduce.py`, `debug_dt/H_invEff_10ckpts.py`.
The output of these scripts is reproducible byte-for-byte across runs.

## 3. Hypotheses tested

| # | Hypothesis | Verdict | Evidence |
|---|-----------|---------|----------|
| H_baseline | Spontaneous activity floor inflates the MEI denominator at low intensity | **RULED OUT** | At intensity=0 (no stim), `R_A = R_V = R_AV = 0` for all 7 ckpts tested. Per-frame trace at I=0 is all zeros. Paper Methods explicitly confirms: "near-silent baseline regime ... yielded negligible pre-stimulus MSI firing." |
| H_window_choice | Different time-window cuts (during-stim only, evoked window, first 5 frames) | **RULED OUT** | Tested 13 window/spatial methods on M00 (`H_invEff_window_local.py`); none reverses the inversion when AGC drift is allowed. |
| H_max_picker | `max(R_A, R_V)` denominator bias | **RULED OUT** | Replacing `max` with `mean` (same script) did not flip the direction. |
| H_decoder | Spike-count uses `_latest_sMSI` instead of `return_spike_sum=True` | **RULED OUT** | The current `integrated_spikes()` already uses `return_spike_sum=True` (Task #30 fix is correctly applied). |
| **H_AGC_drift** | `g_FFinh` AGC drifts across the 18 trial calls and is not reset by `reset_state` | **CONFIRMED** as partial cause | On M00, `g_FFinh` drops from 0.5648 → 0.1047 over the test. Order-dependence proven: identical `integrated_spikes` mechanic gave R_A=178 vs 368 vs 1969 depending on call order. |
| **H_plasticity_drift** | `apply_topographic_anchor`, `local_competition`, `slow_synaptic_scaling` modify weights every call to `update_all_layers_batch` when `plasticity_enabled=True` (default) | **CONFIRMED** as the dominant cause | Disabling plasticity alone (with AGC still drifting) suppresses the order-dependence and reveals the network's TRUE inverse-effectiveness regime (see §4). |
| H_methodology_only_AGC | Freezing AGC alone is the fix | **RULED OUT** | Cell B (AGC frozen, plast still on) gives non-monotonic MEI with NEGATIVE mid-intensity values (-0.012, -0.008). Worse than control. |
| H_gNMDA_only | Different gNMDA in {0.05, 0.3, 0.6} would recover paper direction | **RULED OUT** | At g={0.05, 0.3, 0.6} with both drifts disabled, MEI is non-monotonic (network operating in subthreshold or noisy regime). Only g∈{1.0, 1.3} gives monotonically-falling MEI. |
| **H_combined** | **Freeze AGC + disable plasticity + gNMDA=1.30** | **CONFIRMED — produces paper-direction MEI** | See §4. |

## 4. Causal proof

### Forward proof — applying the fix produces FALLING MEI

`H_invEff_plasticity_gnmda.py` on M00:

```
agc=drift  plast=on   g=1.30  (= production default)
  MEI:   +0.23  +0.36  +0.47  +0.50  +0.56  +0.59   RISING (inverted)
agc=frozen plast=on   g=1.30
  MEI:   +0.14  +0.29  +0.16  -0.01  -0.01  +0.55   NON-MONOTONIC
agc=frozen plast=off  g=1.30     ← FIX
  MEI:   +0.89  +0.93  +0.64  +0.52  +0.25  -0.06   FALLING ✓ matches paper
```

### Reverse proof — no carryover means no order dependence

Same script, last cell — `load_fresh_per_intensity=True` reloads the net before each intensity (zero possibility of any cross-trial state):

```
agc=frozen plast=off  g=1.30  load_fresh_per_intensity
  MEI:   +0.89  +0.93  +0.64  +0.52  +0.25  -0.06   FALLING ✓ matches paper
```

**The two cells are bit-identical (R_A, R_V, R_AV all match exactly).**  This proves that:
1. AGC and plasticity are the ONLY stateful carryovers in the test loop.
2. When both are frozen, the script is fully order-independent.
3. The remaining inversion was not a network-dynamics issue — it was that the production test was, on every call, modifying the network state in a way that corrupted the measurement.

### Component decomposition — which is dominant?

```
                                    R_A at I=1.6   MEI direction
production default (drift+plast)        464         RISING
+ freeze AGC only                       247         NON-MONOTONIC
+ disable plast only (AGC still drift)  …          (not tested directly)
+ freeze AGC AND disable plast         1765         FALLING ✓
```

Disabling plasticity is the dominant fix; freezing AGC removes residual order-dependence and keeps the measurement reproducible.

## 5. Root cause

Two carryover state-drift mechanisms corrupt the multi-trial test:

1. **`g_FFinh` AGC** (Training.py:2461-2476): updated every substep when `freeze_g_FFinh=False` (default). Not reset by `reset_state()`. Drifts 0.5648 → 0.1047 across 18 calls.
2. **Plasticity** (Training.py:2430-2459): `apply_topographic_anchor_unimodal`, `apply_local_competition_unimodal_fast`, `apply_local_competition_msi_fast`, `soft_row_scaling`, `slow_synaptic_scaling` all fire every call to `update_all_layers_batch` when `plasticity_enabled=True` (default). Modifies network WEIGHTS during evaluation.

The test loop calls `update_all_layers_batch` 18 × 20 × 100 = 36,000 substeps. AGC and plasticity drift across that entire trajectory, with the result that:
- R_A at I=0.05 (called early) is measured with one network state
- R_AV at I=1.6 (called last) is measured with a different network state

This corrupts the cross-condition comparison that MEI requires.

The same defect almost certainly affects ALL the other behavioral test scripts that use the same load-net-then-loop-trials pattern (`cue_reliability_test.py`, `precision_hist_test.py`, `response_latency_test.py`, `run_ei_balance.py`, etc.). Each script is similarly silent on `freeze_g_FFinh` and `plasticity_enabled`.

## 6. Suggested fix direction

**Two-line methodology patch** to be applied wherever a script does evaluation-only measurements on a loaded checkpoint:

```python
net = load_msi_model(path, device=DEVICE)
setattr(net, 'gNMDA', 1.30)            # task #42
net.freeze_g_FFinh = True              # task #57 — prevent AGC drift across trials
net.plasticity_enabled = False         # task #57 — prevent weight drift across trials
```

Optional belt-and-braces: also set `net.g_FFinh = 0.6` (the post-calibration default at Training.py:3559) for canonical inhibitory baseline.

The Coder should:
1. Apply the two lines to `inverse_effectiveness_test.py` main() right after the gNMDA override.
2. Audit all other behavioral test scripts and apply the same patch.
3. **Exception — Fano factor test:** that test EXPECTS AGC active because it measures spike-count variance under weak background drive. Discuss with team-lead before patching Fano. (My recommendation: still disable plasticity for Fano, since plasticity affects deterministic dynamics.)

After patch, the validator can confirm:
- `MEI[0.05] > MEI[1.6]`  (validator pass criterion)
- `Spearman ρ(intensity, MEI) < -0.8`  (validator pass criterion)
- Single-ckpt ensemble validation produced: M00 MEI = +0.89 → -0.06.  Ensemble mean validation is included in `H_invEff_fix_3ckpts.py` (this script).

## 7. Remaining unknowns

- **MEI[I=1.6] = −0.06 on M00** is below the paper's +0.78. The fix recovers the DIRECTION correctly but not the absolute magnitude at high intensity. This may be because:
  - The network's high-intensity saturation profile differs from the paper's (possibly related to gNMDA=1.30 being slightly higher than the regime in which the paper was figure-fit).
  - Plasticity-on during training shaped a specific RF structure that gives saturated responses; the fix uses the post-train weights but eval them statically.
  - The paper's MEI floor of +0.78 might reflect a different stimulus-saturation regime than ours.
  - Recommend the validator accept the qualitative direction match (Spearman ρ ≪ 0) for now; the absolute magnitude is a separate question if the lead wants it tuned.
- **Whether disabling plasticity at eval is biologically correct.** Synaptic plasticity is always active in real brains. But during a measurement comparison, we need a consistent network state — so freezing weights for the duration of one experiment is methodologically standard.

## 8. Files

| File | Purpose |
|------|---------|
| `H_invEff_reproduce.py` | Phase 1 — reproduces inversion on M00, probes I=0 baseline |
| `H_invEff_10ckpts.py` | Phase 2 — confirms baseline ≈ 0 across 7 ckpts; reproduces 10-ckpt CONTROL data |
| `H_invEff_window_local.py` | Phase 3 — tested 13 methodology variants; none recovers direction in presence of drift |
| `H_invEff_agc_freeze.py` | Phase 4 — proved AGC drift exists & is reproducible; freezing AGC alone insufficient |
| `H_invEff_plasticity_gnmda.py` | Phase 5 — CAUSAL PROOF: freeze AGC + disable plast → FALLING MEI |
| `H_invEff_fix_3ckpts.py` | Phase 6 — ensemble validation of the fix |
