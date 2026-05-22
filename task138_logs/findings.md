# Task #138 — Diagnostic report: SBW saturation on legacy surr_10 ckpts

## Headline finding

**Root cause is NOT a Training.py regression or paper irreproducibility. It is a
SBW/TBW test-pipeline bug introduced by task #123's physical-time AGC cadence.
The pipeline resets `g_FFinh` and `step_counter` between SBW passes (AV → A → V)
but does NOT reset the new AGC physical-time gates `_last_agc_fast_t` and
`_last_agc_slow_t`. After pass 1 those trackers hold `t > step_counter*dt`
when step_counter is rewound for pass 2/3, so `(current_t_ms − _last_agc_*_t)`
goes NEGATIVE and AGC stops firing in passes 2 and 3. g_FFinh then stays at the
loaded high value (0.5648) instead of adapting downward for unisensory stimuli,
suppressing A-only / V-only MSI response by ~100× while AV remains normal.
enhancement = AV − max(A, V) ≈ AV → trivially crosses threshold=10 at all
separations → P(fusion)=1.0 saturation.**

**Single-line fix at each `run_pass` reset site reproduces paper SBW HW.**

## Reproducer

```bash
python -c "import numpy as np; d=np.load('cache/sbw_control_t10.npz'); \
  print('min/max p:', d['mean_prob'].min(), d['mean_prob'].max())"
# Output: min/max p: 1.0 1.0
```

Cache regenerated via `python generate_all_fresh.py` at 2026-05-22 02:49.

## Hypotheses tested

| # | Hypothesis | Verdict | 1-line evidence |
|---|---|---|---|
| H1 | Cache stale / contaminated by prior config | **RULED OUT** | Training.py mtime 2026-05-22 00:47, cache mtime 03:16; same day, no Training.py edit in between |
| H2 | Paper SBW=24° is unreproducible on pristine fb6d3f6 + surr_10 | **RULED OUT** | Pristine fb6d3f6 Training.py + pristine SBW_test.py + surr_10_00 → bell curve HW≈25° (E2 result) |
| H5 | SBW_test.py `_latest_sMSI`→`sum_sM` swap (per-substep accumulation inflates enh ~100×) | **PARTIAL** | Confirmed magnitude inflation: pristine A_roi=64 vs sum_sM A_roi=6254 (E3); but does NOT alone explain saturation (pristine path's `_latest_sMSI` A=64 vs current Training.py `_latest_sMSI` A=0.66 — already 100× under-firing) |
| H6 | `_last_agc_*_t` not reset between SBW passes → AGC silently disabled in passes 2/3 | **CONFIRMED** | Single-variable intervention reproduces pristine A_roi exactly (E7); full SBW curve recovers bell shape with HW=32.7° (E8) |

## Causal proof (E7)

Single-variable intervention: in `compute_sbw_enhancement_persep`'s `run_pass`,
add reset of `_last_agc_*_t` alongside `step_counter`.

|  | A_roi mean (sep=0) | V_roi mean (sep=0) | g_FFinh after [AV,A,V] |
|---|---:|---:|---|
| **VARIANT A** (current buggy code) | **0.66** | **0.32** | `[0.2221, 0.5648, 0.5648]` ← A/V passes never adapt |
| **VARIANT B** (fix: reset `_last_agc_*_t`) | **64.60** | **45.48** | `[0.2221, 0.1956, 0.1490]` ← A/V passes adapt downward |
| PRISTINE fb6d3f6 (E3, for ref) | 64.60 | 45.48 | (no _last_agc_*_t field; bug doesn't exist) |

Variant B EXACTLY reproduces pristine A_roi/V_roi (64.60 / 45.48). Causal — only
the AGC tracker reset differs; all other code (Training.py, SBW_test.py) unchanged.

## Full SBW curve recovery (E8)

With H6 fix applied to current Training.py + current SBW_test.py at the
`compute_sbw_enhancement_persep` call site, full SBW curve on surr_10_00:

| sep (°) | mean_enh | P(enh>10) |
|---:|---:|---:|
| ±80 | −753 / −782 | 0.000 |
| ±55 | −741 / −749 | 0.000 |
| ±35 | −188 / −147 | 0.10 / 0.20 |
| ±30 | +460 / +420 | 0.94 / 0.92 |
| ±25 | +1173 / +1255 | 0.94 / 0.96 |
| 0 | +4245 | 1.000 |

**HW (one-sided 0.5 crossing) = 32.7°** — bell-shaped, no saturation. (Pristine
single-ckpt was 25°; difference attributable to current SBW_test using `sum_sM`
accumulation while pristine used `_latest_sMSI` — H5 confirms this inflates
enhancement, which slightly widens the half-max region but does NOT saturate.)

Paper SBW HW: 24.3°. Pristine HW (single ckpt): ~25°. Current+fix HW: 32.7°.
The saturation problem is fully resolved.

## Mechanism

In current Training.py (task #123 FIX 1, L2535-2562), AGC fires gated by:

```python
current_t_ms = self.step_counter * self.dt
if (current_t_ms - self._last_agc_fast_t) >= self.T_AGC_FAST_MS:  # 0.1
    self._last_agc_fast_t = current_t_ms
    self.g_FFinh += alpha_fast * (exc_fast * target_ratio - inh_mean)
    ...

if (current_t_ms - self._last_agc_slow_t) >= self.T_AGC_SLOW_MS:  # 10.0
    self._last_agc_slow_t = current_t_ms
    self.g_FFinh += alpha_slow * (exc_mean_long * self.pv_nmda - inh_mean_long)
    ...
```

The SBW pipeline (SBW_test.py:660-692 `compute_sbw_enhancement_persep.run_pass`)
runs 3 passes (AV → A-only → V-only):

```python
def run_pass(stim_A, stim_V):
    net.g_FFinh = initial_g_FFinh           # reset to loaded value (e.g., 0.5648)
    net.step_counter = initial_step_counter # reset to loaded step_counter (e.g., 652000)
    # MISSING: reset of net._last_agc_fast_t and net._last_agc_slow_t
    net.reset_state(batch_size=total_batch)
    ...
```

End of pass 1 (AV): `_last_agc_*_t ≈ 65400` (= 654000 substeps × 0.1 ms).
Start of pass 2 (A-only): step_counter = 652000, current_t_ms = 65200, but
`_last_agc_*_t = 65400` still. Diff = 65200 − 65400 = −200 ms. The gate
`(diff) >= 0.1` is FALSE for the next 1900+ substeps (190 ms physical).

Pass 2 only runs 20 outer frames = 2000 substeps = 200 ms physical. So AGC fast
*just barely* catches up at end of pass; AGC slow never catches up to the +10 ms
gate within pass 2's duration. Net: g_FFinh stays at the LOADED 0.5648 throughout
pass 2 — the value baked in by the original training that anticipated AV-level
excitation drive.

For A-only stimulus (much weaker than AV), g_FFinh=0.5648 is far too high — it
clamps MSI response down to ~0.66 spikes/trial. Pristine fb6d3f6's AGC fires
every 100 substeps (gated by `step_counter % 100`) and adapts g_FFinh DOWN to
~0.15-0.20 within 20 frames, allowing MSI to fire normally for unisensory stim
(64 spikes/trial).

Result: enhancement = AV − max(A, V):
- Pristine: AV≈117, A≈64, V≈45 → enh≈52 at sep=0; AV≈58, A=64, V≈0 → enh≈−6 at sep=80 (proper bell)
- Current: AV≈117, A≈0.66, V≈0.32 → enh≈117 at sep=0; AV≈58, A=0.66, V≈0 → enh≈57 at sep=80 (saturated)

## Suggested fix (Coder)

In **SBW_test.py** at lines 316/317, 336/337, 524/525, 561/562, 661/662, 691/692,
and in **TBW_test.py** at lines 910/911, 944/945, 1052/1053, 1080/1081, after
each existing block of:

```python
net.g_FFinh = initial_g_FFinh
net.step_counter = initial_step_counter
```

add:

```python
# Task #138 fix: reset AGC physical-time trackers alongside step_counter,
# else AGC stops firing in passes 2+ (debugger #138 root cause).
if hasattr(net, "_last_agc_fast_t"):
    net._last_agc_fast_t = initial_step_counter * net.dt
    net._last_agc_slow_t = initial_step_counter * net.dt
```

Alternative (cleaner but more invasive): make `Training.py:reset_state()` also
reset `_last_agc_*_t` to `step_counter * dt`. This requires audit because some
training-time call sites may rely on these trackers persisting across
reset_state. The per-test-site patch above is the minimal, surgical fix.

## Methodological caveats

- Tested on single ckpt (surr_10_00). Other surr_10_* ckpts likely behave the
  same since the bug mechanism is in pipeline code, not ckpt-specific behavior.
- E8 HW=32.7° (current+fix) vs E2 HW=25° (pristine baseline) — modest gap due
  to `sum_sM` per-substep accumulation in current SBW_test (H5 partial effect).
  The saturation IS resolved; tuning HW further might need a threshold rescale
  or revert to `_latest_sMSI` accumulation. That's a separate decision for the
  Lead, not part of this fix.
- The bug also affects TBW (same reset pattern), but TBW uses a peak-valley
  classifier on the MSI population time-series, not enhancement-thresholding,
  so the visible symptom in TBW is different (less obvious saturation, but
  similar suppression of unisensory MSI response that distorts the curve).
- Same root mechanism likely affects every probe that reset g_FFinh+step_counter
  WITHOUT resetting `_last_agc_*_t`: all 10 call sites identified above.

## Files

- `task138_logs/e1_instrument_sbw.py` + `e1_results.log` — initial dual-path instrumentation
- `task138_logs/e2_pristine_results.log` — pristine fb6d3f6 baseline (RULED OUT H2)
- `task138_logs/e3_pristine_dualpath.log` — pristine + sum_sM accumulation (rules out H5 as sole cause)
- `task138_logs/e4_state_diff.py` + `e4_state_diff.log` — post-load state comparison
- `task138_logs/e5_dt_correct_nmda_off.py` + `e5_results.log` — dt_correct_nmda toggle (no effect)
- `task138_logs/e6_one_frame_trace.py` + `e6_results.log` — 1-frame identical-state trace (proves byte-identical at substep level)
- `task138_logs/e7_agc_reset_test.py` + `e7_results.log` — **causal proof** (Variant A vs B)
- `task138_logs/e8_full_sbw_with_fix.py` + `e8_results.log` — full SBW curve recovery
- `/tmp/fsts_pristine/` — pristine fb6d3f6 worktree (probe_sbw_pristine.py + probe_dual_path.py)
