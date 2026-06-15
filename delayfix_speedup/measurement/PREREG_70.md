# PRE-REGISTRATION — Task #70 scientific-equivalence of the bs=250 relax

**Registered 2026-06-15, BEFORE any bs=250/baseline curve exists** (kill-criteria-before-data rule).
Refute-designed: the default verdict is **NO-GO**; GO requires equivalence to be affirmatively shown
on every observable. A NO-GO ships the 19.4-min bit-identical build; a GO ships the 12.2-min bs=250 build.

## What is compared
- **BASELINE** = TBW/SBW/E-I curves of the **bit-identical bs=256** model (byte-equal to the frozen
  delayfix build → its curves ARE the delayfix science).
- **TEST** = same curves of the **relaxed bs=250** model (4×250 even batch; weights no longer
  byte-equal, so torch.equal is replaced by this curve-equivalence test).
- Both measured through ONE apparatus: `val36_traj.py` (delayfix delayed-recurrence forward).

## Locked measurement protocol (identical for both models)
- `VAL36_BUILD=delayfix`; netsrc md5 `5e7d6d20`, TBW md5 `80d33465`, SBW md5 `73b7d136` (logged in each JSON).
- `tau_nmda_inh=21.6` forced; per-epoch `g_rec` (ep79 → 0.1); `plasticity_enabled=False`; measurement batch 256.
- **Same measurement `--seed` for BOTH** → stimulus + membrane-noise draws are COMMON-MODE and cancel
  in the Δ, isolating the weight effect. (If the two ckpts were trained under different seeds, I build
  each with its own training seed and flag the provenance.)
- Probe grids/trials as in harness: TBW 31 SOA pts (−300..+300 ms, 20 ms), SBW 33 sep pts (−80..+80°, 5°),
  E-I 7 offsets; n_trials=50.

## Noise floor → tolerance sizing
Probes are Monte-Carlo (n_trials=50): per-point p_fusion SE = √(p(1−p)/50) ≤ 0.071 (worst p=0.5; ≈0 on
the saturated p∈{0,1} plateau/floor). With the **same seed**, run-to-run draws are common-mode, so the
paired same-model floor is ~1 trial-quantum (1/50 = 0.02) at transition bins, ~0 elsewhere. The relax is
only mini-batch 256→250 (~2.3%) → expect tiny deltas. Tolerances are set a few× the ~0.02 floor: tight
enough to ignore batch wobble, wide enough only for a REAL regime change (edge bins move ≥0.3 / FWHM ≥100 ms
/ E/I flips across 1).

## Per-observable acceptance criteria (GO needs BOTH columns to pass)

| Observable | QUALITATIVE (regime) | QUANTITATIVE (vs baseline) |
|---|---|---|
| **TBW** | curve `shape` class preserved; central fusion intact: P(fusion)@SOA0 ≥ 0.5 | p_fusion curve max\|Δ\| ≤ **0.10**; box_width50 ±**40 ms**; robust-FWHM ±**40 ms**; peak ±**0.07**; P@0 ±**0.07** |
| **SBW** | in-band boolean matches baseline (band sits in [24.5, 40.9]°); non-degenerate band present (finite half-width AND p_fusion relief ≥ 0.2) | p_fusion curve max\|Δ\| ≤ **0.10**; half-width ±**5.0°** |
| **E/I** (membrane ⟨I_M⟩/⟨I_M_gaba⟩, target ~1) | same side of balance=1 as baseline (baseline inh-dominated ~0.46–0.58 → test must stay <1, not flip exc-dominated) | relative \|Δ\| ≤ **0.20** on BOTH ei_ratio_sync and ei_ratio_offsetmean |

## Verdict logic
**GO** iff every qualitative AND quantitative check above passes. **NO-GO** if ANY fails → escalate to lead
→ ship 19.4-min bit-identical fallback. (Encoded in `compare_ab.py`; exit 0=GO / 2=NO-GO / 3=incomplete.)

## Explicit kill list (any one → NO-GO)
- TBW shape flips (box/plateau → graded/degenerate), OR curve max\|Δ\| > 0.10, OR box_width50 shift > 40 ms,
  OR FWHM shift > 40 ms, OR peak drop > 0.07, OR P@0 drop > 0.07, OR P@0 < 0.5.
- SBW band lost (degenerate / half-width NaN / relief < 0.2), OR in-band boolean flips, OR half-width Δ > 5.0°,
  OR curve max\|Δ\| > 0.10.
- E/I flips regime across 1.0, OR relative \|Δ\| > 0.20 on either ratio.

## Tool status
`compare_ab.py` encodes exactly the above; defaults = these numbers; all CLI-overridable. Validated CPU-only:
identical→GO (14/14), regime-change→NO-GO (8 FAIL), within-noise wobble (curve +0.03 / hw +1.9° / E/I +5%)→GO.

## Optional empirical anchor (if lead wants the % measured, not theoretical)
After the baseline ckpt lands (and only with GPU authorization, serial after the coder): measure the
bit-identical baseline TWICE (same seed → expect ≈0 Δ confirming determinism; or two seeds → observe the
true MC band), then keep tol = max(registered, 3× observed floor). The registered numbers stand as the
locked default regardless.
