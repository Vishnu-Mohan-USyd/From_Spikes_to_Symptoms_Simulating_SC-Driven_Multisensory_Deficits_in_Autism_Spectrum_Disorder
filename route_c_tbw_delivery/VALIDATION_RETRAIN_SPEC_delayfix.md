# VALIDATION-RETRAIN SPEC — Recurrent-Delay Fix (route-C MSI SNN)

**Stored 2026-06-13** while pivoting to training-speed optimization. This is the self-contained
recipe to run the **real test** of the recurrence fix once (a) training is sped up and (b) the user
green-lights it. The fix is already PROVEN on the cheap single-seed substrate (GO); this doc exists
so nothing needed for the full retrain is lost in the pivot.

> STATUS: **HELD.** Fix proven on cheap substrate (single-seed pre-screen GO, 2026-06-13). The full
> ep0→79 retrain is the real test and is **gated on (1) the training-speed optimization landing and
> (2) the user's explicit go-ahead.** Do NOT launch without both.

---

## A. THE FIX (what to implement)

**Mechanism:** give the recurrent MSI→MSI synapse a biological **transmission delay** so the
pre-spike leads the post-spike by **1 external step = 10 ms = 100 substeps** (the primary,
biologically-defensible value). This breaks the exact LTP/LTD self-cancellation (proven root cause
H1) that pinned the recurrent weights at init — the delay gives spike-timing plasticity a consistent
direction to learn from. It is the *mechanism* doing the work: **no new knob/gain/cap/Wmax, no
symptom-masking.**

**Two coupled edits** (exact anchors per `dbg_recur_20260612/DIAGNOSTIC_REPORT_recur.md`; the coder
confirms exact lines at implementation against the live build):
1. **Recurrent forward** (~Training.py **L2983**, the instantaneous `F.linear` recurrent injection):
   inject the recurrent current from an sMSI **delayed by 1 external step**, not the same-instant sMSI.
2. **Recurrent STDP pre-trace** (~Training.py **L3479**, where the STDP call has post≡pre≡sMSI):
   compute the pre-trace off the **delayed** sMSI so pre leads post by the delay.

**Base build (delay-fix starts from this exact file):**
`/home/vishnu/coding_proj/fsts_5/fsts_perilog_20260607/code/Training.py` — **md5 466c9a76**
(recurrence ON + plastic; this is the build the pre-screen harness imports untouched). The delay is
the ONLY functional change.

**Biological bounds (do NOT violate):** primary 10 ms; grounded finer range 5–15 ms (50–150 substeps);
**HARD CEILING ~20 ms (2 steps)** — beyond that the pre-lead crosses from the LTP lobe into LTD and
re-breaks the rule. Recurrent edge read as **oligosynaptic** (SC-direct ≥3–6 ms) → 10 ms is the
defensible primary as-is; the finer ~1.4–3.4 ms substep variant is a *flagged fallback only* (needs a
cross-loop code relocation since STDP runs outside the substep loop — not worth it on a proven primary).
Sources: `researcher_sc_recurrent_delay_stdp_grounding_20260613.md` + the Phase-6 grounding entries in the handoff.

---

## B. THE PROOF that justifies the retrain (cheap-substrate GO, 2026-06-13)

Single seed (42), resume ckpt_ep25, lag=1 (10 ms), g_rec=0.1, ep26→39, vitals battery every epoch:
- **LEARNS:** ||S||sym 0.0035→0.3744 (107×, monotonic); antisym_frac 0.99→0.065 (self-cancellation
  broken); off-diag 0→0.021; **distributed** (nE>2×init=0, structural peak does not grow).
- **VITALS healthy:** rate peaks 4.77 (1.17× no-delay baseline) then descends to 1.59 (bounded, not
  climbing); E/I 24→16 (never inverts); F@SOA0=1.00 every epoch; n_active=180 stable; zero NaN.
- **Reproduced** on the A6000 (independent GPU) within ~5% through ep38.
- Archived: coder's per-epoch table (handoff 2026-06-13 entry) + `dbg_recur_20260612/out/harness_lag1_ep26_40*.json`.

Satisfies the §3 single-seed mechanism gate. **Remaining unknown the retrain answers:** does the
delayed rule (a) stay stable across a full ep0→79 *from scratch* and (b) land the biology bar.

---

## C. RETRAIN CONFIG

- **Seeds:** 42,43,44,45,46 (5-seed ensemble). **Epochs:** ep0→79 (n_unsup_epochs=80, 0-indexed;
  ckpts ep0,5,…,75; final ep79 → checkpoint/msi_redone_agc_fix_.pt).
- **Runner / overrides:** `route_c_tbw_delivery/code/run_task192_disynaptic_ep40_retrain.py`
  (runtime overrides ~L429–444). Update the md5 gate in the launch script to the new (delay-fix) hash.
  Confirm the exact launcher used for the recurrence-ON build at implementation.
- **Device:** per the user's then-current directive. **Currently A6000 (cuda:1)** — the 5090 is under
  heavy external dev load and kept free. (Re-evaluate after the speed optimization.)
- **Load hygiene (mandatory every ckpt load):** `net.tau_nmda_inh = 21.6` (ckpts serialize 45.0).
- **SINGLE-SEED-FIRST gate (§3):** run ONE seed to ep40–45 first; confirm (i) recurrent ||S||sym
  clearly leaves init (the delay took) AND (ii) vitals intact; only then commit the full 5-seed fleet.
  Never multi-seed-to-endpoint to *discover* whether it works.
- **Smoke gates + rolling (every retrain):**
  - STAGE-1 (ep5, iSTDP physics): KILL if W_*2msiInh at clamp / monotonic march to clamp; PASS = matches the validated build's ep5 iSTDP trajectory.
  - STAGE-2 (ep30, MSI emergence): KILL if recurrent W pinned-to-clamp/runaway, MSI rate >2× iSTDP target AND climbing, E/I inverted (<1), or fusion degenerate; PASS = recurrent W off-clamp/decelerating, rate bounded near target, E/I healthy, graded fusion.
  - NEW recurrence-specific check: recurrent **||S||sym leaves init by ep40–45** (else the delay didn't take — kill).
  - ROLLING (every 5 ep, ep30→80): KILL if rate diverges, inh OR recurrent weight saturates clamp, n_active drops sharply, or E/I inverts.

---

## D. VALIDATION BAR (the definition of done — validator issues GO/NO-GO)

- **TBW** + **SBW** (band [24.5, 40.9]°, ref ~31°) + **E/I balance** all match the majority of
  biological findings; every number command+output-backed.
- **The TBW `is_temporally_fused` readout is UNTOUCHABLE** (no symptom-masking anywhere).
- Apparatus: TBW `tbw_routec_curve.py` + `TBW_test.py`; SBW `sbw/{sbw_routec_run.py, run_all_sbw.sh,
  aggregate_sbw.py, apparatus/SBW_test.py}`; E/I `repo/EI_balance_test.py` (adapt to route-c
  `constructor_hparams`). Measurement load hygiene: `tau_nmda_inh=21.6` on every load.

---

## E. SOURCES / ARTIFACTS
- Root-cause proof + exact code anchors + reusable open-box harness: `fsts_perilog_20260607/dbg_recur_20260612/DIAGNOSTIC_REPORT_recur.md`, `harness_openbox.py`.
- Biology grounding (delay value + STDP window, PMID-verified): `fsts_perilog_20260607/researcher_sc_recurrent_delay_stdp_grounding_20260613.md` + handoff Phase-6 entries.
- Pre-screen GO evidence: handoff `2026-06-13 — PHASE-6 PRE-SCREEN PASSED → GO` + `dbg_recur_20260612/out/harness_lag1_ep26_40*.json`.
- Running log: `route_c_tbw_delivery/HANDOFF_TBW_FIX.md`.

## F. OPEN ITEMS before launch
1. Training-speed optimization must land first (in progress) — then re-confirm device.
2. Coder confirms the exact L2983/L3479 anchors + the recurrence-ON launcher against the live build.
3. Pre-registered, refute-designed pre-verification PASS (single-seed gate above) immediately before the fleet.
4. User's explicit go-ahead.
5. **Cadence ↔ optimized-build tail-handling (added 2026-06-13; UPDATED same day — now BUILT + validated, no longer an open blocker):** the optimized (lever-4 CUDA-graph) build runs the canonical **1000-cadence** correctly via **per-batch-size PERSISTENT pools** (a B=256 graph pool + a separate B=232 tail pool, neither ever freed/realloc'd; the B=232 tail routes eager). Validated bit-exact on a cheap gate (ep0→15 + the ep25→26 flip: max|Δ state_dict| = 0.0 every epoch incl. ep5 eager + the B=232 tail). **#34 is the full ep0→79 identity at the 1000-cadence** (the cadence this retrain uses). **Measured 5090 speed: 2.08× at 1000-cadence** (l4off 24.5 s/ep ≈ 33 min/network; l4on 11.8 s/ep ≈ 16 min/network) — the eager tail is NOT marginal; **3.3× is the cadence-relaxed ceiling** (nseq=1024, no tail, ≈12 min/network). ⇒ **run this retrain at the canonical 1000-cadence with the optimized build → ~16 min/network, 2.08×, zero tail-handling work remaining** (done). The 1024 path is a ceiling footnote, not a deployment target. (Build at batch_size=256, NOT nseq — a wide-init buffer set is freed on reset and segfaults the graph replay; coder-proven fix.)
