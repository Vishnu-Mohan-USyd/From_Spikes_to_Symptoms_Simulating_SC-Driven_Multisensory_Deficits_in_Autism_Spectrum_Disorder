# delayfix_speedup — Route-C delay-fix trainer + CUDA-graph speedup

This module packages the **route-C disynaptic feed-forward-inhibition (FFI) delay-fix** model
together with a **CUDA-graph training port** that reduces the 80-epoch wall from ~8 h to a
**measured 12.23 min (~39×)** without changing the scientific output.

It contains three things:

1. `code/` — the frozen eager reference trainer and its CUDA-graph speedup port, plus the drivers
   used to measure walls and gate byte-equivalence.
2. `measurement/` — the TBW / SBW / E-I trajectory apparatus and the scientific-equivalence A/B
   harness used to certify that the fast build reproduces the reference curves.
3. `engineering_notes/` — the forensic diagnostic reports and the running performance log that
   document, step by step, how the speedup was built and why each lever was kept or rejected.

---

## 1. The model — route-C disynaptic FFI delay-fix

The route-C lineage routes multisensory-integration (MSI) inhibition through a **disynaptic
feed-forward path** (excitatory drive → inhibitory interneuron → excitatory target) rather than a
direct monosynaptic inhibitory contact. The "delay-fix" is the corrected conduction-delay /
inhibitory-kinetics configuration on that path.

The frozen reference implementation is **`code/Training_delayfix.py`** (md5
`5e7d6d20538592b5fc92165b371949d7`). It is treated as the scientific ground truth: it is **not**
modified by this module — every speedup is validated *against* it.

The model reproduces two validated psychophysical signatures and one balance constraint:

- **TBW (temporal binding window)** — P(fusion) as a function of audio–visual stimulus-onset
  asynchrony (SOA). The route-C build produces a wide box/plateau (box50 ≈ 480 ms) under the
  `is_temporally_fused` readout.
- **SBW (spatial binding window)** — P(fusion) as a function of angular separation; a bell with
  half-width ≈ 44°.
- **E/I balance** — the primary membrane ratio ⟨I_M⟩/⟨I_M,GABA⟩ measured per SOA offset
  (inhibition-dominated regime).

These three observables are the acceptance bar for every change.

---

## 2. The speedup — CUDA-graph training port

`code/Training_graphdf.py` (md5 `6daa16354d47470a27b546460450ecc9`) is the delay-fix model ported
onto a **CUDA-graph capture of the per-substep training loop**.

### Why it works

Profiling (`engineering_notes/perf_profile.md`, task #48) proved the original ~8 h wall was
**launch-bound** — dominated by CPU kernel-dispatch overhead, not GPU compute. Capturing the
substep loop into a CUDA graph collapses that per-substep dispatch cost into a single replay.
Because the bound is the host CPU + PyTorch/CUDA/driver stack rather than GPU FLOPs, the wall is
set by single-thread host performance, so portability depends on the host CPU and software stack,
not merely "a fast GPU".

### What shipped — the L6 build (CERTIFIED)

The shipped configuration is the **L6 lever set at batch size 250** (even batch, no partial tile).

- **Measured 80-epoch wall = 733.75 s = 12.23 min** (seed 42), down from ~8 h ≈ **~39×**.
- It is **byte-identical** to the eager `Training_delayfix.py` reference output, so no equivalence
  drift is possible by construction.
- It was additionally certified by the scientific-equivalence A/B (see §4): **GO, 14/14 checks
  PASS**.

This is the build to use for ongoing experiments: every ~12-min run reproduces the reference
network exactly.

### What was rejected — the L8-full build (NO-GO)

A more aggressive lever (`code/retrain_l8full.py`) folds the unimodal **competition GEMM**
(`s = matmul(M, r)`) into the captured graph. It reaches a **measured 5.999 min** (under a 10-min
stretch target) — but it **fails byte-equivalence**:

- Forensic root cause (task #73, `engineering_notes/DIAGNOSTIC_REPORT_l8.md`): an in-graph GEMM
  selects a **different cuBLAS kernel under graph capture than in eager mode**, so `W_inA`/`W_inV`
  diverge by 1.58e-4. This is fundamental — an in-graph GEMM cannot bit-match the eager path.
- The divergence propagates to the curves: on the equivalence A/B the **TBW P(fusion) curve
  diverges max|Δ| = 0.28** at the −260 ms leading box edge (reference 0.98 → L8-full 0.70),
  tripping the locked anchored tolerance of 0.12 (2.3×). SBW, E/I, and the MSI proxy all stayed in
  range, but TBW alone is a real, weight-induced mismatch.
- **Verdict (task #76): NO-GO.** The 5.999-min build was rejected; the certified 12.23-min L6
  build shipped instead.

The clean dose–response across lever aggressiveness — reference 0.98 → L6 0.88 (Δ0.10, GO) →
L8-full 0.70 (Δ0.28, NO-GO) — is the evidence that the L6 cut is safe and the L8 cut is not.

---

## 3. The metrics — TBW / SBW / E-I

The measurement apparatus lives in `measurement/`:

- **`val36_traj.py`** — the single-checkpoint trajectory driver. For a given checkpoint it loads the
  network (eager `Training_delayfix.py` by default), runs the TBW SOA sweep, the SBW separation
  sweep, and the per-offset E/I panel, and writes a JSON result. Selected by the env var
  `VAL36_BUILD` (`delayfix` = frozen reference, the default).
- **`agg36_traj.py`** — aggregates per-checkpoint JSONs into an across-epoch trajectory.
- **`msi_proxy.py`** — a cheap MSI-rate proxy used as an early pathology flag during screens.
- **`compare_ab.py`** — the scientific-equivalence comparator (see §4).

The TBW/SBW *physics* lives in `TBW_test.py` and `SBW_test.py`. **These are reused from
`route_c_tbw_delivery/` in this same repo** (see the path note in §6) — they are not duplicated
here.

---

## 4. Equivalence validation — the #70 protocol

Because the speedup must be **output-preserving**, every candidate build is judged against the
reference curves by a pre-registered A/B, not by eyeballing.

- **Pre-registration:** `measurement/PREREG_70.md` fixes the comparison and the kill criteria
  *before* the data is seen.
- **Protocol:** both builds are measured with `val36_traj.py` at ep79, `tau_nmda_inh = 21.6`,
  `g_rec = 0.1`, plasticity off, using the **same measurement seed** so Monte-Carlo sampling noise
  is common-mode.
- **Comparator:** `compare_ab.py` checks the full TBW and SBW P(fusion) curves, the box/FWHM
  summaries, the peak and P@SOA0, the SBW half-width and band, and the E/I membrane ratio + regime.
- **Anchored tolerances:** the curve tolerances are anchored at **3× the empirical same-model
  Monte-Carlo noise floor** — `tbw_curve_tol = 0.12`, `sbw_curve_tol = 0.36` — derived from a
  second-seed re-measure of the identical model (`anchor_70.sh`). A FAIL is therefore a real
  weight-induced difference, not sampling noise.
- **Result for the shipped build:** the relaxed bs250 build vs the bit-identical bs256 baseline
  returned **GO, 14/14 checks PASS** (TBW curve max|Δ| = 0.10 < 0.12; SBW max|Δ| = 0.12 < 0.36;
  E/I rel Δ < 0.0015, same inhibition-dominated regime).

Run scripts: `measurement/run_70.sh` (the bs250 vs bs256 certification), `measurement/anchor_70.sh`
(noise-floor anchoring), `measurement/run_76.sh` (the L8-full re-validation that produced the
NO-GO).

---

## 5. File manifest

```
delayfix_speedup/
├── code/
│   ├── Training_delayfix.py     frozen eager reference trainer (md5 5e7d6d20) — scientific ground truth
│   ├── Training_graphdf.py      CUDA-graph speedup port (md5 6daa163) — the L6 12.23-min build
│   ├── retrain_l8full.py        L8-full retrain driver (in-graph competition GEMM; 5.999 min, NO-GO)
│   ├── wall_measure.py          end-to-end 80-epoch wall measurement
│   ├── ladder_measure.py        per-lever wall ladder (which lever buys how much)
│   ├── gate_fused.py            byte-equivalence gate vs the eager reference
│   ├── run_baseline_gated.py    gated baseline run (delayfix vs graphdf)
│   └── epoch_breakdown.py       per-epoch timing breakdown / profiler
├── measurement/
│   ├── val36_traj.py            single-checkpoint TBW+SBW+E/I trajectory driver (VAL36_BUILD env)
│   ├── agg36_traj.py            across-epoch trajectory aggregator
│   ├── msi_proxy.py             cheap MSI-rate pathology proxy
│   ├── compare_ab.py            scientific-equivalence A/B comparator
│   ├── PREREG_70.md             pre-registration + kill criteria for the equivalence A/B
│   ├── run_70.sh               certify bs250 vs bit-identical bs256 (GO)
│   ├── anchor_70.sh            anchor curve tolerances to the same-model MC noise floor
│   └── run_76.sh               L8-full re-validation (NO-GO)
└── engineering_notes/
    ├── perf_profile.md          task #48 — proof the wall is launch-bound, not compute-bound
    ├── DIAGNOSTIC_REPORT_l4.md  L4 alloc-free reset byte-FAIL + NaN forensic
    ├── DIAGNOSTIC_REPORT_l6.md  L6 graph byte-FAIL (_latest_sMSI_inh pointer) — proven benign
    ├── DIAGNOSTIC_REPORT_l7.md  L7 STDP-graph segfault forensic
    ├── DIAGNOSTIC_REPORT_l8.md  L8 fused-graph byte-FAIL (W_inA/W_inV 1.58e-4) — root cause
    ├── DIAGNOSTIC_REPORT_segfault.md   two-coexisting-graphs mutual corruption (root cause)
    ├── DIAGNOSTIC_REPORT_panel66.md    panel-replay buffer forensic
    ├── DIAGNOSTIC_REPORT_pfusion.md    ep0 p_fusion mismatch on graphed panel
    ├── DIAGNOSTIC_REPORT_train_port.md graphdf vs delayfix training-path divergence
    ├── GATE_RESULTS_seed42.md   byte-gate results for seed 42
    └── PERF_PANEL_LOG.md        the full running performance log (every lever, gate, verdict)
```

---

## 6. Path assumptions (read before running)

The measurement drivers use **absolute paths** keyed to the original workstation layout. They are
**not** auto-discovered — adapt them to your checkout before running.

`measurement/val36_traj.py` (lines 37–41) hardcodes:

```python
ROOT     = "/home/vishnu/coding_proj/fsts_5/fsts_perilog_20260607"
CODE     = ROOT + "/code"                                          # canonical Training.py
DELAYFIX = ROOT + "/delayfix_retrain_20260614/Training_delayfix.py"  # the frozen reference trainer
TBW_DIR  = "/home/vishnu/coding_proj/fsts_5/repo/route_c_tbw_delivery/code"        # TBW_test.py
SBW_DIR  = "/home/vishnu/coding_proj/fsts_5/repo/route_c_tbw_delivery/sbw/apparatus"  # SBW_test.py
```

Notes:

- **`DELAYFIX`** points at the frozen reference trainer. The copy shipped in this module is
  `code/Training_delayfix.py` (same md5 `5e7d6d20`); repoint `DELAYFIX` at it, or at your own
  checkout of the original.
- **`TBW_DIR` / `SBW_DIR`** point at the TBW/SBW physics in **`route_c_tbw_delivery/`** in this same
  repository. That apparatus is intentionally shared, not duplicated — keep `route_c_tbw_delivery/`
  present, or repoint these two paths at it.
- The `code/` drivers load the trainers by **`importlib.import_module("Training_graphdf")`** (and
  `"Training_delayfix"`), so they must be run from within `code/` (or with `code/` on
  `sys.path`) — which is why the trainers are kept flat in `code/` alongside their drivers.
- Checkpoints (`.pt`), figures (`.png`/`.npz`), and logs (`.log`) are intentionally **not** tracked
  (see the repo `.gitignore`); regenerate them with the drivers above.

---

## 7. Provenance

Every quantitative claim in this README is backed by a command + output recorded in
`engineering_notes/PERF_PANEL_LOG.md` and the forensic reports. Key anchors:

- 12.23-min L6 wall — PERF_PANEL_LOG, bs250 MEASURED entry (733.75 s, 80 ep, seed 42).
- 5.999-min L8-full wall + NO-GO — PERF_PANEL_LOG, task #75/#76 checkpoint blocks.
- L8 byte-FAIL root cause (1.58e-4 GEMM divergence) — `DIAGNOSTIC_REPORT_l8.md` (task #73).
- Equivalence GO (14/14) + anchored tolerances — the #70 A/B verdict and `anchor_70.sh`.
