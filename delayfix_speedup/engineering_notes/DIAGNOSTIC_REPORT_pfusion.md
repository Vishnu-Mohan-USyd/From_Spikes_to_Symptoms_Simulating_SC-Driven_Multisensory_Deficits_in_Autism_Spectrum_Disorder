# DIAGNOSTIC REPORT — ep0 p_fusion 0.05 bit-identity mismatch (task #56)

**Debugger forensic. Diagnosis only — NO fix applied. Build `Training_delayfix.py` md5
`5e7d6d20…` untouched; `TBW_test.py` untouched. All compute cuda:0 / RTX 5090. Every claim below
is backed by a command + captured output in `diag_out/`. The harness `diag_pf.py` monkeypatches
`np.random.default_rng` *in-process only* — nothing on disk was edited.**

---

## Failure (reproduced)
`measopt_20260614 && CUDA_VISIBLE_DEVICES=0 python -u test_bitidentity.py` (`bitid_gate_lead.log`):
- **ep0 (g_rec=0.0): p_fusion max|Δ| = 5.000e-02 → MISMATCH** (FAIL). All 4 scalar vitals
  (v_rate, v_EI, v_F0, v_TBW) BIT-IDENTICAL; full-panel-dict mismatches (excl P6_bc) = 0.
- ep26 (g_rec=0.1): p_fusion max|Δ| = 0.000e+00 → BIT-IDENTICAL (PASS).
- 0.05 = exactly 1 trial of n_trials=20 flipping at one offset of the 27-offset grid.

`run_case` (test_bitidentity.py:53-124) builds a **fresh `build_net(42)`** for both baseline and
graphed nets at each epoch; "ep0" vs "ep26" differ **only** in `g_rec` (0.0 vs 0.1) — no checkpoint
is loaded.

## Localization
p_fusion is produced by `compute_tbw_temporal_fusion_persep` (TBW_test.py:890-978), called at
B=540. On the graphed pnet the panel wrapper routes **B=540 → eager `ORIG`** (panelgraph.py:271-279;
only B=8 volley is graphed) — so the CUDA graph a priori cannot touch p_fusion.

Inside that function (verified by reading TBW_test.py:938-967):
- **L940 `rng = np.random.default_rng()`** — UNSEEDED (no argument → fresh OS entropy every call;
  NOT controlled by the `np.random.seed(42)` that `direct_tbw_pfusion` / `run_panel` set, because
  `default_rng()` is an independent Generator, decoupled from the legacy global RNG).
- **L948 `all_locs = rng.integers(0, net.space_size, size=(n_offsets, n_trials))`** — the *only* use
  of `rng`; the per-trial stimulus spatial locations.
- L965 `noise_std=0.0`; forward pass in `torch.inference_mode()`.
⇒ `all_locs` is the **sole stochastic input** to p_fusion, and it is redrawn every call.

---

## Hypotheses tested

| # | Hypothesis | Verdict | Evidence (command → output) |
|---|---|---|---|
| A | Inherent non-determinism: baseline p_fusion not reproducible run-to-run | **CONFIRMED** | `diag_pf.py` unseeded: baseline-vs-baseline max|Δ|=**0.15** (g_rec=0) / 0.05 (g_rec=0.1) |
| B | graphdf-eager ≠ delayfix-eager (module FP-order) | **RULED OUT** | seeded: baseline-vs-graphdf_eager max|Δ|=**0.000e+00** at both g_rec |
| C | Graph-induced (volley CUDA graph perturbs p_fusion) | **RULED OUT** | wrapper routes B=540→eager; real gate seeded: ep0 0.05→**0.000 PASS**, ep26 PASS |
| — | GPU kernel non-determinism (atomics) flips shoulder trials | **RULED OUT** | seeded: baseline-vs-baseline = **0.000e+00** (would survive seeding if real) |

### Controlled matrix — single variable = is `np.random.default_rng` seeded
`diag_pf.py` builds two fresh `build_net(42)` delayfix nets (A: baseline-vs-baseline) and one
graphdf-eager net (B), runs `compute_tbw_temporal_fusion_persep` on each, at g_rec ∈ {0.0, 0.1}.
Each call already seeds random/np.random/torch/cuda to 42 (== the gate). The ONLY difference between
the two runs below is the `default_rng` monkeypatch.

**UNSEEDED** (`diag_out/pf_unseeded.log`, == the gate's real condition):
| g_rec | A baseline-vs-baseline | B baseline-vs-graphdf_eager |
|---|---|---|
| 0.0 | **max|Δ| = 1.500e-01** (at offsets −10, +8) | 1.500e-01 |
| 0.1 | max|Δ| = 5.000e-02 | 1.000e-01 |
All variation sits on the fusion-curve **shoulders** (e.g. p_base1[+8]=1.00 vs p_base2[+8]=0.85);
the interior (1.0) and exterior (0.0) offsets are identical every run.

**SEEDED** `np.random.default_rng → default_rng(777)` (`diag_out/pf_seeded.log`):
| g_rec | A baseline-vs-baseline | B baseline-vs-graphdf_eager |
|---|---|---|
| 0.0 | **max|Δ| = 0.000e+00** | **0.000e+00** |
| 0.1 | **max|Δ| = 0.000e+00** | **0.000e+00** |

---

## Proven root cause
**The metric `compute_tbw_temporal_fusion_persep` draws its per-trial stimulus locations from an
UNSEEDED `np.random.default_rng()` (TBW_test.py:940 → L948 `all_locs`), so every invocation uses
different stimuli and p_fusion is non-deterministic run-to-run** — up to **0.15** between two
identical baseline nets. The gate's requirement that graphed p_fusion be *bit-identical* to baseline
is therefore **unsatisfiable: the baseline is not bit-identical to itself.** The ep0 0.05 is one
draw of this noise; ep26's 0.000 was a lucky draw (ep26 baseline-vs-itself also varies — 0.05 above).

**Causal proof (single variable):** monkeypatching `np.random.default_rng` to a fixed seed — and
changing *nothing else* — collapses max|Δ| from **0.15 → 0.000e+00** for BOTH baseline-vs-baseline
(A) and baseline-vs-graphdf-eager (B), at BOTH g_rec values. The four pre-existing seedings
(random/np.random/torch/cuda) do not tame it; only seeding `default_rng` does ⇒ it is necessary and
sufficient. This simultaneously **rules out B** (modules are bit-identical once stimuli are fixed)
and **GPU non-determinism** (which would survive RNG seeding).

**C (graph) ruled out:** the wrapper runs the B=540 TBW eager (panelgraph.py:271-279), and
graphdf-eager == delayfix-eager bit-for-bit (B, seeded). End-to-end confirmation
(`run_gate_seeded.py` = the REAL gate `test_bitidentity.main`, full graphed pipeline incl. volley
graph capture + graphed pnet, single variable = `default_rng` seeded; `diag_out/pf_gate_seeded.log`):
**ep0 flips FAIL → PASS: max|Δ p_fusion| = 0.000e+00 (BIT-IDENTICAL), all 4 vitals OK, P6_bc
Δ=0.000e+00, full-dict mismatches=0**, panel 91.2s→22.0s. Final seeded gate VERDICT: **ep0 PASS /
ep26 PASS** (vs the unseeded gate's ep0 FAIL / ep26 PASS) — seeding `default_rng` is the single
change that flips ep0. With the CUDA graph fully active, p_fusion is bit-identical once — and only
once — the location RNG is seeded. (The lead independently
reproduced `diag_pf.py`: unseeded A=0.10/0.05, seeded A=B=0.000 — and got 0.10 where this run got
0.15, the run-to-run-of-the-difference itself further corroborating A.)

Mechanism: `is_temporally_fused` is a threshold classifier (scipy gaussian_filter1d + find_peaks +
valley/peak ratio). At shoulder offsets (≈±8–10 ms) a trial sits near the fusion threshold; a
different random location flips its boolean → ±1/n_trials = ±0.05 per flipped trial. Interior and
exterior offsets are far from threshold and never flip — exactly the observed pattern.

## Suggested fix direction (Coder decides — NOT applied here)
This is a **test/metric RNG defect, not a graph or module defect** (graph + graphdf proven
bit-identical). Options:
1. Seed the location draw in `compute_tbw_temporal_fusion_persep` — e.g. accept a `seed`/`rng` arg
   and use `np.random.default_rng(seed)` so p_fusion is reproducible (then the gate's bit-identity
   bar becomes satisfiable).
2. Gate p_fusion with a tolerance, not bit-identity: the run-to-run envelope is ≥0.15 at n_trials=20,
   so any bit-identity bar on p_fusion is invalid as written.
3. Increase n_trials to shrink shoulder variance (does not make it bit-identical; combine with #2).
Do NOT "fix" by tuning the graph/module — they are already exact.

## Remaining unknowns
- None material. The shoulder trials are genuinely near the `is_temporally_fused` threshold
  (p=0.95 at offset −10 even when seeded = 19/20 fused), so location changes legitimately flip them;
  that is expected classifier behavior, not a separate bug.
