# DIAGNOSTIC REPORT — L8 fused-graph numeric byte-FAIL on `W_inA`/`W_inV` (task #73)

**Debugger forensic. Diagnosis only — NO fix applied. Build-under-test `Training_graphdf.py` md5
`c4fea86c049f58d8d2a336113be23275`; frozen baseline `Training_delayfix.py` md5
`5e7d6d20538592b5fc92165b371949d7` — both VERIFIED byte-untouched (the GPU experiments use flag-gated
monkeypatches on the imported module in memory; the build FILE is never edited). cuda:0 (RTX 5090),
serialized after #70. Probes: `diag_l8_norm.py`, `diag_l8.py`, `diag_l8b.py`, `diag_l8c.py`; logs in
`diag_out/l8_norm.log`, `l8_expcause{,2,3}.{log,json}`.**

---

## PROVEN ROOT CAUSE (single-variable causal flip)
The byte-divergence is caused by the **in-graph execution of the unimodal lateral-competition's
neighbour-mask matmul** `s = torch.matmul(M, r)` — `apply_local_competition_unimodal_fast`, graphdf
**L520** — once the L8 fusion captures the post-loop plasticity. Run that competition **eager**
post-replay and `W_inA`/`W_inV` become **byte-identical** (max|Δ|=0); leave it **in-graph** and they
diverge by the exact 1.579e-4/1.580e-4. It is a CUDA-graph-vs-eager **GEMM kernel-selection**
difference, NOT a stale pointer, NOT the `.norm`/`.median` row-scaling, NOT the cache, NOT the STDP
tail, NOT the anchor.

> **Correction of my own earlier (code-only) finding.** My pre-GPU report localized the cause to the
> `W_in*`-only `.norm`/`.median` row-scaling by elimination. The decisive GPU flip **REFUTED** that:
> moving the `.norm`/`.median` eager did NOT fix the gate (`l8_norm_eager` still FAILed at the
> identical 1.58e-4). Elimination-by-census was wrong because it assumed "matmul is byte-identical
> in-graph (proven via the MSI weights)" — true only at ep26 for the MSI mask (dist=6); at ep0 nothing
> covered the unimodal matmul (dist=4), and the in-graph GEMM picks a different kernel than eager. The
> protocol's rule (prove by causal flip, never by elimination alone) is what caught this.

---

## Failure (reproduced) — `gate_fused_clean.out` and every probe
`gate_fused.py`: ref (frozen delayfix, eager), l6 (graphdf, substep-loop graphed + EAGER post-loop
plasticity & STDP), l8 (graphdf, FULL body fused into one capture). One train epoch; `torch.equal` at
ep0 (g_rec=0) AND ep26 (g_rec=0.1). `ref==l6` PASS both; `ref==l8` and `l6==l8` FAIL both, on exactly
`W_inA`(1.579e-4) / `W_inV`(1.580e-4) + the four `_latest_*`(|Δ|=1.0). Clean run (no crash) ⇒ numeric,
not a capture failure. The L8 fusion (L6→L8) is the sole changed variable.

## Hypothesis ledger (every verdict by a run, not by argument)

| # | Hypothesis | Verdict | Decisive evidence |
|---|---|---|---|
| H1 | `_latest_*` repoint omission feeds STALE spikes into the in-graph plasticity → wrong `W_in*` | **RULED OUT** | the four `_latest_*` |Δ|=1.0 are the #66 benign escaped pointer; in-graph plasticity reads the correct in-graph spikes (`_latest_*` set L3433-3436 before plasticity L3569); `l8_comp_eager` repoints them and W_in still goes byte-identical |
| H2 | the `W_in*`-only `.norm(p=2,dim=1)`/`.median()` row-scaling (competition L526-534 + soft_row_scaling) diverges in-graph | **RULED OUT (GPU flip)** | `l8_norm_eager` (renorm+srs eager, rest fused) STILL FAILs at 1.579e-4 — `diag_l8.py` |
| H3 | the in-graph STDP tail corrupts `W_in*` | **RULED OUT (GPU flip)** | `l8_stdp_eager` (tail eager) STILL FAILs at 1.579e-4 — `diag_l8.py`; and `_stdp_tail` shares reduction-free `stdp_update_batch` with the byte-identical MSI weights |
| H4 | `_p_add` accumulation differs in-graph | **RULED OUT** | elementwise (L1927); proven byte-identical in-graph by L6's CAPTURED in-loop anchor (uses `_p_add`) being byte-identical to ref |
| H5 | the post-loop topographic anchor (sigma=2.5, Gaussian kernel) diverges in-graph | **RULED OUT (GPU flip)** | `l8_anchor_only` keeps the anchor IN-GRAPH (comp/msi/srs/stdp eager) and W_in is byte-identical — `diag_l8b.py` |
| H6 | the gaussian/neighbour-mask cache VALUES are computed under capture and differ | **RULED OUT (GPU flip)** | `l8_prewarm` populates all caches eagerly, full L8 in-graph, and STILL FAILs at the identical 1.579e-4 — `diag_l8c.py` |
| **H7** | **the in-graph unimodal-competition matmul `s=matmul(M,r)` (L520) diverges (GEMM kernel selection)** | **CONFIRMED (GPU flip)** | competition EAGER → W_in max|Δ|=0 (PASS); competition IN-GRAPH → 1.579e-4 (FAIL); single variable = competition location — `diag_l8b.py`/`diag_l8c.py` |

## Causal proof (single variable: competition in-graph vs eager) — `diag_l8c.py`, both epochs
Direct max|Δ| on `W_inA`,`W_inV` (NOT the benign `_latest_*` pointers):
```
ep0 / ep26   (W_inA, W_inV)            verdict
  l8_base        1.579e-4, 1.580e-4    FAIL   (real L8: competition IN-GRAPH)
  l8_prewarm     1.579e-4, 1.580e-4    FAIL   (caches eager-warmed, competition IN-GRAPH) -> not the cache
  l8_comp_eager  0.000e+0 , 0.000e+0   PASS   (competition EAGER post-replay) -> byte-identical
```
Flipping ONLY whether the unimodal competition runs in-graph vs eager flips W_in FAIL↔PASS; reverting
(`l8_base`) restores the FAIL. That is the causal proof.

## Pinning the op WITHIN the competition (controls, not argument)
`apply_local_competition_unimodal_fast` LTD half (L515-524): `r=spikes.mean(0)` (L517);
`M=neighbour_mask(4)` (L519, cached); `s=matmul(M,r)` (L520); `dW=-β(r·s)·W` (L523, elementwise);
`_p_add` (L524). Plus the renorm L526-534.
- `mean(0)` and `_p_add` are byte-identical in-graph — **proven** by L6's CAPTURED **in-loop** anchor
  (graphdf L3485, sigma=3.0), which uses both `spikes.mean(0)` and `_p_add` and is byte-identical to
  ref at ep0 (the substep loop, incl. the in-loop anchor, is captured by BOTH L6 and L8; L6==ref).
- the renorm `.norm`/`.median` is ruled out by H2's flip; the elementwise `(r·s)·W` is byte-identical;
  the cache is ruled out by H6's flip.
- the anchor has **no matmul** (it uses a Gaussian kernel + elementwise broadcast, L350-352), so the
  matmul `s=matmul(M,r)` (L520) is the ONLY op in the divergent region not covered by a byte-identical
  control ⇒ it is the divergent op. `diag_l8_norm.py` shows this matmul is byte-identical in
  ISOLATION — consistent: the divergence is context-dependent GEMM kernel selection inside the full
  L8 capture, not a property of the op in isolation.

Why the forward matmuls are fine but this one isn't: every substep-loop matmul is captured by BOTH L6
and L8 and certified byte-identical by L6==ref; the competition matmul lives ONLY in L8's post-loop
capture and was never byte-validated — in the full L8 graph it binds a different GEMM kernel than the
eager path.

## FIXABLE vs FUNDAMENTAL
- **FUNDAMENTAL (for that op in a graph):** an in-graph GEMM will not be guaranteed to bit-match the
  eager GEMM (kernel autotuning differs under capture), so FULL fusion of a matmul-containing
  post-loop op cannot be byte-identical. Pre-warming caches does NOT help (H6).
- **FIXABLE (the gate):** run the unimodal competition EAGER post-replay (exclude it from the L8
  capture) — `l8_comp_eager` PROVES this yields byte-identical `W_inA`/`W_inV` at ep0 AND ep26. This
  is the L6 discipline (L6 runs ALL post-loop plasticity eager and is byte-identical/certified). The
  minimal exclusion proven sufficient is the **unimodal competition**; excluding the whole post-loop
  block (= L6) is the conservative superset. Either keeps the substep-loop fusion (the bulk of the
  speedup) while dropping only the post-loop competition from the graph.
- Independently, add the `_latest_*`→`_g_out_*` repoint (L6 does this at L3816-3819; L8 omits it) to
  clear the benign `_latest_*` |Δ|=1.0 escaped-pointer diffs — cosmetic to the gate, not the W_in cause.

## Suggested fix direction (Coder — gated on this proof; the edit is theirs)
In the L8 capture path, do NOT fold the unimodal lateral competition into the graph: capture the
forward substeps (+ in-loop anchor, already byte-safe) + the post-loop anchor, and run
`apply_local_competition_unimodal_fast` (A/V) — at minimum — EAGER after `g.replay()` (mirroring L6),
plus soft_row_scaling and the STDP tail if convenient. Add the `_latest_*`→`_g_out_*` repoint. Re-run
`gate_fused.py` → expect byte-PASS (l8==ref==l6) at ep0 and ep26. NO papering / no tolerance change.

## Remaining unknowns
- None for the cause: confirmed by causal flip + controls at both epochs. The precise cuBLAS kernel
  IDs (eager vs in-graph) are not enumerated — not needed; the flip is dispositive and the fix
  (exclude the op from capture) does not depend on which kernels they are.
