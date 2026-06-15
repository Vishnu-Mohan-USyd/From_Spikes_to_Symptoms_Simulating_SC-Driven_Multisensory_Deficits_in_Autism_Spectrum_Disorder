# DIAGNOSTIC REPORT — two-graph panel replay segfault (task #55)

**Debugger forensic. Diagnosis only — NO fix applied. Build `Training_delayfix.py` md5
`5e7d6d20538592b5fc92165b371949d7` untouched. All compute cuda:0 / RTX 5090. Every claim below is
backed by a command + captured output in `diag_out/`.**

---

## Failure (reproduced)
`cd measopt_20260614 && CUDA_VISIBLE_DEVICES=0 python -u test_bitidentity.py`
→ **EXIT=139 (SIGSEGV, core dumped)** immediately after `[panel-net] capture+install = 9.23s`
(`repro_bitid.log`). Baseline panel vitals reproduce the reference exactly
(v_rate=1.1968166828155518, v_EI=24.789452787300185, v_F0=1.0, v_TBW=180.0) and weight transfer is
exact (max|Δ|=0.000e+00). The crash is on the FIRST graph **replay**, not capture. Deterministic.

## Localization
faulthandler (`diag_out/smoke2.log`): crash at `panelgraph.py:185` = **`tg.replay()`** (the TBW
B=540 graph — the first replay in `_panel_battery`), inside
`torch/cuda/graphs.py:143 → CUDAGraph.replay()`. All variants crash in `CUDAGraph.replay()`.

---

## Hypotheses tested

| # | Hypothesis | Verdict | Evidence (command → output) |
|---|---|---|---|
| H1 | Shared `graph_pool_handle` corrupts replay | **RULED OUT** | separate pools crash identically (diag_matrix D,E) |
| H2 | Capturing the 2nd graph corrupts the 1st | **CONFIRMED** | same TBW graph: OK before volley capture, illegal-access after (diag_corrupt) |
| H3 | The volley graph (B=8+record) is broken on its own | **RULED OUT** | single B=8 record=True captures+replays fine (diag_single) |
| H4 | Batch size B=8 is the trigger | **RULED OUT** | single B=8 (rec & norec) both PASS (diag_single) |
| H5 | The record=True branch is the trigger | **RULED OUT** | single B=540 & B=8 record=True both PASS (diag_single) |

### Controlled matrix (each row = one process, faulthandler on)
**Two graphs — pool × first-replay (`diag_matrix.py`, `diag_out/m_*.log`):**
| pool | replay first | result |
|---|---|---|
| shared   | TBW(B=540) | **SEGFAULT 139** |
| shared   | VOLLEY(B=8)| **SEGFAULT 139** |
| separate | TBW(B=540) | **SEGFAULT 139** |
| separate | VOLLEY(B=8)| **SEGFAULT 139** |
→ crash is independent of pool sharing AND of which graph replays first.

**One graph — batch × recording (`diag_single.py`, `diag_out/s_*.log`):**
| B | record | result |
|---|---|---|
| 540 | False | **PASS** (exit 0, 3 replays) |
| 540 | True  | **PASS** |
| 8   | False | **PASS** |
| 8   | True (==volley) | **PASS** |
→ ANY single graph captures + replays fine. The crash needs a SECOND capture.

**Controls:**
- `diag_pool.py` (G): TBW-only graph, eager volley, **both** reset-caches pre-built → **PASS** (exit 0).
- `test_volley_only.py` (F): volley-only graph but run through `_panel_battery` (eager B=540 TBW whose
  `reset_state(540)` is a FIRST-time full realloc, since cache[540] was never pre-built) → **SEGFAULT**
  at `vg.replay()`.

---

## Proven root cause
**Capturing a second CUDA graph on the same net corrupts the first graph's replay memory** →
`cudaErrorIllegalAddress` on replay. Causal proof (`diag_out/corrupt.log`, single variable = "is the
volley graph captured yet"):
```
TBW captured.
   [BEFORE volley capture] TBW replay OK  sMSI.sum=0.0000
VOLLEY captured.
   [AFTER  volley capture] -> torch.AcceleratorError: CUDA error: an illegal memory access
```
The SAME TBW graph replays cleanly, then — with nothing changed except that the volley graph was
captured in between — its next replay is an illegal access. Pool sharing (H1), B=8 (H4), and the
recording branch (H5) are independently ruled out (separate pools crash; every single graph passes).

**Unified empirical rule** (also explains the volley-only crash F): a captured graph's baked memory
is corrupted by ANY memory-allocation event that occurs between its capture and its replay — a second
graph capture (diag_corrupt), or a first-time full-realloc `reset_state` for a batch size whose
persistent reset-cache was not pre-built (F's eager B=540 pass). With NO intervening allocation event,
the graph replays fine (diag_single, diag_pool).

### Mechanism — USE-AFTER-FREE (compute-sanitizer, proven)
`compute-sanitizer --tool memcheck python diag_corrupt.py` (`diag_out/sanitizer_corrupt.log`):
the BEFORE-volley-capture TBW replay is clean; the AFTER-volley-capture TBW replay throws **6,496
errors**, every one identical in kind:
```
Invalid __global__ read of size 4 bytes
    at void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(...)
    Access to 0x73ceeb980ec0 is out of bounds
    and is 2,617,664 bytes before the nearest allocation at 0x73ceebc00000 of size 2,097,152 bytes
```
- The faulting kernel is a **CUTLASS SGEMM** — one of the TBW graph's baked `F.linear`/`mm` matmuls.
- Its operand pointer reads memory that is **out of bounds** (~2.6 MB *before* the nearest live block),
  and the nearest live block is a *different*, freshly-allocated 2 MB region.
This is a textbook **use-after-free / dangling pointer**: capturing the volley graph FREES the memory
that a TBW-graph matmul operand was baked to point at, and the caching allocator re-hands that region
out (the new 2 MB allocation), so TBW replay reads freed/out-of-bounds memory →
`cudaErrorIllegalAddress` → SIGSEGV. The single-graph sanitizer run (no second capture) =
**0 errors** (`diag_out/sanitizer_8rec.log`), confirming the freeing event is the second capture.
This pins the unified rule to an exact fault: a baked matmul operand becomes a dangling pointer.

---

## Suggested fix direction (Coder decides — NOT applied here)
The panel-net's two independently-captured graphs (TBW B=540 + volley B=8) cannot coexist on one net
as currently built; the crash is structural, not value-dependent (zeros input reproduces it).
1. **Single graph + eager other, with BOTH reset-caches pre-built.** Capture ONE shape and run the
   other eager; call `reset_state(540)` AND `reset_state(8)` before capture so no later first-time
   full-realloc reset can clobber the graph. `diag_pool.py` proves this configuration replays fine.
2. **Proper multi-graph capture** per PyTorch `make_graphed_callables` (all graphs share ONE pool,
   captured back-to-back with NO intervening allocation, all pools held alive). The current
   `pool=graph_pool_handle()` attempt is insufficient — it still segfaults — so this needs the full
   documented protocol, not just passing `pool=`.
3. Drop CUDA graphs for the panel and use the launch-reduction / vectorization levers instead.

## Remaining unknowns
- Mechanism is resolved: **use-after-free of a baked matmul (SGEMM) operand** (sanitizer, above).
- The only open detail is the library-level reason a second `torch.cuda.graph` capture frees the first
  graph's still-baked operand memory EVEN with separate pools (PyTorch private-pool handling during
  capture). A Researcher PyTorch docs/source check would formalize WHY; it does not change the proven
  root cause or the fix direction.
