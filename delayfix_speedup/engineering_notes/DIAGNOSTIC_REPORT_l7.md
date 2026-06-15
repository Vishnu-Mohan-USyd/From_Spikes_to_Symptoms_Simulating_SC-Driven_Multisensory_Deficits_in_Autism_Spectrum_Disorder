# DIAGNOSTIC REPORT — L7 two-graph CUDA-graph segfault (task #68)

**Debugger forensic. Diagnosis only — NO fix applied. Port-under-test `Training_graphdf.py` md5
`79c45ffbf88ec771fe443ed621ebdd82` and frozen baseline `Training_delayfix.py` md5
`5e7d6d20538592b5fc92165b371949d7` both VERIFIED byte-untouched after every run. I created only the
throwaway harness `diag_l7.py`; I edited NO build file. All compute cuda:0 / RTX 5090. Every claim is
backed by a command + captured output in `diag_out/l7_*.log`.**

---

## Failure (reproduced)
`smoke_l7.py` (transfer BEFORE training — no #65 confound) trains base(L6 STDP-eager) vs optg(L7
STDP-graphed), single variable = the L7 STDP-tail graph. base(L6) completes; optg(L7) **segfaults**.
The fault is in the **L6 FORWARD-graph replay** — `g.replay()` at `Training_graphdf.py:3799` inside
`_forward`, reached from `train_unsupervised_batch:4066` — and it fires **only once `_l7_install` has
captured the second graph `g_stdp`**. Reproduced under instrumentation in
`diag_out/l7_v0_baseline.log` (process exits 139, faulthandler trace at graphdf:3799).

## VERDICT
**CONFIRMED #55-class two-graph CUDA-graph use-after-free.** Capturing the second graph `g_stdp`
(graphdf L3862-3864) after the L6 forward graph `g_l6`, before `g_l6`'s first replay, corrupts
`g_l6`'s baked memory → the next `g_l6.replay()` segfaults. The corrupting event is the **`g_stdp`
capture itself** (isolated by a clean 2×2 — the side-stream warmup is exonerated). It is **NOT**
fixable by sharing a memory pool (tested — still crashes), and it is **NOT** `g_stdp`'s replay (that
never runs before the crash). It is the coexistence of two separately-captured, both-replayed graphs:
priming one just moves the crash to the other.

---

## Localization (instrumented)
`diag_l7.py` wraps `torch.cuda.graph` (records capture order + pool handle) and
`torch.cuda.CUDAGraph.replay` (flushed per-replay log; the last line before a segfault names the
crashing graph). `diag_out/l7_v0_baseline.log`:
```
[capture #1] graph id=...7520 pool=None      <- g_l6  (own private pool)
[capture #2] graph id=...6368 pool=None      <- g_stdp (own private pool, captured AFTER g_l6)
[replay  #1] graph id=...7520 per-graph#1     <- g_l6's FIRST replay
Fatal Python error: Segmentation fault  (graphs.py:143 replay <- graphdf:3799 _forward <- :4066)
```
Two facts pinned: (1) both graphs are captured with **`pool=None`** → each gets its **own private
mempool** (no sharing); (2) the crash is on **g_l6's very first replay**, and **g_stdp's replay
(id …6368) is never logged** → g_stdp.replay never executes before the crash ⇒ the trigger is on the
**capture/install side**, not g_stdp's replay.

Build order confirmed by code read (graphdf L4039-4086): `reset_state` → `_l6_install` (capture g_l6,
L4046) → `_l7_install` (3-iter side-stream warmup **+ capture g_stdp**, L4058) → t-loop: `_fwd`
(g_l6.replay, L4066→3799) **→ CRASH** before the first `_l7_stdp_graph.replay()` (L4086). So g_stdp's
capture lands **between** g_l6's capture and g_l6's first replay — violating the capture-last
requirement the build asserts for the forward graph (`_l6_install` docstring L3756-3758: "the graph
capture is the last allocation event before the replays").

---

## Hypotheses tested (each gets a verdict)

| # | Hypothesis | Verdict | Evidence (log) |
|---|---|---|---|
| H1 | The `g_stdp` capture after `g_l6` (capture-last violation) corrupts `g_l6`'s replay | **CONFIRMED** | 2×2 isolates the capture as sole trigger; `recapture` cures it |
| H2 | Separate private pools overlap → fixable by sharing one mempool | **RULED OUT** | `sharedpool`: both pools=(0,1), STILL crashes identically |
| H3 | The 3-iter side-stream **warmup** allocation (L3856-3860) is the corrupting event | **RULED OUT** | `warmuponly` (warmup, no capture) completes byte-identical |
| H4 | `g_stdp`'s **replay** (L4086) corrupts shared `_g_out_*`/state | **RULED OUT** | baseline crashes at g_l6.replay#1 BEFORE any g_stdp.replay |
| H5 | The `g_stdp` capture is **necessary** for the crash | **CONFIRMED** | `noop` (skip g_stdp capture) completes byte-identical |

### The causal proof — a clean 2×2 (warmup × g_stdp-capture), all else identical
Each variant is a single-process run of `diag_l7.py --variant X` (base L6 trained unpatched first as
reference; optg L7 trained under the monkeypatch). Exit 139 = segfault; exit 0 = completed.

| | **capture g_stdp = NO** | **capture g_stdp = YES** |
|---|---|---|
| **warmup = NO**  | `noop` → **OK** (byte-identical) | `nowarmup` → **CRASH** (g_l6.replay#1, :3799) |
| **warmup = YES** | `warmuponly` → **OK** (byte-identical) | `baseline` → **CRASH** (g_l6.replay#1, :3799) |

Logs: `l7_v_noop.log` (EXIT=0, PASS), `l7_v_warmuponly.log` (EXIT=0, PASS), `l7_v_nowarmup.log`
(EXIT=139, crash at :3799), `l7_v0_baseline.log` (EXIT=139, crash at :3799).
**Crash ⟺ the g_stdp capture, in BOTH warmup rows.** Toggling ONLY the capture (holding warmup
fixed) flips OK→CRASH; removing it restores OK. The warmup column has zero effect. This is the
single-variable causal proof: the corrupting event is the **second graph's capture**, not the warmup.

### Confirming the mechanism is capture-ORDER (not pool, not replay)
- **`sharedpool`** (`l7_v_sharedpool.log`, EXIT=139): force both captures into ONE shared pool via
  `torch.cuda.graph_pool_handle()` — log shows `pool=(0,1)` for both — **still crashes** on
  g_l6.replay#1. ⇒ H2 RULED OUT; the naive "#55 = share the pool" remedy does **not** apply here.
- **`recapture`** (`l7_v_recapture.log`, EXIT=0, byte-IDENTICAL): after `_l7_install`, run
  `_l6_install` again so a g_l6-type capture is the LAST capture event. Both replayed graphs (g_l6 and
  g_stdp) then replay fine for the whole run. ⇒ making the forward graph capture-last cures it.
- **`primefirst`** (`l7_v_primefirst.log`, EXIT=139): replay g_l6 ONCE before g_stdp is captured
  (close g_l6's capture→first-replay window first). g_l6 now survives — but the crash **moves to
  g_stdp's first replay at L4086** (faulthandler: graphs.py:143 ← graphdf:4086). ⇒ the hazard is the
  two-graph coexistence itself; relocating one graph's exposure just relocates the crash.

---

## Proven root cause (mechanism)
The training loop captures **two** CUDA graphs that coexist and are **both replayed, interleaved**,
each step: `g_l6` (the forward substep graph, `_l6_install`/`_l6_capture_phase`, replayed at
graphdf **L3799**) and `g_stdp` (the STDP ger-sequence tail, captured at **L3862-3864** inside
`_l7_install`, replayed at **L4086**). `g_stdp` is captured **after** `g_l6` and **before** `g_l6`'s
first replay (build order L4046 → L4058 → L4066). That second capture corrupts `g_l6`'s baked
device memory, so the immediately-following `g_l6.replay()` dereferences clobbered pointers →
segmentation fault. This is the same two-graph use-after-free class proven in #55, now reproduced for
the L6-forward / L7-STDP graph pair.

**Causal proof (single-variable, reversible):** the 2×2 above shows the crash is gated **only** by
whether `g_stdp` is captured — toggling that one event flips OK↔CRASH in both warmup rows, and the
crash site (g_l6.replay, graphdf:3799) is invariant. Reverting the trigger (don't capture g_stdp:
`noop`) restores a clean byte-identical run; reintroducing it (`baseline`/`nowarmup`) brings the
crash back. Three independent controls fix the mechanism in place: it is **not** the warmup (`warmuponly`
OK), **not** pool-overlap (`sharedpool` still crashes), **not** g_stdp's replay (`baseline` crashes
before any g_stdp.replay). Making the forward graph capture-last (`recapture`) removes the crash and
the run is byte-identical to eager L6 — and reordering instead (`primefirst`) just moves the crash to
g_stdp, confirming the two coexisting private-pool graphs are mutually corrupting.

## The minimal mechanism that must change
The build must **not** capture a second coexisting CUDA graph (`g_stdp`) after `g_l6` and before
`g_l6` is replayed. The proven facts constrain the fix:
- a bare **reorder** (capture g_stdp first / g_l6 last) is unsafe — `primefirst` shows the
  last-captured-and-replayed graph is the one that then crashes;
- a bare **pool-share** is unsafe — `sharedpool` still crashes.

## Suggested fix direction (Coder — gated on this proof; build edit is theirs)
Ordered by how directly they restore the proven-safe invariant:
1. **(Recommended) Fuse the STDP tail into the L6 forward graph** — capture ONE graph (forward +
   STDP) capture-last, the originally-proven single-graph L6 configuration (#62/#66). This eliminates
   the two-graph coexistence entirely and is the mechanism most consistent with the proven design.
2. **Drop L7 (keep STDP eager).** L6 is already byte-identical to eager and is the proven speedup; if
   L7's marginal gain is small, the two-graph hazard isn't worth carrying.
3. If two graphs must coexist, treat it as an open CUDA-graphs problem to be **re-validated**, not a
   one-line tweak: the only configuration I proved working is `recapture` (a g_l6 capture as the last
   capture event, with the loop replaying the earlier-finalized g_l6 closure) — empirically
   byte-identical, but fragile and non-obvious; do **not** ship it without a full smoke re-run at
   ep0 AND ep26. Bare reorder and bare pool-share are tested-failing.

After any fix, re-run `smoke_l7.py --epoch 0` and `--epoch 26` → expect PASS (byte-identical, no
segfault), then the #63 gate.

## Remaining unknowns
- The exact CUDA caching-allocator internal by which the second private-pool capture clobbers the
  first graph's region (and why an additional finalizing capture in `recapture` rescues it) is not
  reverse-engineered from outside the allocator — immaterial to the verdict, which is established by
  the single-variable 2×2 + controls. Not chased.
- Whether option 1 (fused single graph) is bit-identical and within capture limits is a build+gate
  question for the Coder/Validator, not this forensic.
```
