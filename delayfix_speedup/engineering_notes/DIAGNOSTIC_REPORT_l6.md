# DIAGNOSTIC REPORT — L6 CUDA-graph 1-tensor byte-FAIL on `_latest_sMSI_inh` (task #66)

**Debugger forensic. Diagnosis only — NO fix applied. Port-under-test `Training_graphdf.py` md5
`7266db3a36632e9169dc55d66cd2097e` and frozen baseline `Training_delayfix.py` md5
`5e7d6d20538592b5fc92165b371949d7` both VERIFIED byte-untouched. I created only the throwaway
harness `diag_l6.py`; I edited NO build file. All compute cuda:0 / RTX 5090. Every claim is backed
by a command + captured output in `diag_out/l6_*.log`.**

---

## Failure (reproduced)
`smoke_l6.py` (transfer BEFORE training — no #65 confound) trains base(L4 eager) vs optg(L6 graph),
single variable = graph capture. At ep0 AND ep26 the compare FAILs on **exactly one** tensor attr,
`_latest_sMSI_inh`, max|Δ|=1.0. Reproduced in `diag_l6.py` (`diag_out/l6_ep26_k2_probe.log`,
`l6_propagation.log`). Correction to the framing: the magnitude 1.0 is per-entry (a 0↔1 spike flip);
the COUNT of flipped (batch×inh-neuron) entries is **479** at ep26 K=2 — all of them inside that one
escaped attribute, nothing else.

## VERDICT
**BENIGN escaped/stale-pointer artifact — NOT a real graph-capture divergence.** The inh spike is
computed correctly inside the captured graph (every consumed/persistent tensor byte-matches eager);
`_latest_sMSI_inh` is a write-only attribute that escaped capture, was never repointed to a
persistent `_g_out_*` buffer, and is refreshed by NO replay and read by NO post-replay consumer.
The byte-identity gate trips on a stale pointer, not on wrong math.

---

## Hypotheses tested (each gets a verdict)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Benign escaped/stale pointer (inh computed right; attr dangles, unread) | **CONFIRMED** | Q1 downstream byte-equal + Q2 frozen escaped ptr + Q3 no reader + Q4 no propagation |
| H2 | Real graph-capture divergence in the inh pathway (wrong inh spike) | **RULED OUT** | Q1: W_msiInh2Exc_GABA + pre/post_trace_msiInh2Exc + W_MSI_inh byte-identical at ep26 — a wrong inh spike would move them; Q4: never propagates |
| H3 | Live-but-reused-temporary value (tracks replay, just a different cell) | **RULED OUT** | Q2 liveness: `_latest_sMSI_inh` is FROZEN across two different replay inputs |

### Q1 — downstream-of-the-inh-spike correctness (independent confirmation)
`diag_l6.py` explicitly `torch.equal`-checks the tensors DRIVEN BY the inh spike. At ep26 (iSTDP
active; reads `_latest_sMSI_inh` at graphdf L3436/L3447) — `diag_out/l6_ep26_k2_probe.log`:
```
W_msiInh2Exc_GABA=EQ  pre_trace_msiInh2Exc=EQ  post_trace_msiInh2Exc=EQ  W_MSI_inh=EQ
```
The iSTDP weight + both its traces are byte-identical → the inh spike CONSUMED inside the graph each
replay is correct. (Holds at ep0 too, where iSTDP is off.)

### Q2 — where `_latest_sMSI_inh` points, and whether replay refreshes it (mechanism)
`diag_out/l6_ep26_k2_probe.log`. data_ptr after training:
```
_latest_sA/sV/sMSI/sOut/dA2M/dV2M : ptr == its _g_out_*  (==_g_out? True)   <- repointed (L3785-3787)
_latest_sMSI_inh                  : ptr NOT in _g_out set; _g_out_sMSI_inh exists? False
```
LIVENESS (replay twice with two DIFFERENT inputs):
```
repointed sibling _g_out_sMSI changes with input?  True   (live)
escaped attr _latest_sMSI_inh changes with input?  False  (FROZEN — replay never refreshes it)
```
The 6 siblings are repointed to persistent `_g_out_*` buffers the captured loop fills in-place every
replay (graphdf L3515-3523, L3753-3757) → live → byte-match. `_latest_sMSI_inh` has no `_g_out`
twin and no repoint, so it dangles at a stale graph-pool address that `g.replay()` does not refresh.

### Q3 — every reader of `_latest_sMSI_inh` after replay (code-proven)
`grep` enumeration of all references (only 6 sites): `__init__`/`reset_state` (writes, L1538/L2629);
the in-loop write `self._latest_sMSI_inh = sMi` (L3416, INSIDE the captured graph); the in-loop
iSTDP reads (L3436, L3447, INSIDE the captured graph — they read the LIVE sMi each replay); and one
reader at L406 inside `apply_topographic_anchor_msi_inh`, which is **DEAD CODE** (graphdf L3486
comment: "apply_topographic_anchor_msi_inh remains as dead code (L377-412)"; it has no caller on the
training path). The eager post-replay tail (L3790-3797: anchor_unimodal, competition, soft_row_scaling)
does NOT read it; recording (L4060/L4640) reads `_latest_sMSI` (exc), not `_inh`. **=> zero
post-replay readers of `_latest_sMSI_inh`.**

### Q4 — propagation at higher K / more ext-steps (`diag_out/l6_propagation.log`)
```
ep0  K=2 -> ONLY _latest_sMSI_inh differs   (downstream EQ)
ep0  K=4 -> ONLY _latest_sMSI_inh differs   (downstream EQ)
ep26 K=4 -> ONLY _latest_sMSI_inh differs   (downstream EQ)
ep26 K=8 -> ONLY _latest_sMSI_inh differs   (downstream EQ)
```
The diff set stays EXACTLY `{_latest_sMSI_inh}` and never reaches any weight/state, through 8×
more mini-batches/replays. No latent divergence.

---

## Proven root cause (mechanism)
During capture, `self._latest_sMSI_inh = sMi` (graphdf **L3416**) binds the attribute to the
graph-internal last-substep inh-spike tensor. The replay closure `_forward` repoints the six sibling
`_latest_*` attrs to persistent `_g_out_*` buffers (**L3785-3787**) that the captured loop fills
in-place each replay (**L3515-3523**), but there is **no `_g_out_sMSI_inh`** (absent from the
alloc list L3753-3754 and the copy block L3515-3523) and **no repoint for `_latest_sMSI_inh`**.
So after replay the attribute dangles at a stale graph-pool address. The in-graph iSTDP still reads
the live `sMi`, so all consumed/persistent state is byte-identical; only the escaped attribute is
stale. `_l6_restore` (L637-646) restores in-place (`t.copy_`), so it never rebinds the attribute
off the stale graph tensor either.

**Causal proof (single-variable, structural + liveness):** the repointed-vs-not structural split
maps EXACTLY onto matches-vs-differs — all 6 repointed siblings byte-match; the one non-repointed
attribute is the one and only tensor that differs, in every config tested (ep0/ep26 × K=2/4/8). The
liveness contrast is the direct causal demonstration: under the same replays, the repointed sibling
`_g_out_sMSI` tracks the input (live → matches eager) while the non-repointed `_latest_sMSI_inh`
does not (frozen → differs). (A flip-the-fix confirmation — add `_g_out_sMSI_inh` + repoint and watch
the gate pass — requires a build edit, which is the Coder's domain, gated on this proof.)

## Suggested fix direction (Coder — gated on this proof; build edit is theirs)
The established escaped-tensor repoint, mirroring the six siblings:
1. allocate persistent `_g_out_sMSI_inh` shaped `(B, self.n_inh)` (note: n_inh, not n) — add it
   alongside the `_g_out_*` allocation at graphdf L3753-3757;
2. `self._g_out_sMSI_inh.copy_(sMi)` in the `_g_out` fill block (L3515-3523);
3. `self._latest_sMSI_inh = self._g_out_sMSI_inh` in the replay closure (L3785-3787).
This is a COMPLETENESS fix for the byte-identity bar only — it changes NO training dynamics (the
attribute is write-only-escaped with no consumer; Q3). After it, re-run `smoke_l6.py --epoch 0` and
`--epoch 26` → expect PASS.

## Remaining unknowns
- Exact provenance of the stale bits (why 479 flips rather than the full field or zeros) — immaterial
  to the verdict; the attribute is never consumed. Not chased.
- Scientific note (not a byte-identity issue): in REAL training (no compare harness reading the
  attr), L6 dynamics are ALREADY byte-identical to eager — every weight/trace/neuron-state tensor
  matches (Q1, Q4). The FAIL is strictly the gate reading a stale escaped pointer.
