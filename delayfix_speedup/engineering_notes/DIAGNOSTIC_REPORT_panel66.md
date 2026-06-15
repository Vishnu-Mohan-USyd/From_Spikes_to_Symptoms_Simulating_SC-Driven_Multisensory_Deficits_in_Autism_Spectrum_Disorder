# DIAGNOSTIC REPORT — panel-net `AttributeError: _g_out_sMSI_inh` (task #71)

**Debugger forensic. Diagnosis only — NO fix applied. Port-under-test `Training_graphdf.py` md5
`79c45ffbf88ec771fe443ed621ebdd82` and frozen baseline `Training_delayfix.py` md5
`5e7d6d20538592b5fc92165b371949d7` both VERIFIED byte-untouched. I created only the throwaway
harness `diag_panel66.py` and edited NO build/harness file. Per the lead's constraint, I ran NO GPU
repro (cuda:0 is busy with the bs=250 timing run): the proof is (1) full static code analysis and
(2) a CPU-ONLY single-variable causal test run with `CUDA_VISIBLE_DEVICES=""` (`cuda.is_available()
== False`), which cannot touch or contend with cuda:0. Evidence in `diag_out/panel66_cpu_proof.log`.**

---

## Failure (reproduced)
The lever-e panel worker dies on its **first panel**:
`AttributeError: 'MultiBatchAudVisMSINetworkTime' object has no attribute '_g_out_sMSI_inh'`
at `Training_graphdf.py:3528` (`self._g_out_sMSI_inh.copy_(sMi)  # #66`), reached via
`panelgraph.capture_panel_graph → update_all_layers_batch` during the panel-net's B=8 volley-graph
capture. The CPU proof reproduces the identical attribute error from the verbatim L3523-3528 block
(`diag_out/panel66_cpu_proof.log`, CASE 1).

## VERDICT
**CONFIRMED — hypothesis (i): panel-harness setup gap.** The #66 completeness fix added BOTH a write
to `_g_out_sMSI_inh` inside the *shared* eager forward `update_all_layers_batch` (graphdf **L3528**)
AND the allocation of that buffer inside the *training-only* `_l6_install` (graphdf **L3773-3776**).
The PANEL net builds its `_g_out_*` set through a SEPARATE path —
`panelgraph._alloc_gouts`/`_bind_gouts` over `GOUT_NAMES` (panelgraph **L95-103**) — which was
written for #52 (pre-#66) and lists only the 6/7 same-width siblings, **never `_g_out_sMSI_inh`**.
The write block's guard checks only `_g_out_sA is not None` (graphdf **L3523**), so for the panel net
the guard PASSES (it bound `_g_out_sA`) and the very next line dereferences the never-allocated
`_g_out_sMSI_inh`. Not a deeper graph/capture fault. (ii) RULED OUT.

---

## Evidence chain (static — every claim is a file:line)

**1. The crash site is an UNCONDITIONAL deref inside a partial guard** (graphdf L3523-3528, inside
`update_all_layers_batch`, def L2783):
```python
3523  if (_l4 and getattr(self, '_g_out_sA', None) is not None
3524          and sA.shape[0] == self._g_out_sA.shape[0]):
3525      self._g_out_sA.copy_(sA)
...
3528      self._g_out_sMSI_inh.copy_(sMi)   # #66  <-- guard never checked THIS attr exists
```
The guard's existence-check covers `_g_out_sA` only and assumes the whole `_g_out_*` set co-exists.

**2. `_l4` (the guard's first term) is ON for the panel.** graphdf L2881:
`_l4 = getattr(self, '_lever_L4_graph', False)`; panel sets `net._lever_L4_graph = True`
(panelgraph **L215**). ⇒ guard term 1 = True.

**3. `_g_out_sA` IS bound on the panel (guard term 2 passes), but `_g_out_sMSI_inh` is NOT.**
panelgraph `GOUT_NAMES = ('sA','sV','sMSI','sOut','dA2M','dV2M','sumM')` (**L95**) — no `sMSI_inh`;
`_alloc_gouts` allocates each at `(B, net.n)` (**L97-98**); `_bind_gouts` sets
`net._g_out_sA … _g_out_sumM` (**L100-103**). `setup_panel_net` allocates `net._vol_gout =
_alloc_gouts(net, 8)` (**L227**) and the volley wrapper calls `_bind_gouts(net, net._vol_gout)`
(**L243**) immediately before `capture_panel_graph(... B=8 ...)` (**L246**). So `_g_out_sA` exists
(B=8) and `sA.shape[0]==8==_g_out_sA.shape[0]` ⇒ guard terms 2,3 = True. The whole guard is True →
L3525-3528 execute → L3528 hits the missing attribute.

**4. The crash fires in the capture warmup at B=8.** `capture_panel_graph` calls
`net.update_all_layers_batch(...)` in its 3-iter warmup (panelgraph **L132-134**) and `call_cap`
(**L140-141**) at B=8 — the first such call raises, matching "died on its first panel / B=8 volley
capture."

**5. The full reference set of `_g_out_sMSI_inh` (grep, graphdf) is only 3 logical sites** —
write L3528; allocate L3773-3776 (`_l6_install`); repoint L3806 (`_l6_install` closure). It is
created **nowhere** in `__init__` (L1244), `reset_state` (L2564), or anywhere in `panelgraph.py`.
⇒ on the panel net the attribute is genuinely never created.

**6. Why TRAINING is clean (the asymmetry).** `_l6_install` allocates the COMPLETE set —
all six siblings (L3762-3766) **and** `_g_out_sMSI_inh = torch.zeros((B, self.n_inh), …)`
(L3773-3776) — before the training graph is captured (L3781). So under training the attribute
always exists when L3528 runs. The panel never calls `_l6_install`; it uses `_alloc_gouts`, which #66
did not update.

**7. The buffer width is `n_inh`, not `n`.** `sMi` is `(batch, self.n_inh)` (graphdf L2816) and the
alloc is `(B, n_inh)` (L3775); `_alloc_gouts` uses uniform `(B, net.n)` (panelgraph L98). So the gap
cannot be closed by merely appending `'sMSI_inh'` to `GOUT_NAMES` — that would mis-size it to
`(B, n)` and the `copy_` would then raise a shape error instead (proven below, CASE 3).

## Causal proof (single-variable, CPU-only, reversible) — `diag_out/panel66_cpu_proof.log`
The harness transcribes the L3523-3528 block verbatim and toggles ONLY whether the net carries
`_g_out_sMSI_inh` (`cuda.is_available()==False`):
```
[panel]  hasattr _g_out_sA=True  hasattr _g_out_sMSI_inh=False
[panel]  write_block -> AttributeError: 'Net' object has no attribute '_g_out_sMSI_inh'   (== graphdf:3528)
[panel+inh(B,n_inh)] write_block -> WROTE        <<< crash GONE  (single variable flipped)
[panel+inh(B,n)  WRONG] -> RuntimeError: size of tensor a (5) must match b (3) dim 1   <<< must be (B,n_inh)
[train/_l6_install]  write_block -> WROTE         <<< control: training carries the buffer -> clean
```
Toggling the single variable (presence of `_g_out_sMSI_inh`) flips AttributeError ↔ clean write;
restoring its absence brings the crash back; the training path (buffer present) is the working
control. Width must be `(B, n_inh)` (CASE 3). This is the causal proof.

---

## Proven root cause (mechanism)
`update_all_layers_batch` (the eager forward shared by training-capture AND the panel-volley capture)
unconditionally writes `self._g_out_sMSI_inh.copy_(sMi)` (graphdf L3528) once its partial guard
(`_l4 and _g_out_sA is not None and shape-match`, L3523) is satisfied. That buffer is allocated only
by the training-path `_l6_install` (L3775, the #66 addition). The panel net allocates its `_g_out_*`
via `panelgraph._alloc_gouts`/`_bind_gouts`/`GOUT_NAMES` (L95-103), a pre-#66 (#52-era) path that
omits `_g_out_sMSI_inh` while still binding `_g_out_sA`. So the guard passes on the panel net and
L3528 dereferences a missing attribute → `AttributeError`. #66 updated the write and the *training*
allocator but not the *panel* allocator.

## The minimal mechanism that must change (whose net · which buffer · where)
- **Whose net:** the **PANEL** net (built/configured by `panelgraph.setup_panel_net`); the training
  net is already correct.
- **Which buffer:** **`_g_out_sMSI_inh`**, shape **`(8, net.n_inh)`** — width `n_inh`, NOT `n`
  (CASE 3 proves the wrong width still crashes).
- **Where:** the panel volley g-out allocation/bind path — `panelgraph._alloc_gouts`(L97-98) /
  `_bind_gouts`(L100-103), populated at `setup_panel_net` L227 (`_vol_gout = _alloc_gouts(net, 8)`)
  and bound at L243 (`_bind_gouts`). Because `_alloc_gouts` is uniform-width `(B, net.n)`, the inh
  buffer needs a dedicated `(8, net.n_inh)` allocation added to `_vol_gout` and a corresponding
  `net._g_out_sMSI_inh = …` in `_bind_gouts` (mirroring the #66 `_l6_install` addition).

## Suggested fix direction (Coder — gated on this proof; edit is theirs)
Allocate + bind the panel's `_g_out_sMSI_inh` at `(8, net.n_inh)` alongside the other volley g-outs
(panelgraph `_alloc_gouts`/`_bind_gouts` / `_vol_gout`). Optional defensiveness (does NOT replace the
above): make the graphdf L3523 guard also check `getattr(self,'_g_out_sMSI_inh',None) is not None`,
so the shared forward degrades gracefully if any future net binds a partial `_g_out_*` set — but the
real gap is the panel allocator, which must match the post-#66 invariant. After the fix, re-run the
lever-e panel worker (and the #52 panel bit-identity check) → expect no AttributeError; the buffer is
write-only-escaped (#66-proven: no consumer), so panel outputs/bit-identity are unchanged.

## Remaining unknowns
- None for the crash cause: the AttributeError is deterministic attribute-existence logic, proven
  static + CPU. A GPU repro of the live panel worker (and the post-fix re-run) is deferred until the
  lead clears cuda:0; I expect it to confirm, but I have not run it (constraint honored).
- Whether other post-#52 additions wrote further training-only attributes into the shared forward
  that the panel allocator also lacks — not in scope here; only `_g_out_sMSI_inh` is implicated by
  this traceback and the grep shows no other `_g_out_*` sibling missing from `GOUT_NAMES`.
