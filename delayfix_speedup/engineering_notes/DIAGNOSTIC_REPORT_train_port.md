# DIAGNOSTIC REPORT — graphdf training path diverges from delayfix baseline, levers OFF (task #60)

**Debugger forensic. Diagnosis only — NO fix applied. Frozen baseline `Training_delayfix.py` md5
`5e7d6d20538592b5fc92165b371949d7` and port-under-test `Training_graphdf.py` md5
`50d69e200caaa57164d7e43d99b40036` both VERIFIED byte-untouched. The only file I edited is a
throwaway COPY `Training_graphdf_rev.py` (a controlled single-variable revert). All compute cuda:0 /
RTX 5090. Every claim is backed by a command + captured output in `diag_out/`.**

---

## Failure (reproduced)
`train_bitid_gate.py --K 4 --bs 256 --epoch 26` FAILs PORT FIDELITY: opt(graphdf) levers OFF does not
byte-match base(delayfix) — 36 tensors differ, worst |Δ|=2.649e+02 @ I_M (`train_bitid_lead.log`).
Reproduced minimally (`diag_train.py`, K=1, `diag_out/train_bisect.log`): epoch=26 → **FAIL, 31
tensors differ, worst |Δ|=2.316e+02 @ I_M** (v_msi 130, u_msi 38, post_trace_msi_rec 2.71,
pre_trace_msi_rec 2.69). Determinism control passes and all nets start byte-identical (pre-train
max|Δ|=0.000e+00), so the divergence is the port, not RNG.

## Localization
Source diff base↔port = 626 lines (`/tmp/dfgd.diff`), ~30 hunks. Classifying every hunk by whether
it is active in this gate (plasticity ON, L1=0, L2=0, L3 default-off, L4 unset):
- **Lever-gated** (L1 `_lever_L1_drop_inloop_log`, L2 `_lever_L2_vector_reset`, L3
  `_lever_L3_stdp_matmul`): each has an `else:` = verbatim baseline; with levers OFF the baseline
  branch runs. Inactive.
- **L4-graph-gated** (`if _l4:` reads/scatter/gpos/persistent-buffers, reset cache): `_lever_L4_graph`
  is **ABSENT** on the gate's nets (`getattr default False`) — verified live. Inactive.
- **Recording redesign** (`_ei_record`/`_panel_inh_accum` `.append(.item())` → `_rec_gpu[k][sub_i]`):
  guarded `is not None`; both are **None** during the gate's direct `train_unsupervised_batch`, and
  `_rec_gpu` is never created — verified live. Inactive.
- **Debug-counter swap** (`torch.tensor(0.0,device)` → `torch.zeros((),device)`): bookkeeping spike
  counters, value-identical (0.0), do not feed weights/dynamics. (Ruled out empirically below.)
- **ONE active functional change:** `train_unsupervised_batch`, recurrent MSI→MSI STDP, gated
  `if epoch_idx > 25:` (port `Training_graphdf.py:3729-3738`; baseline `Training_delayfix.py`
  ~`3509-3530`). The port deleted "**delay-fix EDIT #2**":

| | baseline (delayfix) | port (graphdf) |
|---|---|---|
| recurrent STDP `pre_spk` | `self._prev_sMSI_rec` (PREVIOUS external step's MSI spikes; 1-step delay so pre LEADS post) | `sMSI` (CURRENT step) |
| per-step state update | `self._prev_sMSI_rec = sMSI.detach()` | (deleted) |
| field init / seq reset | `self._prev_sMSI_rec = None` in `__init__` (~L1376) + `reset_state` (~L2488) | (both deleted) |

This block writes `W_MSI_exc`, `pre_trace_msi_rec`, `post_trace_msi_rec` directly, and is entered
only at epoch>25 (the gate runs epoch=26, g_rec=0.1).

---

## Hypotheses tested

| # | Hypothesis | Verdict | Evidence (command → output) |
|---|---|---|---|
| H1 | Recurrent MSI→MSI STDP pre-spike change (`pre_spk=sMSI` vs delayed `_prev_sMSI_rec`) | **CONFIRMED** | revert ONLY this → byte-identity (`train_revert.log`) |
| H2 | Debug-counter swap `torch.tensor`→`torch.zeros` | **RULED OUT** | epoch-independent; epoch=25 byte-identical (`train_bisect.log`) |
| H3 | Recording redesign (`_ei_record`/`_panel_inh_accum`→`_rec_gpu`) | **RULED OUT** | both None + `_rec_gpu` absent during training; epoch=25 PASS |
| H4 | L4 graph-path leakage (`_lever_L4_graph`) | **RULED OUT** | `_lever_L4_graph` ABSENT (off); epoch=25 PASS |

### Experiment 1 — epoch bisect (rules out every epoch-INDEPENDENT change at once)
`diag_train.py` (`diag_out/train_bisect.log`), base(delayfix) vs opt(graphdf), levers OFF:
```
epoch=25  g_rec=0.0  -> PASS (byte-identical)      # recurrent-STDP block NOT entered
epoch=26  g_rec=0.1  -> FAIL (31 differ; worst 2.316e+02 @ I_M)   # block entered
[live] opt._lever_L4_graph=ABSENT  _ei_record=None  _panel_inh_accum=None  has_rec_gpu=False
       (identical before AND after training)
```
H2/H3/H4 changes all execute at epoch 25 too (they are epoch-independent), yet epoch=25 is
byte-identical → none of them causes divergence. The only behavior that differs between 25 (PASS) and
26 (FAIL) is the `epoch>25` recurrent-STDP block. (The g_rec=0.1 forward injection also turns on at
epoch>25, but its code is byte-identical between builds — no diff — so it cannot itself diverge;
Experiment 2 confirms by leaving g_rec=0.1 on.)

### Experiment 2 — single-variable causal revert (definitive)
`diag_train_rev.py` (`diag_out/train_revert.log`), epoch=26, levers OFF. `Training_graphdf_rev.py` =
byte-copy of the port differing ONLY at the 3 recurrent-STDP-pre sites (verified by `diff`: lines
1393, 2559, 3732-3733/3737/3743 — the `pre_spk` swap + `_prev_sMSI_rec` field):
```
base vs Training_graphdf      (original port) -> FAIL (31 differ; worst 2.316e+02 @ I_M)
base vs Training_graphdf_rev  (reverted copy) -> PASS (byte-identical)
```

---

## Proven root cause
**`Training_graphdf.py:3733` (recurrent MSI→MSI STDP in `train_unsupervised_batch`, gated
`epoch_idx>25`) passes `pre_spk=sMSI` (current-step MSI spikes) where the frozen baseline passes
`pre_spk=self._prev_sMSI_rec` (the PREVIOUS external step's MSI spikes — the 1-external-step
conduction delay, "delay-fix EDIT #2"). The port also deleted the `_prev_sMSI_rec` field init
(`__init__`) and per-sequence reset (`reset_state`) and the per-step `self._prev_sMSI_rec =
sMSI.detach()` store.** This is a functional change (zero-lag pre==post vs lag-1 pre→post), not an
FP-ordering difference.

**Causal proof (single variable):** reverting ONLY those 3 sites in a copy of the port — changing
nothing else (levers OFF, g_rec=0.1, epoch=26 all held fixed) — flips base-vs-opt from **FAIL (worst
|Δ|=232 @ I_M) to PASS (byte-identical)**. The copy differs from the port at exactly those lines
(`diff` output in the run log). The build md5s are unchanged (`5e7d6d20…`, `50d69e20…`).

**Mechanism / fingerprint (consistent, not assumed):** the changed call writes `W_MSI_exc` (the
largest *weight* drift in the K=4 gate log, 1.31e-2) and `pre_trace_msi_rec`/`post_trace_msi_rec`
(2.69/2.71 — top of the divergence list). The altered `W_MSI_exc` then changes the recurrent drive
`g_rec_syn = F.linear(delayed_spikes_msi_rec, W_MSI_exc)` → `I_M` (232) → `v_msi`/`u_msi` (130/38).
First divergence is on the FIRST recurrent-STDP call (baseline pre = zeros at sequence start vs port
pre = current sMSI), so it desyncs immediately and cascades to the FF traces/weights via the altered
sMSI spike train.

## Suggested fix direction (Coder — NOT applied here)
Restore delay-fix EDIT #2 in `Training_graphdf.py` (exactly the proven revert, ~13 lines):
1. `__init__`: add `self._prev_sMSI_rec = None`.
2. `reset_state`: add `self._prev_sMSI_rec = None` (clears the delayed pre at each sequence start).
3. recurrent STDP block (`epoch>25`): `pre_spk_rec = self._prev_sMSI_rec if not None else
   torch.zeros_like(sMSI)`; pass `pre_spk=pre_spk_rec`; after `fill_diagonal_`, add
   `self._prev_sMSI_rec = sMSI.detach()`.
This is a PORT-FIDELITY restoration, independent of the perf levers (L1/L2 are clean — they byte-match
at epoch 25 and inside the reverted copy at epoch 26). `Training_graphdf_rev.py` is the working
reference; `diag_train_rev.py` re-proves byte-identity after the change.

## Remaining unknowns
- None material to the byte-match cause. The divergence is fully explained and reverts to zero.
- Functional note (lead's call, not byte-identity): delay-fix EDIT #2 is the documented mechanism
  that breaks the zero-delay antisymmetric LTP=LTD cancellation freezing `W_MSI_exc`; the port as-is
  (pre==post==sMSI) reintroduces that zero-lag recurrent rule. That is a scientific regression beyond
  the byte mismatch, but I report it as context, not a tested claim of mine.
