# DIAGNOSTIC REPORT — L4 (alloc-free reset + gpos cursors) byte-FAIL + NaNs under training (task #65)

**Debugger forensic. Diagnosis only — NO fix applied. Frozen baseline `Training_delayfix.py` md5
`5e7d6d20538592b5fc92165b371949d7` and port-under-test `Training_graphdf.py` md5
`2d14feb782a6643674a8fab755b82d81` both VERIFIED byte-untouched. I created only throwaway
diagnostic harnesses (`diag_l4.py`, `diag_l4b.py`, `diag_l4c.py`); I edited NO build and NO gate
file. All compute cuda:0 / RTX 5090. Every claim is backed by a command + captured output in
`diag_out/l4_split{1,2,3}.log`.**

---

## Failure (as reported)
`train_bitid_gate.py --check-l4 --K 4 --bs 256 --epoch 0` FAILs the L4 gate: opt(graphdf) with
`l1=0 l2=1 l4=1` does not byte-match base(delayfix) — **65 tensors differ, worst |Δ|=1.680e+03 @
attr:nmda_m_inh, NaN in v_msi_inh / u_msi_inh**; data_ptr stability PASSES (4 resets × 68 tensors
stable). Reproduced verbatim (`diag_out/l4_split2.log`, every combo; `diag_out/l4_split3.log`
CONFIG 2): same 65 tensors, same worst 1.680e+03 @ nmda_m_inh, same NaNs, same v_uniA=9.973e+01 /
v_uniV=9.452e+01 / W_inA=2.562e-2 / W_inV=2.867e-2.

## Headline result
**The L4 alloc-free reset + gpos cursors are BYTE-CLEAN. The `--check-l4` byte-FAIL is a HARNESS
artifact in `train_bitid_gate.py`: opt4 is given base's ALREADY-TRAINED weights, so the gate
compares a single-trained net against a double-trained net.** With the transfer placed correctly
(opt starts from the same init as base), L4=1 byte-matches the baseline — with the alloc-free
RESTORE path proven to have executed.

---

## Hypotheses tested (every one gets a verdict)

| # | Hypothesis | Verdict | Evidence (command → output) |
|---|---|---|---|
| H1 | alloc-free RESET (persistent pool restore) causes the FAIL/NaN | **RULED OUT** | `diag_l4.py` ALLOCFREE_ONLY (l4=0 **l5=1**, gpos OFF) → PASS byte-identical; `diag_l4c.py` CONFIG1 (reset path executed: 4 resets, cache present, data_ptr stable) → PASS |
| H2 | gpos cursors (incl. the `_inh` cursors with no SHIP-PARITY rebind) cause it | **RULED OUT** | `diag_l4.py` FULL_L4 (l4=1, gpos ON) → PASS byte-identical |
| H3 | alloc-free reset × gpos INTERACTION causes it | **RULED OUT** | FULL_L4 is both together → PASS (l4_split1.log) |
| H4 | prior-net allocator pressure (gate builds/trains 4 nets before opt4) | **RULED OUT** | `diag_l4b.py` PRIOR=0 and PRIOR=1 both FAIL identically (1.680e+03 @ nmda_m_inh) — pressure does not change the verdict |
| H5 | the data_ptr probe shadowing `net.reset_state` perturbs values | **RULED OUT** | `diag_l4b.py` PROBE=0 and PROBE=1 both FAIL identically |
| H6 | gate harness transfers base's TRAINED weights into opt4 (opt4 double-trains) | **CONFIRMED** | `diag_l4c.py`: move ONLY the `transfer_weights` call → BEFORE base-train = PASS, AFTER base-train = FAIL (exact signature), L4 exercised in both |

### Experiment 1 — clean rebuild, split L4 into its two bundled changes (`diag_out/l4_split1.log`)
Harness `diag_l4.py` builds base+opt, **transfers BEFORE training**, trains each once, compares.
The existing `_lever_L5_allocfree_reset` (port line 2484: reset runs under `L4 OR L5`; line 2787:
gpos forward is `L4`-only) gives a no-edit single-variable split:
```
FULL_L4        (l1=0 l2=1 l4=1 l5=0)  alloc-free reset + gpos  -> PASS (byte-identical)
ALLOCFREE_ONLY (l1=0 l2=1 l4=0 l5=1)  alloc-free reset, gpos OFF-> PASS (byte-identical)
```
Both byte-clean → neither the reset nor the gpos cursors diverges. (This is the first sign the
reported FAIL is not in the L4 code.)

### Experiment 2 — rule out allocator pressure & the probe (`diag_out/l4_split2.log`)
`diag_l4b.py` mirrors the gate's structure (transfer AFTER base-train) with two toggles, fresh
process per combo:
```
PRIOR=0 PROBE=0 -> FAIL (65; 1.680e+03 @ nmda_m_inh)   PRIOR=0 PROBE=1 -> FAIL (identical)
PRIOR=1 PROBE=0 -> FAIL (identical)                     PRIOR=1 PROBE=1 -> FAIL (identical)
```
All four identical → prior-net allocator pressure (H4) and the reset_state-shadow probe (H5) are
NOT the cause. The verdict tracks ONLY the harness structure shared by all four (transfer-after-train).

### Experiment 3 — single-variable transfer timing, definitive (`diag_out/l4_split3.log`)
`diag_l4c.py`: identical build order, L4=1 in BOTH, opt trained WITH the data_ptr probe so the
alloc-free RESTORE path is proven to run in both. The ONLY thing moved is the `transfer_weights`
line:
```
CONFIG 1  transfer BEFORE base training (correct):
   resets_recorded=4  cache_present=True   data_ptr stable: PASS (4×68 stable)
   base vs opt (L4=1)  -> PASS (byte-identical)
CONFIG 2  transfer AFTER base training (gate's order, = gate line 185 then 202):
   resets_recorded=4  cache_present=True   data_ptr stable: PASS (4×68 stable)
   base vs opt (L4=1)  -> FAIL (65 differ; worst 1.680e+03 @ nmda_m_inh; v_msi_inh/u_msi_inh NaN)
```
Alloc-free reset executed and addresses were stable in BOTH; only the comparison's starting weights
differ. FAIL ↔ PASS flips with that single line.

### Experiment 4 — the decisive 2×2 (refutes "levers-off/lever-b PASS ⇒ fault inside L4")
`diag_l4d.py` (`diag_out/l4_2x2.log`). The premise's inference has a confound: in the gate,
opt0/opt2 transfer BEFORE base-train (`:180-181`) but opt4 transfers AFTER (`:202`) — so opt2-vs-opt4
differs in TWO variables (lever AND transfer timing). Decoupling them, fresh base+opt per cell:
```
           | transfer BEFORE | transfer AFTER (gate opt4 order)
  L4=0     |      PASS       |      FAIL  (65 differ; 1.680e+03 @ nmda_m_inh)
  L4=1     |      PASS       |      FAIL  (65 differ; 1.680e+03 @ nmda_m_inh)
```
The verdict is set ENTIRELY by the transfer column; the L4 row is FLAT. The key new cell is
**L4=0, transfer-AFTER → FAIL with the identical signature** — the FAIL happens with NO L4 at all.
So opt2 (lever-b) passed in the gate *only* because it used transfer-before-train; run opt2's config
under transfer-after-train and it FAILs too. L4 contributes nothing to the byte-mismatch.

---

## Proven root cause
**`train_bitid_gate.py:202` (`PG.transfer_weights(base, opt4)`) runs AFTER `:185`
(`train(base, …)`) has already trained `base` in place. So opt4 starts from base's TRAINED weights
and then trains again, while the reference `base` was trained once. `compare(base, opt4)` therefore
pits a single-trained net against an effectively double-trained net → 65-tensor divergence, growing
to NaN in the inhibitory MSI state.** The same flaw is at `:216` (`transfer_weights(base, optr)`
for `--check-reset`). The correctly-built nets in the same gate — opt0/opt2, transferred at
`:180-181` BEFORE training at `:185` — PASS, which is the in-file control.

This is NOT a defect in the L4 alloc-free reset or gpos cursors. Those are byte-identical to the
baseline when opt starts from the same init as base (Experiments 1 & 3, CONFIG 1), with the
alloc-free restore path proven executed (4 resets, cache present, data_ptr stable).

**Causal proof (single variable):** in `diag_l4c.py`, moving ONLY the `transfer_weights(base→opt)`
call from after-base-training to before-base-training — nothing else changed (same build order,
L4=1, same seed/K/bs/epoch, alloc-free restore confirmed running in both) — flips base-vs-opt from
**FAIL (worst |Δ|=1.680e+03 @ nmda_m_inh, NaN) to PASS (byte-identical)**. Build md5s unchanged
(`5e7d6d20…`, `2d14feb7…`).

**Why the reported fingerprints are consistent:** data_ptr stability PASSED (the alloc-free reset is
correct — addresses never move); the value divergence and NaN come entirely from the wrong starting
weights, independent of L4. The divergence reaches the unimodal layer (v_uniA/v_uniV, W_inA/W_inV)
because the wrong starting weights are the FF input weights themselves — exactly what a second
training pass perturbs first; that the *unimodal* layer diverges is itself proof the cause is
upstream of (and unrelated to) the MSI-only gpos cursors.

## Suggested fix direction (Coder — applies to the GATE harness, NOT the build)
In `train_bitid_gate.py`, give opt4/optr the SAME init as base (as opt0/opt2 already get):
- Snapshot base's init state before any training (e.g. capture `base.state_dict()` deep-copied, or
  keep a pristine `base_init` net), and transfer THAT pristine init into opt4 (`:201-202`) and optr
  (`:215-216`); **or** build+transfer opt4/optr up front alongside opt0/opt2 (before `:185`).
- No change to `Training_graphdf.py` is warranted by this gate FAIL. After the harness fix, re-run
  `--check-l4` and `--check-reset`; expect PASS — `diag_l4.py`/`diag_l4c.py` CONFIG 1 already prove
  L4 (reset+gpos) and L5 (reset-only) are byte-clean.

## Remaining unknowns
- None for the byte-FAIL: it is fully explained and reverts to byte-identity with the one-line
  harness correction.
- I did not separately characterize the dynamics of why double-training reaches NaN; it is a
  consequence of starting training from already-trained weights (not a code defect) and is not
  needed for the diagnosis.
