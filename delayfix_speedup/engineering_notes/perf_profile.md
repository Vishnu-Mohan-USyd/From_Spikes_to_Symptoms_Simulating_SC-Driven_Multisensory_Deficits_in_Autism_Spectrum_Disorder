# DIAGNOSTIC REPORT — where the training compute goes (task #48)

**Debugger forensic profiling. Diagnosis only — no fixes applied. All numbers are
command+output-backed (RTX 5090 / cuda:0, single proc unless stated). Build =
`delayfix_20260614/Training_delayfix.py` md5 `5e7d6d20538592b5fc92165b371949d7`, imported AS-IS.**

Harnesses (this dir, `prof_20260614/`): `prof_decompose.py` (A), `prof_inside.py` (B),
`prof_profiler.py` (C), `prof_panel.py` (F), `prof_nonzero.py` (H), `prof_stdp.py` (I),
`prof_ep5_microbench.py` + `prof_trainonly.py` (E), `prof_concurrency.sh` (D). Raw outputs in `out/`.

---

## Failure / question
Training a single network is reported at ~8 h. Localize, with measured evidence, exactly where
the wall-clock goes (per-epoch components AND inside `train_unsupervised_batch`), and state the
mechanism (kernel-launch-bound vs sync-stall vs compute-bound).

## Reproducer
`R = retrain_delayfix`; `net = R.build_net(42)` (sets tau_nmda_inh=21.6, n_substeps=100);
per-epoch loop body = `train_unsupervised_batch(1000, batch_size=256, epoch_idx=ep)` then
`W_stats` / `measure_vitals`(→`net._panel_battery(seed, full=False)`) / `inh_clamp` / `json.dump`.
Training generates T_max=20 ext steps × 4 mini-batches → 8,000 substeps/epoch.

---

## TL;DR — the compute budget (clean single proc, ~85–89 s/epoch)

| Layer | Component | Time | Share |
|---|---|---|---|
| **per-epoch** | **`_panel_battery` (measure_vitals)** | **55.4 s** | **65 %** |
| | `train_unsupervised_batch` | 29–32 s | 35 % |
| | W_stats / inh_clamp / json / ckpt | <0.01 s ea | ~0 % |
| panel → | single-volley (B=8, 12,000 substeps) | 38.4 s | 70 % of panel |
| | TBW sub-sim (B=540, 6,000 substeps) | 16.3 s | 30 % of panel |
| | numpy classify + stim-gen | 0.07 s | 0.1 % of panel |
| train → | `update_all_layers_batch` (100-substep fwd) | 24–27 s | 84 % of train |
| | `stdp_update_batch` | 4.3–5.3 s | 15 % of train |
| | data-gen / poisson / reset / slow-updates | <0.03 s ea | ~0 % |

**Both dominant costs are the SAME `update_all_layers_batch` substep loop**, run launch-bound:
the panel is 99.9 % that loop, and training is 84 % that loop. Nothing else is material.

---

## Hypotheses tested

| # | Hypothesis | Verdict | 1-line evidence |
|---|---|---|---|
| H1 | The untimed measurement battery dominates per-epoch wall | **CONFIRMED** | panel 55.4 s = 65 % vs train 29 s (Exp A) |
| H2 | W_stats/inh_clamp/json/ckpt are non-negligible | **RULED OUT** | each <10 ms (Exp A) |
| H3 | Inside train, the 100-substep fwd loop dominates (not data-gen/STDP) | **CONFIRMED** | update 84 %, stdp 15 %, data-gen 0.08 % (Exp B) |
| H4 | The substep loop is kernel-launch-bound (not compute-bound) | **CONFIRMED** | GPU-busy 33.6 % of wall; 384 launches/substep; 1.36 µs/kernel; B=8≈B=540 per-substep (Exp C,F) |
| H5 | A host-sync storm serializes the GPU each substep | **CONFIRMED** | 35 DtoH/substep; `nonzero` 10/substep from `u[mask]+=d`; idiom swap 4.14× (Exp C,H) |
| H6 | STDP is launch-bound via a Python per-batch loop | **CONFIRMED** | `for b in range(256): ger(...)` = 212.9× slower than matmul, identical math (Exp I) |
| H7 | The real ~74 s train (vs clean 29 s) is GPU contention from 5 concurrent seeds | **CONFIRMED** | controlled 1-way=28.7 s → 5-way=74.2 s = **2.59×**, reproduces real 74.7 s exactly (Exp D) |
| H8 | ep5 is anomalously slow due to a dead-debug `.item()` block | **CONFIRMED (small clean, large under contention)** | `if epoch_idx==5:` 4 unused `.item()`/substep (L3063-74); clean ep5=29.6 s vs ep4/6=28.4 s (**+1.1 s**); real 5-way ep5≈88 s vs ~75 s (**+13 s**, contention-amplified) |

---

## Proven root causes (with causal evidence)

### 1. The measurement battery is 65 % of per-epoch wall — and it is the same substep loop
`measure_vitals` → `_panel_battery(seed, full=False)` runs two sub-simulations through
`update_all_layers_batch`:
- **single-volley**: `_panel_single_volley(B=8, T_frames=120)` → 120 frames × 100 = **12,000 substeps** → **38.4 s (70 % of panel)**
- **TBW**: `compute_tbw_temporal_fusion_persep(27 offsets × 20 trials = 540, T=60)` → 60 × 100 = **6,000 substeps** → **16.3 s (30 %)**
- numpy classify (`is_temporally_fused` ×540) + stim-gen = **0.07 s (0.1 %)**

Evidence (Exp F, `out/panel_ep0.json`): panel 54.8 s; update_all_layers_batch = 54.7 s = **99.9 %**.
**Per-substep: B=8 volley = 3.20 ms vs B=540 TBW = 2.72 ms.** A 67× larger batch costs the SAME
per substep → the cost is the substep COUNT, not the work → launch-bound (causal: batch size is
not the variable; substep count is).

### 2. `update_all_layers_batch` is kernel-launch-bound + sync-stalled
Exp C (`prof_profiler.py`, 400 substeps, B=256, `out/profiler_ep0.json`):
- **GPU-busy (self CUDA) = 33.6 % of wall** → GPU idle 2/3 of the time
- **384.6 cudaLaunchKernel / substep**, 746.8 device kernels/substep, **avg kernel 1.36 µs**
- **13,904 Memcpy-DtoH host-syncs / 400 substeps (~35/substep)**; `aten::nonzero` = 10/substep
- Top CPU op = `cudaLaunchKernel` (576 ms). Top CUDA ops are tiny elementwise (`mul`, vectorized_elementwise).
Tiny kernels (1.36 µs) with a 384-launch/substep dispatch load and the GPU idle 66 % ⇒ the wall is
CPU dispatch + launch latency, not compute. (ep26/recurrence-ON identical: 34.8 % busy, 0.94 µs/kernel.)

### 3. The sync storm: `u[mask] += d` (Izhikevich reset) forces `nonzero` host-syncs — 5×/substep
L2956/2966/2980/3090/3102: `self.u_xxx[spike_mask] += d` (boolean-mask index-assign). Each does a
read+write → 2 `aten::nonzero` → 2 DtoH syncs; 5 sites × 2 = 10 nonzero/substep (matches Exp C's 10).
Exp H single-variable proof (`prof_nonzero.py`, only the idiom changes):
- `u[mask] += d` = **4.350 s** vs `u += mask*d` = **1.052 s** → **4.14×**
- profiler/100 substeps: idiom A = 1000 nonzero + **2000 DtoH** + 7000 launches; idiom B = **0 nonzero, 0 DtoH**, 2000 launches.

### 4. `stdp_update_batch` loops over the batch in Python (212.9× penalty)
L3313-3323: `for b in range(B=256): dW += ger(post[b],pre[b]) - ger(post_trace[b],pre_spk[b])`
— 2·B=512 tiny `ger` kernels per call, ×480-560 calls/epoch.
Exp I single-variable proof (`prof_stdp.py`, math identical, diff 2.7e-5):
- for-`b` ger loop = **3.603 s** vs `post.T@pre_trace - post_trace.T@pre_spk` = **0.017 s** → **212.9×**
- profiler/call: loop = 1024 ger kernels; matmul = 4 mm. Same dW.

### 5. The ~8 h is 5 concurrent seeds on ONE GPU (contention), not one slow network
`run_retrain.sh` L27-32 launches **5 simultaneous** `retrain_delayfix.py` on `CUDA_VISIBLE_DEVICES=0`.
- Real epochs.tsv train-only `sec` (5-way) = median **74.7 s**; my clean 1-way train = **29 s** → 2.55×.
- `gpu_sample.csv` (2107 samples, 71 min, 5-way): GPU util **mean 89.9 %, median 92 %**; mem 9.1 GB
  of 32.6 GB (5 × 1.8 GB). Packing 5 launch-bound procs fills the GPU (1-proc util ~26-33 %).
- Real per-epoch WALL (5-way batch) = 4282 s / 15 epochs = **285 s/epoch** → 80 epochs ≈ **6.3 h**
  (recurrence-ON epochs slightly heavier → ~the reported ~8 h).
- **Controlled 1×-vs-5× (Exp D — single variable = #procs on cuda:0, same code/epoch/build, train-only no panel):
  1-way = 28.71 s → 5-way = 74.2 s mean (73.8/74.1/74.1/74.5/74.7, tight) = 2.59×.** This REPRODUCES the
  real `epochs.tsv` 74.7 s median ⇒ the real ~74 s train time is GPU contention, proven causally
  (only the proc count changed). `out/concurrency.log`.

### 6. (secondary) ep5 dead-debug `.item()` block — small alone, amplified by contention
L3063-3074 runs ONLY at `epoch_idx==5`: 4 `(...).sum().item()` host-syncs/substep whose results are
**unused** (dead code). Measured (Exp E, `prof_trainonly.py`):
- **clean 1-proc**: ep4=28.41 s, **ep5=29.57 s**, ep6=28.44 s → **+1.1 s** (small — the GPU is
  already idle 66 %, so the extra per-substep syncs are mostly absorbed).
- **real 5-way run** (epochs.tsv): ep5 = 87.5/89.0/87.8/88.2/88.6 s vs ~75 s neighbours → **+13 s**.
The 12× amplification is the same per-substep sync becoming expensive once 5 procs saturate the GPU
(a cheap drain when idle → a contended stall at ~90 % util). Trivial to remove (dead code) but only
a ~0.2 % single-proc / contention-dependent effect — lowest priority.

---

## Suggested fix directions (for the Coder — NOT applied here; lead decides)
Ranked by measured payoff. All are vectorization/scheduling, not algorithmic — outputs unchanged.
1. **Slash the per-epoch panel (65 % of wall).** It runs 18,000 launch-bound substeps every epoch.
   Options: run `_panel_battery` every N epochs (not every epoch); shrink the single-volley
   (12,000 substeps @ B=8 = 45 % of the whole epoch) and the TBW grid; or batch volley wider.
2. **Kill the substep sync storm.** Replace the 5× `u[mask] += d` with `u += mask.to(dtype)*d`
   (or `torch.where`) — removes ~10 nonzero + ~20 DtoH/substep (proven 4.14×).
3. **Vectorize STDP.** Replace `for b in range(B): ger(...)` with `post.T @ pre_trace` (proven 212.9×).
4. **Reduce launch count / enable pipelining.** The substep loop launches ~384 kernels/substep at
   1.36 µs each (GPU 33.6 % busy) → fuse elementwise ops / CUDA-graph the substep / cut per-ext-step
   `.item()` syncs (L3228-31) so the CPU can run ahead.
5. Remove the ep5 dead-debug `.item()` block (L3063-3074).

## Remaining unknowns
- None outstanding for the diagnosis (Exp D filled in §5; Exp E clean ep4/5/6 in §6).
- ~15 of the ~35 DtoH/substep are accounted for by `nonzero` (×2 each = 20); the residual DtoH source
  was not separately itemized (immaterial to the verdict — sync-storm is already proven by the 4.14× idiom swap).
- Fix payoffs above are *component measurements*, not end-to-end post-fix runs (debugger does not fix).
