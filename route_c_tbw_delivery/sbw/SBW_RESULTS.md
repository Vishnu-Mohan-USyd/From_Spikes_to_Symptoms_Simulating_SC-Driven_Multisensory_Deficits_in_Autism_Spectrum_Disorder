# Route-c SBW measurement — asymrc (+ asymd control)

Spatial Binding Window (SBW) half-widths for the **route-c** retrain (`asymrc`,
the same 5 ep80 checkpoints whose TBW box is documented in
[`../DOCUMENTATION.md`](../DOCUMENTATION.md)), measured on this host after the
route-c TBW delivery. `asymd` (the sibling retrain *without* route-c,
`tau_nmda_inh` absent) is included as a paired control.

The TBW work left SBW unmeasured on the retrained model; this directory closes
that gap. **The route-c fix touches only the interneuron NMDA decay
(`tau_nmda_inh=25`); SBW is a spatial metric, so the expectation was little
movement — confirmed: a small, consistent widening, both lineages well inside
the biology band.**

---

## Headline

| lineage | per-seed half-width (deg) | mean ± SD | grand-fit | degenerate? |
|---|---|---|---|---|
| **asymrc** (route-c) | 29.83, 31.00, 31.70, 32.02, 31.18 | **31.15 ± 0.75°** | 31.16° | No |
| **asymd** (no route-c, control) | 28.07, 28.62, 30.73, 31.34, 29.71 | 29.69 ± 1.23° | 29.70° | No |

- **asymrc SBW = 31.1 ± 0.7°** — a clean, **non-degenerate, symmetric**
  fusion bell: P(fusion) rises from 0 at large separation to 1.0 at 0°
  separation and falls symmetrically back to 0 (grand P(fusion) 0.00 → 1.00,
  max |Pf(+s) − Pf(−s)| ≈ 0.04). This is unlike the TBW result on the same
  checkpoints, which is a **flat-top box** (see `../DOCUMENTATION.md`); SBW is
  well-behaved.
- **Both lineages sit comfortably inside the #39 biology band [24.5, 40.9]°**
  (ledger-#39 reference SBW half-width 27.29°).
- **Route-c vs control: +1.45° paired (≈ +5%), 5/5 seeds positive**
  (+1.76, +2.38, +0.97, +0.68, +1.47°). The effect is **modest — on the order
  of the per-seed SD** — and should be reported as a small consistent widening,
  not a large effect.

## Independent validation

The number was independently reproduced by a second run with the same frozen
apparatus and route-c code on the same 5 local asymrc ep80 checkpoints:

- independent per-seed: 29.87, 30.98, 31.23, 31.60, 31.47° → grand-fit
  **31.04°**, mean 31.03 ± 0.62°, `degenerate=False`.
- per-seed |Δ| vs the run in this directory: max **0.47°** (all ≪
  unseeded-Monte-Carlo tolerance); grand 31.16 vs 31.04 → Δ 0.12°.
- curve matches point-by-point within ~0.03 in P(fusion).
- per-ckpt wiring confirmed identical on both runs:
  `tau_nmda=80.0 / tau_nmdaVolt=100.0 / tau_nmda_inh=25.0 / gNMDA=0.700 /
  plasticity_enabled=False / sigma_in=6.50`.

Verdict: **GO** on reporting asymrc SBW half-width ≈ **31° (31.1 ± 0.7°)**.

---

## Method (frozen, identical to `make_curves.make_sbw`)

- **Apparatus**: `apparatus/SBW_test.py`, md5 `73b7d13626964d851cc090818b728311`
  (the validated build — plasticity-OFF via `load_msi_model` `.eval()` +
  `plasticity_enabled=False`; `is_fused` = `gaussian_filter1d(sigma=2, wrap)` +
  `find_peaks(height=0.5, dist=10)`, fused if ≤1 peak **or** valley/peak > 0.6).
- **Metric**: `compute_sbw_fused_persep` → P(fusion) per separation;
  half-width = `abs(fit_pedestal_curve(...)[2])` (the fitted pedestal `w`,
  exactly the `make_curves` half-width).
- **Grid / sampling**: `separations_deg = range(-80, 85, 5)` (33 points),
  `n_trials=50`, `intensity=0.5`, `duration=20`, `linear=False`.
- **Parallelism**: one checkpoint per process, all 10 (5 asymrc + 5 asymd) run
  concurrently. `compute_sbw_fused_persep` uses an **unseeded**
  `np.random.default_rng()` with no cross-checkpoint state, so per-checkpoint
  parallelization is semantically identical to `make_curves`' sequential loop
  (each seed is an independent Monte-Carlo estimate). This is why the two
  independent runs differ by ≤0.47° per seed — that spread *is* the MC noise.
- **Per-lineage code path**: each process imports `Training.py` from the
  lineage's own `code_dir` (route-c `Training.py`, md5
  `6b649d153850cfc3c872f4dcafd71b95`, for asymrc) so `SBW_test`'s
  `from Training import *` resolves to the correct net definition.

### `asymd` control — the `tau_nmda_inh=nan` print is benign

asymd was trained **before** route-c, so its net has no `tau_nmda_inh`
attribute; the run script's `getattr(net, "tau_nmda_inh", nan)` therefore prints
`nan`. This is a missing-attribute fallback, **not** dynamics corruption — the
asymd run reproduces the previously-delivered asymd SBW (29.83°) to within
0.14°. It does not touch the asymrc measurement, where `tau_nmda_inh=25` is
present and verified on every checkpoint.

---

## Files

```
sbw/
  SBW_RESULTS.md          this file
  sbw_routec_run.py       measure SBW on ONE checkpoint (parallel-friendly)
  aggregate_sbw.py        combine 5 per-seed jsons -> mean±SD + grandmean-curve fit
  run_all_sbw.sh          launch all 10 (5 asymrc + 5 asymd) in parallel, then aggregate
  apparatus/
    SBW_test.py           validated apparatus, md5 73b7d136
  out/
    sbw_asymrc_m{0..4}.json   per-seed asymrc results (Pf curve, hw, wiring)
    sbw_asymd_m{0..4}.json    per-seed asymd control
  sbw_asymrc_AGG.json     asymrc aggregate (per-seed hw, mean/SD, grandmean Pf + hw)
  sbw_asymd_AGG.json      asymd aggregate
  logs/                   per-seed run logs + aggregate logs + orchestrator log
```

## Reproduce

The route-c checkpoints and `Training.py` live in this same delivery
(`../checkpoints/asymrc_m{0..4}_ep80.pt`, `../code/Training.py`). From this
`sbw/` directory:

```bash
# one checkpoint (route-c / asymrc), to ../code's Training.py:
python sbw_routec_run.py \
    --eval_app ./apparatus --code_dir ../code \
    --ckpt ../checkpoints/asymrc_m0_ep80.pt --tag asymrc_m0 \
    --out_json ./out/sbw_asymrc_m0.json

# aggregate the 5 seeds:
python aggregate_sbw.py \
    --eval_app ./apparatus --code_dir ../code \
    --glob './out/sbw_asymrc_m*.json' --prefix asymrc \
    --out_json ./sbw_asymrc_AGG.json
```

**Hardcoded-path note (adjust for another host).** `run_all_sbw.sh` hardcodes
absolute paths from the host this ran on — `W=/home/vishnu/coding_proj/routec_sbw_work`,
`PY=/home/vishnu/miniconda3/bin/python3`, and the `RC_CODE`/`RC_CK`/`AD_CODE`/`AD_CK`
checkpoint+code roots. To reproduce inside this repo, ignore `run_all_sbw.sh`
and call `sbw_routec_run.py` / `aggregate_sbw.py` directly with the relative
`--code_dir ../code` / `--ckpt ../checkpoints/...` / `--eval_app ./apparatus`
arguments as shown above (`sbw_routec_run.py`'s only non-relative default is
`--eval_app /scratch/eval_apparatus`, which the explicit `--eval_app ./apparatus`
overrides). The torch-load shim in both scripts sets `weights_only=False` to load
these trusted research checkpoints under torch ≥ 2.6; it changes only the
unpickling flag, not the weights or the metric.

## md5 manifest (committed sbw artifacts)

```
49113f562a7f5271f113a324dd48d32f  aggregate_sbw.py
73b7d13626964d851cc090818b728311  apparatus/SBW_test.py
7829b0f49af4d81642018ac8ed2573c4  run_all_sbw.sh
8a2afe1f1e8119e8833d4303192e10f7  sbw_asymd_AGG.json
1f20928a20e2f2f0185f4245e3b7dd9a  sbw_asymrc_AGG.json
66ff2409388777a08402b98e82036d76  sbw_routec_run.py
```
