# FSTS Route-C SC Multisensory Network — tau40 10-model ensemble, 7 validations

A **clean, self-contained, reproducible bundle** of the route-C superior-colliculus (SCi) multisensory
(MSI) spiking network: the **10-seed tau40 ensemble** (seeds 42–51, epoch 79) and the **seven biological
validations** it has been measured against, each rendered with across-seed error bars.

Everything needed to **train**, **reproduce all seven validations**, and **regenerate every figure** lives
in this directory and resolves by bundle-relative paths — no `/tmp`, no external scratch, no network.

> Supersedes the earlier `5of6 / seed42-only` snapshot: now 10 models with mean±SD, the latency criterion
> reframed to **inverse effectiveness**, and a **7th** validation (cue-reliability / Bayesian cue
> integration) added.

---

## What it is

Route-C is a disynaptic feed-forward-inhibition (FFI) model of multisensory integration in the deep
superior colliculus. Auditory (A) and visual (V) afferents drive a shared MSI population on an `n = 180`
neuron spatial ring (Izhikevich units); cross-modal inhibition is **disynaptic** (A/V → interneuron →
MSI), and integration is carried by **slow NMDA** with a voltage-dependent dendritic Mg-gate.

**tau40 substrate (as-trained operating point):**

- NMDA `tau_nmda = 40 ms`; dendritic Mg-gate `dend_coupling_alpha = 2`, `mg_vhalf_exc = -48`,
  `mg_vhalf_inh = -30`, `mg_k = 0.15`; `gNMDA = 0.5`; recurrent `g_rec = 0.03` post-warmup.
- Divisive **surround-shunt GABA**: `k_shunt_surr = 0.026`, `E_gaba = -70 mV`, `tau_gaba = 10 ms`.
- **Asymmetric V-leading conduction delays** (auditory arrives ~15 ms before visual at the MSI) — the source
  of the V-leading temporal-binding-window asymmetry. **Do not alter the A/V delays.**
- **Correlated common-mode A–V latency jitter** `sigma_dL = 3 frames` (~30 ms), trained in.
- Form-A dV/dt-adaptive spike threshold is **inert** (`K_DVDT = 0.0`).
- Trained `bs = 250`, `ep0..79`, `delay = 52` substeps.

The full 16-variable substrate env is set automatically by every measurement script (see
`FROZEN-READOUT FIREWALL` and the `ENV` dict at the top of each `measure/` script). It **must** be set
before `routec_net_io` is imported (substrate params are read at net-build time).

---

## Layout

```
fallback_5of6_tau40/
├── README.md                 ·  this file
├── CKPT_MD5.txt              ·  md5 manifest (10 ckpts + 2 frozen readouts); `md5sum -c` verifiable
├── .gitignore
│
│   ── importable core (flat siblings: self-relative HERE/ROOT imports) ──
├── routec_net_io.py          ·  net-IO: load_ckpt (restores as-trained operating point), ckpt_path_for_seed,
│                                 frozen-readout md5 firewall, house-style plotting helpers
├── val36_traj_d52.py         ·  load/measure harness; registers the delayfix build as "Training"
├── Training_delayfix_d52.py  ·  eager MEASURE build (substep first-spike probe)
├── TBW_test.py  SBW_test.py  ·  FROZEN readouts — md5-LOCKED (80d33465 / 73b7d136), run-only
├── inverse_effectiveness_routec.py  response_latency_routec.py  cue_reliability_routec.py
├── diag_136b_cre_5seed.py  measure_107_convergence.py  measure_develop_check.py
├── q5_53_scorecard.py  harden_85.py  grade_88_noise.py
├── Training.py  Training_graphdf_d52.py  panelgraph.py  run34.py  retrain5_jitter.py   · TRAIN chain
│
├── checkpoint/               ·  the 10 trained models — ckpt_ep79_seed{42..51}_bs250_delay52_tau10_dL3.pt
├── fonts/Roboto-Regular.ttf  ·  house-style font
│
├── measure/                  ·  ensemble drivers (run from here; resolve ROOT from __file__)
│   ├── run_all.sh            ·  ONE command → all 7 validations + all figures
│   ├── parallel_measure.py   ·  gates 1–6 across BOTH local GPUs, aggregate, plot
│   ├── measure_ens_main.py   ·  gates 1–4 (RATE / TBW / E-I / MEI) per seed
│   ├── measure_ens_cre.py    ·  gate 5 (SBW / CRE) per seed
│   ├── measure_ens_latency_sweep.py  ·  gate 6 (latency benefit vs intensity)
│   ├── run_gate7_cuerel.py   ·  gate 7 (cue-reliability); --gain_exp 1 (headline) | 2 (transparency)
│   └── combine_ens.py        ·  merge per-GPU-lane JSONs → canonical aggregates
│
├── plots/                    ·  house-style figure scripts (pure: read results/ JSON → figures/)
│   ├── _bridge.py            ·  shared bridge (substrate env, frozen-md5 firewall, fonts)
│   └── run_gate1_ens.py … run_gate7_ens.py
│
├── results/                  ·  measurement outputs the plots consume (committed, JSON)
│                                gates_main.json, gate5_cre.json, gate6_latency_sweep.json,
│                                gate7_cuerel_g{1,2}_seed*.json, gate7_cuerel_g{1,2}_aggregate.json
├── figures/                  ·  the 7 delivered validations (PNG + SVG)
└── records/                  ·  provenance: grade_5of6_tau40.out, train_log_k0_tau40.out
```

---

## Dependencies

- Python 3.13, PyTorch 2.x (CUDA build), NumPy, SciPy, Matplotlib.
- One NVIDIA GPU to **measure or train** (the ensemble was measured on an RTX 5090 + RTX A6000; ~0.7%
  cross-device FP differences are far inside the across-seed spread).
- **No GPU needed to regenerate the figures** — `plots/run_gate*_ens.py` render purely from the committed
  `results/*.json`.

---

## How to run the validations

**Everything, one command** (gates 1–6 in parallel across both GPUs, then gate 7 both variants, then
all figures):

```bash
bash measure/run_all.sh                 # default seeds 42..51
bash measure/run_all.sh 42              # single-seed sanity
```

**Just regenerate the figures from the committed measurements (no GPU):**

```bash
for g in 1 2 3 4 5 6; do python plots/run_gate${g}_ens.py; done
python plots/run_gate7_ens.py gate7_cuerel_g1     # paper-faithful headline
python plots/run_gate7_ens.py gate7_cuerel_g2     # transparency variant
```

**Individual gates** (write JSON → `results/`, GPU):

```bash
python measure/measure_ens_main.py 42 43 44 45 46 47 48 49 50 51     # gates 1–4
python measure/measure_ens_cre.py  42 ... 51                          # gate 5
python measure/measure_ens_latency_sweep.py 42 ... 51                 # gate 6
python measure/run_gate7_cuerel.py --seed 42 --gain_exp 1 --device cuda:0   # gate 7 (per seed)
python measure/run_gate7_cuerel.py --aggregate --gain_exp 1                  #        (pool)
```

Ckpts resolve automatically via `routec_net_io.ckpt_path_for_seed(seed)` → `checkpoint/`.

---

## How to train

Single seed, ~13 min on an RTX A6000, one process per seed (one CUDA-graphed net per process avoids the
two-graph segfault). Writes `checkpoint/ckpt_ep{5,30,79}_seed{SEED}_bs250_delay52_tau10_dL3.pt`.

```bash
env CUDA_VISIBLE_DEVICES=1 RUN_DIR_OVERRIDE=/tmp/repro_tau40 \
    DEND_COUPLING_ALPHA=2 MG_VHALF=-48 MG_VHALF_INH=-30 MG_K=0.15 \
    GABA_SHUNT_SURR=1 K_SHUNT_SURR=0.026 E_GABA=-70.0 TAU_GABA=10 SIGMA_DL_FRAMES=3 \
    GNMDA=0.50 TAU_NMDA=40 G_REC=0.03 K_DVDT=0.0 TAU_DVDT=3.0 V_THRESH_FLOOR=20.0 DVDT_CAP=50.0 \
    python retrain5_jitter.py 42
```

Repeat for seeds 43–51 (the only change vs the certified delay52/tau10/bs250 reference lineage is
`SIGMA_DL_FRAMES=3`). `RUN_DIR_OVERRIDE` writes to scratch so a retrain never clobbers the committed
checkpoints; drop it to write straight into `checkpoint/`. Train build chain:
`retrain5_jitter.py → panelgraph.py → run34.py → Training.py` (pristine canonical) + `Training_graphdf_d52.py`
(CUDA-graphed). The published checkpoints were trained with the env above; reproduction is seed-deterministic
on a fixed device.

---

## Results — 10 models, mean ± SD across seeds 42–51

Reproduced bit-for-bit from the committed `results/` (frozen TBW/SBW md5 asserted BEFORE == AFTER on
every gate; weights never mutated).

| # | Validation | Ensemble result (mean ± SD, n = 10) | Read it honestly |
|---|---|---|---|
| 1 | **RATE** — bimodal MSI population rate | **17.43 ± 0.49 Hz**, in the re-grounded SCi band [12, 32] → **PASS** | band re-grounded from a stale [25,45] label (researcher #119) |
| 2 | **TBW** — temporal binding window | raw half-max width **258 ± 11 ms**; half-max crossings ≈ [−149, +121] ms | **V-leading asymmetry is real** (asymmetric A/V delays) — keep it |
| 3 | **E/I** balance (shunt-aware) | (E, I) = (21.30 ± 0.39, 20.52 ± 0.42); **E/I = 1.038 ± 0.024** | reported as **one averaged sync point** vs the 5 biological references (no scatter cloud) |
| 4 | **POP-IE** (MEI, inverse effectiveness) | MEI/intensity 0.05:**4.00** · 0.1:**7.76 ± 0.30** · 0.2:6.30 ± 0.44 · 0.4:1.73 · 0.8:0.95 · 1.6:0.93 — **mid-peak at weak I** | enhancement largest for weak stimuli = inverse effectiveness |
| 5 | **SBW** — AV cross-modal enhancement (CRE) | peak **76.0 ± 1.2 %**; zero-cross **29.9 ± 0.8°**; centre HWHM **19.3 ± 0.3°**; surround trough **−11.3 ± 1.5 %** @ 39° | negative surround lobe is **real cross-modal surround suppression**, KEPT; see caveats |
| 6 | **Latency** — onset facilitation (inverse effectiveness) | benefit min(A,V)−B = **+19.93 ± 1.08 ms at I = 0.05**, collapses to ≈ 0 by I = 0.1; full-I anchor L_A/L_V/L_B = 33.7/48.6/33.7 ms | benefit is real but **confined to the weakest near-threshold intensity**; see caveats |
| 7 | **Cue-reliability** — MLE / inverse-variance cue integration | **gain_exp = 1 (headline): R² = 0.866** (per-seed 0.841 ± 0.011 SEM), MAE = 0.093, RMSE = 0.133; equal-reliability w_V = 0.49, corner (σ_A=2, σ_V=20) w_V = 0.14; **100/100 cells** | every seed beats paper R² = 0.71; see gain_exp note |

### Caveats (read these before quoting a number)

- **Gate 3 (E/I):** a single network-averaged (E, I) point at the sync condition — compared to the
  biological references as one point, deliberately **without** a scatter cloud.
- **Gate 5 (SBW):** the spatial axis is an **RF-border-crossing proxy**, not a validated degree-width —
  zero-cross ~30° / trough ~39° mark "cue beyond its RF border." The negative surround lobe is genuine
  excitatory-centre / inhibitory-surround suppression (Meredith & Stein 1996; Kadunce et al. 1997, cat SC;
  Wallace et al. 1996, macaque). Our **−11 % trough is conservative** vs biology (mean depression ~46 %,
  up to ~100 %).
- **Gate 6 (latency):** the inverse-effectiveness benefit (+20 ms) is **reproducible but confined to the
  weakest, near-threshold intensity (I = 0.05) and collapses by I = 0.1.** It does **not** overturn the
  magnitude limit (the multisensory first spike is capped by the #137 regenerative-leap gap, the one
  remaining open mechanism). At full intensity the bimodal first spike sits **at** the race-model bound
  (B = fastest unimodal) — reported as an honest null, never cherry-picked.
- **Gate 7 (cue-reliability):** **`gain_exp = 1` is the paper-faithful headline** (constant-area Gaussian,
  gain ∝ σ_ref/σ — the manuscript Methods). `gain_exp = 2` (the repo's exploratory "set 1.0 or 2.0" knob)
  scores a higher R² = 0.910 **but only 75/100 cells are measurable** (24 weak/wide stimuli give empty MSI
  profiles, honestly NaN-excluded — never a spurious w_V = 0). It is kept for transparency, rendered as the
  `_g2` figures, and is **not** the headline (apples-to-oranges vs the paper's gain_exp = 1).

---

## Frozen-readout firewall

`TBW_test.py` and `SBW_test.py` are the **frozen biological readouts** and are **md5-LOCKED**:

```
TBW_test.py = 80d33465c4bf55d6e85b5990acb92da7
SBW_test.py = 73b7d13626964d851cc090818b728311
```

They are **run-only and must never be edited to pass.** Every gate (and every plot) asserts both md5s
**BEFORE == AFTER** via `routec_net_io.assert_frozen_readouts`, and the measurement gates additionally
assert **weight bit-identity** before == after (the net is inference-only, never mutated). `md5sum -c
CKPT_MD5.txt` re-verifies the readouts and all 10 checkpoints from the bundle root.

---

## Checkpoint strategy

**All 10 checkpoints are committed** (`checkpoint/`, ≈ 19.3 MB total, 1.93 MB each). This gives exact,
bit-identical reproduction of every validation with no retraining and no GPU. The checkpoints are not large
for git, `.gitignore` does not exclude `*.pt`, and the earlier snapshot already tracked seed 42 — so
committing the full set is the consistent, lowest-friction choice. To regenerate any model from scratch
instead, use `retrain5_jitter.py` / `measure/run_all.sh` (see **How to train**); a single seed is ~13 min on
an A6000. `CKPT_MD5.txt` pins every checkpoint's md5.

---

## Records (provenance)

- `records/grade_5of6_tau40.out` — the original seed-42 scorecard the earlier snapshot summarised.
- `records/train_log_k0_tau40.out` — the seed-42 training log (config header + per-epoch vitals).
