# route-c deliverable — interneuron-only fast NMDA decay (`tau_nmda_inh = 25`)

FSTS (From Spikes to Symptoms) · TBW disynaptic-inhibition lineage.
Self-contained snapshot: route-c code, the 5 `asymrc` ep80 checkpoints, the route-c TBW
curves + aggregate, and this document. Every artifact is md5-listed in §(f); the md5s were
re-computed on the delivered files (not recalled). This document records facts and
evidence only.

---

## (a) What route-c is — the code change

route-c gives the **disynaptic interneuron's NMDA conductance** its own, faster decay
time-constant, decoupled from the excitatory NMDA pool. Intent: **shorten the sustained
disynaptic-GABA (PV-type interneuron → MSI_exc) inhibition** that follows a stimulus.

Three loci in `code/Training.py` (md5 `6b649d153850cfc3c872f4dcafd71b95`); the line numbers
below are the ones present in the delivered file:

| # | Line | Code (verbatim, as delivered) |
|---|------|-------------------------------|
| 1 | **1507** | `self.tau_nmda_inh = 25.0  # task#51 route-c: interneuron NMDA decays faster than exc/volt (tau_nmda=80)` |
| 2 | **2234** | `nmda_decay_inh = 1.0 - self.dt / self.tau_nmda_inh  # task#51 route-c (interneuron NMDA decay)` |
| 3 | **2496** | `self.nmda_m_inh.mul_(nmda_decay_inh)` |

Mechanism, as verified in the delivered file:
- Edit 1 declares `tau_nmda_inh = 25.0` ms.
- Edit 2 (L2234) builds the per-substep decay factor `nmda_decay_inh = 1 − dt/tau_nmda_inh`
  (= 0.99600 at `dt = 0.1`).
- Edit 3 (L2496) applies that factor to the **interneuron** NMDA state `self.nmda_m_inh`.
- The **excitatory** NMDA pool `self.nmda_m` decays separately with
  `nmda_decay = 1 − dt/self.tau_nmda` (built L2233, applied L2394). The two pools now use
  different time-constants — that decoupling is route-c.

> **Runtime NMDA time-constants (constructor defaults are overridden before training).**
> The `Training.py` constructor defaults `self.tau_nmda = 40.0` (L1506, excitatory) and
> `self.tau_nmdaVolt = 200.0` (L1511, voltage) are **both overridden before training**: the
> excitatory NMDA tau is set to **80.0** and the voltage NMDA tau to **100.0**. This happens
> in the build/training path (`Training.py` L3913 / L3916) and, for the actual `asymrc`
> retrain, in the runner `run_task192_disynaptic_ep40_retrain.py` L431 / L434 — applied on
> the freshly-built net (constructed L409 with the asym delays) before training;
> `launch_asymrc.sh` L22 invokes that runner. route-c's `tau_nmda_inh = 25.0` (L1507) is
> **never** overridden. Therefore the delivered `asymrc` ep80 checkpoints were trained with
> **excitatory `tau_nmda = 80`, voltage `tau_nmdaVolt = 100`, interneuron
> `tau_nmda_inh = 25`** — i.e. the interneuron NMDA (25 ms) decays faster than the runtime
> excitatory NMDA (80 ms), exactly as the L1507 comment "(tau_nmda=80)" states. The comment
> is accurate.
>
> Two distinct "80"s exist and must not be conflated: **(i)** the training-time
> *excitatory* `tau_nmda = 80` above — a stored runtime constant (runner L431 / Training.py
> L3913); and **(ii)** the task #54 forensic (§(d) item 2), which separately swapped the
> *interneuron* NMDA tau to 80 at inference (vs route-c's 25) to measure box width — a
> different pool and a different experiment.

---

## (b) Provenance & model configuration

**Source.** Pod = runai workspace `dev` (`MYGPU_WS=dev`), path
`/scratch/fsts_retrain_asym_20260530/`. Code from `code/`, checkpoints from
`checkpoints_asymdelay/`. (At the time this delivery was finalized the pod was offline; all
pod-sourced files had already been pulled byte-exact and md5-verified in a prior pass — see
§(g).) The TBW curves were copied from the local directory
`/home/vishnu/coding_proj/routec_tbw_plot/`; their `.npz`/`.png` md5s are byte-identical to
the pod-pulled curve outputs.

**Model configuration (verified by grep on the delivered `Training.py`):**
- **No `g_tonic`.** `grep -c g_tonic Training.py` = **0** (the content-free tonic-leak term
  is absent).
- **No AGC.** Automatic-gain-control loops were removed in task #185 (minimal-arch); only
  documentation comments and one retained-but-unused variable name (`self.pv_nmda`) remain.
- **Real disynaptic feed-forward inhibition.** A/V → interneuron (`*_inh`) → MSI_exc, with
  the interneuron→MSI_exc GABA edge `self.W_msiInh2Exc_GABA` (`nn.Parameter`, L1290).
- **Asymmetric, visual-leading conduction delays** (substeps × `dt = 0.1` ms; L1459–1485):

  | path | substeps | ms |
  |------|----------|----|
  | `conduction_delay_a2msi` (auditory → MSI) | 250 | **25.0** |
  | `conduction_delay_v2msi` (visual → MSI) | 400 | **40.0** |
  | `conduction_delay_a2msi_inh` | 270 | 27.0 |
  | `conduction_delay_v2msi_inh` | 420 | 42.0 |
  | `conduction_delay_msi_inh2exc` (disynaptic IPSC) | 450 | 45.0 |

  Visual arrives 15 ms after auditory, which widens the TBW on the visual-leading
  (negative-SOA) side.

**Naming.** `asymrc` = the route-c retrain (asymmetric delays **+** `tau_nmda_inh = 25`),
fresh ep0→80, seeds 42–46. `asymd` = the sibling retrain with the asymmetric delays but
**without** route-c (used as the #52 inference-precheck baseline). Only `asymrc`
checkpoints are bundled here.

**Checkpoints (`checkpoints/`, 1672126 bytes each), md5:**

| seed | file | md5 |
|------|------|-----|
| m0 | asymrc_m0_ep80.pt | `bd114bb20da14bf8bf7dcf8cdbeffefa` |
| m1 | asymrc_m1_ep80.pt | `6f1ad1d9960616cf1f56873e2e0be1d8` |
| m2 | asymrc_m2_ep80.pt | `047e2e50ea6fd925c4b734268d26cd24` |
| m3 | asymrc_m3_ep80.pt | `5b66792c932d48193567e299467cffc2` |
| m4 | asymrc_m4_ep80.pt | `c020299429f67da9d9ea33f0cdfea203` |

**Training (epoch-80 line; source: the `asymrc` training stdout log — not bundled in this
delivery, cited for provenance):** all 5 seeds completed 80 epochs; MSI rate 7.46–10.63 Hz;
`W_a2Inh` pinned at 0.0124–0.0125 / 0.0250 across seeds (FF→interneuron frozen-identity
held); no divergence.

---

## (c) Measured route-c P(fusion) TBW — flat-top box

Read directly from `curves/asymrc_tbw_ep80_AGGREGATE.csv` (5-seed across-nets aggregate,
20 ms SOA grid; SOA sign convention: **− = visual-leading**). Apparatus =
`run_fusion_across_models(fusion_method='temporal_fusion')` →
`is_temporally_fused` binary P(fusion).

The aggregate `mean_P_fusion` column:

```
SOA(ms): … -260   -240   -220   -200 … +180   +200   +220   +240 …
mean_P :  0.004  0.052  0.860  1.000 … 1.000  0.300  0.020  0.000
```

**The curve is a flat-top box, not a graded bell.** P(fusion) = 1.000 across the entire
interior and steps sharply to 0 at the edges:

- **On-grid P = 1.000 plateau:** SOA **[−200, +180] ms** (380 ms wide).
- **On-grid points with P ≥ 0.5:** SOA **[−220, +180] ms** (400 ms wide).
- **Interpolated half-maximum (P = 0.5) crossings:** left **−228.9 ms**
  (between −240 @ 0.052 and −220 @ 0.860), right **+194.3 ms** (between +180 @ 1.000 and
  +200 @ 0.300) → **full-width ≈ 423.2 ms**, centred at ≈ −17.3 ms (visual-leading shift,
  consistent with the 25/40 ms delays).

**Per-seed ep80 P = 1.000 plateau** (from `curves/asymrc_tbw_ep80_m{0..4}.npz`; SOA in ms):

| seed | P = 1.0 plateau | bounding edge values (off-plateau) |
|------|-----------------|------------------------------------|
| m0 | [−200, +180] | P(−220) = 0.60, P(+200) = 0.08 |
| m1 | [−200, +180] | P(−220) = 0.74, P(+200) = 0.10 |
| m2 | [−220, +180] | P(−220) = 1.00, P(+200) = 0.46 |
| m3 | [−220, +180] | P(−220) = 1.00, P(+200) = 0.50 |
| m4 | [−200, +180] | P(−220) = 0.96, P(+200) = 0.36 |

Per-seed robust-Gaussian fits over the same curve return `sigma` railed at the fit's upper
bound (200 ms) and `robust-FWHM = 471 ms`; the strictly-intermediate-P count is 2–3 of 41
SOA points per seed — i.e. the response is essentially binary (box), not graded.

**The width numbers in this lineage are different estimators / conditions and are not
directly comparable:**
- **471 ms** — per-seed robust-Gaussian FWHM on the retrained `asymrc` box; `sigma` is
  railed at the 200 ms fit upper bound, so this is an upper-rail artifact, not a measured σ.
- **≈ 423 ms** — the aggregate plateau half-maximum (P = 0.5) crossing width (above); a
  *different estimator* (plateau-crossing, not a Gaussian fit) on the across-nets curve.
- **396–419 ms** (σ 168–178 ms) — the task #51 inference-time proof: `tau_nmda_inh = 25`
  applied at inference on the pre-existing ep80 checkpoints, robust-Gaussian FWHM near the
  human ≈ 414 / 176. Same estimator as the 471 ms but a *different condition*
  (inference-patched existing checkpoints vs retrained-from-scratch).

For reference, the across-nets reads at earlier epochs: ep30 plateau SOA [−100, +50] ms
(`sigma` 113.7, FWHM 267.7); ep45 plateau SOA [−200, +150] ms (`sigma` railed 200, FWHM
471). The box widens from ep30 to ep45/80. (Files: `curves/asymrc_tbw_ep30.{npz,png}`,
`asymrc_tbw_ep45.{npz,png}`, `asymrc_tbw_ep80_m{0..4}.{npz,png}`, the aggregate
`.csv`/`.png`, and the per-run stdout logs `curve_ep30/45.log`, `curve_ep80_m{0..4}.log`.)

---

## (d) Proven causal findings (forensic; evidence as reported)

These are results from the forensic tasks #53/#54. The forensic output files are **not**
bundled in this delivery; they are recorded here with the evidence as reported.

1. **route-c training does not cause the box.** Swapping the *training* lineage
   (`asymd` ↔ `asymrc`) at a fixed inference tau leaves the flat-top box unchanged.
2. **Inference tau changes box *width* only.** Swapping the interneuron NMDA tau at
   inference between **25 and 80** changes the box width (≈ **471 ms** at tau 25 ↔
   ≈ **800 ms** at tau 80) but does not convert the box into a graded bell. route-c
   (tau 25) therefore narrows the box; it does not remove it.
3. **Root cause of the box = the binary fusion criterion.** `is_temporally_fused`
   (`code/TBW_test.py`, L878–879) scores a response as fused whenever it has **≤ 1 peak**.
   A single MSI response peak persists across the whole conduction-overlap SOA range, so the
   criterion reports P(fusion) = 1.0 throughout that range — manufacturing the flat top.
4. **Why a single peak.** The single peak arises from a **sustained disynaptic-GABA
   plateau** (the interneuron → MSI_exc inhibition). Zeroing the disynaptic GABA weight
   `W_msiInh2Exc_GABA` restores a proper **±~100 ms** temporal window in the response.

---

## (e) How to reproduce (from `code/`)

Environment (source pod `/scratch/calibenv`): Python 3.12, `torch 2.6.0+cu124`, `numpy`,
`scipy 1.17.1`, `matplotlib`; CUDA GPU required.

**Retrain the 5 `asymrc` checkpoints** (route-c is already in `Training.py`):
```bash
cd code
bash launch_asymrc.sh asymrc      # preflights Training.py md5 == 6b649d15…, then launches
                                  # 5 seeds (42..46), 80 epochs each, via:
#   python run_task192_disynaptic_ep40_retrain.py --model_idx {0..4} --prefix asymrc --n_epochs 80
```
(`launch_asymrc.sh` and the runner reference absolute pod paths under
`/scratch/fsts_retrain_asym_20260530/`; adjust `CODE`/`LOGS` in the launcher and the
runner's output paths for another host.)

**Regenerate a TBW fusion curve** (loads checkpoints with plasticity off; restores the asym
delays and `tau_nmda_inh = 25` from the constructor):
```bash
cd code
# 5-seed pooled, coarse 50 ms grid (as ep30/ep45):
python tbw_routec_curve.py --epoch 80 --ckpt_dir ../checkpoints --out_dir ../curves --ntrials 50
# single seed, fine 20 ms grid (as the per-seed ep80 here):
python tbw_routec_curve.py --epoch 80 --only_model 0 --ostep 2 --ckpt_dir ../checkpoints --out_dir ../curves
```
Each run prints pooled + per-model P(fus@{0,±200}), robust-FWHM / sigma / mu / R², the
`n_graded` (strictly-intermediate-P) count, and the full SOA:P curve; it writes
`<prefix>_tbw_ep{NN}[_m{K}].npz` + `.png`. The aggregate CSV/PNG
(`asymrc_tbw_ep80_AGGREGATE.*`) pools the five per-seed `.npz`.

---

## (f) md5 manifest (re-computed on the delivered files)

```
# code/
6b649d153850cfc3c872f4dcafd71b95  code/Training.py                              180322
a7c8a2827a44a642f3333a7f83d1ef60  code/TBW_test.py                               73760
905a2fe56e80e00aa104a93917175d54  code/run_task192_disynaptic_ep40_retrain.py    28759
b881a7fcc77b34fba3bdd802f953fa96  code/tbw_routec_curve.py                        5679
aa32b1d706b3e458e9c5d01cf97e8dc5  code/launch_asymrc.sh                           1186

# checkpoints/   (1672126 bytes each)
bd114bb20da14bf8bf7dcf8cdbeffefa  checkpoints/asymrc_m0_ep80.pt
6f1ad1d9960616cf1f56873e2e0be1d8  checkpoints/asymrc_m1_ep80.pt
047e2e50ea6fd925c4b734268d26cd24  checkpoints/asymrc_m2_ep80.pt
5b66792c932d48193567e299467cffc2  checkpoints/asymrc_m3_ep80.pt
c020299429f67da9d9ea33f0cdfea203  checkpoints/asymrc_m4_ep80.pt

# curves/
d14e04caa46ba0521b38bacea9656467  curves/asymrc_tbw_ep30.npz                      3110
6da1a7b50212629d3c4800432d5c355d  curves/asymrc_tbw_ep30.png                     44728
a45c8c5cbc92bfed84396443e6fd721b  curves/asymrc_tbw_ep45.npz                      3110
98135beb8cfad2f60f51ebcef7a83b6e  curves/asymrc_tbw_ep45.png                     40816
d7d116d1391254bccc5fd7ae7d3a6d01  curves/asymrc_tbw_ep80_AGGREGATE.csv            2069
3384ab29253f45a13589b4bd62e864d4  curves/asymrc_tbw_ep80_AGGREGATE.png          106705
fb8b80369cc5e76ed192feff0261b840  curves/asymrc_tbw_ep80_m0.npz                   3590
d5a7f204b976ea395758e46b49508b94  curves/asymrc_tbw_ep80_m0.png                  43580
8910958733d7f0ee67d6f9e587b7622b  curves/asymrc_tbw_ep80_m1.npz                   3590
60644d45c088ec8cee315fd615f0fc4e  curves/asymrc_tbw_ep80_m1.png                  43383
fa7ea2df7fe55a4d0bb63cba3a21dbbd  curves/asymrc_tbw_ep80_m2.npz                   3590
0dff3c6ccf0bc62c7e76df1251d378e1  curves/asymrc_tbw_ep80_m2.png                  43028
d3239beb13c97ab21bc3de781a24f785  curves/asymrc_tbw_ep80_m3.npz                   3590
96135d796121576d5e61b85225541967  curves/asymrc_tbw_ep80_m3.png                  42427
ade499fa0593ed555e03a09f6eecc210  curves/asymrc_tbw_ep80_m4.npz                   3590
862d4e874c523696306ce9913e79c877  curves/asymrc_tbw_ep80_m4.png                  43208
175312e26ec39b685f08354f96754b71  curves/curve_ep30.log                           1557
6e72e1b6f4aa9ffa32862a146c1c2b2d  curves/curve_ep45.log                           1557
be5ba946fcdfd16f25023d056ed0de15  curves/curve_ep80_m0.log                        1645
9fd7a837b9916e61b4fb579bb20c7ec7  curves/curve_ep80_m1.log                        1647
54e80b1991344fc964f876aff85120df  curves/curve_ep80_m2.log                        1647
baab625a666aa8946848dd7e15067381  curves/curve_ep80_m3.log                        1647
243c951b5a2ffe2d58cbd00f69bdab51  curves/curve_ep80_m4.log                        1647
```
**33 manifest entries — code 5, checkpoints 5, curves 23 (excludes DOCUMENTATION.md
itself).** The 7 `curve_*.log` files are the raw stdout of the **TBW-probe** curve runs (the
source of the per-seed fit numbers in §(c)) — they are **not** training logs; they are
included as evidence in addition to the curve files the brief listed.

---

## (g) Transfer integrity

Code + checkpoints were pulled from the pod via staged `tar` → `base64 -w0` →
`split -b 800000` (8 pieces) → per-piece file pull (stdout-redirected) → per-piece md5
verify (pod == local, all 8) → reassemble (6 000 716 b64 bytes) → `base64 -d` → tarball md5
`ef2720671b8ec4e4dbaded6c59ea7552` (pod == local) → untar → per-file md5 re-verified against
the manifest above. No direct scp was used. TBW curves were copied from the local
`routec_tbw_plot/` directory and md5-verified source == destination. Consolidated
2026-06-01.

### Cross-references (forensic outputs not bundled)
| task | finding |
|------|---------|
| #51 | route-c adopted into production `Training.py`; 5 ckpts retrained ep0→80 |
| #53 | flat-top box = binary `is_temporally_fused` (≤1 peak ⇒ fused) reading a single GABA-suppressed peak |
| #54 | training lineage does not cause the box; inference tau 25↔80 sets box width (≈471↔800 ms); zeroing `W_msiInh2Exc_GABA` restores a ±~100 ms window |
