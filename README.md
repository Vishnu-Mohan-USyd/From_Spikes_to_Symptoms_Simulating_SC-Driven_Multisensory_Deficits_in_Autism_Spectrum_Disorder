# FSTS Route-C — Audiovisual Multisensory Integration SNN (TBW + SBW)

A spiking neural network model of audiovisual multisensory integration in the superior
colliculus. **n = 180** Izhikevich-style neurons on a **180° spatial ring** receive auditory (A)
and visual (V) inputs through a disynaptic feed-forward-inhibition (**"route-C"**) circuit. The
network is trained to fuse A and V stimuli; we then measure two psychophysical signatures of
multisensory binding:

- **TBW (Temporal Binding Window)** — how the probability/strength of AV fusion depends on the
  A–V stimulus-onset asynchrony (SOA, in ms).
- **SBW (Spatial Binding Window)** — how AV cross-modal enhancement depends on the A–V spatial
  separation (in degrees).

### Headline results (5 seeds, checkpoint `dL3` epoch 79, validated)

| Metric | Result |
|---|---|
| **TBW** | graded bell, **FWHM 185.09 ± 0.70 ms**, GO 5/5, never a box (`n_box=0`) |
| **SBW** (AV cross-modal enhancement) | bell, **peak +272 % at 0°**, **HWHM 15.3°** |
| **E/I balance** | **0.92** (membrane ⟨I_M⟩/⟨I_M,gaba⟩), in-band 5/5 |
| Canonical MSI firing rate | 17.3 Hz |

The three findings reproduce exactly from the included checkpoints (see **§ 6 Validation**).

---

## 1. The model / architecture

**Neurons & space.** `MultiBatchAudVisMSINetworkTime`, `n_neurons = 180` over `space_size = 180`
(a 180° ring, 1°/neuron). Integration `dt = 0.1 ms`, `n_substeps = 100` (so **1 "frame" = 10 ms**),
`tau_m = 20 ms`, `v_thresh = 0.3`.

**Circuit ("route-C").** Two afferent streams (A, V) project to an MSI (multisensory) layer through
a disynaptic feed-forward-inhibition motif: each afferent excites MSI directly **and** drives a fast
inhibitory interneuron onto MSI. The defining "route-C" change is that the **inhibitory
interneuron's NMDA decay is decoupled to its own, faster time constant** `tau_nmda_inh = 21.6 ms`,
while excitatory NMDA keeps `tau_nmda = 80 ms` (`gNMDA = 1.30`, `Erev_nmda = 20`, Mg block
`mg_vhalf = -35`). GABA `g_GABA = 10`, `tau_gaba = 10 ms` (trained value, stored per-checkpoint).
Short-term depression `u = 0.7`, `tau_rec = 400 ms`; `input_scaling = 400`; recurrent `g_rec = 0.1`
(operating point for epoch > 25).

**Asymmetric conduction delays.** `A→MSI = 250 substeps (25 ms)`, `V→MSI = 400 substeps (40 ms)`.
Visual conduction is *slower*, so the visual stimulus must **lead** to arrive coincidentally — this
gives the visual-leading TBW.

**The TBW fix (graded window, not a box).** A **correlated, common-mode A–V latency jitter**
(`σ = 30 ms` ≡ `SIGMA_DL_FRAMES = 3`) is baked into **training**: each trial, both streams receive
the *same* random latency offset ~ N(0, 30 ms). This converts the fusion-vs-SOA curve from a flat-top
box into a **graded bell (FWHM 185 ms)**, matching biology. σ is fixed by design, never tuned to the
metric.

**Plasticity.** Inhibitory plasticity is ON during training (`set_inhib_plasticity(True)`) and frozen
at measurement (`plasticity_enabled = False`). Learning rates `lr_unimodal = lr_msi = 2e-2`,
`lr_readout = 8e-4`.

### Two numerically-equivalent model builds

- **TRAINING build** — `Training_graphdf_d52.py` (CUDA-graphed fast forward) + `panelgraph.py`.
  Used *only* to train the checkpoints.
- **MEASUREMENT build** — `Training_delayfix_d52.py` (faithful eager forward). **All metrics are
  measured on this build.** `val36_traj_d52.py` loads it as `Training` when `VAL36_BUILD=delayfix`
  (the default). The two builds are verified-equivalent forwards: *train fast, measure faithful.*

---

## 2. Repository layout

Flat: every Python module, both Training builds, the two frozen readouts, the checkpoints, and the
data all live in one directory. Scripts resolve their paths relative to their own location, so the
repo works from any clone path (no env vars required; set `CUDA_VISIBLE_DEVICES` to choose a GPU).

```
Training.py                     canonical (non-delayfix) build
Training_delayfix_d52.py        MEASUREMENT build — faithful eager forward (all metrics measured here)
Training_graphdf_d52.py         TRAINING build — CUDA-graphed fast forward
panelgraph.py, run34.py         training-build helpers (graph wiring + snapshot/restore utilities)

launch_retrain5.py              TRAIN: orchestrator — 5 seeds across available GPUs
retrain5_jitter.py              TRAIN: per-seed driver (σ=30 ms jitter + tau_gaba baked in)

val36_traj_d52.py               MEASURE: harness — build_net / load_ckpt + measure_tbw / sbw / ei
harden_85.py, grade_88_noise.py MEASURE: TBW helpers (frozen-readout wrappers)
measure_107_convergence.py      MEASURE: TBW driver (per seed → TBW + EI + MSI + legacy P-fusion)
ensemble_92.py                  MEASURE: TBW 5-seed aggregator → GO/NO-GO
measure_develop_check.py        MEASURE: env/import shim used by the TBW driver
diag_136b_cre_5seed.py          MEASURE: SBW stage 1 — capture spike profiles → out/*.npz
sbw_clean_curve_5seed.py        MEASURE: SBW stage 2 — analyse the npz → curve PNG + numbers

TBW_test.py                     FROZEN TBW readout (md5 80d33465…) — never edited
SBW_test.py                     FROZEN SBW readout (md5 73b7d136…) — never edited

checkpoint/                     the 5 trained checkpoints (ckpt_ep79_seed{42..46}_…_dL3.pt)
out/                            SBW inputs (diag_136*_profiles*.npz, diag_141_pedestal.json) +
                                where measurement outputs (JSON/PNG/logs) are written
fonts/                          Roboto/Helvetica TTFs used by the readouts' figure styling
```

---

## 3. Dependencies

Python 3.13, PyTorch 2.10 (CUDA 13.0 build), `numpy`, `scipy`, `matplotlib`, and one CUDA GPU
(any model; the network needs ~2 GB). No install script:

```
pip install torch numpy scipy matplotlib
```

---

## 4. How to TRAIN the checkpoints

The 5 trained `ep79` checkpoints are already in `checkpoint/`, so you can skip this and go straight
to measurement. To retrain from scratch, from the repo root:

```
python launch_retrain5.py
```

This trains seeds 42–46 to epoch 79 (batch 250) with the σ = 30 ms correlated jitter and
`tau_gaba = 10` baked in (`SIGMA_DL_FRAMES=3`, `TAU_GABA=10`), splitting the seeds across available
GPUs (~13 min/seed). It writes, into `checkpoint/`:

```
ckpt_ep{5,30,50,79}_seed{SEED}_bs250_delay52_tau10_dL3.pt
```

The `ep79` checkpoints are the ones measured below. A single seed directly:

```
SIGMA_DL_FRAMES=3 TAU_GABA=10 CUDA_VISIBLE_DEVICES=0 python retrain5_jitter.py 42
```

---

## 5. How to MEASURE / TEST

Every measurement loads the faithful `delayfix` build and the **frozen** readouts, whose md5s are
asserted before *and* after each run. All commands run from the repo root.

### TBW — temporal binding window (+ E/I + MSI)

One process per seed (`CK` = `checkpoint/ckpt_ep79_seed<SEED>_bs250_delay52_tau10_dL3.pt`):

```
CUDA_VISIBLE_DEVICES=0 python measure_107_convergence.py \
    --ckpt checkpoint/ckpt_ep79_seed42_bs250_delay52_tau10_dL3.pt \
    --seed 42 --sigma_dL 3.0 --tag ep79_seed42 --label off92_jit_seed42_ep79
```

Repeat for seeds 43–46 with labels `off92_jit_seed43_ep79`, `off92_jit_seed44_ep79`,
`jit_seed45_ep79`, `jit_seed46_ep79`. Each writes `out/measure_107_<label>.json` and prints a
per-seed scorecard (FWHM, max_step, EI, MSI). Then aggregate the 5 seeds:

```
python ensemble_92.py --mode official
```

→ `out/ensemble_92_official.json` and the summary line:

```
REVAL TBW FWHM=185.09±0.70 n_box=0 n_graded=5 verdict=GO
REVAL EI  sync=0.9228±0.0321 in_band=5/5
REVAL SBW-Pfusion=31.34 MSI=17.29
```

> The per-seed scorecard's `GB5_fwhm` sub-gate is a stricter single-seed band; the **official**
> GO/NO-GO is the `ensemble_92` aggregate above.

### SBW — spatial binding window (AV cross-modal enhancement)

The spike-profile `npz` files are already in `out/`, so stage 2 reproduces the curve directly:

```
python sbw_clean_curve_5seed.py
```

→ `out/SBW_curve_5seed_dL3_ep79.png` and:

```
peak@0° = 271.7 ± 6.4 %   HWHM = 15.3 ± 0.6 °
```

To regenerate the profiles from the checkpoints first (stage 1, one process per seed):

```
SEED=42 CUDA_VISIBLE_DEVICES=0 python diag_136b_cre_5seed.py   # repeat SEED=43..46
```

`diag_136b` writes `out/diag_136b_profiles_seed{N}.npz`; `sbw_clean_curve` reads all 5.

### E/I balance

Computed inside the TBW driver (the `EI sync` line): the membrane ⟨I_M⟩/⟨I_M,gaba⟩ ratio, balanced
≈ 0.92, in-band 5/5.

---

## 6. Results & validation

1. **TBW — graded temporal binding window.** Fusion-vs-SOA is a graded **bell, FWHM 185.09 ± 0.70 ms**,
   GO on 5/5 seeds, `n_box = 0` (never the flat-top box), with a slight visual-leading shift from the
   asymmetric delays. *Caveat:* 185 ms sits at the **low edge** of the targeted biological band
   [185, 270] ms — a narrow-edge GO, not band-center.

2. **SBW — spatial AV enhancement.** Cross-modal enhancement vs A–V separation is a **bell peaking
   +272 % at 0° separation, HWHM 15.3°**. *Important:* this is the AV **enhancement** window (per-neuron
   best-aligned multisensory enhancement, ME %), **not** the behavioral P(fusion) spatial window. The
   legacy P(fusion) spatial halfwidth the TBW driver still prints (~31°) is a different, superseded
   ruler from an earlier model lineage.

3. **E/I balance — 0.92.** Membrane excitation/inhibition ratio, in-band on all 5 seeds; readouts
   frozen.

### Re-running the § 5 commands reproduces the recorded reference exactly

| Metric | Reference | Re-validated |
|---|---|---|
| TBW FWHM | 185.09 ± 0.70 ms, GO, n_box=0/5 | **185.09 ± 0.70 ms, GO, n_box=0/5** ✓ |
| E/I sync | 0.9228 ± 0.0321, 5/5 | **0.9228 ± 0.0321, 5/5** ✓ |
| MSI | 17.29 Hz | **17.29 Hz** ✓ |
| SBW (legacy P-fusion) | 31.36° | 31.34° (within the readout's unseeded-RNG noise) |
| SBW enhancement | peak +272.1 ± 6.8 %, HWHM 15.3 ± 0.6° | **peak +271.7 ± 6.4 %, HWHM 15.3 ± 0.6°** ✓ |

**Frozen-readout firewall.** The two readouts are byte-frozen and md5-asserted before and after
every measurement — `TBW_test.py` md5 `80d33465c4bf55d6e85b5990acb92da7`, `SBW_test.py` md5
`73b7d13626964d851cc090818b728311`. They are never edited; no metric is ever changed to pass.

> **Note on lineage.** The current build is `dL3`-ep79: `tau_nmda_inh = 21.6`, the σ = 30 ms
> correlated-jitter TBW fix (graded 185 ms bell), and SBW reported as the AV cross-modal
> **enhancement** (HWHM 15.3°). It supersedes an earlier lineage whose TBW was a flat-top box and
> whose SBW was reported as the P(fusion) spatial window (~31°).

---

## 7. Provenance, fonts & citing

This is a self-contained `dL3`-ep79 TBW/SBW deliverable within the Mohan & Rideaux superior-colliculus
multisensory-integration project (*From Spikes to Symptoms: Simulating SC-Driven Multisensory Deficits
in Autism Spectrum Disorder*). Its model build and frozen readouts are a **distinct lineage** from the
manuscript's main branch (different `Training.py`, different `TBW_test.py` / `SBW_test.py` readouts, and
a different checkpoint naming scheme), so this branch ships its own complete, runnable set of files.

The `fonts/` TTFs are used only for figure styling and are provided under their own license
(`fonts/LICENSE.txt`). If you use this code in academic work, please cite the accompanying manuscript;
corresponding author **reuben.rideaux@sydney.edu.au**.
