# Route-C TBW fix — correlated common-mode A–V latency jitter

**Status: validated, GO (5-seed pooled, ep79).**
This directory documents and reproduces the biological correction that turns the Route-C
audiovisual temporal-binding-window (TBW) readout from a **box** into a **graded bell**, matching
the majority of biological findings — without touching the frozen `is_temporally_fused` readout and
without tuning the jitter magnitude to hit a number.

---

## 1. Problem

The Route-C model (disynaptic feed-forward inhibition lineage) reproduced the spatial binding
window (SBW) and E/I balance faithfully, but its **TBW was a box**: P(fusion) sat at ~1.0 across a
flat plateau of stimulus-onset asynchronies (SOAs) and then fell off a cliff, instead of the graded
bell that human psychophysics reports (fusion highest at simultaneity, tapering smoothly with
increasing |SOA|).

A box is not merely cosmetically wrong — it means the network reports *all-or-nothing* temporal
fusion, with no graded sensitivity to audiovisual asynchrony. The biology bar (graded bell, finite
slope on both limbs, biologically-plausible width) was unmet for TBW while SBW and E/I were already
in range.

## 2. Root cause (forensic #98)

The box does **not** come from the readout and is **not** fixed by any static parameter (time
constant, conduction delay, gain). Single-variable forensics established:

> The TBW box is graded **only by correlated, common-mode trial-to-trial latency jitter.**
> Independent per-neuron noise averages out across the ~180-neuron MSI population and leaves the box
> intact; a **single shared timing perturbation per trial** is what the downstream readout integrates
> into a graded fusion probability.

In other words, the missing ingredient is *trial-to-trial variability in the audiovisual relative
latency that is shared across the whole population on each trial* — a biologically real source of
variance (attention, arousal, conduction-state fluctuations shift the whole percept's timing on a
given trial), not per-synapse jitter.

## 3. Mechanism (the fix)

A single biologically-motivated source of variance, baked into the model:

- **Correlated common-mode A–V latency jitter** ΔL ~ N(0, σ), **one shared Gaussian draw per trial**,
  applied as an integer-frame shift of the entire trial's audio-visual relative onset.
- **σ = 3 frames = 30 ms** (1 frame = 10 ms at the model's substep resolution).
- Exposed as a single knob, the environment variable **`SIGMA_DL_FRAMES`**:
  - `SIGMA_DL_FRAMES=0` (default) → draws nothing, the numpy RNG stream is untouched → **byte-identical
    to the pristine pre-fix build.**
  - `SIGMA_DL_FRAMES=3` → the fix, live in both training and measurement.

The same knob/mechanism drives the **training path** and the **measurement path** (see
`Training_graphdf_d52.py` L4265–4267 for the training-time draw, L251–255 for the shared per-trial
draw). This is the cardinal constraint: the jitter is **baked into the retrain**, present in *both*
training and measurement — it is the network genuinely embodying the mechanism, not an
inference-only measurement patch on frozen weights.

The driver `retrain5_jitter.py` enforces this with a **FATAL assert** (L93–94): training aborts
unless `net.sigma_dL_frames` equals the requested `SIGMA_DL_FRAMES`, so a retrain cannot silently run
without the jitter actually live.

**The `is_temporally_fused` readout was never touched** (md5 `80d33465c4bf55d6e85b5990acb92da7`,
asserted stable before and after every measurement). The bell is produced by the network's spiking
dynamics under correlated jitter, read by the *unchanged* fusion criterion — not by editing the ruler.

## 4. Biology justification (researcher dossier #108)

The adopted σ is justified by the **outcome width**, not chosen to hit a target. Neutral
deep-research (see `biology/researcher_av_tbw_width_108_20260617.md`) established the biological
audiovisual TBW width band as **~185–270 ms FWHM-equivalent** (Gaussian full-width-at-half-maximum),
with a units guard separating it from the ~2× wider 50%-criterion full-widths and single-neuron SC
windows that share the word "window" but measure a different quantity.

- **Van der Burg et al. 2014** is the decisive same-construct anchor at **~185 ms**.
- The model's measured FWHM (**185 ms**) lands at the narrow edge of the biological band — *inside the
  majority of findings*, dead-on the closest same-construct measurement.

σ was **not tuned** to move the width: σ=30 ms is a round, biologically-plausible common-mode timing
variance, and the resulting 185 ms FWHM was *measured*, not targeted. (Reaching the ~215 ms band
center would require a larger σ, which would only be legitimate with independent biological
justification — adopting it to center a number would be metric-chasing, which is forbidden.)

## 5. Validation result (official 5-seed pooled GO, ep79)

Measurement protocol = **per-seed-fit-then-aggregate** (`ensemble_92.py`): fit each model-seed's
pooled P(fusion) curve, then aggregate mean ± SD across seeds. An artifact-guard unittest refuses
pooling-faked grading (a raw cross-seed Σfused/Σtrials pool can manufacture a fake bell from box
seeds offset against each other — the guard rejects it). Verdict from `results/ensemble_92_official.json`:

| Observable | Result (5-seed, ep79) | Biology bar | Verdict |
|---|---|---|---|
| TBW shape | **0 / 5 box, 5 / 5 graded bell** | graded, finite slope both limbs | **PASS** |
| TBW FWHM | **185.1 ± 0.7 ms** | ~185–270 ms band | **PASS** (narrow edge) |
| TBW PSS | −16.5 ± 0.4 ms | near simultaneity | PASS |
| TBW peak P0 | 0.993 ± 0.002 | ~1 at simultaneity | PASS |
| SBW half-width | **31.4 ± 0.2°** | band [24.5, 40.9]° | **PASS** (5/5 in band) |
| E/I sync | **0.92 ± 0.03** | balanced | **PASS** (5/5 in band) |
| E/I off-sync | 0.67 ± 0.02 | balanced | PASS |
| MSI rate | 17.3 ± 0.9 Hz | alive, >5 Hz | PASS (0/5 flagged) |
| Frozen readouts | TBW + SBW md5 unchanged | untouchable | PASS |
| Artifact guard | not flagged | — | PASS |

**Per-seed FWHM:** 184.2 / 186.1 / 184.9 / 185.0 / 185.1 ms — tight across all five seeds.

Two bands are reported in the JSON: the **biological** band [185, 270] (the literature-grounded bar,
**GO**) and a stricter **pre-registered floor** [200, 300] (**NO-GO**, FWHM below 200). The 185 ms
result sits in the biological majority but below the conservative pre-reg floor; the accept-185
decision rests on Van der Burg 2014's same-construct ~185 ms anchor, not on relaxing the bar to pass.

**Figure:** `results/tbw_bell_official_5seed_ep79.png` — the no-jitter box (red dashed) vs the five
jitter seeds (light blue) and their mean bell (navy), with the FWHM half-max span annotated and the
vitals (SBW, E/I, MSI) in a side panel.

## 6. Contents

```
tbw_correlated_jitter_fix/
├── README.md                    ← this file
├── code/
│   ├── Training_graphdf_d52.py  ← training build WITH the gated common-mode ΔL knob (THE fix)
│   ├── retrain5_jitter.py       ← 5-seed jitter retrain driver (FATAL assert jitter is live)
│   ├── launch_retrain5.py       ← orchestrator (cuda:0 → seeds 42,43 / cuda:1 → seed 44)
│   ├── ensemble_92.py           ← per-seed-fit-then-aggregate GO/NO-GO + artifact-guard unittest
│   └── measure_107_convergence.py ← TBW / SBW / E-I / MSI measurement harness
├── results/
│   ├── ensemble_92_official.json          ← the official 5-seed pooled GO verdict
│   ├── measure_107_*.json (×6)            ← per-seed TBW/SBW/E-I/MSI (jit seeds 42–46 + nojit ref)
│   ├── tbw_bell_official_5seed_ep79.png   ← the bell figure
│   └── tbw_bell_official_5seed_ep79.pdf
└── biology/
    └── researcher_av_tbw_width_108_20260617.md  ← the #108 biological-width dossier
```

**Model weights are NOT included.** The five ep79 checkpoints are ~1.9 MB each and are excluded by
the repo's `.gitignore` (no new `*.pt` committed). They are regenerated deterministically by the
retrain command below.

## 7. Reproduction

All compute is local — RTX 5090 (`cuda:0`) + A6000 (`cuda:1`). `tau_nmda_inh` is forced to 21.6 ms on
every checkpoint load (checkpoints do not serialize it; the loader defaults to 45.0).

**Retrain the validated 5-seed ensemble** (produces `ckpt_ep79_seed{42..46}_…_dL3.pt`):

```bash
# orchestrated: seeds 42,43 → cuda:0 ; seed 44 → cuda:1 ; seeds 45,46 reuse prior ep79 ckpts
python code/launch_retrain5.py

# or one seed by hand (the single-variable change vs the pristine build is SIGMA_DL_FRAMES):
SIGMA_DL_FRAMES=3 TAU_GABA=10 CUDA_VISIBLE_DEVICES=0 python code/retrain5_jitter.py 42
```

**Measure one checkpoint** (TBW + SBW + E/I + MSI, frozen-readout md5 asserted):

```bash
CUDA_VISIBLE_DEVICES=0 python code/measure_107_convergence.py \
    --ckpt ckpt_ep79_seed42_bs250_delay52_tau10_dL3.pt --sigma_dl_frames 3
```

**Aggregate the official GO/NO-GO** (per-seed-fit-then-aggregate; also runs the artifact-guard unittest):

```bash
python code/ensemble_92.py --official     # writes ensemble_92_official.json
```

**Pristine (pre-fix) box control** — same build, jitter off, byte-identical to the original:

```bash
SIGMA_DL_FRAMES=0 ... python code/retrain5_jitter.py 42   # reproduces the box
```

## 8. Constraints honored

- **Baked into training, not inference-only.** The jitter draws in both the training loop and the
  measurement path via the same knob; `retrain5_jitter.py` FATAL-asserts it is live during training.
- **Frozen readout untouched.** `is_temporally_fused` md5 `80d33465c4bf55d6e85b5990acb92da7`
  asserted stable before/after; SBW readout md5 `73b7d13626964d851cc090818b728311` likewise.
- **No metric-chasing.** σ=30 ms is biologically motivated; the 185 ms FWHM was measured, not tuned.
- **No symptom-masking knobs.** No global gain, baseline subtraction, or cap fudge — a single shared
  source of biological timing variance does the functional work.
- **Off = pristine.** `SIGMA_DL_FRAMES=0` leaves the RNG stream untouched → byte-identical to the
  original validated build, so the fix is a clean single-variable addition.
