# Research and validation log

## Frozen anchor

This closeout applies to **Mohan & Rideaux, _From Spikes to Symptoms: Simulating
SC-Driven Multisensory Deficits in Autism Spectrum Disorder_**, revision 2:
`manuscript_rev_2.pdf` (57 pages; document timestamp 2026-05-07 23:42:20 AEST;
SHA-256
`9d9f156e802462bac0c238c47f59749302204d8b5bc1b3c3424d4c040d6378eb`).
The repository anchor is commit
`ecadfd4d472cead684c2d49d18657df4c1b6525b`. The standalone assay validation
ran on 2026-07-26; the final independent regression completed on 2026-07-27.
This document and the versioned
[validation protocol](validation_protocol.md) are the durable release record.
Full command transcripts were captured during that session under
`/tmp/fsts10-final-validation.SfSiSn/logs/`; that path is session-local and is
not an authoritative or durable dependency of the repository.

## Acceptance rule and evaluation lifecycle

The primary gate is reproduction of the named **biological phenomenon**, not
exact equality to a manuscript number. A stronger valid result is accepted—for
example, the final cue-weighting agreement (`R² = .969`, `MAE = .063`) is retained
rather than weakened to match an earlier value. TBW and SBW perturbations are
accepted by their prespecified signed direction (or stability, where specified);
their absolute widths are operational model outputs, not estimates of human
clinical effect size.

All assays use one ten-checkpoint ensemble,
`checkpoint/msi_model_surr_10_00.pt` through
`checkpoint/msi_model_surr_10_09.pt`. The lifecycle is
**fresh load → apply assay-local configuration → run → discard**. Configurations
are never saved back to a checkpoint. The control calibration is `gNMDA = 1.30`;
stationary estimators additionally disable plasticity and freeze feed-forward
inhibitory adaptation where required. The TBW/SBW perturbation workflow applies
only the named in-memory condition. Before/after hashes were identical for all 12
checkpoint files covered by the integrity gate.

## Durable reproduction record

The final run selected physical GPU 1, which exposed an NVIDIA RTX A6000 as
logical CUDA device 0, and used a headless plotting backend. From the repository
root, the exact scientific commands were:

```bash
env CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
env CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 python EI_balance_test.py
env CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 python fano_factor_test.py
env CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 python precision_hist_test.py
env CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 python cue_reliability_test.py
env CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 python inverse_effectiveness_test.py
env CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 python response_latency_test.py
env CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 python generate_all_fresh.py
```

All commands exited 0. Their exact accepted results are recorded in the matrix
below. The final independent A6000 suite reported
`7 passed in 268.02s (0:04:28)`; the
standalone wall times were 1m01s (E/I), 3m40s (Fano), 19m16s (localization),
35s (cue reliability), 15m43s (inverse effectiveness), 32s (latency), and
26m56s (TBW/SBW). These commands create or overwrite generated outputs; use an
isolated copy when artifact hashes must remain clean.

The frozen run invoked the `EI_balance_test.py` compatibility entry point. Its
command-line route delegated to the canonical `run_ei_balance.main`, so the
measured implementation was canonical. New users should run
`python run_ei_balance.py` directly.

Checkpoint integrity was evaluated by computing SHA-256 values for
`checkpoint/*.pt` before the first command and after the last command, then
comparing the two manifests. All 12 entries matched. The primary scientific
ensemble is the ten `msi_model_surr_10_*.pt` files; the two additional
checkpoint files were included only in the repository-wide immutability check.

## Mechanisms, findings, and assumptions

- **E/I balance.** Separate AMPA+NMDA excitatory and GABAergic inhibitory current
  magnitudes are pooled over the 125 ms evoked window. Their order-one ratio
  supports dynamically balanced drive; it does not establish cell-type-specific
  balance in vivo.
- **Localization and cue weighting.** Auditory, visual, and audiovisual spatial
  uncertainty is summarized by pooled localization-error sigma. Reliability-based
  weighting is compared with the maximum-likelihood inverse-variance prediction.
  The narrower AV distribution and strong cue-weight agreement support precision
  enhancement and reliability weighting; model degrees and sensitivity are
  operational quantities.
- **Inverse effectiveness.** Multisensory enhancement index (MEI) is evaluated
  from weak to strong cues under stationary fresh-load evaluation. The accepted
  phenomenon is the strong low-to-high decline, not strict pointwise monotonicity:
  the first two ensemble points show a small rise before the sustained decline.
- **Latency.** Latency is the first threshold-crossing time. AV is 7.5 ms earlier
  than the mean of A and V but ties the fastest (auditory) route within time-step
  resolution. This supports the manuscript-defined mean-unimodal benefit; it does
  **not** support a claim that AV is faster than both unimodal routes.
- **Fano factor.** The primary measure uses physical 10 ms frames (100 simulation
  substeps) and active neurons in a fixed stimulus-centred ±2σ ROI; zero-mean bins
  are undefined and omitted. The conventional all-neuron ROI is secondary. The
  ensemble shows stimulus-onset variability quenching; a rate rise is an ensemble
  result (8/10 checkpoints), not a per-checkpoint invariant.
- **Temporal and spatial binding windows.** TBW is the half-width of the temporal
  fusion curve; SBW is the half-width of the spatial enhancement curve at the
  canonical threshold. Both are operational classifiers. Coincidence-dependent
  integration and the signed perturbation responses are the accepted mechanisms;
  exact magnitudes and cross-species equivalence are not assumed.
- **ASD-related perturbations.** Conditions are assay-local engineering probes:
  reduced feed-forward inhibition (`pv_nmda = 0.8`, `targ_ratio = 0.8`), reduced
  adaptation (`aM = 0.001`, `bM = 0.2`, `cM = -60`, `dM = 0.01`), reduced NMDA
  (`gNMDA = 0.50`), and increased NMDA (`gNMDA = 5.00`). They support **internal
  mechanistic consistency**, not a unique ASD etiology, patient-level causality,
  or the exclusion of other molecular and circuit mechanisms.

## Final observed matrix

| Gate | Final observation | Acceptance | Durable reproducer |
|---|---|---|---|
| Test and device gate | 7 tests passed in 268.02 s (0:04:28); physical GPU1 exposed as logical CUDA 0, NVIDIA RTX A6000 | Pass | `python -m pytest -q tests` with the exact environment above |
| E/I | E `3.8158 ± .0526`, I `3.6328 ± .0399`; E/I `1.050 ± .004` | Pass: order-one balance | Frozen compatibility route `python EI_balance_test.py`; canonical route `python run_ei_balance.py` |
| Localization | σ A/V/AV `8.99°/10.28°/8.26°`; positive AV sensitivity advantage `+8.8–9.5%` across accepted summaries | Pass: AV sigma below both unimodal sigmas and sensitivity improvement over best unimodal ≥5% | `python precision_hist_test.py` |
| Cue reliability | MLE deviation `17.7%`; MAE `.063`; RMSE `.079`; R² `.969` | Strong pass; stronger valid result retained | `python cue_reliability_test.py` |
| Inverse effectiveness | Ensemble MEI `.914724 → .030694` from intensity `.05 → 1.6`; low–high margin `.884030` | Pass: strong inverse trend | `python inverse_effectiveness_test.py` |
| Latency | A/V/AV `26.7/41.7/26.7 ms`; benefit vs mean `+7.5 ms`; benefit vs fastest `0.0 ms` | Pass under mean-unimodal definition | `python response_latency_test.py` |
| Fano | Primary active-ROI FF `1.123541 → .783592` in the first 100 ms; quenching 10/10; mean rate `.095922 → .121753`, rising 8/10 | Pass: variability quenching | `python fano_factor_test.py` and `python -m pytest -q tests` |
| TBW | Control `110 ms`; reduced FF inhibition `158` (`+48`); reduced adaptation `229` (`+119`); reduced NMDA `94` (`−16`); increased NMDA `112` (`+2`, stable) | Pass: 4/4 directional/equivalence predictions | `python generate_all_fresh.py` |
| SBW | Control `26.7°`; reduced FF inhibition `29.0` (`+2.3`); reduced adaptation `30.2` (`+3.5`); reduced NMDA `13.2` (`−13.5`); increased NMDA `30.6` (`+3.9`) | Pass: 4/4 directional predictions | `python generate_all_fresh.py` |
| Integrity | 12/12 checkpoint SHA-256 values identical before and after; scoped source hashes unchanged | Pass: read-only evaluation | Before/after `sha256sum checkpoint/*.pt` manifests and `cmp` |

## Primary literature

### E/I balance

- Wehr & Zador (2003), “Balanced inhibition underlies tuning and sharpens spike
  timing in auditory cortex.” [DOI: 10.1038/nature02116](https://doi.org/10.1038/nature02116)
- Haider et al. (2006), “Neocortical network activity in vivo is generated through
  a dynamic balance of excitation and inhibition.” [PMID: 16641233](https://pubmed.ncbi.nlm.nih.gov/16641233/)

### Localization and reliability-based cue weighting

- Battaglia et al. (2003), “Bayesian integration of visual and auditory signals for
  spatial localization.” [PMID: 12868643](https://pubmed.ncbi.nlm.nih.gov/12868643/)
- Alais & Burr (2004), “The ventriloquist effect results from near-optimal bimodal
  integration.” [PMID: 14761661](https://pubmed.ncbi.nlm.nih.gov/14761661/)
- Fetsch et al. (2011), “Neural correlates of reliability-based cue weighting during
  multisensory integration.” [PMID: 22101645](https://pubmed.ncbi.nlm.nih.gov/22101645/)

### Inverse effectiveness

- Meredith & Stein (1983), “Interactions among converging sensory inputs in the
  superior colliculus.” [PMID: 6867718](https://pubmed.ncbi.nlm.nih.gov/6867718/)
- Stanford et al. (2005), “Evaluating the operations underlying multisensory
  integration in the cat superior colliculus.”
  [DOI: 10.1523/JNEUROSCI.5095-04.2005](https://doi.org/10.1523/JNEUROSCI.5095-04.2005)

### Response latency

- Rowland et al. (2007), “Multisensory integration shortens physiological response
  latencies.” [PMID: 17537958](https://pubmed.ncbi.nlm.nih.gov/17537958/)
- Diederich & Colonius (2004), “Bimodal and trimodal multisensory enhancement:
  effects of stimulus onset and intensity on reaction time.”
  [PMID: 15813202](https://pubmed.ncbi.nlm.nih.gov/15813202/)

### Fano-factor dynamics

- Churchland et al. (2010), “Stimulus onset quenches neural variability: a
  widespread cortical phenomenon.” [PMID: 20173745](https://pubmed.ncbi.nlm.nih.gov/20173745/)
- Ponce-Alvarez et al. (2013), “Stimulus-dependent variability and noise
  correlations in cortical MT neurons.” [PMID: 23878209](https://pubmed.ncbi.nlm.nih.gov/23878209/)

### Temporal binding window

- Meredith et al. (1987), “Determinants of multisensory integration in superior
  colliculus neurons. I. Temporal factors.” [PMID: 3668625](https://pubmed.ncbi.nlm.nih.gov/3668625/)
- Powers et al. (2009), “Perceptual training narrows the temporal window of
  multisensory binding.”
  [DOI: 10.1523/JNEUROSCI.3501-09.2009](https://doi.org/10.1523/JNEUROSCI.3501-09.2009)

### Spatial binding window

- Meredith & Stein (1986), “Spatial factors determine the activity of multisensory
  neurons in cat superior colliculus.” [PMID: 3947999](https://pubmed.ncbi.nlm.nih.gov/3947999/)
- Meredith & Stein (1986), “Visual, auditory, and somatosensory convergence on
  cells in superior colliculus results in multisensory integration.”
  [PMID: 3537225](https://pubmed.ncbi.nlm.nih.gov/3537225/)

### ASD-relevant temporal processing and inhibitory evidence

- Foss-Feig et al. (2010), “An extended multisensory temporal binding window in
  autism spectrum disorders.” [PMID: 20390256](https://pubmed.ncbi.nlm.nih.gov/20390256/)
- Stevenson et al. (2014), “Multisensory temporal integration in autism spectrum
  disorders.” [PMID: 24431427](https://pubmed.ncbi.nlm.nih.gov/24431427/)
- Robertson et al. (2016), “Reduced GABAergic action in the autistic brain.”
  [PMID: 26711497](https://pubmed.ncbi.nlm.nih.gov/26711497/)
- Puts et al. (2017), “Reduced GABA and altered somatosensory function in children
  with autism spectrum disorder.” [DOI: 10.1002/aur.1691](https://doi.org/10.1002/aur.1691)

## Inhibitory-input STP correction and step-size validation (2026-07-27)

### Proven defect and mechanistic mapping

The A/V→MSI-inhibitory AMPA path previously stored resource variables with
postsynaptic shape `(B, n_inh)`, depleted every entry by the population spike
count, and multiplied the projected current by that postsynaptic resource. A
fail-to-pass production-path regression showed that synchronous presynaptic
spikes could therefore consume more than the available resource, producing
negative resources and non-finite state. This is inconsistent with
presynaptic, terminal-local short-term depression.

The corrected path is the depression-only Tsodyks–Markram specialization used
elsewhere in the model. For batch index `b` and presynaptic terminal `j`, each
substep computes

```text
R^-_bj = R_bj + (1 - R_bj) Δt / τ_rec
q_bj   = u_bj R^-_bj s_bj
R^+_bj = R^-_bj - q_bj
I_bi   = Σ_j W_ij q_bj.
```

Here `R`, `u`, binary delayed spike `s`, and released fraction `q` all have
shape `(B, n)`; `W` has shape `(n_inh, n)`; and `I` has shape `(B, n_inh)`.
`R`, `u`, and `q` are dimensionless, while `Δt` and `τ_rec` are in ms. Thus
release is computed at each presynaptic terminal before `F.linear(q, W)`.

The minimal change only moved the four inhibitory-input `R/u` tensors to
presynaptic shape at initialization/reset and reordered AMPA release/depletion
before projection. It did not add clipping, change checkpoints or parameters,
or alter the parallel NMDA path.

Assumptions are that delayed spikes are binary per substep, production uses the
initialized `u=.2` without a facilitation update, Euler recovery is valid at
the validated step sizes, and each paired condition starts from the same fresh
checkpoint with matched stimuli and seed. This is a depression-only engineering
realization of the Tsodyks–Markram family, not a claim that facilitation is
dynamically fit here.

Primary mechanistic sources are Tsodyks & Markram (1997), “The neural code
between neocortical pyramidal neurons depends on neurotransmitter release
probability” [DOI: 10.1073/pnas.94.2.719](https://doi.org/10.1073/pnas.94.2.719),
and Tsodyks, Pawelzik & Markram (1998), “Neural networks with dynamic synapses”
[DOI: 10.1162/089976698300017502](https://doi.org/10.1162/089976698300017502).

### Corrected validation result

The paired control audit used 10 checkpoints, 51 SOAs, 50 trials per SOA, and
seeds `12345 + model_index`. Physical frames remained 10 ms (`dt=.1 ms` × 100
substeps; `dt=.05 ms` × 200 substeps); the same configured physical delays were
realized within half a substep.

| Paired metric | Corrected result |
|---|---:|
| Mean per-checkpoint full width, `dt=.1/.05 ms` | `219.404083 / 221.688583 ms` |
| Mean paired full-width drift | `+2.284500 ms` |
| Mean paired half-width drift | `+1.142250 ms` |
| 90% CI, paired full-width drift | `[1.482331, 3.086670] ms` |
| Worst per-checkpoint full-width drift | `+5.487398 ms` |
| Pooled-curve Pearson correlation | `.999712` |
| Pooled-curve NRMSE | `.010152` |

All 1,200/1,200 paired frames had finite audited dynamic state; all four STP
resources remained in `[0,1]` at both step sizes. All 20/20 fits had valid
brackets, two crossings, finite parameters/covariance, and no fallback. The
before/after SHA-256 digests of registered `state_dict` parameter/buffer content
were identical for 20/20 conditions. In the five-condition
canonical `dt=.1 ms` regression, 600/600 audited frames per condition passed:

| Condition | Pooled full width | Reported half-width |
|---|---:|---:|
| Control | `219.762088 ms` | `109.881044 ms` |
| Reduced feed-forward inhibition | `314.283169 ms` | `157.141585 ms` |
| Reduced adaptation | `456.880614 ms` | `228.440307 ms` |
| Reduced NMDA | `190.388656 ms` | `95.194328 ms` |
| Increased NMDA | `223.093334 ms` | `111.546667 ms` |

Width terminology matters. Supplementary Figure 1's historical labels
`215.8 ms` at `dt=.1 ms` and `256.5 ms` at `dt=.05 ms`—a `+40.7 ms` drift—are
left-to-right **full widths**, despite older notes calling them “HW.” Current
release tables report **half-width = full width / 2**. The corrected paired
result above is a new validation result; its `+2.284500 ms` mean full-width
drift (`+1.142250 ms` in half-width units) must not be presented as a
manuscript threshold or substituted into the historical figure.
