# Biological validation protocol

- **Model:** SC-driven audiovisual spiking neural network
- **Protocol version:** 2.1
- **Frozen validation date:** 2026-07-27
- **Manuscript:** Mohan & Rideaux, _From Spikes to Symptoms: Simulating
  SC-Driven Multisensory Deficits in Autism Spectrum Disorder_, revision 2
- **Manuscript SHA-256:**
  `9d9f156e802462bac0c238c47f59749302204d8b5bc1b3c3424d4c040d6378eb`

## 1. Purpose and acceptance hierarchy

This protocol defines the current public commands, measurement contracts, and
validation criteria for the supplied pretrained ensemble. It supersedes historical
criteria in `CHANGES.md` and diagnostic notes.

The acceptance hierarchy is:

1. Reproduce the named biological phenomenon with the specified estimator.
2. Preserve checkpoints and registered parameter/buffer (`state_dict`) content;
   allow dynamic state to evolve while auditing its finiteness and bounds.
3. Apply the quantitative criterion below.
4. Compare with manuscript values as context, not as a requirement to degrade a
   stronger valid result.

Inverse effectiveness is a low-versus-high intensity phenomenon; strict
pointwise monotonicity across every intermediate intensity is **not** required.
TBW/SBW perturbations are evaluated by their prespecified direction or
equivalence. Their absolute model widths are not treated as human clinical
effect sizes.

## 2. Execution and state contract

Run commands from the repository root. Primary validation uses exactly:

```text
checkpoint/msi_model_surr_10_00.pt
…
checkpoint/msi_model_surr_10_09.pt
```

Each independent condition follows:

```text
fresh checkpoint load → assay-local configuration → measurement → discard
```

No assay may save modified model state back to a checkpoint. The calibrated
control evaluation sets `gNMDA = 1.30` in memory. Stationary Fano, inverse, and
latency estimators additionally disable plasticity and freeze feed-forward
inhibitory adaptation. The canonical TBW/SBW pipeline retains its within-run
network dynamics but reloads checkpoints between conditions.

The four perturbation configurations are:

| Condition | Assay-local configuration |
|---|---|
| Reduced feed-forward inhibition | `gNMDA = 1.30`, `pv_nmda = 0.8`, `targ_ratio = 0.8` |
| Reduced adaptation | `gNMDA = 1.30`, `aM = 0.001`, `bM = 0.2`, `cM = -60`, `dM = 0.01` |
| Reduced NMDA | `gNMDA = 0.50` |
| Increased NMDA | `gNMDA = 5.00` |

These are internal mechanistic probes. Passing their directional gates does
not identify a unique ASD etiology.

## 3. Canonical commands

With a selected CUDA device and a headless plotting backend where needed:

```bash
python run_ei_balance.py
python fano_factor_test.py
python precision_hist_test.py
python cue_reliability_test.py
python inverse_effectiveness_test.py
python response_latency_test.py
python generate_all_fresh.py
```

These commands write figures and, for the full TBW/SBW pipeline, caches. Run an
isolated copy if generated-file hashes must remain unchanged.

## 4. Assay contracts and gates

### 4.1 E/I balance

**Property:** AMPA+NMDA excitation and feed-forward+recurrent+lateral
inhibition have comparable magnitudes during the evoked response.

**Test:** `run_ei_balance.py`

**Public API:**

```python
from run_ei_balance import run_ei_evoked

summary = run_ei_evoked(model_paths, device="cuda")
ratio = summary["ei_ratio_mean"]
```

The probe uses the first 125 ms (50 ms input plus 75 ms tail) of separated
substep current traces. Mandatory gates are `0.5 <= ratio <= 2.0` and
`abs(ratio - 1.0) < 0.15`, with no checkpoint or parameter mutation.

**Final ensemble:** E `3.8158 ± .0526`, I `3.6328 ± .0399`, E/I
`1.050 ± .004`.

### 4.2 Localization precision and multisensory enhancement

**Property:** The audiovisual localization distribution is more precise than
either unimodal distribution, and AV sensitivity exceeds the best unimodal
sensitivity.

**Command:** `python precision_hist_test.py`

The primary precision statistic is the standard deviation of pooled signed
localization error. Sensitivity is `1 / sigma`; its enhancement is measured
relative to the better unimodal sensitivity. Both mandatory gates must hold:
AV sigma is below both unimodal sigmas **and** AV sensitivity improves on the
best unimodal sensitivity by at least 5%.

**Final ensemble:** σ A/V/AV `8.99°/10.28°/8.26°`; accepted positive AV
sensitivity advantage `+8.8–9.5%` across the pooled audit and direct script
summary.

### 4.3 Reliability-based cue weighting

**Property:** Empirical audiovisual weights track the inverse-variance
maximum-likelihood prediction as cue reliability changes.

**Command:** `python cue_reliability_test.py`

Mandatory gates are `R² >= .70` and `MAE < .15`; `RMSE < .15` is an additional
quality check. A stronger valid agreement must not be weakened merely to match
an earlier report.

**Final ensemble:** `R² = .969`, `MAE = .063`, `RMSE = .079`, mean relative
deviation `17.7%`.

### 4.4 Inverse effectiveness

**Property:** Proportional multisensory enhancement is substantially larger for
weak cues than for strong cues.

**Command:** `python inverse_effectiveness_test.py`

**Evaluation configuration:**

```python
from inverse_effectiveness_test import configure_inverse_eval

configure_inverse_eval(net)
```

The mandatory gate is `MEI(intensity=.05) > MEI(intensity=1.6)`. The focused
regression additionally requires a low-to-high margin greater than `.75` on
M00. Intermediate points may show local variation; neither strict pointwise
monotonicity nor a Spearman threshold is a validation requirement.

**Final ensemble:** approximately `.914724 → .030694`, low-to-high margin
`.884030`. The first two ensemble points rise modestly before the sustained
decline, which remains consistent with this gate.

### 4.5 Response latency

**Property:** AV reaches the response threshold before the mean unimodal
latency.

**Command:** `python response_latency_test.py`

**Public API:**

```python
from response_latency_test import run_latency_test

latencies = run_latency_test()
```

Each A, V, and AV measurement starts from a fresh, stationary checkpoint. The
mandatory mean-unimodal benefit is positive and in the validated 5–10 ms range.
The fastest-unimodal comparison is reported separately and must remain within
one time step of zero in the control ensemble. There is no latency perturbation
in this protocol and no claim that AV is faster than both unimodal
routes.

**Final ensemble:** A/V/AV `26.7/41.7/26.7 ms`; mean-unimodal benefit
`+7.5 ms`; fastest-unimodal benefit `0.0 ms`.

### 4.6 Fano-factor dynamics

**Property:** Trial-to-trial spike-count variability is quenched at stimulus
onset while mean activity rises at the ensemble level.

**Command:** `python fano_factor_test.py`

For programmatic validation, request labelled diagnostics rather than relying
on positional return values:

```python
from fano_factor_test import run_fano_factor_test_bio

result = run_fano_factor_test_bio(
    model_paths,
    n_trials=32,
    baseline_frames=30,
    stim_frames=30,
    device="cuda",
    seed_base=0,
    return_diagnostics=True,
)
primary = result["primary"]
```

The primary Fano statistic excludes zero-mean neurons and uses a fixed
stimulus-centred ±2σ ROI. Frames must be physical 10 ms bins
(`dt * n_substeps = 10 ms`, with `n_substeps = 100`). The conventional
epsilon/all-neuron ROI is secondary.

Mandatory focused ranges are baseline active-neuron FF `.7–1.5`, early FF
`.4–1.0`, and a decrease greater than `.2`. Across ten checkpoints, every early
FF must be below its baseline, the ensemble rate must rise, and rate must rise
in at least 8/10 checkpoints.

**Final ensemble:** primary active-neuron FF `1.123541 → .783592`; fixed-ROI
rate `.095922 → .121753`; FF quenching 10/10 and rate rise 8/10. M00's
conventional all-neuron ROI `1.095657 → .224841` is a secondary diagnostic and
must not be substituted for the primary `.783592` result.

### 4.7 Temporal and spatial binding windows

**Property:** Integration is strongest for coincident cues, and assay-local
perturbations change the fitted binding half-width in the prespecified
direction.

**Command:** `python generate_all_fresh.py`

TBW uses temporal-fusion classification across 51 offsets and reports the 50%
fusion half-width. SBW uses AV enhancement above the calibrated threshold
across 33 separations, control-floor subtraction, and a symmetric pedestal fit.
Each condition pools 10 checkpoints × 50 trials.

| Perturbation | TBW gate | Final TBW | SBW gate | Final SBW |
|---|---:|---:|---:|---:|
| Control | Reference | `109.881044 ms` | Reference | `26.7°` |
| Reduced feed-forward inhibition | Wider | `157.141585 ms` | Wider | `29.0°` |
| Reduced adaptation | Wider | `228.440307 ms` | Wider | `30.2°` |
| Reduced NMDA | Narrower | `95.194328 ms` | Narrower | `13.2°` |
| Increased NMDA | Stable/equivalent | `111.546667 ms` | Wider | `30.6°` |

All eight signed/equivalence perturbation predictions passed. The two control
widths are model-operational references, not clinical norms.

### 4.8 TBW step-size invariance

**Property:** The control TBW is stable when the integration step is halved
without changing physical frame duration, delays, stimuli, or trial labels.

The canonical paired protocol uses checkpoints `00..09`, 51 SOAs, 50 trials
per SOA, and seed `12345 + model_index`, reset identically for each step size.
Compare `dt=.1 ms` × 100 substeps with `dt=.05 ms` × 200 substeps, so both
external frames span 10 ms. Express drift first as the fitted left-to-right
**full width**; the reported TBW half-width is exactly half that value.

The predeclared acceptance gates are:

- absolute mean paired full-width drift `<= 5 ms`;
- absolute drift at every checkpoint `<= 10 ms`;
- the complete 90% CI of paired full-width drift lies within `[-10, +10] ms`;
- pooled-ensemble absolute full-width drift `<= 10 ms` (observed
  `+2.210524 ms`);
- the `dt=.1 ms` pooled control half-width lies within `110 ± 10 ms` (observed
  `109.881044 ms`);
- pooled fit `R² >= .90` (observed `.998386/.998502`) and every individual fit
  `R² >= .80` (observed minimum `.995995`);
- pooled-curve Pearson `r >= .95` and NRMSE `<= .10`; and
- paired location hashes match at both step sizes (observed 10/10 checkpoint
  pairs) and corrected-versus-prepatch location hashes match (observed 20/20
  conditions).

Invariant gates require exact 10 ms frames, physical delays preserved within
half a substep, finite audited dynamic tensors at every frame, each STP
resource in `[0,1]`, unchanged registered parameter/buffer (`state_dict`)
content, and valid two-crossing fits with finite parameters/covariance and no
fallback. Dynamic simulation state is expected to evolve; its finiteness and
resource bounds, rather than immobility, are audited.

**Final paired result:** mean full-width drift `+2.284500 ms` (90% CI
`[1.482331, 3.086670] ms`), mean half-width drift `+1.142250 ms`, worst
checkpoint `+5.487398 ms`, `r=.999712`, and NRMSE `.010152`; all acceptance and
invariant gates passed. These are repository validation gates and results, not
thresholds reported by the manuscript or its Supplementary Figure 1.

## 5. Recommended regression check

The focused tests are under `tests/`:

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m pytest -q tests
```

The RTX A6000 validation run produced the result below; its literal command
and environment are preserved in the [research log](research_log.md):

```text
7 passed in 268.02s (0:04:28)
```

The current tests deliberately assert A6000 hardware for exact release
reproduction. The assay implementation itself does not use A6000-specific
kernels.

Coverage includes:

- canonical E/I routing, value range, checkpoint hash, `state_dict`
  preservation, and finite dynamic state;
- Fano silent-neuron semantics, physical time, primary/secondary separation,
  fixed ROI, seeds, one-checkpoint ranges, and ten-checkpoint directions;
- inverse-effectiveness low/high direction and stationary state; and
- latency mean/fastest semantics, reporting, fresh modality state, and removal
  of the old forced adaptation perturbation; and
- presynaptic inhibitory-input STP shape, release, finite state, and resource
  bounds under synchronous production-path input.

## 6. Integrity and reporting

For validation, hash checkpoint files before the first command and
after the final command. The frozen run compared 12 checkpoint files and found
12/12 SHA-256 values unchanged. It also confirmed that scoped source hashes did
not change during execution.

Analysis commands write generated figures and caches. These outputs are
expected side effects and should be produced in an isolated copy when a clean
artifact tree is required. Report the exact command, device mapping, result,
runtime, and any deviations from this protocol.

Primary biological sources, mechanism assumptions, interpretation limits, and
the current result record are maintained in the
[research log](research_log.md).
