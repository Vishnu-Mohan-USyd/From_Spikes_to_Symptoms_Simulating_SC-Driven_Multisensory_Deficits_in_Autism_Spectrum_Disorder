# Clean trainable multisensory integration network (CUDA/C++)

This branch contains a fresh, self-contained spiking multisensory integration
(MSI) model implemented entirely in CUDA C++. Python is not required to build,
train, evaluate, or test the network.

The model is deliberately small: 180 auditory relay neurons, 180 visual relay
neurons, 180 multisensory excitatory neurons, and 60 recruited inhibitory
neurons. It is trained over repeated sensory presentations using local synaptic
plasticity. TBW and SBW are measured from the trained neural response; neither
curve is supplied as a target during learning.

The detailed biological rationale, assumptions, equations, and primary-source
citations are in
[`docs/research_log_clean_msi.md`](docs/research_log_clean_msi.md).

## Validated five-seed result

The authoritative run used five independently initialized models and 2,000
developmental presentations per model. Evaluation pooled 1,000 trials at each
TBW SOA and 1,000 trials per SBW disparity/orientation condition.

| Neural measure | Result |
|---|---:|
| Empirical TBW50 | **184.507477 ms** |
| Empirical TBW75 | **151.851410 ms** |
| Gaussian-primary SBW HWHM | **23.6659412 degrees** |
| Raw SBW half-height crossing | **26.0182571 degrees** |
| Held-out neural-fusion observer AUROC | **0.996797025** |
| Held-out Brier score | **0.0196785014** |

The prespecified TBW50 interval is 100–300 ms. The prespecified Gaussian SBW
HWHM interval is 15–25 degrees. The raw SBW crossing is retained separately as
a shape audit; it is not silently substituted for the Gaussian-primary width.

These are run measurements, not hard-coded constants or training labels. The
published source does not contain frozen checkpoints or raw trial logs, so a
new run is a new stochastic experiment controlled by the supplied seed.

## Circuit architecture

All populations are evolved on the GPU with vectorized CUDA kernels in 1-ms
simulation/model steps. The event-resolved neuron solver uses adaptive
conductance-dependent substepping within each model step.

| Population | Count | Model | Role |
|---|---:|---|---|
| A | 180 | heterogeneous regular-spiking Izhikevich | auditory relay sheet |
| V | 180 | heterogeneous regular-spiking Izhikevich | visual relay sheet |
| MSI-E | 180 | heterogeneous regular-spiking Izhikevich | multisensory output sheet |
| MSI-I | 60 | heterogeneous fast-spiking Izhikevich | recruited feed-forward/lateral inhibition |

The A, V, and MSI-E sheets share a one-dimensional azimuth axis spanning
approximately -90 to +90 degrees. The 60-neuron inhibitory sheet spans the
same axis at lower resolution.

### Pathways and receptor types

| Projection | Receptors/transmitter | Plasticity |
|---|---|---|
| A -> MSI-E | AMPA + voltage-gated NMDA | local Clopath voltage rule |
| V -> MSI-E | AMPA + voltage-gated NMDA | local Clopath voltage rule |
| A -> MSI-I | AMPA + voltage-gated NMDA | fixed transmitting scaffold |
| V -> MSI-I | AMPA + voltage-gated NMDA | fixed transmitting scaffold |
| MSI-E -> MSI-E | recurrent AMPA + NMDA | local Clopath rule at reduced rate |
| MSI-E -> MSI-I | AMPA + NMDA | local Oja rule |
| MSI-I -> MSI-E | GABA-A | local Vogels-like inhibitory STDP |

AMPA/NMDA describe excitatory receptors on the postsynaptic target. The
inhibitory neurons are excited through those glutamatergic receptors and then
release GABA onto MSI-E; their output is not modeled as AMPA/NMDA inhibition.
The minimal circuit includes GABA-A but omits GABA-B and I-to-I connectivity.

Every excitatory contact has paired AMPA and NMDA state. NMDA uses slower
kinetics and a voltage-dependent magnesium gate. MSI-E neurons also have the
Izhikevich recovery variable and spike-triggered recovery increment, providing
a phenomenological spike-frequency-adaptation mechanism. No extra AHP channel
is claimed.

### Recurrence and inhibition

MSI-E recurrence is sparse. Its candidate mask is position-blind; any measured
distance structure in the final recurrent efficacies is therefore a trained
outcome rather than a wired local-neighborhood rule. Recurrence provides
limited amplification rather than a separate attractor or memory subsystem.
Recruited inhibition is explicitly disynaptic:

```text
A/V --AMPA+NMDA--> MSI-I --GABA-A--> MSI-E
```

The network does not include a default-mode network, diencephalic module,
global controller, hand-written fusion equation, or other auxiliary mechanism.
TBW and SBW are read from MSI-E spiking after the minimal circuit evolves.

## What learns

Training changes A-to-E, V-to-E, E-to-E, E-to-I, and I-to-E efficacies through
contact-local rules. It does not optimize TBW, SBW, fusion labels, preferred
disparity, or a desired final map.

The sensory candidate masks start from a coarse biological alignment prior:
each target receives 45 candidate contacts, with the auditory candidate field
broader than the visual field. Active efficacies begin weak and heterogeneous
and are refined by experience. Consequently, final topography is not an
entirely structure-free emergence claim: coarse candidate geometry is a
developmental guidance prior, while synaptic efficacy and measured response
organization are activity dependent. E-to-E and I-to-E candidate masks are
position-blind sparse controls.

No network-global weight normalization is used. Excitatory plasticity is
voltage based; inhibitory plasticity is a local timing rule; E-to-I competition
uses a local Oja update. Weights remain within configured pathway-specific
bounds.

## Developmental presentations

Each presentation is independently sampled from:

- 40% common-cause AV events;
- 20% independent AV events;
- 20% auditory-only events;
- 20% visual-only events.

Common-cause AV events share a latent azimuth; independent events draw the two
locations separately. Auditory and visual observations receive independent
spatial noise. Event salience is log-uniform from 25 to 100 Hz. Common-cause
physical SOA (`tV - tA`) is sampled from a truncated distribution centered at
-50 ms, while independent AV SOA is broad. Positive, modality-specific sensory
latencies are then added before spikes reach the circuit.

The model receives only afferent spike trains. The latent source category,
physical SOA, and spatial correspondence are never passed to plasticity as
labels.

## Neural measurements

### Temporal binding window (TBW)

TBW is measured as neural `P(fusion)` over physical audiovisual onset
disparity, where physical SOA is `tV - tA`. The observer uses MSI-E PSTH
structure and response episodes defined from matched A-only and V-only neural
controls. The empirical primary width is the distance between the two
contiguous 50% crossings. A 75% width is serialized as a secondary diagnostic.

The evaluation grid is -500 to +500 ms in 25-ms increments. The saved curve is
unimodal and asymmetric, with TBW50 184.507477 ms.

### Spatial binding window (SBW)

SBW is a neural enhancement curve, not a behavioral fusion probability:

```text
G(d) = R_AV(d) - max(R_A(d), R_V(d))
```

Matched component conditions are measured at unsigned disparities from 0 to
60 degrees in 2.5-degree increments and in both spatial orientations. The
primary width is the HWHM of a fixed-center Gaussian fitted to pooled neural
gain. The direct half-height crossing is reported separately.

## Source layout

| File | Purpose |
|---|---|
| `clean_msi.cuh` | public types, constants, model/evaluation interfaces |
| `clean_msi.cu` | CUDA kernels, training, calibration, evaluation, and native CLI |
| `tests/test_clean_msi.cu` | native C++/CUDA scientific and mechanics tests |
| `docs/research_log_clean_msi.md` | biological rationale, equations, assumptions, citations, and detailed result contract |

Generated binaries and plots are intentionally not source-controlled.

## Requirements

- Linux with a CUDA-capable NVIDIA GPU
- NVIDIA driver compatible with the installed CUDA toolkit
- CUDA toolkit with `nvcc`
- A C++17 host compiler supported by that CUDA toolkit

No third-party neural-network library, Python runtime, package manager, or
build framework is required.

## Build

From the repository root:

```bash
nvcc -O3 -std=c++17 -arch=native -Xcompiler=-pthread \
  clean_msi.cu -o clean_msi

nvcc -O3 -std=c++17 -arch=native -Xcompiler=-pthread \
  -DCLEAN_MSI_NO_MAIN clean_msi.cu tests/test_clean_msi.cu \
  -o test_clean_msi
```

`-arch=native` builds for the GPU visible on the build machine. On a headless
cross-compilation host, replace it with the required explicit CUDA architecture.

## Run

Calibrate receptor event quanta on GPU 0:

```bash
./clean_msi calibrate --device 0 --seed 0
```

Train five models for 2,000 presentations each:

```bash
./clean_msi train --device 0 --seed 0 --seeds 5 \
  --presentations 2000 --chunk 100
```

Train and immediately run the full neural validation:

```bash
./clean_msi train-validate --device 0 --seed 0 --seeds 5 \
  --presentations 2000 --chunk 100 \
  --observer-train 800 --observer-holdout 400 \
  --tbw-trials 200 --sbw-trials 200 \
  > clean_msi_run.jsonl
```

The CLI emits newline-delimited JSON records for calibration/training, the
observer, TBW, SBW, receptive fields, topology, inverse effectiveness, and
causal controls. Redirect stdout as shown to preserve the full experiment.

Available commands are `calibrate`, `train`, `validate`, and
`train-validate`. Important controls include:

| Option | Meaning |
|---|---|
| `--device N` | CUDA device index |
| `--seed N` | base model/random seed |
| `--seeds N` | number of independent models |
| `--presentations N` | training presentations per model |
| `--chunk N` | training presentations per reporting chunk |
| `--observer-train N` | observer training trials per class/model |
| `--observer-holdout N` | held-out observer trials per class/model |
| `--tbw-trials N` | trials per SOA per model |
| `--sbw-trials N` | trials per disparity/orientation per model |
| `--rf-trials N` | receptive-field trials per location/model |
| `--inverse-trials N` | inverse-effectiveness trials per condition/model |
| `--evaluation-seed N` | independent evaluation RNG seed |
| `--control NAME` | one reversible causal intervention |
| `--all-controls` | paired baseline/intervention evaluation for all controls |

The causal interventions remove NMDA, GABA-A output, E-to-E recurrence, the
recruited-inhibition pathway, the MSI-E spike-triggered adaptation increment,
or shuffle paired sensory AMPA/NMDA source rows. They are evaluation tools, not
alternate mechanisms silently enabled in the validated baseline.

## Test

List available native tests:

```bash
./test_clean_msi --list
```

Run the fast core suite and the evaluation-contract suites:

```bash
./test_clean_msi --core
./test_clean_msi --evaluation-synthetic
./test_clean_msi --evaluation-mechanics
```

Run every suite, including performance and training checks:

```bash
./test_clean_msi --all
```

Tests are native CUDA/C++; no Python test harness is part of this model.

## Scientific boundaries

- This is a mechanistic abstraction of a deep-SC-like circuit, not a fitted
  reconstruction of a particular species, lamina, or cell catalogue.
- Population counts and one-dimensional azimuth geometry are operational.
- The exact developmental mixture, scaffold widths, degrees, and acceptance
  intervals are explicit experimental choices, not universal biological
  constants.
- A coarse aligned candidate scaffold is present before learning; the model
  must not be described as learning spatial organization from a completely
  unstructured graph.
- GABA-B, I-to-I connectivity, laminar circuitry, and identified adaptation
  ion channels are outside this minimal model.
- The validated TBW/SBW values establish behavior of the measured five-seed
  run. Causal interpretation requires the separately emitted interventions.

For receptor biology, plasticity equations, timing order, calibration targets,
measurement validity rules, and complete citations, use the research log rather
than inferring beyond these stated boundaries.
