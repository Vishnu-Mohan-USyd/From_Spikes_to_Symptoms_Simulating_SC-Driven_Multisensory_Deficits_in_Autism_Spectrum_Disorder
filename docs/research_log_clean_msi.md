# Clean MSI network: biological rationale and measurement contract

## Scope

This is a clean-room, cross-species abstraction of a deep-superior-colliculus-like
multisensory circuit, not a reconstruction of one animal, lamina, or cell type.
The model asks a narrow scientific question: given a coarse developmental
alignment prior and local exposure-driven plasticity, can auditory excitation,
visual excitation, recurrence, and inhibition support spatial and temporal
multisensory selectivity? The validated artifact contains only the frozen
post-training state, so it does not by itself quantify refinement from the
initial scaffold.

The simulated populations are 180 auditory relay neurons, 180 visual relay
neurons, 180 multisensory excitatory neurons, and 60 inhibitory neurons. The
relay sheets and MSI-E sheet share a one-dimensional azimuth coordinate from
-89.5 to 89.5 degrees. Population counts are an operational discretization, not
an anatomical cell-count claim.

## Mechanism choices

| Choice | Biological mapping and evidence | Implementation boundary |
|---|---|---|
| Spiking neurons | The Izhikevich model captures regular-spiking and fast-spiking response classes at low computational cost ([Izhikevich 2003](https://doi.org/10.1109/TNN.2003.820440)). GABAergic neurons in the SC motor-output layer include a large fast-spiking class and local/horizontal interneurons ([Sooksawate et al. 2011](https://doi.org/10.1111/j.1460-9568.2010.07535.x)). | MSI-E, A, and V use heterogeneous regular-spiking parameters; MSI-I uses heterogeneous fast-spiking parameters. These are phenotype abstractions, not fitted SC membrane models. |
| Excitatory transmission | Deep-SC multisensory responses are sensitive to NMDA antagonism ([Binns & Salt 1996](https://doi.org/10.1152/jn.1996.75.2.920)), and NMDA signalling participates in activity-dependent collicular map refinement ([Huang & Pallas 2001](https://doi.org/10.1152/jn.2001.86.3.1179); [Johnson et al. 2023](https://doi.org/10.1523/JNEUROSCI.1473-22.2022)). Voltage-dependent magnesium block is grounded in direct channel measurements ([Jahr & Stevens 1990](https://doi.org/10.1523/JNEUROSCI.10-09-03178.1990)). | Every excitatory pathway uses paired fast AMPA and slower voltage-gated NMDA conductances: A/V→E, A/V→I, E→E, and E→I. The model does not claim identical receptor fractions across these pathways; event quanta are calibrated by explicit response assays. |
| Recruited inhibition | GABA-A receptors mediate important collicular surround inhibition, whereas GABA-B has a distinguishable slower role ([Binns & Salt 1997](https://doi.org/10.1111/j.1469-7793.1997.629bd.x)). Anatomical/electrophysiological work identifies local and long-range GABAergic neurons in intermediate/deep SC ([Sooksawate et al. 2011](https://doi.org/10.1111/j.1460-9568.2010.07535.x)). | A and V excite MSI-I through AMPA/NMDA; MSI-I inhibits MSI-E through GABA-A. Thus AMPA/NMDA are receptors *onto the inhibitory neurons*, while their output transmitter is GABA. GABA-B and I→I are omitted from this minimal model, not asserted absent in biology. |
| Recurrent excitation | Patch-clamp photostimulation in rat intermediate SC found strong local excitatory interactions, usually within 500 µm, consistent with neighboring cells sharing recurrent excitation ([Lee & Hall 2006](https://doi.org/10.1523/JNEUROSCI.0724-06.2006)). Local-circuit burst amplification in rat SGI is NMDA-receptor-dependent ([Saito & Isa 2003](https://doi.org/10.1523/JNEUROSCI.23-13-05854.2003)). Neither study selected multisensory neurons or measured connection probability. | Sparse local E→E AMPA/NMDA recurrence is a model hypothesis extrapolated from intermediate/deep-SC physiology, not an established multisensory-SC microcircuit. Its causal contribution is tested by reversible E→E ablation; exact sparsity and receptor pairing are modelling choices. |
| Experience-dependent maturation | Early multisensory SC neurons lack adult-like integration; integration and receptive fields mature over development ([Wallace & Stein 1997](https://doi.org/10.1523/JNEUROSCI.17-07-02429.1997)). Removing normal visual experience leaves responsive multisensory neurons unable to synthesize their inputs ([Wallace et al. 2004](https://doi.org/10.1523/JNEUROSCI.2535-04.2004)). | Sensory→E contacts begin from a coarse topographic candidate scaffold and then change through local learning. Sensory→I uses the same coarse candidate geometry but remains fixed while transmitting. No TBW, SBW, preferred disparity, fusion label, or final target map enters plasticity. |
| Local excitatory plasticity | Spike-timing-dependent changes at glutamatergic synapses depend on relative timing, postsynaptic state, and initial strength ([Bi & Poo 1998](https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998)). Voltage-based local plasticity can produce localized receptive fields and code-dependent recurrent structure ([Clopath et al. 2010](https://doi.org/10.1038/nn.2479)). | A/V→E and E→E use the ungated Eq. 3 form with delayed contact-local presynaptic traces, additive updates, and hard path bounds; recurrent E→E uses `0.01` times the feedforward Clopath learning rate. A/V→I is not plastic. The detailed causal trace order and full-square homeostasis are documented below. |
| Excitation of inhibition and inhibitory plasticity | Oja's local normalized Hebbian rule is a formal model of competitive feature learning ([Oja 1982](https://doi.org/10.1007/BF00275687)). Timing-dependent inhibitory plasticity can stabilize excitation/inhibition and sensory representations ([Vogels et al. 2011](https://doi.org/10.1126/science.1211095)). | Fixed A/V→I AMPA/NMDA scaffolds recruit inhibition; Oja updates only E→I, and I→E uses local Vogels-like inhibitory STDP with fixed primary `eta_iSTDP=1e-4`. No network-global weight normalization is used. |

### Clopath amplitude, traces, and causal ordering

[Clopath et al. 2010, Table 1b](https://doi.org/10.1038/nn.2479) gives the
visual-cortex set `theta_minus=-70.6 mV`, `theta_plus=-45.3 mV`,
`A_LTD=14e-5 mV^-1`, `A_LTP=8e-5 mV^-2`, `tau_x=15 ms`,
`tau_minus=10 ms`, and `tau_plus=7 ms`. The implementation uses
`tau_x=15 ms`, `tau_minus=10 ms`, `tau_plus=7 ms`, each model neuron's stable
rest as `theta_minus`, and `theta_plus=-45 mV`. With each positive voltage
factor normalized by `V0=20 mV`, the dimensional amplitudes become
`D=A_LTD*V0=0.0028` and `P=A_LTP*V0^2=0.032`, so `D/P=0.0875`. A unit-jump
presynaptic trace gives the corresponding dimensionless LTD coefficient
`(tau_x/dt)*(D/P)=15*0.0875=1.3125`. Additive deltas are hard-clipped to each
path's bounds, consistent with the published voltage-rule construction. This
mapping is a model implementation of the cited cortical rule, not a measured
SC plasticity parameter set.

The slow homeostatic state uses `tau_H=100000 ms` and the **full square**
`H <- exp(-dt/tau_H) H + (1-exp(-dt/tau_H)) (v_pre-reset-v_rest)^2`.
There is no `max(...,0)` rectification. LTD is multiplied once by `H/H_ref`,
where `H_ref=16.3214752406 mV²` is the preregistered frozen equal-salience
100,000-step seed-0 reference. **Explicit modelling assumption:** the
full-square voltage statistic and shared reference are normalization choices,
not measured SC constants or fits to TBW/SBW.

The accepted-step order is deliberately causal. The delayed presynaptic event
first advances the contact-local `x` trace. Plasticity then reads that updated
presynaptic trace and the current `pre_reset` voltage gate, but reads the
**prior-step** filtered `u_minus`, `u_plus`, and `H`. A protective barrier
follows plasticity; only then are the Clopath voltage/homeostasis traces
advanced with the current `pre_reset` voltage, concurrently with training-step
completion. Thus a spike's current voltage excursion cannot enter the
low-pass voltage factors and potentiate that same event ("current-spike
self-LTP"), while the causal presynaptic arrival remains event-inclusive.

## Developmental environment

Each presentation is drawn independently from:

- 40% common-cause AV, 20% independent AV, 20% auditory-only, and 20%
  visual-only events;
- every present modality has a latent azimuth drawn uniformly from
  `[-60, 60]` degrees: common AV shares one draw, independent AV draws A and V
  separately, and a unimodal event draws its one present modality. Thus the
  per-modality latent marginals match across event classes. Every present A or
  V draw then receives independent `N(0, 8²)` or `N(0, 2²)` degree noise,
  respectively, and the noisy coordinate is reflected at ±90 degrees; absent
  channels remain inert;
- log-uniform event salience from 25 to 100 Hz, shared across modalities for a
  common event and sampled separately for independent events;
- physical SOA `tV - tA`: truncated `N(-50 ms, 60² ms²)` within ±250 ms for a
  common event, and uniform within ±600 ms for independent events;
- positive sensory latencies `A ~ N(21, 5²) ms` and `V ~ N(69, 10²) ms`.

The approximately -48 ms mean physical offset compensates the difference in
mean sensory latency, but the network receives only spikes and never the
generating SOA. The precise mixture, noise, salience, and latency distributions
are ecological *priors for this experiment*, not universal measurements of SC
input statistics.

Developmental inputs use reflected Gaussian afferent profiles sampled on the
180-bin lattice (auditory sigma 8 degrees, visual sigma 2 degrees); these widths
are operational encoding resolutions. At each stimulus location, the visual
profile receives one finite-lattice scalar so its discrete population sum
matches the auditory profile at equal nominal salience. This is an engineering
salience-control approximation, motivated by the measured short
counterfactual that the unequal widths otherwise change total afferent
population mass, not a fit to TBW or SBW ([Miller et al.
2015](https://doi.org/10.1523/JNEUROSCI.4771-14.2015)). Controlled evaluation
probes apply the same population-mass control while preserving the requested
center exactly and adding no developmental location jitter, so the SBW target
is not injected through the training-noise sampler; SBW noise-control
sensitivity must be reported separately. Each event lasts 50 ms and has 100 ms of
pre-stimulus activity, at least 250 ms post-stimulus activity, and a real
250-ms zero-afferent receptor-update interval after the one-step afferent
pipeline flush; the `valid` mask remains true so neural and synaptic state
continues to evolve. The validated final development run was sequential
(`B=1`): 2,000 presentations for each of five seeds. The category name is
recorded for auditing only.

## What is biological and what is operational

SC physiology establishes spatial, temporal, and effectiveness constraints on
multisensory enhancement ([Meredith & Stein 1986](https://doi.org/10.1152/jn.1986.56.3.640);
[Meredith, Nemitz & Stein 1987](https://doi.org/10.1523/JNEUROSCI.07-10-03215.1987);
[Perrault et al. 2005](https://doi.org/10.1152/jn.00926.2004)). Those findings
motivate the *phenomena* tested here. The exact observer and thresholds below
are operational measurement definitions:

- A logistic observer is fitted only after network learning. Its inputs are
  MSI-E spikes in 20-ms bins from -100 to +700 ms relative to the earlier
  latency-shifted event. L2 is `1e-3`.
- The neural fusion rule pools the raw auditory-only and visual-only MSI-E
  control PSTHs in those same 20-ms bins. Their pooled raw prestimulus mean
  plus three sample standard deviations defines one strict response threshold.
  For each modality, the primary peak is selected only in the preregistered
  0–260-ms response bins and the contiguous raw supra-threshold episode defines
  its control window. No centered smoother or fallback participates in that
  window selection. This preserves brief onset responses rather than spreading
  or attenuating them before response detection: classic SC work identifies
  relative response timing as a determinant of multisensory interaction
  ([Meredith, Nemitz & Stein
  1987](https://doi.org/10.1523/JNEUROSCI.07-10-03215.1987)), and awake mouse
  SC recordings found that most auditory response peaks occurred within the
  first 20 ms and analyzed that fast component explicitly ([Ito et al.
  2020](https://doi.org/10.1038/s41467-020-14897-7)). Centered three-bin
  smoothing remains an operational, symmetric A/V rule only for classifying
  each trial's response episode after the raw control windows are fixed.
- Designated observer positives have physical SOA -50 ms; negatives have
  physical SOA -500 or +500 ms. The default split is 800 training and 400
  held-out trials **per model seed**, giving 4,000 training and 2,000 held-out
  trials over the validated five seeds. Acceptance requires held-out AUROC
  ≥0.80 and Brier ≤0.20. These frozen-logistic-observer metrics are separate
  evaluation diagnostics; the observer does not generate the empirical TBW.
- TBW is measured at physical SOAs `0, ±25, ..., ±500 ms`, with 200 fresh
  trials per SOA per model seed (1,000 trials per SOA over five seeds).
  For each SOA, empirical `P(fusion)` is the fraction of trials whose
  PSTH-derived `neural_fusion_event` is true. The primary metric is the full
  width between contiguous empirical crossings of `P(fusion)=0.50`.
  Acceptance requires a valid unimodal curve, finite empirical `TBW50` in the
  inclusive 100–300-ms interval, and finite peak-minus-tail probability at
  least 0.25. Crossings at `P(fusion)=0.75` and `TBW75` are independently
  computed and serialized as secondary diagnostics; they may be unavailable
  when the empirical peak is below 0.75 and never determine acceptance. A
  bounded asymmetric Gaussian is also diagnostic because
  audiovisual temporal windows can be asymmetric ([Meredith et al.
  1987](https://doi.org/10.1523/JNEUROSCI.07-10-03215.1987); [Cecere, Gross &
  Thut 2016](https://doi.org/10.1111/ejn.13242)). Human work using a
  three-quarter maximum criterion provides an illustrative asymmetric window
  ([Hillock et al.
  2012](https://doi.org/10.1111/j.1467-7687.2012.01171.x)), not this model's
  acceptance gate. The target is not injected into learning or the observer.
- SBW uses unsigned disparities `0, 2.5, ..., 60 degrees`, both A-left/V-right
  and A-right/V-left orientations, and 200 trials per disparity/orientation
  per model seed (1,000 trials per disparity/orientation over five seeds).
  The neural gain is
  `G(d)=R_AV(d)-max(R_A(d),R_V(d))` from matched component conditions.
  This is a neural multisensory-enhancement measure, not a behavioral
  probability of perceptual fusion. Percent enhancement and deviation from
  additivity remain secondary.

  The **acceptance-primary** width is a fixed-center, fixed-baseline symmetric
  Gaussian fit. The baseline `B` is the mean of the final three pooled gain
  samples, amplitude is fixed to `A=G(0)-B`, center is fixed at zero disparity,
  and only `sigma` is searched in
  `G_hat(d)=B+A exp(-d^2/(2 sigma^2))`. The reported Gaussian SBW is
  `HWHM=sigma*sqrt(2 ln 2)`. Acceptance requires a valid fit, valid finite
  HWHM, `A>1e-6`, finite MSE, HWHM greater than zero and no larger than the
  sampled disparity range, and an inclusive HWHM of 15–25 degrees. No MSE
  quality threshold is invented.

  The original raw half-height crossing is retained transparently as a
  **raw-crossing audit**. It still checks finite increasing inputs, positive
  center/contrast endpoints, a near-center peak, a contiguous above-half
  prefix, one outward crossing without recrossing, and a valid interpolated
  width. Those raw shape prerequisites are required by the top-level spatial
  gate, but the fitted Gaussian HWHM is the top-level `sbw50_deg`; the nested
  raw width and `direct.spatial_gate` remain separately serialized.

  These operational choices follow the SC spatial rule that enhancement
  depends on cross-modal spatial alignment and can turn to suppression with
  disparity ([Meredith & Stein
  1996](https://doi.org/10.1152/jn.1996.75.5.1843); [Kadunce et al.
  1997](https://doi.org/10.1152/jn.1997.78.6.2834)) and the developmental
  emergence of aligned multisensory spatial representations ([Xu et al.
  2018](https://doi.org/10.1113/JP275427)). The model treats zero disparity as
  expected correspondence because its developmental A/V statistics and coarse
  scaffold are aligned. That is not a universal biological claim:
  disparity-reared animals can learn enhancement for a displaced
  correspondence ([Wallace & Stein
  2007](https://doi.org/10.1152/jn.00497.2006)). The 15–25-degree interval and
  Gaussian measurement contract are operational design choices, not
  hard-coded receptive fields or learning signals.
- Inverse effectiveness is checked at 25, 50, and 100 Hz using both raw and
  percent enhancement. Greater relative enhancement for weak component
  responses is the biologically motivated expectation ([Perrault et al.
  2005](https://doi.org/10.1152/jn.00926.2004); [Alvarado et al.
  2009](https://doi.org/10.1523/JNEUROSCI.0525-09.2009)), but
  weakest-salience dominance is a diagnostic rather than an acceptance
  criterion.

## Falsification and causal checks

The implementation reports failures rather than forcing target curves.
Topography is measured from the frozen post-training A/V→E weights and the
effective inhibitory maps `(A/V→I AMPA) @ (I→E GABA-A)`. RF sweeps retain the
full 180-neuron MSI-E response at locations -60:5:60 degrees. The final high
correlations therefore establish organization in the final state; without an
emitted pre-training snapshot they do not establish a learned delta beyond
the coarse alignment prior.

Reversible controls can remove all NMDA, GABA-A output, E→E recurrence, or the
whole recruited-inhibition pathway; a fifth control sets the MSI-E adaptation
increment to zero. A paired AMPA/NMDA source-row shuffle is available as a
separate topology intervention that preserves marginal weights. Mechanics
tests execute and restore the adaptation-off setting, but the validated
five-seed run is a baseline run with all causal-control flags equal to `0`.
No directional effect of adaptation removal on TBW is asserted.

The top-level acceptance flag combines held-out observer quality with the
temporal and spatial gates. Topography and causal controls are reported as
mechanistic audits, not silently folded into that flag. A numerical fit by
itself is not evidence of a biological mechanism, and this log does **not**
claim that every causal control passed.

## Coarse developmental guidance prior and pathway contract

The sensory candidate masks contain a coarse alignment prior. For every
postsynaptic target, a seeded deterministic Gumbel ranking selects exactly
`K=45` active presynaptic candidates from a Gaussian distance score. Auditory
candidates use `FWHM=105 degrees`; visual candidates use
`FWHM=45 degrees`. Active A/V weights are independently initialized from the
weak distribution `U[0.05,0.15]`; inactive contacts have zero efficacy. The
same auditory and visual kernels are applied to sensory→E and sensory→I, with
the 60-neuron I coordinate rescaled over the same 179-degree span.

Developmental SC evidence supports coarse cross-modal alignment and
experience-dependent refinement, while auditory spatial representations are
broader than visual ones ([Wallace & Stein
1997](https://doi.org/10.1523/JNEUROSCI.17-07-02429.1997); [Xu et al.
2018](https://doi.org/10.1113/JP275427)). Work on developing collicular
topographic projections and auditory spatial maps supports a guidance prior
rather than an initially structure-free sheet ([Simon & O'Leary
1992](https://doi.org/10.1523/JNEUROSCI.12-04-01212.1992); [Yates et al.
2001](https://doi.org/10.1523/JNEUROSCI.21-21-08548.2001); [Nodal et al.
2005](https://doi.org/10.1002/cne.20478)).

**Explicit design assumptions:** the exact `105/45-degree` widths, exact
`K=45`, one-dimensional geometry, deterministic Gumbel realization, weak
weight interval, and reuse of the same sensory kernels for E and I targets are
model choices. The cited biology motivates only coarse alignment and the
qualitative ordering "auditory broader than visual"; it does not identify
these numerical values. Sensory candidate masks carry the stated alignment
prior, while their active efficacies can be modified by the documented
plasticity rules; E→E and I→E candidate masks remain position-blind sparse
controls. Because the validated output is a final-only snapshot, its
topographic correlations cannot apportion the observed organization between
the scaffold and subsequent learning.

The authoritative pathway/plasticity contract is:

| Path | Projection | Runtime contract |
|---:|---|---|
| 0 | A→E | Clopath plasticity |
| 1 | V→E | Clopath plasticity |
| 2 | E→E | Clopath plasticity at `0.01` of feedforward rate |
| 3 | A→I | Fixed, transmitting AMPA/NMDA scaffold |
| 4 | V→I | Fixed, transmitting AMPA/NMDA scaffold |
| 5 | E→I | Oja plasticity |
| 6 | I→E | Vogels-like inhibitory STDP |

Paths 3 and 4 each contain exactly `45*60=2700` active contacts per seed.
Their weights transmit throughout training, but their final
`changed_contacts` count is exactly zero. AMPA and NMDA share every
excitatory contact's mask and efficacy. No network-global normalization is
used.

## Fixed-primary inhibitory-plasticity calibration

The 5-Hz constant defines the Vogels alpha setpoint
`alpha=2*r_setpoint*tau=0.2` for a 20-ms trace
([Vogels et al. 2011](https://doi.org/10.1126/science.1211095)); it is not a
target imposed on the isolated assay's output rates. The applied inhibitory
learning rate is fixed-primary at exactly `eta_iSTDP=1e-4` (row 16 of the
native 81-point grid). The assay records weak/strong rates and bound fractions
from that row only. It is a finite/nonnegative and nonsaturation diagnostic
(both bound fractions must remain below `0.10`), not a selector that searches
for simulated rates between 4 and 6 Hz.

## Validated five-seed p2000 result

The final authoritative run used five seeds and 2,000 presentations per seed
with pruning disabled. These are run-specific measurements, not biological
constants:

| Measure | Validated value |
|---|---:|
| Overall criteria | `true` |
| `finite_rf_coverage` | `1` |
| `direct.finite` | `true` |
| Finite-batch evaluation | completed |
| Empirical `TBW50` | `184.507477 ms` |
| Empirical `TBW75` | `151.851410 ms` |
| Gaussian-primary SBW HWHM | `23.6659412 degrees` |
| Gaussian `sigma` | `20.1000004 degrees` |
| Gaussian amplitude | `0.507896245 Hz` |
| Gaussian baseline | `0.363203675 Hz` |
| Gaussian MSE | `0.000434781366` |
| Direct raw-crossing audit | `26.0182571 degrees` |
| Held-out AUROC | `0.996797025` |
| Held-out Brier score | `0.0196785014` |
| Held-out ECE | `0.0177110191` |
| Auditory / visual RF order correlation | `0.978469 / 0.976200` |
| Auditory / visual weight order correlation | `0.989149 / 0.992576` |
| Effective inhibitory correlation | `0.610573` |
| Recurrent distance correlation | `0.494548` |
| RF mismatch | `3.98965 degrees` |
| Mean A→E / V→E weight | `0.598837733 / 0.578493237` |
| Mean E→E / I→E weight | `0.0246553514 / 0.106403865` |

All three inverse-effectiveness gains were positive, but the
weakest-salience condition did not have the greatest relative gain. That
ordering is not an acceptance criterion. No contacts were pruned. The
Gaussian HWHM is the accepted top-level neural SBW; the `26.0182571-degree`
direct crossing is reported separately as the raw audit. These are baseline
measurements with causal-control flags equal to `0`, not evidence that every
causal intervention passed.

## MSI-E phenomenological spike-frequency adaptation

MSI-E follows the standard two-state Izhikevich dynamics
([Izhikevich 2003](https://doi.org/10.1109/TNN.2003.820440)):

`dv/dt = 0.04 v^2 + 5 v + 140 - u + I`

`du/dt = a (b v - u)`

At a spike, `v <- c` and `u <- u + d`. For MSI-E, `a = 0.02`,
`b = 0.20`, and `c = -65 + 15 r^2` for the seeded heterogeneity draw `r`.
Exactly 171 of 180 MSI-E neurons per seed use `d = 0.10`; the remaining 9 use
`d = 1.0`.

The spike-triggered increment of the recovery variable is a phenomenological
spike-frequency-adaptation mechanism, not a channel-identified AHP current.
There is no separate `g_AHP`, `E_AHP`, `tau_AHP`, or `q_AHP` state,
calibration, or emitted result in this model.

The adaptation-off causal setting changes only the MSI-E reset increment to
`d = 0`. It does not remove the recovery variable `u` and does not set
`a` or `b` to zero. The intervention therefore isolates the spike-triggered
increment; it is not a removal of all intrinsic adaptation dynamics.
