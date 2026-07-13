# Research log

This log records the biological mapping, equations and scope assumptions used
for the current `dm10` documentation. It is a scientific interpretation log,
not a substitute for the exact checkpoint or result manifests.

| Mechanism | Model mapping / equation | Primary evidence | Assumptions and limits |
|---|---|---|---|
| Spiking dynamics | `dv/dt = 0.04v^2 + 5v + 140 - u + I`; `du/dt = a(bv-u)`; on spike, `v <- c`, `u <- u+d`. All populations use `dt=0.1 ms`. | [Izhikevich 2003](https://doi.org/10.1109/TNN.2003.820440); [Izhikevich, Gally & Edelman 2004](https://doi.org/10.1093/cercor/bhh053) | Computationally efficient phenomenology; `aM` and `dM` do not uniquely identify ion channels. |
| Excitatory integration | A/V inputs carry AMPA plus slow voltage-dependent NMDA current to both MSI populations; `gNMDA` is a global conductance scale. | [Binns & Salt 1996](https://doi.org/10.1152/jn.1996.75.2.920); [Truszkowski et al. 2017](https://doi.org/10.7554/eLife.25392) | Supports NMDA-sensitive integration, but not the in-vivo direction of a global `gNMDA` perturbation or a unique clinical mechanism. |
| Inverse effectiveness | Formal RMI is measured across intensity; the large effect is confined to `I=.05`, with only a small later-quantile residual at `I=.10`. | [Stanford, Quessy & Stein 2005](https://doi.org/10.1523/JNEUROSCI.5095-04.2005); [Truszkowski et al. 2017](https://doi.org/10.7554/eLife.25392) | Model intensity is dimensionless and is not calibrated to sound pressure or luminance. Qualitative comparison only. |
| Feed-forward inhibition | A/V → MSI-inhibitory delays are 27/42 ms; MSI-inhibitory → MSI-excitatory GABA adds 5.2 ms. `pv_gaba_scale` affects this edge. | [Miller, Stein & Rowland 2017](https://doi.org/10.1523/JNEUROSCI.2767-16.2017); [Kaneda & Isa 2013](https://doi.org/10.1016/j.neuroscience.2012.12.061); [Villalobos et al. 2018](https://doi.org/10.3389/fncir.2018.00035) | The 54-neuron population is PV/GABA-like; biological SC PV cells are heterogeneous and not uniformly inhibitory. |
| Lateral inhibition | MSI spikes drive a lateral/surround GABA state scaled by `g_GABA`; it first acts on the next 0.1 ms substep. | [Kaneda & Isa 2013](https://doi.org/10.1016/j.neuroscience.2012.12.061); [Miller, Stein & Rowland 2017](https://doi.org/10.1523/JNEUROSCI.2767-16.2017) | Cannot change the spike that created it. It can affect later firing and TBW. |
| Excitatory plasticity | Local spike-timing-dependent updates train feed-forward edges; recurrent MSI plasticity is delayed and staged after warm-up. | [Bi & Poo 1998](https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998); [Izhikevich, Gally & Edelman 2004](https://doi.org/10.1093/cercor/bhh053) | Local timing rules are biology-inspired abstractions, not fitted SC synapse-identification models. |
| Inhibitory plasticity | The MSI-inhibitory → MSI-excitatory GABA edge receives symmetric timing-dependent updates around a target trace. | [D'Amour & Froemke 2015](https://doi.org/10.1016/j.neuron.2015.03.014); [Vogels et al. 2011](https://doi.org/10.1126/science.1211095) | The exact rule and target are engineering choices grounded in cortical evidence, not direct SC measurements. |
| Formal Miller RMI | `F_AV(t) <= min(F_A(t)+F_V(t),1)` under the race/context-invariance null; reported statistic `G(q)=Q_bound(q)-Q_AV(q)`. | [Miller 1982](https://doi.org/10.1016/0010-0285(82)90010-X) | A violation rejects the joint null assumptions; it does not uniquely locate coactivation. The model endpoint is neural latency, not human RT. |
| Autism comparison | Compare reduced simple-stimulus facilitation only; elevated `gNMDA` is a frozen model perturbation that reproduces that endpoint direction. | [Brandwein et al. 2013](https://doi.org/10.1093/cercor/bhs109); [Ostrolenk et al. 2019](https://doi.org/10.1038/s41598-019-48413-9) | Not a diagnosis, whole-condition model, unique mechanism or direct behavioural reproduction. |

## Evidence ledger

- Baseline acquisition: revision `4da0644`; tie-corrected analysis: `7cbe8d8`.
- Seed-42 perturbation screen: revision `ddaaae5`; exploratory selection only.
- Held-out confirmation: revision `19133d6`; checkpoints 43–51, seed 42 excluded.
- Triggered adaptation attribution and timing work: revision `dbcf447`.
- Canonical GPU evidence for these analyses: NVIDIA GeForce RTX 5090.
- Compact committed outputs and archive metadata: [`results/race_model/`](../results/race_model/).
- Historical acquisition hashes remain pinned to their recorded revisions.
  Current-main acquisition requires a new versioned protocol, not a rewrite of
  the completed provenance record.
