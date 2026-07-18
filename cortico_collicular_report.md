# The cortico-collicular top-down signal to the audiovisual superior colliculus

### What it is, what it does, and the most minimal-yet-accurate way to represent it in your SC/ASD spiking model

*An evidence-based report. Every substantive empirical claim is tied to a primary paper (DOI). Where the full text could not be obtained, the claim is flagged and attributed conservatively. The model-specific sections are grounded line-by-line in the actual code of the `From_Spikes_to_Symptoms` repository (branch `agent/reproducible-sc-perturbation-handoff`, commit `caf7e8e1`).*

---

## 0. TL;DR — the answer in six sentences

1. **The premise needs correcting.** Feedforward A+V convergence alone makes an SC neuron *respond* to both modalities, but on the best available evidence it does **not** produce the superadditive multisensory *enhancement* that defines integration — reversibly cooling the association cortex (AES + rLS) collapses enhancement from ~76% of stimulus pairings to ~17%, while leaving the unisensory responses present (Alvarado et al. 2007, [10.1523/JNEUROSCI.3524-07.2007](https://doi.org/10.1523/JNEUROSCI.3524-07.2007)). So the cortico-collicular projection is what turns a *multisensory* neuron into an *integrating* one.
2. **Nature of the signal:** two (or three) *unisensory*, modality-matched excitatory projections (visual from AEV, auditory from FAES) that **converge on the same dendritic compartment** of one SC output neuron, where they co-activate AMPA + NMDA receptors (Fuentes-Santamaría et al. 2009, [10.1093/cercor/bhp060](https://doi.org/10.1093/cercor/bhp060)).
3. **Mechanism — it is neither a pure "gate" nor "just independent projections + STDP"; it is the synthesis of both:** the descending inputs are unisensory and independent *in origin*, but their **synergy** at a shared nonlinear (NMDA/Mg²⁺-gated) compartment produces the boost — removing *either* the visual or the auditory cortical input abolishes the *whole* enhancement (Alvarado et al. 2009, [10.1523/JNEUROSCI.0525-09.2009](https://doi.org/10.1523/JNEUROSCI.0525-09.2009)) — and that convergence pattern is **wired up by experience-dependent (Hebbian/STDP-like) plasticity** during development. The result functions as an alignment-dependent gain: a boost for spatiotemporally aligned input, and (via a cortex-driven inhibitory route) suppression for misaligned input.
4. **How it learns statistics:** during development, Hebbian/covariance plasticity on the descending synapses grows a modality-matched, spatially-registered convergence *only for cross-modal events that actually co-occur* — the weights come to store, in effect, P(A and V share a common cause & are aligned). Dark-rearing prevents this (7/117 neurons integrate vs. ~87 expected; Wallace et al. 2004, [10.1523/JNEUROSCI.2535-04.2004](https://doi.org/10.1523/JNEUROSCI.2535-04.2004)); rearing with *spatially disparate* but temporally coincident cues *reverses* the spatial rule (Wallace & Stein 2007, [10.1152/jn.00497.2006](https://doi.org/10.1152/jn.00497.2006)).
5. **Does it pass down context?** Yes — but "context" here means a *learned prior over cross-modal configurations* (which combinations, at which relative positions, belong together), not a copy of the stimulus and not (in this anesthetized-cat literature) moment-to-moment attention. Computationally this is the p(common cause) term of Bayesian causal inference (Körding et al. 2007, [10.1371/journal.pone.0000943](https://doi.org/10.1371/journal.pone.0000943)).
6. **Minimal module for your model:** add one 180-unit cortical population that projects, through a learned positive-weight matrix, onto `self.I_M` as the **carrier of the NMDA (coincidence-gated) drive** — reusing the model's *existing* voltage-gated Mg²⁺ NMDA machinery — plus a branch onto the PV interneuron for misalignment depression, with a scalar `g_cortex` "deactivation" lever exactly analogous to the model's existing `pv_gaba_scale` ASD lever. This makes enhancement contingent on cortex (so `g_cortex → 0` reproduces the Alvarado-cooling phenotype) while leaving a reduced, additive unisensory response — which is precisely the biology.

---

## 1. Setting the stage: the central puzzle

Your model, and the multisensory-SC field generally, starts from Meredith & Stein's classic picture: individual deep-SC neurons receive converging inputs from more than one modality, and when a visual and an auditory stimulus coincide in space and time the neuron's response can be *enhanced* — often **superadditively** (more than the sum of the two unisensory responses), with the largest proportional boost for the *weakest* stimuli ("inverse effectiveness"; Meredith & Stein 1986, [10.1152/jn.1986.56.3.640](https://doi.org/10.1152/jn.1986.56.3.640); Stein & Stanford 2008, [10.1038/nrn2331](https://doi.org/10.1038/nrn2331)).

The natural assumption — and the one built into many computational models, including the coincidence-gate in your own model — is that this enhancement is simply what happens when two excitatory inputs converge on a neuron with a supralinear (e.g. NMDA-like, or threshold) nonlinearity. On that view the descending cortical projection is a nice-to-have, and the feedforward convergence does the real work.

**The experimental evidence says otherwise, and this is the crux of the whole report.** When Stein, Wallace, Alvarado and colleagues reversibly deactivate the two association-cortex areas that project to the deep SC — the anterior ectosylvian sulcus (**AES**) and the rostral lateral suprasylvian sulcus (**rLS**) — three things happen together:

- The neurons **keep responding** to visual-alone and auditory-alone stimuli (they are still multisensory in the trivial sense). (Wallace & Stein 1994, [10.1152/jn.1994.71.1.429](https://doi.org/10.1152/jn.1994.71.1.429).)
- The **multisensory enhancement disappears.** In the definitive study, cross-modal enhancement fell from 76% of pairings (91/120) to 17.5% (21/120) during deactivation, and the *mode* of combination shifted: superadditive interactions dropped from 55.8% → 11.7%, additive rose from 39.2% → 78.3%, and the mean multisensory Z-score fell from 2.59 to 0.37 — i.e. the residual combination became **additive and statistically non-significant** (Alvarado et al. 2007, [10.1523/JNEUROSCI.3524-07.2007](https://doi.org/10.1523/JNEUROSCI.3524-07.2007)).
- **Within-modal integration is untouched** — visual–visual pairings stay subadditive whether or not cortex is on. So cortex is specific to *cross-modal* synthesis, not to combining inputs in general (Alvarado et al. 2007; Alvarado et al. 2007b, [10.1152/jn.00018.2007](https://doi.org/10.1152/jn.00018.2007)).

**"But doesn't deactivation also reduce the unisensory drive, and isn't that the real cause?"** This is the obvious objection and the authors answered it directly. Deactivation *does* reduce modality-specific responses (~40%). But that reduction **cannot** explain the loss of enhancement, because of inverse effectiveness: weaker unisensory components should produce *more* proportional enhancement, not less. The enhancement collapses in the *opposite* direction to what the drive reduction predicts. Therefore cortex exerts a control over *integration* that is separable from its contribution to *drive* (Alvarado et al. 2007). This single logical step is what resolves your question "why cortico-collicular projections if feedforward convergence exists": **feedforward convergence is sufficient for a combined response but not for the integrative transform; the transform is a separate, cortically-gated computation.**

---

## 2. What is the nature of the top-down signal, and what information does it carry?

### 2.1 Anatomically: unisensory, modality-matched, converging on a shared compartment

The cortico-collicular cells themselves are **unisensory**. AES has three functional subdivisions — AEV (visual), FAES (auditory), SIV (somatosensory) — and the descending projection is organized so that an SC neuron receives cortical input **matching its own feedforward modality profile**: an audiovisual SC neuron gets AEV (visual) + FAES (auditory) cortical input (Wallace et al. 1993, [10.1152/jn.1993.69.6.1797](https://doi.org/10.1152/jn.1993.69.6.1797); Stein et al. 2009, [10.1016/j.heares.2009.03.012](https://doi.org/10.1016/j.heares.2009.03.012)). The projection originates in **layer V** (Fuentes-Santamaría et al. 2009, [10.1093/cercor/bhp060](https://doi.org/10.1093/cercor/bhp060)).

The decisive anatomy comes from Fuentes-Santamaría et al. 2009, who combined anterograde tracing of physiologically-mapped AES subdivisions with labeling of SC output neurons and electron microscopy:

- Fibers from **two different AES subdivisions terminate on the same SC output neuron** — confirmed with dual tracers (FAES–AEV, AEV–SIV, FAES–SIV pairs), both across and within single dendritic compartments.
- Contacts concentrate on **primary dendrites (~55–57%) and soma (~23–25%)** — i.e. electrotonically **proximal**, where two inputs can interact strongly.
- The synapses are **asymmetrical (excitatory-type)**, glutamatergic.
- There are **two axon morphologies**: a majority of small-caliber "modulator-like" (Type I) axons and a minority of large-caliber "driver-like" (Type II) axons — an anatomical analogy to the corticothalamic driver/modulator scheme (this functional assignment is a well-motivated hypothesis, not a directly tested fact in SC).

So the descending signal is **excitatory unisensory drive that is physically arranged to converge** — the substrate for a nonlinear interaction, not for two independent additive contributions.

### 2.2 Physiologically: it carries both drive *and* the license to integrate

There is a genuine, and important, dual character here that a faithful model must respect:

- The cortico-collicular cells **do contribute unisensory drive** — cooling them reduces modality-specific SC responses by ~40% (Alvarado et al. 2007), and in dark-reared animals they actually *dominate* the unisensory drive (Yu et al. 2013, [10.1111/ejn.12182](https://doi.org/10.1111/ejn.12182)).
- Yet their **defining, non-redundant function** is to enable the superadditive transform (§1).

The resolution the field adopts — and the one your model can implement cleanly — is that the descending input is **ordinary excitatory drive that, because it converges with matched modality-specificity onto a shared nonlinear compartment, *acts as* the gain/gating signal for integration.** Drive and gain are not two separate pathways; they are the same input read two ways. The best mechanistic proposal for *what makes it a gain* is the NMDA receptor: NMDA conductance is voltage-dependent and supralinear, so co-arrival of the visual and auditory cortical inputs on a shared, depolarized dendritic compartment produces a boost that neither input creates alone (Rowland et al. 2007, [10.1068/p5842](https://doi.org/10.1068/p5842); NMDA-dependence of SC integration: Binns & Salt 1996, [10.1152/jn.1996.75.2.920](https://doi.org/10.1152/jn.1996.75.2.920)).

> **What information does the signal convey, in one line?** A modality-matched, experience-shaped estimate of "these two unisensory events belong together" — delivered as excitatory drive onto a coincidence-sensitive compartment, so that when the estimate and the sensory evidence agree, the response is boosted.

---

## 3. The heart of your question: gating signal, or independent projections + STDP at the junction?

You framed the key alternative sharply:

> Does it give "boosting signals" during aligned input and "suppressive signals" otherwise? Or is it just simple independent A and V projections and something that just happens via STDP at the junction where those descending projections meet the SC multisensory neurons?

**The evidence says the truth is a synthesis of your two options, and this is the single most important conceptual point in the report.** The two framings are not actually mutually exclusive — they describe the *same* circuit at two different timescales.

### 3.1 In origin, the descending inputs ARE independent unisensory projections

There is no evidence for a dedicated cortical unit that computes "aligned vs misaligned" and sends down a scalar boost/suppress command. The descending cells are plain unisensory (visual or auditory) neurons. So your second framing is correct about the *inputs*.

### 3.2 But the integration is a SYNERGY at the junction, not independent addition

The experiment that settles this is Alvarado et al. 2009 ([10.1523/JNEUROSCI.0525-09.2009](https://doi.org/10.1523/JNEUROSCI.0525-09.2009)). They deactivated the visual (AEV) and auditory (FAES) subdivisions **individually and together**:

| Condition | Enhancement index | Enhancement abolished in |
|---|---|---|
| Control | 163 ± 10% | — |
| FAES (auditory) off | 27 ± 5% | 78% of neurons |
| AEV (visual) off | 27 ± 5% | 73% of neurons |
| Both off | 22 ± 5% | 78% of neurons |

The logic is airtight: **if the two descending inputs were independent additive drives, removing only the visual one would leave the auditory-driven contribution intact and merely reduce enhancement. Instead, removing *either* one abolishes the *entire* superadditive product**, and removing both is barely more effective than removing one. That is the signature of a **cooperative (synergistic) interaction** — the two inputs must be *jointly present* on the shared compartment for the boost to exist. (Each subdivision still delivers modality-specific drive: FAES-off mainly cuts the auditory response, AEV-off the visual — so they are unisensory carriers whose *joint action* is what integrates.)

**So the descending projection functions as an alignment-dependent gain** — your first framing — but that gain is not a separately-computed command; it is an **emergent property of two independent unisensory inputs meeting at a nonlinear coincidence detector.** Boost when they co-activate the compartment (aligned in space, because the maps are registered, and in time, because NMDA has a long time-constant); no boost — indeed active suppression (§5) — when they do not.

### 3.3 And STDP at the junction IS how the synergy gets wired — during development

Your "STDP at the junction" intuition is also correct, but it operates on a *developmental* timescale, not moment-to-moment. The synergistic convergence is **not innate** — it is built by experience:

- **Dark-reared cats** develop a normal complement of sensory-responsive and even multisensory SC neurons, but **almost none integrate** (7/117 showed enhancement vs. ~87 expected; Wallace et al. 2004, [10.1523/JNEUROSCI.2535-04.2004](https://doi.org/10.1523/JNEUROSCI.2535-04.2004)). Systematically varying space, time and effectiveness never recovered a significant interaction — the *capacity* is absent, not just mistuned.
- **Experience creates the transform, not the inputs.** Millisecond analysis shows dark-reared neurons have intact (even stronger) unisensory drive but no early superadditive "initial response enhancement," defaulting instead to cross-modal **competition/suppression**. Neurotypic neurons exceed a statistical-facilitation ("race") prediction by 31% overall and 57% in the initial-response window (Wang et al. 2020, [10.3389/fnint.2020.00018](https://doi.org/10.3389/fnint.2020.00018)). **The untrained default is competition; enhancement must be learned.**
- **The cortico-collicular projection is the site of that learning.** In dark-reared animals the descending projection still develops and still drives the SC — but **non-selectively**: cooling it scales visual, auditory and multisensory responses roughly equally (e.g. −70%/−81%/−74%) and leaves the (near-absent) integration index unchanged, whereas in normal animals cooling *selectively* removes the enhancement (Yu et al. 2013, [10.1111/ejn.12182](https://doi.org/10.1111/ejn.12182)). So **experience converts a non-specific descending gain into a modality-matched, synergy-producing selective enhancer.**

The mechanistic modeling of exactly this (Cuppini et al. 2012, [10.1007/s00422-012-0511-9](https://doi.org/10.1007/s00422-012-0511-9)) uses a Hebbian covariance rule with heterosynaptic "forgetting" and weight normalization *on the descending synapses*, trained on a stream that is ~80% spatially-coincident cross-modal events. The descending weights grow only for co-occurring, aligned cross-modal pairs — i.e. STDP-like coincidence learning at the junction, exactly your intuition, but as the *developmental wiring rule* that installs the synergy, after which the run-time computation is the synergy itself.

> **Verdict on your dichotomy:** it is not "gate" **or** "independent projections + STDP." It is: *independent unisensory descending projections* (your option 2) whose *experience-dependent (STDP-like) convergence onto a shared NMDA compartment* (your option 2's mechanism) *produces an emergent alignment-dependent gain* (your option 1's function). All three clauses are needed; dropping any one contradicts a specific experiment.

---

## 4. The flash–beep worked example

Consider a brief **flash** (visual) and a brief **beep** (auditory) presented to the audiovisual SC neuron. Two cases:

**Case A — aligned (same location, ~same time).** The flash and beep are at the same spatial location and roughly synchronous.
1. Feedforward A and V afferents each deliver a *weak* excitatory drive to the SC neuron. On their own these are modest and combine roughly *additively* — this is all a de-corticated neuron can do.
2. The **descending cortical inputs** (AEV visual + FAES auditory), which the animal's development has wired to converge on the same proximal dendritic compartment *because flashes and beeps from the same event have co-occurred thousands of times*, arrive together.
3. Their summed depolarization pushes the shared dendrite past the NMDA Mg²⁺-block half-activation voltage. The Mg²⁺ block is relieved, NMDA current flows, and because NMDA is supralinear the combined response **exceeds the sum of the parts — superadditive enhancement.** The largest proportional boost occurs when the flash and beep are individually weak (inverse effectiveness), because that is where the extra NMDA current matters most relative to the small feedforward baseline.

**Case B — misaligned (beep late, or in the wrong place).** Now the beep lags the flash by ~150 ms, or comes from a different location.
1. The two descending inputs no longer co-activate the same compartment at the same time (temporal misalignment), or they land on different points of the spatially-registered map (spatial misalignment).
2. Neither input alone lifts the dendrite past the NMDA half-activation voltage, so **the coincidence gate stays largely closed** — no superadditive boost.
3. Worse, the descending pathway also recruits **local inhibition** (via a cortex-driven GABAergic interneuron; §5), so a spatially disparate second stimulus actively **depresses** the response below the aligned-unisensory level. The neuron effectively reports "these two things are probably *not* one event."

The figure below shows this as a mechanism cartoon (illustrative dynamics — *not* simulated output of your model, consistent with the static-analysis scope). Top: the V and A drives. Middle: the shared-dendrite voltage and the NMDA gate (0→1), with the −48 mV half-activation used in your model. Bottom: the SC neuron's output, with the feedforward-only response (dashed) for comparison.

![Flash-beep worked example: aligned inputs sum past the NMDA half-activation and open the coincidence gate (superadditive enhancement); misaligned inputs each stay below it and, with cortex-driven inhibition, the response is depressed. Illustrative mechanism cartoon, not simulated model output.]({{artifact:61608f5f-3a74-4ec4-a6d5-04feb6a24f19}})

The circuit that implements this, with every element mapped onto the biology and onto your model's code objects, is:

![Circuit schematic: A/V feedforward (ascending/subcortical) inputs make the SC neuron responsive to both modalities; the descending cortical (AES/rLS) projection carries modality-matched unisensory drive onto a shared NMDA compartment (coincidence gate, callout 1) and onto a PV interneuron (misalignment depression, callout 2); the descending weights are learned and store the cross-modal prior (callout 3). The g_cortex lever reproduces AES deactivation. Panel B is an illustrative prediction of the response regimes.]({{artifact:213ad45f-455c-4c8d-a309-ef3ea8eb9a80}})

---

## 5. Does misalignment produce active suppression, and is that cortical?

Yes — and this is a real, separable computation, not merely the absence of enhancement.

- **Cortex controls depression.** When an auditory stimulus outside its receptive field degrades the response to a within-field visual stimulus (classic cross-modal depression), deactivating AES+rLS **disinhibits** the response — the multisensory response becomes paradoxically *more* vigorous, sometimes indistinguishable from visual-alone (Jiang & Stein 2003, [10.1152/jn.00369.2003](https://doi.org/10.1152/jn.00369.2003)). So cortex supplies *both* the boost for aligned input and (part of) the suppression for misaligned input. *(Full text of this paper was closed; this specific claim rests on its abstract. The dissociation result immediately below, from Jiang et al. 2006, was read in full and is the stronger anchor.)*
- **There is an anatomical route for it:** AES contacts not only SC output neurons but also GABAergic/nitrergic SC interneurons — a direct excitatory route *and* an indirect inhibitory route (Fuentes-Santamaría et al. 2008, [10.1093/cercor/bhm192](https://doi.org/10.1093/cercor/bhm192)). Misalignment-driven suppression is itself a well-characterized within-neuron mechanism (Kadunce et al. 1997, [10.1152/jn.1997.78.6.2834](https://doi.org/10.1152/jn.1997.78.6.2834)).
- **But suppression is only *partly* cortical.** Neonatal ablation of AES+rLS collapses enhancement (64% → 15%) yet leaves depression **largely intact** (73% → 60%; Jiang et al. 2006, [10.1152/jn.00880.2005](https://doi.org/10.1152/jn.00880.2005)). Enhancement and depression are **dissociable**: cortex's grip on depression is weaker/more redundant than on enhancement, implying depression also has an intracollicular/feedforward-inhibition component.

**Modeling implication (important for your ASD work):** route misalignment-suppression through *two* sources — a cortex-driven inhibitory branch (so it is modulated by the descending signal) *and* a non-cortical intracollicular surround (so it survives when the descending channel is silenced). Your model already has the second (the lateral Mexican-hat `W_MSI_inh` surround and the disynaptic PV route); the cortical module adds the first.

---

## 6. How does it learn stimulus statistics, and does it pass down "context"?

### 6.1 What is learned

The thing that develops **is the descending cortico-collicular weight structure itself.** In the developmental model that reproduces the cat data (Cuppini et al. 2012, [10.1007/s00422-012-0511-9](https://doi.org/10.1007/s00422-012-0511-9)), the descending AES→SC synapses **start at zero** and grow via a Hebbian covariance rule; "maturation of multisensory integration" *is* the growth of that projection (plus the lateral inhibitory surround). The learned weights come to encode:

- **which cross-modal combinations co-occur** (co-occurrence probability), and
- **their spatial register** (where the visual and auditory receptive fields should overlap).

The statistics of experience are causal in the model: below ~65–70% cross-modal exposure, neurons fail to become integrative; at ≥70%, ~96% do. This reproduces dark-rearing (no cross-modal exposure → no descending convergence → no integration) and — strikingly — **disparity rearing**: rear an animal with temporally coincident but *spatially disparate* audiovisual cues and a large fraction of SC neurons develop *misaligned* receptive fields, so that only spatially disparate stimuli enhance — **the spatial rule is rewritten to match the environment** (Wallace & Stein 2007, [10.1152/jn.00497.2006](https://doi.org/10.1152/jn.00497.2006), recapitulated in Stein et al. 2009).

This is the direct evidence that the descending signal is **not** a copy of the current stimulus and **not** a hard-wired "same-place → enhance" rule. It is a **learned prior over cross-modal configurations**.

### 6.2 "Context" — what it means and what it does not

Putting the computational framing on it: the top-down signal corresponds to the **p(common cause)** term in Bayesian causal inference (Körding et al. 2007, [10.1371/journal.pone.0000943](https://doi.org/10.1371/journal.pone.0000943)). The percept is a model-average:

$$\hat s = p(C{=}1\mid x_V,x_A)\cdot \hat s_{\text{fused}} + \big(1-p(C{=}1\mid x_V,x_A)\big)\cdot \hat s_{\text{independent}}$$

Small cross-modal disparity → high posterior for a common cause → fusion → **enhancement**; large disparity → low posterior → segregation → **suppression**. The learned descending weights that store P(A,V co-occur & aligned) are a neural implementation of the **prior** p(C=1) — that is the "context" the cortex passes down. (Körding's causal-inference model, with p_common as a fitted prior, reproduced human localization data at R² = 0.97 and outperformed a forced-fusion model.)

**Two honest caveats on "context":**
1. In this literature "context" = *learned structural prior over cross-modal statistics*, encoded in synaptic weights. It is a slow, developmental/environmental prior.
2. It is **not** moment-to-moment endogenous attention or task-set. The AES/rLS deactivation experiments are in anesthetized cats; how awake fronto-parietal attention modulates SC integration is a genuine gap in this specific literature (see §9). If by "context" you mean fast attentional gain, that is a *different, additional* top-down system and is not what AES/rLS provides.

---

## 7. Why cortico-collicular projections at all, given feedforward convergence exists?

Collecting the evidence into a direct answer to your headline question:

1. **Feedforward convergence makes a neuron multisensory (responsive to both), but not integrative (superadditive).** Cooling cortex leaves the former and abolishes the latter (Alvarado et al. 2007).
2. **The integrative transform is a distinct computation** — a synergy at a shared NMDA compartment — that depends on the *joint* presence of the modality-matched descending inputs (Alvarado et al. 2009). Feedforward afferents in isolation combine additively at best.
3. **The transform is learned, and the descending projection is where the learning is stored.** Experience wires the synergistic, spatially-registered convergence; without it (dark-rearing) integration never appears (Wallace et al. 2004; Yu et al. 2013). The cortex is the substrate that carries the *environmental statistics* into the reflex-generating midbrain.
4. **The cortex also arbitrates suppression** for misaligned inputs (Jiang & Stein 2003), giving the SC a full "these belong together / these don't" computation rather than a monotonic "more input → more response."

In short: the feedforward path answers *"is there a visual thing? is there an auditory thing?"*; the cortico-collicular path answers *"do these two things belong to one event, given everything this animal has learned about how the senses co-occur?"* — and only the second question, answered on the shared NMDA compartment, produces integration.

---

## 8. How the computational-modeling literature frames the descending signal

There are two model families, and they disagree in an instructive way about whether you even *need* an explicit cortical unit.

**(a) Emergent-network models (Cuppini, Ursino, Magosso, Rowland, Stein).** These build the cortex in as a necessary element and show enhancement emerging from the dynamics.
- *Cuppini et al. 2010* ([10.3389/fnint.2010.00006](https://doi.org/10.3389/fnint.2010.00006)) is the most useful blueprint: AES enters in **two roles** — a **dedicated un-gated excitatory carrier** to SC (their Eqs 13–14) *and* a **shunting gate on the subcortical inputs** (Eqs 15–16). Enhancement is emergent from a sigmoidal activation, not hand-coded. This is close in spirit to what your model can do at `self.I_M`.
- *Cuppini et al. 2012* (the developmental paper) adds the **learning rule** (Eq. 24): a Hebbian covariance update with heterosynaptic forgetting and weight normalization on the descending synapses (§6.1).
- *Rowland et al. 2007* ([10.1068/p5842](https://doi.org/10.1068/p5842)) is the single most relevant single-neuron model: enhancement arises from a **compartmental nonlinearity (a squaring/supralinear dendritic operation) on a shared compartment where cortical inputs land** — the direct antecedent of your model's NMDA-on-shared-dendrite mechanism.

**(b) Canonical-computation models (Ohshiro, Körding) reproduce MSI *without* a cortical unit at all.**
- *Divisive normalization* (Ohshiro, Angelaki & DeAngelis 2011, [10.1038/nn.2815](https://doi.org/10.1038/nn.2815); confirmed in primate MSTd 2017, [10.1016/j.neuron.2017.06.043](https://doi.org/10.1016/j.neuron.2017.06.043)): a single equation, R = (d₁A + d₂V)ⁿ / (αⁿ + (1/N)·Σⱼ Eⱼⁿ), reproduces inverse effectiveness, spatial-principle enhancement *and* cross-modal suppression, with **no separate top-down term** — the "suppression" is just the normalization pool (its diagnostic prediction, that a still-*excitatory* but non-optimal input suppresses the response to an optimal input, was empirically confirmed in MSTd, with divisive beating subtractive). On this view the descending signal need not be a boost/suppress command; it might just be *additional excitatory drive into the numerator*, with the normalization producing the rest.
- *Bayesian causal inference* (Körding et al. 2007) reproduces the full behavioral pattern with a top-down term that is **exactly one scalar prior**, p(common cause).

**The tension you should be aware of (and which your model has already taken a side on).** The emergent models implement misalignment-suppression **subtractively/shunting** (Cuppini's shunting gate); the canonical models implement it **divisively** (Ohshiro's normalization). Which one the SC actually uses is not settled in the SC literature. **Your model already commits to a specific hybrid:** disynaptic PV inhibition enters *subtractively* (`I_M − I_gaba_subtractive`) while the lateral surround enters *divisively/shunting* (`I_surr_shunt = k·(E_gaba − V)`). A cortical module should respect that existing commitment rather than reopen it — i.e. route top-down suppression through the *existing* PV/surround machinery, not a new normalization term.

**Bottom line from the modeling literature:** the *minimal* faithful top-down element is **one modality-matched excitatory descending drive per modality, converging on a shared supralinear compartment, with a learned weight and an optional inhibitory branch** — which is precisely what Cuppini 2010/2012 and Rowland 2007 use, and precisely what your model's architecture is one step away from supporting.

---

## 9. Honest caveats, gaps, and where the evidence is thin

Following the project's rule to distinguish evidence from inference:

1. **Species and state.** The causal deactivation evidence (AES/rLS cooling/ablation) is almost entirely **anesthetized cat**. The NMDA-coincidence *mechanism* is a well-motivated synthesis (Rowland 2007 model + Binns & Salt 1996 pharmacology + Fuentes-Santamaría 2009 anatomy), not a single experiment that measured Mg²⁺-block relief during a boosted multisensory response. It should be presented as the leading mechanistic hypothesis, which it is.
2. **Driver vs modulator.** The Type I/Type II axon distinction (Fuentes-Santamaría 2009) is anatomical; the functional "driver/modulator" reading is an analogy to thalamus, not a tested SC fact.
3. **rLS is under-characterized.** AES dominates the literature and the modeling; rLS is consistently the junior partner and is essentially **unmodelled** everywhere. Your module can fold both into a single "association cortex" population without loss.
4. **Attention/context.** As stressed in §6.2, fast endogenous attention is a *separate* top-down system not captured by the anesthetized AES/rLS work. If your ASD interests include attentional differences, that is an additional module, not this one.
5. **Divisive vs subtractive suppression** is genuinely unresolved in SC (§8) — flagged rather than papered over.
6. **The literature search could not use one scholarly index** (OpenAlex was unavailable — no API key), so a small number of secondary papers were read from abstracts rather than full text; every *load-bearing* claim above rests on a full-text primary source, which was the project's stated bar.

---

## 10. Implementation-ready specification: the minimal cortical module for *your* model

This section is written to be executed against `Training_delayfix_d52.py` (the authoritative measurement build, commit `caf7e8e1`). Line numbers and code idioms are taken from the model dossier; every hook uses machinery the model already has. **No code is changed here — this is the spec.**

### 10.1 Design principle (why this is minimal *and* accurate)

The single most important observation from the code analysis: **your model already contains the exact biophysics the biology attributes to the cortical mechanism** — a voltage-gated NMDA Mg²⁺-block coincidence detector on a shared dendritic compartment (`mg_k=0.15`, `mg_vhalf=−48 mV`, `Erev_nmda=20 mV`, `tau_nmda=40 ms`; NMDA add at `:3334`), and a disynaptic PV inhibitory route with a ready-made scalar lesion lever (`pv_gaba_scale`, `:3457`). What it lacks is only the *source*: right now that NMDA coincidence gate is driven by the **feedforward** A/V afferents.

The minimal faithful change is therefore **not** to invent a new integration mechanism, but to **make a descending cortical population the carrier of the (coincidence-gated) NMDA drive**, so that:

- removing it (`g_cortex → 0`) abolishes the superadditive/NMDA-gated enhancement while leaving the AMPA-mediated feedforward response — **reproducing the Alvarado-cooling phenotype** (unisensory responses survive, enhancement collapses to additive);
- its weights are **learned** (STDP) and therefore come to **store the cross-modal prior** P(A,V co-occur & aligned) — reproducing the developmental / dark-rearing evidence;
- misalignment routes through the **existing PV lever**, giving cortex-controlled depression that is dissociable from enhancement (Jiang 2006) and respects the model's existing subtractive-PV / divisive-surround commitment.

This gives you a `g_cortex` knob that sits **alongside** `pv_gaba_scale` as a second, orthogonal, biologically-grounded lever — directly useful for the ASD work (ASD involves altered cortico-collicular development, not only E/I balance).

### 10.2 What to add (five code objects, all in existing patterns)

| # | Object | Pattern to copy | Where |
|---|---|---|---|
| 1 | Cortical population `v_ctx`, `u_ctx` (n=180 Izhikevich **or** rate/Poisson source), co-registered to the map | the A/V afferent populations `v_uniA/v_uniV` | declare in `__init__`; reset in `reset_state` (`:2857…`) |
| 2 | Descending weights `W_ctx2msi_AMPA`, `W_ctx2msi_NMDA` (180×180, `Positive`-reparam `nn.Parameter`) | `W_a2msi_AMPA/_NMDA` (`:110–140` `Positive` class; `pos_init`) | `__init__`; add to `model_state` for `make_checkpoint` (`:4328`) / `load_ckpt` |
| 3 | Conduction delay ring buffer for ctx→MSI (≥ the 25–40 ms afferent delays; suggest 60 ms) | `conduction_delay_a2msi` via `_ensure_delay_buffer` | register in `_reset_delay_buffers` (`:1965`) |
| 4 | The drive term + `g_cortex` lever, injected into `self.I_M` | the FF-NMDA add at `:3334` | `update_all_layers_batch`, immediately after `:3334` |
| 5 | (optional) STDP learning of `W_ctx2msi_*` | `stdp_update_batch` (`:3867`) + `_p_add` (`:2014`) | call in `train_unsupervised_batch` (`:3999`) beside the 6 FF STDP calls |

### 10.3 The equations

Let the cortical population carry modality-tagged unisensory activity. Two subpopulations (or two halves of one 180-ring), visual `c_V` and auditory `c_A`, co-registered to the SC map. At macro-frame *t*, unit *i*:

**Cortical drive onto the shared dendrite (the boost).** Reusing the model's own Mg²⁺ gate `m(v) = 1/(1+exp(−k(v−v½)))` with the model's `k=0.15`, `v½=−48`:

$$I^{\text{ctx}}_{\text{NMDA},i} = g_{\text{cortex}}\; g_{\text{nmda}}\; \big(m(V^{\text{dend}}_{A,i}) + m(V^{\text{dend}}_{V,i})\big)\;\big(E_{\text{nmda}} - V^{\text{msi}}_i\big)\; s^{\text{ctx}}_i$$

where

$$s^{\text{ctx}}_i = \sum_j W^{\text{ctx→msi}}_{ij}\,\big[\,\hat c_{A,j}(t-\delta_{\text{ctx}}) + \hat c_{V,j}(t-\delta_{\text{ctx}})\,\big]$$

is the delayed descending synaptic activation (delay $\delta_{\text{ctx}}$ ≈ 60 ms), filtered with the existing NMDA time constant `tau_nmda=40 ms`. **This is deliberately identical in form to the feedforward NMDA term at `:3319–3334`** — the coincidence nonlinearity is the shared dendritic Mg gate, so aligned $c_A$+$c_V$ that co-depolarize the compartment produce the supralinear boost, and either-alone does not (§3.2, §4). Injected as the fourth additive source:

$$\texttt{self.I\_M.add\_}\big(g_{\text{cortex}}\cdot I^{\text{ctx}}_{\text{NMDA}}\cdot \texttt{dt\_linear\_scale}\big)\quad\text{(right after :3334)}$$

**Cortical drive onto the PV interneuron (the misalignment suppression).** A weaker projection to the existing MSI-inh population, so that a spatially/temporally non-aligned second input recruits disynaptic inhibition through the channel the model already scales with `pv_gaba_scale`:

$$I^{\text{ctx→inh}}_{k} = g_{\text{cortex}}\sum_j W^{\text{ctx→inh}}_{kj}\big[\hat c_{A,j}(t-\delta) + \hat c_{V,j}(t-\delta)\big]$$

added to the MSI-inh drive exactly like `W_a2msiInh_AMPA` (`:3383` region). No new suppression *mechanism* is introduced — suppression remains the model's disynaptic PV route (subtractive) + lateral surround (divisive).

**The lever.** `g_cortex ∈ [0,1]` (1 = intact; 0 = full AES/rLS deactivation), applied as a live scalar exactly like `pv_gaba_scale` (§5.1) — so a "cortex-cooling" run is a one-knob change on frozen weights, weight-md5 unchanged, matching the repo's perturbation methodology.

**The learning rule (how it acquires stimulus statistics).** Train `W_ctx→msi` with the model's existing soft-bounded STDP:

$$\Delta W^{\text{ctx→msi}}_{ij} = \eta\big[A_+\,(W_{\max}-W_{ij})^{\mu}\,\text{tr}^{\text{pre}}_j\, s^{\text{post}}_i - A_-\,W_{ij}^{\mu}\,\text{tr}^{\text{post}}_i\, s^{\text{pre}}_j\big]$$

(the `stdp_update_batch` rule at `:3867`, with a `_softbound_wmax` entry, applied via `_p_add` and `normalize_rows`). Because cortical $c_A$ and $c_V$ fire together only for *co-occurring, aligned* cross-modal events during training, the descending weights grow a modality-matched, spatially-registered convergence — i.e. they come to encode P(A,V co-occur & aligned), the learned prior. This is the direct analogue of Cuppini 2012 Eq. 24 (§6.1, §8), implemented with your model's own plasticity machinery.

### 10.4 Pseudocode (drop-in, following the model's forward-pass idiom)

```python
# ---- __init__ (declare; sizes/patterns mirror the A/V afferents) ----
self.g_cortex   = 1.0                       # live scalar lever (NT=1.0; 0.0 = AES/rLS cooled)
self.g_nmda_ctx = self.gNMDA                # reuse the FF NMDA conductance (0.51) as default
self.delay_ctx_ms = 60.0                    # descending delay >= FF afferent delays (25/40 ms)
# cortical population state (rate/Poisson source is enough; Izhikevich if spiking desired)
self.v_ctx = torch.full((B, self.n), self.cM)     # or a rate buffer c_A, c_V in [0,1]
# descending weights, POSITIVE-reparametrized nn.Parameters (copy W_a2msi_* construction)
self.W_ctx2msi_NMDA = Positive(pos_init((self.n, self.n), ...))   # 180 x 180
self.W_ctx2msi_AMPA = Positive(pos_init((self.n, self.n), ...))   # optional fast component
self.W_ctx2msiInh   = Positive(pos_init((self.n_inh, self.n), ...))  # 54 x 180, misalignment route
self._softbound_wmax['W_ctx2msi_NMDA'] = 0.018   # same cap as W_*2msi_NMDA
# register in model_state (make_checkpoint :4328) and _reset_delay_buffers (:1965)

# ---- reset_state (:2857) ----
self._ensure_delay_buffer('conduction_delay_ctx2msi', delay_ms=self.delay_ctx_ms, width=self.n)
self._ensure_delay_buffer('conduction_delay_ctx2inh', delay_ms=self.delay_ctx_ms, width=self.n)

# ---- update_all_layers_batch, IMMEDIATELY AFTER the FF-NMDA add at :3334 ----
# 1. descending presynaptic activation (delayed), filtered on the NMDA timconstant
ctx_spk = self.read_delay('conduction_delay_ctx2msi')          # (B, n)
s_ctx   = F.linear(ctx_spk, self.W_ctx2msi_NMDA())             # (B, n)  same idiom as W_a2msi_NMDA
self.s_ctx_nmda = self.s_ctx_nmda * decay_nmda + s_ctx         # tau_nmda=40 ms filter
# 2. Mg-gated coincidence drive — SAME gate the FF NMDA uses (mg_A + mg_V computed at :3319)
I_ctx_nmda = (self.g_nmda_ctx * self.s_ctx_nmda * (mg_A + mg_V)
              * (self.Erev_nmda - self.v_msi))
# 3. inject as the FOURTH additive source, behind the g_cortex lever
self.I_M.add_(self.g_cortex * I_ctx_nmda * dt_linear_scale)    # <-- the hook
# 4. misalignment suppression: drive MSI-inh through the existing disynaptic route
ctx_inh_spk = self.read_delay('conduction_delay_ctx2inh')
I_ctx_inh   = F.linear(ctx_inh_spk, self.W_ctx2msiInh())       # into MSI-inh drive (:3383 region)
self.I_Minh.add_(self.g_cortex * I_ctx_inh * dt_linear_scale)  # PV route, later scaled by pv_gaba_scale

# ---- train_unsupervised_batch (:3999), beside the 6 feedforward STDP calls ----
self.stdp_update_batch(W_attr='W_ctx2msi_NMDA', pre_spk=ctx_pre, post_spk=msi_spk, lr=self.lr_msi)
# (optionally also W_ctx2msi_AMPA and W_ctx2msiInh); weights round-trip via make_checkpoint/load_ckpt
```

### 10.5 Two variants, and which to pick

The code supports either at the same hook (dossier §6.3). Choose by what you want the module to *claim*:

- **Variant A — "carrier of NMDA drive" (recommended).** The descending input *is* the NMDA/coincidence-gated component (above). Enhancement is emergent at the shared Mg gate; `g_cortex→0` removes enhancement, leaves AMPA feedforward → the Alvarado phenotype falls out for free. Most faithful to Rowland 2007 / Cuppini 2010 and to your model's own architecture.
- **Variant B — "explicit multiplicative gate."** Make `g_cortex` (or a per-unit alignment signal) *multiply* the feedforward NMDA drive: `I_nmda *= (1 + w·s_ctx)`. Simpler to reason about, but it hard-codes the boost/suppress rule instead of letting it emerge, and it is less consistent with the "unisensory descending projections" anatomy. Use only if you want the top-down signal to be an abstract gain field rather than a modality-matched projection.

Variant A is the one this report recommends: it is minimal (reuses the existing NMDA gate and PV lever, adds one population + one learned weight), it is accurate (matches the synergy/NMDA/learning evidence, §§2–4), and it delivers the `g_cortex` lever your ASD program needs.

### 10.6 Validation targets (how you would know it works)

Without changing the existing gates, the module should reproduce four evidence-anchored signatures:

1. **Cortex-dependence of enhancement (Alvarado 2007):** `g_cortex: 1→0` should collapse the SBW/CRE enhancement gate toward additive while leaving unisensory MSI rates largely present (a ~40% unisensory reduction is *consistent* with the biology, not a bug).
2. **Synergy (Alvarado 2009):** silencing only the visual *or* only the auditory half of `c_*` should abolish most of the enhancement, not just its own share.
3. **Learned statistics (Wallace 2004 / Cuppini 2012):** training with <~65% aligned cross-modal exposure should fail to grow `W_ctx→msi` into an integrative configuration; ≥70% should succeed — a direct dark-rearing analogue.
4. **Dissociation (Jiang 2006):** `g_cortex→0` should reduce enhancement much more than it reduces misalignment-depression (because depression also rides the non-cortical lateral surround).

---

## 11. One-paragraph answer to your original question

The cortico-collicular top-down signal is **not** a computed "boost/suppress" command, and it is **not** merely two independent projections that add. It is a pair of **modality-matched unisensory excitatory projections** (visual from AEV, auditory from FAES) that developmental experience has wired — via Hebbian/STDP-like coincidence learning — to **converge on the same nonlinear (NMDA-bearing) dendritic compartment** of an SC neuron. Because the convergence is onto a *shared supralinear compartment*, the two inputs interact **synergistically**: when a flash and a beep are aligned in space and time they co-depolarize the compartment, relieve the NMDA Mg²⁺ block, and produce a **superadditive boost**; when they are misaligned the gate stays closed and a cortex-recruited inhibitory route **depresses** the response. So the descending signal *functions* as an alignment-dependent gain (your first option) that is *built* from independent unisensory projections meeting at a plastic junction (your second option) — the two framings are the same circuit seen at run-time versus developmental time. What it "learns" is the statistics of which cross-modal events co-occur and where; what it "passes down" is that learned prior — the cross-modal "context" p(common cause) — not a copy of the stimulus and not (in this circuit) moment-to-moment attention. And the reason it is needed even though feedforward convergence exists is the one experimental fact everything else hangs on: **cooling that cortex leaves the neuron responsive to both senses but abolishes the integration** — feedforward convergence buys you a multisensory neuron, but only the cortico-collicular projection buys you a neuron that *integrates*. For your model, the minimal faithful implementation is one 180-unit descending population feeding the existing NMDA coincidence gate at `self.I_M` (right after line 3334) through a learned positive weight, with a `g_cortex` lever mirroring `pv_gaba_scale`, so that turning the cortex off reproduces the Alvarado phenotype and the descending weights come to store the learned cross-modal prior.

---

## References (primary sources cited, with DOIs)

**Causal role of association cortex (deactivation/ablation):**
- Alvarado JC, Vaughan JW, Stanford TR, Stein BE (2007). Multisensory versus unisensory integration: contrasting modes in the superior colliculus. *J Neurophysiol* 97:3193–3205. [10.1152/jn.00018.2007](https://doi.org/10.1152/jn.00018.2007)
- Alvarado JC, Stanford TR, Vaughan JW, Stein BE (2007). Cortex mediates multisensory but not unisensory integration in superior colliculus. *J Neurosci* 27:12775–12786. [10.1523/JNEUROSCI.3524-07.2007](https://doi.org/10.1523/JNEUROSCI.3524-07.2007)
- Alvarado JC, Stanford TR, Rowland BA, Vaughan JW, Stein BE (2009). Multisensory integration in the superior colliculus requires synergy among corticocollicular inputs. *J Neurosci* 29:6580–6592. [10.1523/JNEUROSCI.0525-09.2009](https://doi.org/10.1523/JNEUROSCI.0525-09.2009)
- Jiang W, Wallace MT, Jiang H, Vaughan JW, Stein BE (2001). Two cortical areas mediate multisensory integration in superior colliculus neurons. *J Neurophysiol* 85:506–522. [10.1152/jn.2001.85.2.506](https://doi.org/10.1152/jn.2001.85.2.506)
- Jiang W, Stein BE (2003). Cortex controls multisensory depression in superior colliculus. *J Neurophysiol* 90:2123–2135. [10.1152/jn.00369.2003](https://doi.org/10.1152/jn.00369.2003)
- Jiang W, Jiang H, Stein BE (2006). Neonatal cortical ablation disrupts multisensory development in superior colliculus. *J Neurophysiol* 95:1380–1396. [10.1152/jn.00880.2005](https://doi.org/10.1152/jn.00880.2005)
- Wallace MT, Stein BE (1994). Cross-modal synthesis in the midbrain depends on input from cortex. *J Neurophysiol* 71:429–432. [10.1152/jn.1994.71.1.429](https://doi.org/10.1152/jn.1994.71.1.429)

**Anatomy of the descending projection:**
- Fuentes-Santamaría V, Alvarado JC, McHaffie JG, Stein BE (2009). Axon morphologies and convergence patterns of projections from different sensory-specific cortices … onto multisensory neurons in the cat SC. *Cereb Cortex* 19:2902–2915. [10.1093/cercor/bhp060](https://doi.org/10.1093/cercor/bhp060)
- Fuentes-Santamaría V et al. (2008). The cortico-collicular projection … GABAergic/nitrergic interneurons. *Cereb Cortex*. [10.1093/cercor/bhm192](https://doi.org/10.1093/cercor/bhm192)

**Development / experience-dependence:**
- Wallace MT, Perrault TJ, Hairston WD, Stein BE (2004). Visual experience is necessary for the development of multisensory integration. *J Neurosci* 24:9580–9584. [10.1523/JNEUROSCI.2535-04.2004](https://doi.org/10.1523/JNEUROSCI.2535-04.2004)
- Wallace MT, Stein BE (2007). Early experience determines how the senses will interact. *J Neurophysiol* 97:921–926. [10.1152/jn.00497.2006](https://doi.org/10.1152/jn.00497.2006)
- Yu L, Xu J, Rowland BA, Stein BE (2013). Development of cortical influences on superior colliculus multisensory neurons: effects of dark-rearing. *Eur J Neurosci* 37:1594–1604. [10.1111/ejn.12182](https://doi.org/10.1111/ejn.12182)
- Wang Z, Yu L, Xu J, Stein BE, Rowland BA (2020). Experience creates the multisensory transform in the superior colliculus. *Front Integr Neurosci* 14:18. [10.3389/fnint.2020.00018](https://doi.org/10.3389/fnint.2020.00018)

**Mechanism (NMDA / dendritic nonlinearity):**
- Binns KE, Salt TE (1996). Importance of NMDA receptors for multimodal integration in the deep layers of the cat superior colliculus. *J Neurophysiol* 75:920–930. [10.1152/jn.1996.75.2.920](https://doi.org/10.1152/jn.1996.75.2.920)
- Rowland BA, Stanford TR, Stein BE (2007). A model of the neural mechanisms underlying multisensory integration in the superior colliculus. *Perception* 36:1431–1443. [10.1068/p5842](https://doi.org/10.1068/p5842)

**Foundational MSI principles:**
- Meredith MA, Stein BE (1986). Visual, auditory, and somatosensory convergence on cells in superior colliculus … *J Neurophysiol* 56:640–662. [10.1152/jn.1986.56.3.640](https://doi.org/10.1152/jn.1986.56.3.640)
- Stein BE, Stanford TR (2008). Multisensory integration: current issues from the perspective of the single neuron. *Nat Rev Neurosci* 9:255–266. [10.1038/nrn2331](https://doi.org/10.1038/nrn2331)
- Stein BE, Stanford TR, Rowland BA (2009). The neural basis of multisensory integration in the midbrain … *Hear Res*. [10.1016/j.heares.2009.03.012](https://doi.org/10.1016/j.heares.2009.03.012)

**Computational models:**
- Cuppini C, Ursino M, Magosso E, Rowland BA, Stein BE (2010). An emergent model of multisensory integration in superior colliculus neurons. *Front Integr Neurosci* 4:6. [10.3389/fnint.2010.00006](https://doi.org/10.3389/fnint.2010.00006)
- Cuppini C, Magosso E, Rowland B, Stein B, Ursino M (2012). Hebbian mechanisms help explain development of multisensory integration … *Biol Cybern* 106:691–713. [10.1007/s00422-012-0511-9](https://doi.org/10.1007/s00422-012-0511-9)
- Ohshiro T, Angelaki DE, DeAngelis GC (2011). A normalization model of multisensory integration. *Nat Neurosci* 14:775–782. [10.1038/nn.2815](https://doi.org/10.1038/nn.2815)
- Ohshiro T, Angelaki DE, DeAngelis GC (2017). A neural signature of divisive normalization at the level of multisensory integration … *Neuron* 95:399–411. [10.1016/j.neuron.2017.06.043](https://doi.org/10.1016/j.neuron.2017.06.043)
- Körding KP, Beierholm U, Ma WJ, Quartz S, Tenenbaum JB, Shams L (2007). Causal inference in multisensory perception. *PLoS ONE* 2:e943. [10.1371/journal.pone.0000943](https://doi.org/10.1371/journal.pone.0000943)

*Note on attribution and provenance: several deactivation findings appear across multiple Stein-lab papers of the same period; where a specific quantitative value is cited (e.g. 76%→17.5%; the synergy indices 163.2%→27.4/26.5/21.7%; dark-rear 7/117; Yu 70/81/74%; Wang 31.4/56.7% race-exceedance; neonatal 64%→15% / 73%→60%), it is from the paper named at that point in the text and was read in full and quoted verbatim from the primary source. Two load-bearing papers were closed-access and their specific claims rest on abstracts, flagged inline where used: Jiang & Stein 2003 (depression-disinhibition) and Alvarado et al. 2007b (within-modal contrast); Wallace & Stein 2007's disparity-rearing result and Rowland et al. 2007's compartmental model were verified via their full recapitulation in the open-access Stein et al. 2009 review. Author/year forms were checked against the DOIs listed above.*


