# Biological Validation Protocol
## SC-Driven Multisensory Deficits in ASD — Spiking Neural Network Model

**Version:** 1.1 (updated with Researcher cross-reference and run_training override analysis)
**Date:** 2026-03-17
**Author:** Validator Agent

---

## 1. Overview

This document establishes validation protocols for the spiking neural network (SNN) model of the superior colliculus (SC) multisensory integration (MSI) layer, as described in *Mohan & Rideaux — From Spikes to Symptoms*. The model uses Izhikevich-type neurons with explicit AMPA, NMDA, and GABA synaptic currents, Tsodyks-Markram short-term depression, STDP-based learning, and conduction delays to simulate audiovisual integration and ASD-related perturbations.

---

## 2. Biological Properties That MUST Hold

### 2.1 Multisensory Enhancement (MSE)

**Property:** Bimodal (AV) responses must exceed the best unisensory response.

**Literature:**
- Meredith & Stein (1983, 1986): SC neurons show superadditive responses to cross-modal stimuli. Enhancement = (AV - max(A,V)) / max(A,V) > 0.
- Stein & Stanford (2008): Review confirming MSE as a fundamental SC property.
- Wallace et al. (1998): Cat SC deep-layer neurons show 75-120% enhancement for spatially coincident AV stimuli.

**Test:** `precision_hist_test.py` → sensitivity(Bimodal) > max(sensitivity(Audio), sensitivity(Visual))

**Tolerance:**
- MSE index (MEI) > 0 for matched-location, moderate-intensity stimuli (mandatory).
- For low-intensity stimuli: MEI should be substantially higher (inverse effectiveness).
- Bimodal sensitivity (1/sigma) should exceed best unisensory by >= 10% (based on MLE optimality).

**How to test:**
```python
# From precision_hist_test.py
mean_ctrl, sem_ctrl = pool_hybrid_sensitivity_fast(paths)
# mean_ctrl[0]=Audio, [1]=Visual, [2]=Bimodal
assert mean_ctrl[2] > max(mean_ctrl[0], mean_ctrl[1])
```

---

### 2.2 Inverse Effectiveness

**Property:** Multisensory enhancement is inversely related to stimulus intensity — weaker stimuli produce proportionally greater enhancement.

**Literature:**
- Meredith & Stein (1986): Original demonstration in cat SC.
- Stein & Stanford (2008): Review; the principle holds across species and paradigms.
- Holmes & Spence (2005): Behavioral analogue in humans.

**Test:** `inverse_effectiveness_test.py` → MEI monotonically decreases as intensity increases.

**Tolerance:**
- MEI at lowest intensity (0.05 a.u.) > MEI at highest intensity (1.6 a.u.) (mandatory).
- Monotonic decrease in MEI across the 6 intensity levels tested (0.05, 0.1, 0.2, 0.4, 0.8, 1.6).
- Spearman rank correlation between intensity and MEI should be significantly negative (rho < -0.8).

**How to test:**
```python
# From inverse_effectiveness_test.py
mei_mean = mei.mean(0)  # shape (6,) across intensities
assert mei_mean[0] > mei_mean[-1]  # low > high intensity
from scipy.stats import spearmanr
rho, p = spearmanr(INTENSITIES, mei_mean)
assert rho < -0.8 and p < 0.05
```

---

### 2.3 Spatial Rule (Spatial Binding Window)

**Property:** Multisensory integration (fusion) occurs only when stimuli are spatially coincident; P(fusion) decreases monotonically with spatial disparity.

**Literature:**
- Meredith & Stein (1986, 1996): Spatial coincidence rule in cat SC.
- Wallace & Stein (1997): Spatial tuning of multisensory responses in neonatal cat SC.
- Stein & Stanford (2008): Spatial rule as a fundamental principle.

**Test:** `SBW_test.py` → P(fusion) curve shows monotonic decline from 0° to 80° disparity.

**Tolerance:**
- P(fusion) at 0° disparity > 0.8 (mandatory; spatially coincident stimuli should nearly always fuse).
- P(fusion) at 60° disparity < 0.3 (mandatory; widely separated stimuli should rarely fuse).
- 50% fusion threshold (SBW half-width) should fall between 15° and 45° (literature range for SC neurons).
- The pedestal fit should produce a half-width parameter in this range.

**How to test:**
```python
# From SBW_test.py
pooled_ctrl = run_spatial_binding_across_models(model_paths, separations_deg=separations, device="cuda:0")
assert pooled_ctrl["mean_prob"][0] > 0.8   # 0° disparity
assert pooled_ctrl["mean_prob"][-1] < 0.3  # 80° disparity
# Check pedestal half-width
xs_ref, ys_ref, popt = fit_pedestal_curve(pooled_ctrl)
half_width = popt[2]
assert 15 <= half_width <= 45
```

---

### 2.4 Temporal Rule (Temporal Binding Window)

**Property:** Multisensory integration is strongest for temporally coincident stimuli; MSI response (integrated spikes) peaks near zero asynchrony and falls off with temporal disparity.

**Literature:**
- Meredith et al. (1987): Temporal coincidence rule in cat SC.
- Wallace et al. (2004): Temporal binding window in humans ~100-250 ms.
- Stevenson & Wallace (2013): TBW in typical adults ~100-200 ms FWHM for simple stimuli.
- Powers et al. (2009): TBW for audiovisual speech can be wider (~250 ms).

**Test:** `TBW_test.py` → Gaussian fit to integrated MSI spikes vs. AV asynchrony.

**Tolerance:**
- Peak integrated spikes should occur within ±20 ms of zero asynchrony (mandatory).
- Gaussian FWHM should be 80-300 ms (literature range).
- Response at ±500 ms asynchrony should be < 30% of peak response.

**How to test:**
```python
# From TBW_test.py
fit = fit_tbw_curve(offs_ms, mean_int_spikes, model="gaussian")
mu = fit["params"][2]
assert abs(mu) < 20  # peak near zero
fwhm = fit["fwhm"]
assert 80 <= fwhm <= 300
```

---

### 2.5 Cue Reliability Weighting (Bayesian Integration)

**Property:** The decoded location should weight each modality inversely proportional to its spatial uncertainty (sigma). This is the MLE/Bayesian prediction: w_V = (1/sigma_V^2) / (1/sigma_A^2 + 1/sigma_V^2).

**Literature:**
- Ernst & Banks (2002): Optimal cue integration in vision-haptics (MLE framework).
- Alais & Burr (2004): Audiovisual localisation follows MLE weighting.
- Fetsch et al. (2012): Bayesian cue weighting in SC-relevant circuits.

**Test:** `cue_reliability_test.py` → visual weight w_V vs. predicted w_V* across sigma combinations.

**Tolerance:**
- R^2 correlation between predicted and empirical weights >= 0.7 (mandatory).
- Mean absolute error (MAE) between w_V and w_V* < 0.15 (good agreement).
- RMSE < 0.15.

**How to test:**
```python
# From cue_reliability_test.py output
# After pooled = pool_to_mean_sem(all_results)
R2 = np.corrcoef(w_pred_all, w_emp_all)[0, 1] ** 2
assert R2 >= 0.7
MAE = np.mean(mae_list)
assert MAE < 0.15
```

---

### 2.6 E/I Balance

**Property:** The ratio of excitatory to inhibitory currents in the MSI layer should be approximately balanced, consistent with cortical/subcortical recordings.

**Literature:**
- Xue et al. (2014): Mouse V1 E/I ratio ~1.06.
- Okun & Lampl (2008): Mouse V1, tight E/I tracking.
- Wehr & Zador (2003): Rat A1, E/I ratio ~1.09.
- Barral & Bhatt (2016): Mouse S1, E/I ~0.91.
- Priebe & Ferster (2005): Cat V1, E/I ~0.94.

**Test:** `EI_balance_test.py` → E/I ratio from pooled models.

**Tolerance:**
- E/I ratio should be in range [0.5, 2.0] (mandatory; extreme imbalance is non-biological).
- I/E ratio (inhibition-dominant regime) is acceptable for SC deep layers where inhibition can slightly exceed excitation.
- The ratio should be relatively stable across temporal offsets (±200 ms).

**How to test:**
```python
# From EI_balance_test.py
results = pool_ei_fast(model_paths, device="cuda", test_averaged=True)
ei = results["averaged"]["ei_ratio_mean"]
assert 0.5 <= ei <= 2.0
```

---

### 2.7 Fano Factor Dynamics

**Property:** Spike-count variability (Fano factor) should be ~1.0 during baseline (Poisson-like) and decline below 1.0 during stimulus presentation (stimulus-induced variability reduction).

**Literature:**
- Churchland et al. (2010): Fano factor drops from ~1.3 to ~0.6 upon stimulus onset across 14 cortical areas.
- Softky & Koch (1993): Baseline cortical FF ~1.0-1.5.
- Nawrot et al. (2008): FF reduction is a signature of sensory processing.

**Test:** `fano_factor_test.py` → Fano factor time course.

**Tolerance:**
- Baseline Fano factor (pre-stimulus): 0.5 - 2.0 (mandatory).
- Stimulus-evoked Fano factor should be lower than baseline (mandatory).
- The magnitude of FF reduction should be >= 10% relative to baseline.

**How to test:**
```python
# From fano_factor_test.py
t, meanF, semF, meanR, semR, dt = run_fano_factor_test_bio(...)
baseline_ff = meanF[:30].mean()  # first 30 frames = baseline
stim_ff = meanF[30:60].mean()    # frames 30-60 = stimulus
assert 0.5 <= baseline_ff <= 2.0
assert stim_ff < baseline_ff
assert (baseline_ff - stim_ff) / baseline_ff >= 0.1
```

---

### 2.8 Response Latency Reduction

**Property:** Bimodal stimuli should elicit faster MSI responses than unimodal stimuli (latency facilitation).

**Literature:**
- Rowland et al. (2007): Multisensory latency reduction in cat SC of 5-25%.
- Diederich & Colonius (2004): Race model / coactivation accounts for ~20-50 ms speedup.
- Hershenson (1962): Audiovisual reaction times are faster than either modality alone.

**Test:** `response_latency_test.py` → Latency(B) < mean(Latency(A), Latency(V)).

**Tolerance:**
- ΔLatency = UniMean - Bimodal > 0 (mandatory; bimodal must be faster).
- ΔLatency should be 5-50 ms (typical SC range).

**How to test:**
```python
# From response_latency_test.py
df = run_latency_test()
mean_delta = df["ΔLatency_ms"].mean()
assert mean_delta > 0
assert 5 <= mean_delta <= 50
```

---

## 3. ASD Perturbation Predictions

### 3.1 NMDA Hypofunction (gNMDA = 0.02, baseline = 0.05 per run_training)

**Expected effects (based on literature):**
- Wider TBW (reduced temporal precision) — Foss-Feig et al. (2010), Brandwein et al. (2013)
- Narrower or disrupted SBW — Hypothesized from reduced spatial integration
- Reduced multisensory enhancement (lower MEI)
- Increased localization error (lower sensitivity)
- Altered E/I balance (shifted toward less excitation)

**References:**
- Foss-Feig et al. (2010): Extended TBW in ASD children.
- Brandwein et al. (2013): Multisensory integration deficits in ASD.
- Lee et al. (2015): NMDA receptor hypofunction model of ASD.
- Gandal et al. (2012): NMDA receptor changes in ASD postmortem tissue.

### 3.2 Reduced Feedforward Inhibition (g_FFinh scaling)

**Expected effects:**
- Altered E/I balance (core ASD biomarker)
- Potentially broader tuning / reduced spatial selectivity
- Changed TBW shape

**References:**
- Rubenstein & Merzenich (2003): E/I imbalance hypothesis of ASD.
- Gogolla et al. (2009): Inhibitory circuit deficits in ASD mouse models.

### 3.3 Reduced Spike-Frequency Adaptation

**Expected effects:**
- Hyperexcitability, increased spike counts
- Potentially disrupted temporal dynamics

---

## 4. Parameter Cross-Check: Code vs. Literature

### 4.1 Izhikevich Neuron Parameters

| Parameter | Code Value | Expected (Izhikevich 2003) | Status |
|-----------|-----------|---------------------------|--------|
| a (excit RS) | 0.02 | 0.02 | OK |
| b (excit RS) | 0.2 | 0.2 | OK |
| c (excit RS) | -65.0 | -65.0 | OK |
| d (excit RS) | 8.0 | 8.0 | OK |
| a (inhib FS) | 0.1 | 0.1 | OK |
| b (inhib FS) | 0.2 | 0.2 | OK |
| c (inhib FS) | -65.0 | -65.0 | OK |
| d (inhib FS) | 2.0 | 2.0 | OK |
| dt | 0.1 ms | 0.1 ms typical | OK |
| spike threshold | 30.0 mV | 30.0 mV | OK |
| n_substeps | 100 | N/A (model choice) | OK (100 × 0.1 ms = 10 ms frame) |

**Assessment:** All Izhikevich parameters match standard RS/FS classifications from Izhikevich (2003).

### 4.2 NMDA Parameters

**IMPORTANT:** The constructor sets default values, but `run_training()` (line 3264-3273) overrides several NMDA parameters before training. The **run_training overrides** are what trained checkpoints use.

| Parameter | Constructor | run_training Override | Expected Range | Source | Status |
|-----------|------------|----------------------|---------------|--------|--------|
| gNMDA | 0.6 | **0.05** | 0.1-1.0 (relative) | Model-specific | WARNING (very low) |
| tau_nmda | 40.0 ms | **80.0 ms** | 50-150 ms (decay) | Jahr & Stevens (1990) | OK (NR2A range) |
| nmda_alpha | 0.1 | **0.1** | N/A (model) | — | OK |
| mg_k | 0.062 /mV | 0.062 /mV | 0.062 /mV | Jahr & Stevens (1990), Nowak et al. (1984) | OK |
| Erev_nmda | 10.0 mV | **20.0 mV** | 0 mV typical | Dingledine et al. (1999) | **CRITICAL** |
| mg_vhalf | -35.0 mV | **-35.0 mV** | N/A (custom) | Not in standard Jahr-Stevens | **NON-STANDARD** |
| tau_nmdaVolt | 200.0 ms | **100.0 ms** | N/A (model) | — | OK |

**Concerns:**
1. **Erev_nmda = 20.0 mV (run_training override)** is 20 mV above the standard 0 mV reversal potential. This adds substantial extra driving force to NMDA currents, making them unnaturally excitatory. **CRITICAL** — must be justified or corrected.
2. **gNMDA = 0.05 (run_training override)** is very low compared to constructor default of 0.6. Combined with Erev_nmda=20mV, the net effect may partially compensate, but the individual parameters are non-standard.
3. **mg_vhalf = -35 mV** is a non-standard modification to the Mg²⁺ block curve not found in the standard Jahr & Stevens (1990) formulation. Should be documented as an engineering approximation.
4. **tau_nmda = 80 ms (run_training)** corrects the constructor default of 40 ms and IS within biological range for NR2A-containing receptors.

### 4.3 AMPA Parameters & AMPA/NMDA Split

| Parameter | Code Value | Expected Range | Source | Status |
|-----------|-----------|---------------|--------|--------|
| tau_syn (AMPA decay) | 2.5 ms | 1-5 ms | Hestrin et al. (1990) | OK |
| Erev_ampa | 0 mV | 0 mV | Standard | OK |
| gAMPA | 1.0 | Model-specific | — | OK |

**CRITICAL — AMPA/NMDA Weight Split:**
During training, the AMPA/NMDA weight re-normalization (line 2417-2436) enforces a **25% AMPA / 75% NMDA** split:
```python
set_param_weight('W_a2msi_AMPA', 0.25 * W_tot_a)
set_param_weight('W_a2msi_NMDA', 0.75 * W_tot_a)
```
**Literature:** The biological AMPA/NMDA ratio is approximately **80%/20%** (Myme et al. 2003; Lester & Jahr 1992). The code has this **inverted**.
This means 75% of excitatory drive comes through the slow, voltage-dependent NMDA channel, which fundamentally alters integration dynamics (making neurons more sensitive to coincident pre/post activity and membrane potential state). This must be either justified as an intentional modeling choice or corrected.

### 4.4 Tsodyks-Markram STP Parameters

**IMPORTANT:** `run_training()` overrides u_a/u_v from 0.2 to 0.7 (lines 3276-3277).

| Parameter | Constructor | run_training Override | Expected Range | Source | Status |
|-----------|------------|----------------------|---------------|--------|--------|
| tau_rec | 400 ms | 400 ms | 200-800 ms | Tsodyks & Markram (1997) | OK |
| tau_fac | 20 ms | 20 ms | 10-100 ms | Tsodyks & Markram (1997) | OK |
| u_0 (initial) | 0.2 | **0.7** | 0.2-0.5 | Tsodyks & Markram (1997) | **WARNING** |

**Concern:** u_0 = 0.7 (run_training) is substantially higher than biological range (0.2-0.5). This produces very strong short-term depression — 70% of available resources are released on the first spike. This may be intentional to produce rapid adaptation, but should be documented. The high u combined with tau_rec=400ms means recovery from depression is slow, heavily filtering sustained input.

### 4.5 GABA / Inhibition Parameters

**IMPORTANT:** `run_training()` overrides g_FFinh and g_GABA (lines 3284-3285).

| Parameter | Constructor | run_training Override | Expected Range | Status |
|-----------|------------|----------------------|---------------|--------|
| g_GABA | 2.0 | **10.0** | Model-specific | WARNING (5× increase) |
| g_FFinh | 5.0 | **0.6** | Model-specific | WARNING (reduced ~8×) |
| n_inh ratio | 30% | 30% | 20-30% (cortex) | OK |

**Concerns:**
- **g_GABA = 10** (run_training) is 5× the constructor default — very strong lateral/recurrent inhibition.
- **g_FFinh = 0.6** (run_training) is 8× lower than constructor default — weak feedforward inhibition.
- The combination suggests the model relies on strong recurrent (GABA) inhibition rather than feedforward inhibition for E/I balance.
- The 30% inhibitory fraction is at the upper end of cortical estimates but reasonable for SC deep layers.

### 4.5.1 STDP Parameters

| Parameter | Code Value | Expected Range | Source | Status |
|-----------|-----------|---------------|--------|--------|
| tau_post_i | 150.0 ms | ~20 ms | Vogels et al. (2011) | **WARNING** (7.5× slower) |

**Concern:** tau_post_i = 150 ms is 7.5× slower than the Vogels (2011) iSTDP time constant (~20 ms). This dramatically broadens the inhibitory plasticity window, allowing inhibition to track excitation on much longer timescales than biologically expected.

### 4.6 Conduction Delays

**IMPORTANT:** Constructor defaults are 5 substeps, but `run_training()` overrides A→MSI to 250 and V→MSI to 400 substeps (lines 3235-3236).

| Path | Constructor (substeps) | run_training Override | At 0.1ms/substep | Expected | Status |
|------|----------------------|---------------------|------------------|----------|--------|
| A→MSI | 5 | **250** | **25 ms** | ~21 ms (auditory brainstem) | OK (close) |
| V→MSI | 5 | **400** | **40 ms** | ~69 ms (retino-collicular) | **WARNING** (low) |
| Unimodal→MSI_inh | +20 from A/V | +20 from A/V | +2 ms | Longer than direct | OK |
| MSI_inh→MSI_exc | +50 from max(A,V) | +50 from max(A,V) | +5 ms | Model-specific | OK |

**Concerns:**
- **V→MSI delay = 40 ms** is shorter than expected retino-collicular conduction time (~69 ms). This compresses the A-V temporal offset, which could affect TBW shape and the natural auditory-leading advantage.
- **A→MSI delay = 25 ms** is close to biological estimates for auditory brainstem-to-SC conduction (~21 ms).
- The A-V offset is 40-25=15 ms (V arrives 15ms after A), which is biologically plausible but compressed compared to the ~48ms typical offset.

### 4.7 Network Architecture

| Feature | Code Value | Expected | Status |
|---------|-----------|----------|--------|
| n_neurons (default) | 30 (constructor) | N/A (model choice) | See checkpoints |
| space_size | 180° | Auditory azimuth range | OK |
| sigma_in | 5.0 neurons | Receptive field width | OK |
| Architecture | A→MSI, V→MSI, MSI_inh | SC deep layers | OK |

---

## 5. Validation Checklist for All Future Changes

### Pre-merge Checklist

- [ ] **Multisensory Enhancement:** Bimodal sensitivity > best unisensory sensitivity
- [ ] **Inverse Effectiveness:** MEI at low intensity > MEI at high intensity; negative Spearman correlation
- [ ] **Spatial Binding Window:** P(fusion) at 0° > 0.8; P(fusion) at 60° < 0.3; half-width 15-45°
- [ ] **Temporal Binding Window:** Peak within ±20 ms of zero; FWHM 80-300 ms
- [ ] **Cue Reliability:** R² >= 0.7 for w_V vs predicted; MAE < 0.15
- [ ] **E/I Balance:** E/I ratio in [0.5, 2.0]
- [ ] **Fano Factor:** Baseline FF in [0.5, 2.0]; stimulus FF < baseline FF; reduction >= 10%
- [ ] **Response Latency:** ΔLatency > 0 (bimodal faster); ΔLatency in [5, 50] ms
- [ ] **ASD Perturbation:** NMDA hypofunction (gNMDA=0.02) produces measurable deficits relative to control
- [ ] **Parameter Sanity:** All Izhikevich params match RS/FS classifications; NMDA/AMPA/GABA params in literature ranges

### Post-perturbation Checklist (ASD Conditions)

- [ ] NMDA hypofunction → reduced MSE and/or altered SBW/TBW
- [ ] FF inhibition reduction → altered E/I balance
- [ ] Adaptation reduction → changed temporal dynamics
- [ ] All perturbations produce graded, not catastrophic, effects

---

## 6. Prioritized Biological Accuracy Concerns

**NOTE (v1.1):** This section has been updated after cross-referencing Researcher findings. The `run_training()` function (lines 3211-3309) overrides many constructor defaults before training begins. The values below reflect the **actual trained checkpoint parameters**, not constructor defaults.

### Critical (Must Address)

1. **AMPA/NMDA weight split is INVERTED: 25%/75% (code) vs ~80%/20% (biology).** During training (line 2429-2436), AMPA weights are forced to 25% and NMDA to 75% of total. Literature (Myme et al. 2003; Lester & Jahr 1992) reports ~80% AMPA / 20% NMDA at most glutamatergic synapses. This fundamentally changes integration dynamics — the model is NMDA-dominated rather than AMPA-dominated. **Recommendation:** Either justify as intentional (e.g., to enhance coincidence detection) or correct to biologically standard ratio.

2. **Erev_nmda = 20 mV (run_training override, line 3269) vs standard 0 mV.** This is 20 mV above the standard NMDA reversal potential, adding substantial extra driving force. Combined with the inverted AMPA/NMDA split, NMDA currents are both dominant AND stronger per unit than expected. **Recommendation:** Must be justified or corrected.

3. **Hardcoded perturbation in response_latency_test.py (line 685).** The main function modifies Izhikevich parameters directly: `net.aM, net.bM, net.cM, net.dM = 0.001, 0.2, -60.0, 0.1`. This is uncommented in the main path, meaning the "default" latency test runs with non-standard MSI neuron dynamics. **Recommendation:** Verify this is intentional; if testing adaptation perturbation, make it configurable/conditional.

### Warning (Should Document)

4. **u_0 (release probability) = 0.7 (run_training override, lines 3276-3277) vs literature 0.2-0.5.** This produces very strong short-term depression (70% of vesicles released per spike). Combined with tau_rec=400ms, this creates severe depression for sustained inputs. **Recommendation:** Document as intentional calibration choice or adjust to biological range.

5. **tau_post_i = 150 ms vs literature ~20 ms (Vogels 2011).** The inhibitory STDP postsynaptic trace is 7.5× slower than biologically expected. This dramatically broadens the inhibitory plasticity window. **Recommendation:** Document or justify.

6. **V→MSI conduction delay = 40 ms vs expected ~69 ms.** This compresses the natural A-V temporal offset (model: 15 ms; biology: ~48 ms), which could affect TBW shape. **Recommendation:** Document this compression.

7. **g_GABA = 10 (run_training) but g_FFinh = 0.6 (run_training).** Strong recurrent inhibition but weak feedforward inhibition — opposite of typical cortical/SC circuits where feedforward inhibition provides fast temporal control. **Recommendation:** Document rationale for this balance.

8. **mg_vhalf = -35 mV** is a non-standard modification to the Mg²⁺ block curve, not found in the standard Jahr & Stevens (1990) formulation. **Recommendation:** Document as engineering approximation.

9. **No automated validation suite.** The test scripts are analysis/figure-generation scripts, not automated pass/fail tests. There is no pytest or unittest infrastructure. Changes could silently break biological properties. **Recommendation:** Add automated regression tests with the tolerances defined in this document.

### Minor

10. **Code duplication across test files.** `load_msi_model`, `spatial_binding_curve_fast`, and other functions are redefined in nearly every test file. **Recommendation:** Refactor shared utilities into a common module.

11. **Random seed control.** Test scripts do not set global random seeds (except `fano_factor_test.py` which uses `np.random.seed(42)` for synthetic data). **Recommendation:** Add seed control to all test scripts.

12. **Fano factor test uses different checkpoint patterns** in `main_fano_fast()` vs `main_fano_fast_with_bio()`. **Recommendation:** Unify checkpoint naming.

---

## 7. References

1. Alais, D. & Burr, D. (2004). The ventriloquist effect results from near-optimal bimodal integration. *Current Biology*, 14(3), 257-262.
2. Barral, J. & Bhatt, D. (2016). Synaptic scaling rule preserves excitatory-inhibitory balance in barrel cortex. *Nature Neuroscience*, 19, 1690-1696.
3. Brandwein, A. B. et al. (2013). The development of multisensory integration in high-functioning autism. *Journal of Autism and Developmental Disorders*, 43, 2295-2309.
4. Churchland, M. M. et al. (2010). Stimulus onset quenches neural variability. *Nature Neuroscience*, 13, 369-378.
5. Dingledine, R. et al. (1999). The glutamate receptor ion channels. *Pharmacological Reviews*, 51(1), 7-61.
6. Ernst, M. O. & Banks, M. S. (2002). Humans integrate visual and haptic information in a statistically optimal fashion. *Nature*, 415, 429-433.
7. Fetsch, C. R. et al. (2012). Neural correlates of reliability-based cue weighting during multisensory integration. *Nature Neuroscience*, 15, 146-154.
8. Foss-Feig, J. H. et al. (2010). An extended multisensory temporal binding window in autism spectrum disorders. *Experimental Brain Research*, 203, 381-389.
9. Gandal, M. J. et al. (2012). GABAB-mediated rescue of altered excitatory-inhibitory balance, gamma synchrony and behavioral deficits following constitutive NMDAR-hypofunction. *Translational Psychiatry*, 2, e142.
10. Hestrin, S. et al. (1990). Mechanisms generating the time course of dual component excitatory synaptic currents recorded in hippocampal slices. *Neuron*, 5, 247-253.
11. Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on Neural Networks*, 14(6), 1569-1572.
12. Jahr, C. E. & Stevens, C. F. (1990). Voltage dependence of NMDA-activated macroscopic conductances predicted by single-channel kinetics. *Journal of Neuroscience*, 10(9), 3178-3182.
13. Lee, E. et al. (2015). Excitation/inhibition imbalance in animal models of autism spectrum disorders. *Biological Psychiatry*, 77(10), 880-890.
14. Meredith, M. A. & Stein, B. E. (1983). Interactions among converging sensory inputs in the superior colliculus. *Science*, 221(4608), 389-391.
15. Meredith, M. A. & Stein, B. E. (1986). Visual, auditory, and somatosensory convergence on cells in superior colliculus results in multisensory integration. *Journal of Neurophysiology*, 56(3), 640-662.
16. Meredith, M. A. et al. (1987). Determinants of multisensory integration in superior colliculus neurons. I. Temporal factors. *Journal of Neuroscience*, 7(10), 3215-3229.
17. Nowak, L. et al. (1984). Magnesium gates glutamate-activated channels in mouse central neurones. *Nature*, 307, 462-465.
18. Okun, M. & Lampl, I. (2008). Instantaneous correlation of excitation and inhibition during ongoing and sensory-evoked activities. *Nature Neuroscience*, 11, 535-537.
19. Rubenstein, J. L. R. & Merzenich, M. M. (2003). Model of autism: increased ratio of excitation/inhibition in key neural systems. *Genes, Brain and Behavior*, 2, 255-267.
20. Rowland, B. A. et al. (2007). Multisensory integration shortens physiological response latencies. *Journal of Neuroscience*, 27(22), 5879-5884.
21. Stein, B. E. & Stanford, T. R. (2008). Multisensory integration: current issues from the perspective of the single neuron. *Nature Reviews Neuroscience*, 9, 255-266.
22. Stevenson, R. A. & Wallace, M. T. (2013). Multisensory temporal integration: task and stimulus dependencies. *Experimental Brain Research*, 227, 249-261.
23. Tsodyks, M. V. & Markram, H. (1997). The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability. *PNAS*, 94(2), 719-723.
24. Wallace, M. T. & Stein, B. E. (1997). Development of multisensory neurons and multisensory integration in cat superior colliculus. *Journal of Neuroscience*, 17(7), 2429-2444.
25. Wallace, M. T. et al. (1998). Multisensory integration in the superior colliculus of the alert cat. *Journal of Neurophysiology*, 80(2), 1006-1010.
26. Wehr, M. & Zador, A. M. (2003). Balanced inhibition underlies tuning and sharpens spike timing in auditory cortex. *Nature*, 426, 442-446.
27. Xue, M. et al. (2014). Equalizing excitation-inhibition ratios across visual cortical neurons. *Nature*, 511, 596-600.
