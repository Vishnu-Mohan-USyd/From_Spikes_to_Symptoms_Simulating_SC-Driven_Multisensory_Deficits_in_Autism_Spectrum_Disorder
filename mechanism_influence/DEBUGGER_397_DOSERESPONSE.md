# DEBUGGER #397 — dm10 FROZEN-weight single-variable TBW dose-response (adaptation / GABA / NMDA)

## QUESTION (from lead/user)
On the FINAL shipped dm10 model, do adaptation, inhibition(GABA), and NMDA EACH still causally move TBW,
or did the TBW fix flatten/remove any of them? Answer = clean single-variable dose-response on FROZEN dm10
weights (inference-time lesion of an EXISTING mechanism — NOT a retrain, NOT a modeling change).

## METHOD (evidence only, no fixes)
- Model FROZEN: primary /tmp/dbg_sfa_isolate/dm10/ckpt_ep79_seed42…pt (md5 8ee948f5); robustness /tmp/dm10_ensemble/ckpt_ep79_seed44…pt
- Harness: /tmp/dbg_lat/tbw_point.py — ONE point per process; VERBATIM drace_leverD load+TBW path; BASE_ENV = dm10 point.
- Measure: TBW raw-FWHM ONLY (0.5×peak width), 3 run-seeds (rs 0,1,2), frozen readout MDC.tbw_fused_counts_jit (TBW_test.py md5 80d33465).
- Firewall md5 80d33465/73b7d136 asserted BEFORE==AFTER; weight_bit_identical asserted per point (overrides are LIVE SCALARS, not weights).
- Single-variable knobs (all live net scalars, overridden AFTER load; verified by the printed EFFECTIVE line):
    adaptation aM (L1489) + dM (L1487)  -> u_msi update L3260/L3266
    GABA conductance scale pv_gaba_scale (L1556) -> I_M_gaba.add_(pv_gaba_scale*I_M_inh2exc) L3205
    GABA decay tau_gaba (L1555) -> gaba_decay=1-dt/tau_gaba L2856
    NMDA gNMDA (L4792 env) -> I_nmda = gNMDA*… L3071

## IMPORTANT CONFIG NOTE — "G_GABA" is a NO-OP env in this net
grep G_GABA/g_gaba/gGABA in Training_delayfix_d52.py => NOTHING. The ENVBLK's G_GABA=5.56 is not consumed by the net.
The mechanism-true, inference-time GABA-conductance scale is pv_gaba_scale (L1556 "disynaptic PV(MSI_inh)->MSI_exc
GABA conductance scale; 1.0 = NT byte-identical"; wired live at L3205). Swept {0, 0.5, 1.0, 1.5} as the faithful
analog of "G_GABA scale" — a scalar multiplier, so NO weight mutation (weight_bit_identical holds). Reported as such.

## HARNESS VALIDITY GATE (run FIRST — a FLAT result is only trustworthy if the harness can move TBW)
- s42_anchor  aM0.02 dM10 pv1.0 tau18 gNMDA0.51 -> TBW [260,240,240] med 240  bit_id=True fw_ok=True
    => reproduces dm10's known TBW ([260,240,240] from #390) EXACTLY. Harness faithful.
- s42_adaptOFF aM0.0 dM0 (all else dm10)         -> TBW [280,300,300] med 300  bit_id=True fw_ok=True
    => adaptation OFF WIDENS TBW 240->300 (+60ms, non-overlapping run-seeds). Override plumbing PROVEN to move TBW.
BOTH validity checks PASS => flats measured below are real (mechanism-removed), not harness artifacts.

## RESULTS — all 15 points COMPLETE. ALL weight_bit_identical=True, ALL firewall md5 intact BEFORE==AFTER.
## Run-seed noise floor: anchor = [260,240,240] => ~20ms run-to-run; effects >~40ms with non-overlapping seeds are REAL.

### 1. ADAPTATION (aM,dM co-varied; all else dm10)  ->  NON-FLAT, span 120 ms
   off       aM0.0   dM0   -> med 300  [280,300,300]
   baseline  aM0.008 dM8   -> med 360  [340,360,400]
   dm10      aM0.02  dM10  -> med 240  [260,240,240]  (anchor)
   Direction: adaptation present & strong (dm10) => NARROWEST TBW (240). Non-monotone: baseline(0.008,8)=local max 360.
   seed44: adaptOFF=300 [280,300,300] == seed42 => SEED-ROBUST.

### 2a. GABA disynaptic-PV conductance (pv_gaba_scale, L3205; all else dm10)  ->  FLAT (span 20 ms ~= noise)  [FLAG]
   pv 0    -> med 260 [260,240,260]      pv 0.5 -> 240 [260,240,240]
   pv 1.0  -> med 240 [260,240,240]      pv 1.5 -> 240 [240,240,240]
   NOTE: pv_gaba_scale lesions ONLY the disynaptic PV->MSI_exc feed-forward GABA (I_M_inh2exc); surround/lateral GABA untouched.
   => GABA CONDUCTANCE AMPLITUDE has negligible effect on TBW-width. seed44 pv0=260 [280,240,260] => seed-robust flat.

### 2b. GABA decay kinetics (tau_gaba, L2856; all else dm10)  ->  NON-FLAT, span 40 ms
   tau 10 -> med 240 [260,240,240]   tau 18 -> 240 (anchor)   tau 40 -> med 280 [280,300,280]
   Direction: SLOWER GABA decay (18->40) WIDENS TBW +40ms (non-overlapping vs anchor). Matches retrained-slow-GABA precedent + biology.

### 3. NMDA (gNMDA, L3071; all else dm10)  ->  NON-FLAT, STRONGEST lever
   gNMDA 0     -> med 600 [600,600,600]  peakP=0.0   *** FUSION BELL COLLAPSES: no fused response; TBW degenerate ***
   gNMDA 0.255 -> med 220 [200,220,220]  peakP=1.0
   gNMDA 0.51  -> med 240 [260,240,240]  peakP=1.0   (anchor)
   gNMDA 0.765 -> med 260 [260,240,260]  peakP=1.0
   Direction: NMDA REQUIRED for fusion to exist (off => peakP 0). Within viable range more NMDA => WIDER TBW (220->260,+40ms).
   seed44 gNMDA0 = 600/peakP0 => seed-robust. (peakP 1.0->0.0 at gNMDA=0 also PROVES the live-scalar override bites.)

### GABA WITNESSES (#398 — close the split airtight; frozen dm10 s42, TBW-only, firewall+bit-id clean)
  AMPLITUDE extreme  pv_gaba_scale=4.0 -> med 240 [260,240,240] peakP1.0  => amplitude TRULY FLAT across [0 -> 4.0], not just [0,1.5]
  TIMING extreme     tau_gaba=60       -> med 300 [300,300,300] peakP1.0  => GRADED/monotone with tau40(280): 10:240 18:240 40:280 60:300 (slower->wider)
  => AIRTIGHT: inhibition moves TBW via TIMING (graded tau_gaba, +60ms over 18->60) NOT amplitude (flat even at 4x).
     "amplitude FLAT" is NOT "inhibition doesn't affect TBW" — inhibition DOES, through its decay kinetics.

## VERDICT — do all three still functionally influence TBW on frozen dm10?  YES for all three; none flattened/removed.
- ADAPTATION: NON-FLAT (120ms). Still the shipped narrowing lever (dm10 aM=0.02 => narrowest).
- NMDA:       NON-FLAT (strongest). Off => fusion collapses (peakP 0); graded => widens TBW.
- GABA:       Still influences TBW via DECAY KINETICS (tau_gaba +40ms), NOT via disynaptic-PV amplitude (pv_gaba FLAT, ~noise). [FLAG the amplitude sub-knob as flat — expected: temporal width is set by inhibition TIMING, not amplitude.]
Plumbing proven: (a) adaptation positive-control moves TBW; (b) gNMDA=0 collapses peakP to 0 (override unambiguously bites); every EFFECTIVE line shows the intended single-variable value; all 15 bit-identical + firewall-clean.
HARNESS FIX (my /tmp harness only, not production): load_ckpt (val36 L128/L148) asserts ENV TAU_GABA/GNMDA == ckpt trained value;
so ENV stays at dm10 trained (18/0.51) and the tau_gaba/gNMDA LESION is applied post-load as a live scalar. 6 points re-run clean.
