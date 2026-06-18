# #108 — Biological audiovisual TBW WIDTH (FWHM): is the model's ~185 ms FWHM in the majority band, or is biology firmly wider?

**Researcher, READ-ONLY, deep-research workflow wf_9fd9791a-926 (104 agents, 2.89M tokens, 6 angles,
21 primary sources fetched, 81 claims → 25 verified → 20 confirmed / 5 KILLED). NEUTRAL framing — NOT
cherry-picked toward 185. Web content treated as untrusted; PMIDs/DOIs carried verbatim from verified
findings — none invented. The `is_temporally_fused` readout is frozen/UNTOUCHABLE; this only judges
whether the converged FWHM is biologically in-band.**

---

## 0 · VERDICT

**~185 ms FWHM is INSIDE the majority central band of biological AV-TBW findings — *provided the
comparison is held apples-to-apples in Gaussian-FWHM-equivalent units*. Biology is NOT firmly wider
under that comparison → Phase-6 WIDTH criterion is MET (GO).**

- **Central FWHM-equivalent band (simple adult AV): ~185–270 ms.** 185 ms sits at its **narrow/lower
  edge**, not its center (the center is closer to ~215–250 ms).
- The literature looks "firmly wider" (~300–500 ms) **only under non-comparable metrics**: speech
  stimuli, lenient 50%-criterion full-widths, youngest-child/developmental data, or the single-neuron
  SC enhancement range. **None of these is a Gaussian FWHM of a simple-stimulus adult behavioral
  curve**, so none contradicts 185 ms apples-to-apples.
- **The single strongest anchor is exact-construct-matched:** Van der Burg, Cass & Alais 2014 report a
  literal **~185 ms full bandwidth (FWHM = 2.355·SD, SD = 79 ms)** for simple AV synchrony — the same
  metric and nearly the same construct as our P(fusion) bell FWHM.
- The lead's reference window — the route-C source paper's **−121/+93 = 215 ms** — also sits inside the
  band; our 185 ms is in fact **slightly narrower** than the source paper's own stimulus window.

**Implication:** the width is already biologically defensible. **Do NOT tune the jitter σ to push the
FWHM wider** — σ is biologically fixed (#99: σ_ΔL≈30 ms), and the resulting 185 ms FWHM lands in-band
as a downstream consequence. Widening toward the ~215–250 center would be optional and must never be
done by metric-chasing.

---

## 1 · Width table (every value converted to an FWHM-equivalent)

| Finding | Population | Metric reported | Value (ms) | Half / full | **FWHM-equiv (ms)** | Primary source |
|---|---|---|---|---|---|---|
| Van der Burg, Cass & Alais 2014 | Human psychophys — AV synchrony search | Gaussian SD=79 → "full bandwidth" | 185 (±93 about PSS) | FULL — *already an FWHM* | **185** | Sci Rep 4:5098 · PMID 24872325 · DOI 10.1038/srep05098 |
| van Wassenhove, Grant & Poeppel 2007 | Human AV-SPEECH (McGurk) | fusion-prevalence span −30→+170 | 200 | FULL (prevalence span ≈ FWHM-approx) | **~200** | Neuropsychologia 45(3):598 · PMID 16530232 · DOI 10.1016/j.neuropsychologia.2006.01.001 |
| Zampini, Guest, Shore & Spence 2005 | Human AV SJ | fitted Gaussian SD ~91–114 ms (in-text 117/137) | — | SD (half-width param) | **~214–268** (up to ~276–323 from in-text SDs) | Percept Psychophys 67(3):531 · PMID 16119399 · DOI 10.3758/BF03193329 |
| Noel, De Niear, Van der Burg & Wallace 2016 | Human AV SJ — lifespan | TWS = Gaussian SD; adult-min 96.88 (~51 yr) | — | SD | **~228** (narrowest adult); youngest SD 222 → ~524 (developmental) | PLOS ONE · DOI 10.1371/journal.pone.0161698 |
| Stevenson & Wallace 2013 — flashbeep | Human AV (simple) | 50%-criterion full-width (25%→75% sigmoid span) | 322 (150 AL + 173 VL) | FULL — **lenient 50% criterion, NOT a 2.355·SD FWHM** | not FWHM-comparable (wider operationalization) | Exp Brain Res 227:249 · PMID 23604624 · DOI 10.1007/s00221-013-3507-3 |
| Stevenson & Wallace 2013 — speech | Human AV-SPEECH | 50%-criterion full-width | 461 | FULL — lenient, speech-specific | not FWHM-comparable | (same) |
| Stevenson & Wallace 2013 — task spread | Human AV | 50%-criterion full-width by task | SJ 420 / PF 425 / TOJ 313 / 2IFC 280 | FULL — lenient criterion | not FWHM-comparable | (same) |
| Meredith, Nemitz & Stein 1987 | **DIRECT cat deep SC neurons** | SOA range of response enhancement (peak-discharge overlap) | window exists; maximal at overlap, decays monotonically, depression at large SOA | full SOA range — *construct ≠ behavioral FWHM* | NOT a Gaussian FWHM | J Neurosci 7(10):3215 · PMID 3668625 · DOI 10.1523/JNEUROSCI.07-10-03215.1987 |
| Wallace & Stein 1997 | **DIRECT cat deep SC neurons** | avg temporal window; minority tail | ~250 central; 500–700 tail | full SOA enhancement range — *NOT behavioral FWHM* | NOT FWHM-comparable | J Neurosci 17(7):2429 · PMID 9065504 · DOI 10.1523/JNEUROSCI.17-07-02429.1997 |

**Apples-to-apples (Gaussian-FWHM) anchors:** 185 (Van der Burg) · ~200 (van Wassenhove speech, narrow
end) · ~214–268 (Zampini) · ~228 (Noel adult-min). → **central band ≈ 185–270 ms; 185 at the narrow edge.**

---

## 2 · Why the wide-looking numbers are NOT a real gap (units are the crux)

The field reports **three non-interchangeable width metrics that differ by ~2×**; conflating them is
the single dominant error this question had to guard against:

1. **Gaussian FWHM = 2.355·SD** — the apples-to-apples object. Simple adult AV ≈ **185–270 ms**.
2. **Lenient 50%-criterion full-widths** (SOA span between sigmoid 25%/75% crossings) ≈ **280–461 ms** —
   inherently ~1.5–2× wider than an FWHM by construction; widened further by speech (461) vs flashbeep
   (322). Comparing 185 ms FWHM against these would *falsely* make biology look "firmly wider."
3. **Single-neuron SC SOA-enhancement range** ≈ **250 ms central, 500–700 ms tail** — a different
   construct (range over which one neuron enhances), explicitly **not** a behavioral Gaussian FWHM. The
   adversarial panel **REFUTED (1-2)** the framing of the ~250 ms cat-SC window as an order-of-magnitude
   contradiction of a 185 ms perceptual FWHM.

Held to metric #1, 185 ms is in-band. The "firmly wider" impression evaporates under unit discipline.

---

## 3 · ⚠ HONESTY FLAGS

1. **185 ms is at the NARROW/LOWER edge, not the center.** The center of the apples-to-apples band is
   closer to ~215–250 ms (Zampini ~214–268, Noel adult-min ~228, source-paper 215). 185 is defensible
   and in-band but closer to the floor — report it as "in-band, narrow edge," not "dead center."
2. **The verdict depends entirely on holding the metric to Gaussian FWHM.** If anyone compares our 185
   against a 50%-criterion or single-neuron number, biology will look ~2× wider — that is a **units
   error, not a real FWHM gap.** Flag this in any validation comparison.
3. **A pro-185 argument was itself REFUTED.** The "symmetric window centered on the AV/VA asymmetry
   midpoint puts 185 inside the band" claim was killed (1-2). The verdict rests on the **direct FWHM
   anchors** (Van der Burg 185, van Wassenhove 200, Zampini 214–268, Noel 228) — NOT on the asymmetry
   midpoint.
4. **Asymmetry: only DIRECTION is firmly established** (window extends further for visual-leading /
   auditory-lagging SOAs; PSS shifted to the visual-lead side). Specific per-side magnitudes (e.g. VA
   200–250 / AV 100–150 ms) were **REFUTED (1-2)** — do not rely on exact half-widths. Our model's
   target should preserve the *direction* of asymmetry if it represents it at all.
5. **McGurk ±267 / ~534 ms tolerance REFUTED 0-3** as a secondary mis-citation (Jiang & Bernstein
   citing Massaro/Munhall, not a direct measurement) — excluded; do not cite it as a wide bound.
6. **DIRECT SC gives shape, not an FWHM.** Meredith/Stein establish the SC window EXISTS and is set by
   peak-discharge overlap, but no DIRECT mammalian-SC source yields a legitimately FWHM-convertible
   number — so the in-band verdict rests on HUMAN-PSYCHOPHYSICS, with SC as mechanistic corroboration
   only. (Open question: a half-maximum of the SC enhancement-vs-SOA function could be made
   apples-to-apples — none reported one.)

---

## 4 · Bottom line for Phase-6

- **WIDTH criterion MET (GO).** ~185 ms FWHM is inside the central biological band (~185–270 ms
  FWHM-equivalent); biology is not firmly wider apples-to-apples.
- **No FWHM gap to close** — and therefore **no license to tune σ to widen the bell.** σ stays
  biologically fixed at #99's σ_ΔL≈30 ms; the 185 ms FWHM is its downstream consequence and is in-band.
- If the lead wants the bell nearer the band *center* (~215–250 ms), that is a discretionary biological
  choice, not a correctness requirement — and it must come from a biologically-justified σ, never from
  metric-chasing. The source paper's own 215 ms window is the natural reference if a center target is
  ever wanted.

---

## 5 · Primary sources (verified; no invented IDs)

- Van der Burg, Cass & Alais 2014, *Sci Rep* 4:5098 — PMID 24872325, DOI 10.1038/srep05098 [HUMAN — ~185 ms full bandwidth, simple AV synchrony; exact-construct anchor]
- van Wassenhove, Grant & Poeppel 2007, *Neuropsychologia* 45(3):598-607 — PMID 16530232, DOI 10.1016/j.neuropsychologia.2006.01.001 [HUMAN AV-speech — ~200 ms McGurk fusion window, asymmetric −30/+170]
- Zampini, Guest, Shore & Spence 2005, *Percept Psychophys* 67(3):531-544 — PMID 16119399, DOI 10.3758/BF03193329 [HUMAN AV SJ — FWHM-equiv ~214–268 ms]
- Noel, De Niear, Van der Burg & Wallace 2016, *PLOS ONE* — DOI 10.1371/journal.pone.0161698 [HUMAN AV SJ lifespan — adult-min ~228 ms FWHM-equiv; developmental spread to ~524 ms]
- Stevenson & Wallace 2013, *Exp Brain Res* 227:249-261 — PMID 23604624, DOI 10.1007/s00221-013-3507-3 [HUMAN — 50%-criterion full-widths: flashbeep 322 / tool 317 / speech 461 ms; task spread 280–425 ms; NOT FWHM-comparable]
- Stevenson, Zemtsov & Wallace 2012, *J Exp Psychol HPP* — PMID 22390292, DOI 10.1037/a0027339 [HUMAN — right-side (visual-lead) width correlates with illusion strength; asymmetry direction]
- Cecere, Gross & Thut 2016, *Eur J Neurosci* 43(12):1561-1568 — PMID 27003546, DOI 10.1111/ejn.13242 [HUMAN — asymmetry direction: narrower window when audio leads]
- Meredith, Nemitz & Stein 1987, *J Neurosci* 7(10):3215-3229 — PMID 3668625, DOI 10.1523/JNEUROSCI.07-10-03215.1987 [DIRECT cat deep SC — temporal window exists, peak-discharge-overlap set; shape not FWHM]
- Wallace & Stein 1997, *J Neurosci* 17(7):2429-2444 — PMID 9065504, DOI 10.1523/JNEUROSCI.17-07-02429.1997 [DIRECT cat deep SC — ~250 ms avg window, 500–700 ms minority tail; NOT a behavioral FWHM]

**KILLED (excluded):** VA 200–250 / AV 100–150 ms per-side asymmetry magnitudes (1-2); ~250 ms SC
window as contradiction of 185 ms FWHM (1-2); SJ-preferred-over-TOJ paradigm claim (1-2); McGurk
±267/~534 ms tolerance (0-3, secondary mis-citation).
