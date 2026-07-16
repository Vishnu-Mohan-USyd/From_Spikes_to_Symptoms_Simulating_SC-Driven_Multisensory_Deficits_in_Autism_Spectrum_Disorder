# Superseded max-normalised TBW observer (v1)

This directory preserves the lineage of the earlier exploratory seed-42 panel
that reported approximately `+60/+60/+20 ms` changes for adaptation, slow GABA
and NMDA. The v1 observer normalized every trial by its own maximum and treated
a response with at most one detected peak as fused. Under a suppressive lesion,
that rule can mistake a weak surviving unisensory-like response for fusion.

The JSON records are retained under `results/seed42/`; the corresponding panel
is under `figures/`. They are provenance only. Do not use them as current
biological evidence or combine them with corrected-v2 results.

The active observer, acquisitions, results and plotting command are documented
in [`mechanism_influence/README.md`](../../mechanism_influence/README.md#corrected-v2-temporal-fusion-perturbations).
