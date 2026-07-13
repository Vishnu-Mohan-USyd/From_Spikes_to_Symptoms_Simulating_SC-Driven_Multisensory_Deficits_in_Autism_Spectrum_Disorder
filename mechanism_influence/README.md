# Mechanism-influence analyses

This directory contains two distinct frozen-checkpoint workflows. The original
TBW dose response measures the SOA half-maximum width of a later fused response.
The formal race workflow measures the first MSI-excitatory population spike and
tests Miller's distributional bound. A scalar can affect one endpoint without
affecting the other; see the [scientific overview](../docs/SCIENTIFIC_OVERVIEW.md).

## File map

| Path | Role |
|---|---|
| `tbw_point.py` | One frozen-checkpoint TBW dose point per process |
| `dose/` | Committed seed-42/44 TBW dose-point JSON |
| `DEBUGGER_397_DOSERESPONSE.md` | Historical TBW dose-response report |
| `race_model_measure.py` | Frozen trial-level A/V/AV/catch first-spike acquisition |
| `race_model_analysis.py` | Formal baseline RMI analysis; one record or a checkpoint family |
| `race_model_perturbation_manifest_v1.json` | Declared screen/confirmation design and scalar cells |
| `race_model_perturbation_measure.py` | Isolated perturbation acquisition and ledger utilities |
| `race_model_completed_analysis.py` | Portable CPU-only held-out and adaptation reanalysis |
| [`../results/race_model/gaba_timing_summary.json`](../results/race_model/gaba_timing_summary.json) | Path-independent summary of the archived seed-42 GABA timing diagnostics |
| [`../results/race_model/`](../results/race_model/) | Compact held-out/adaptation summaries, endpoint tables and provenance checksums |

No trial-level race records are committed. They live in the external archive
with ID `race_model_2026-07-13`; its committed description is
[`archive_manifest.json`](../results/race_model/archive_manifest.json). The
archive has 96 checksummed entries; the two full timing diagnostics are
`timing/gaba_timing_trace_seed42.json` and
`timing/gaba_recruitment_probe_seed42.json`.

The frozen historical protocol manifest intentionally retains its original
`selection_source.analysis_path`; that path is acquisition provenance, not a
portable command. The completed analyzer instead accepts every archive input
as an explicit path and writes the compact outputs linked above.

## Baseline acquisition and analysis

Run from the repository root. Each loop iteration starts a fresh process, as
required by the acquisition protocol. The default formal design is shown
explicitly: seven intensities, 1,000 trials per A/V/AV/catch cell, run seed
`20260712`, SOA 0 and a 400 ms horizon.

The archived baseline acquisition hashes are historical and remain tied to
revision `4da0644`. To reproduce those hashes, run the acquisition command from
a clean checkout of that revision and the tie-corrected analysis from revision
`7cbe8d8`. Running the interface from a newer HEAD is a new acquisition and
requires a new versioned protocol/output set; do not repin the archived hashes.
A fresh baseline analysis also embeds checkout provenance, so a newer or dirty
HEAD is not expected to hash-match or pass the historical claim gate even when
its numerical endpoints agree.

```bash
RACE_MODEL_WORK=race_model_reanalysis
mkdir -p "$RACE_MODEL_WORK/baseline/raw"

for seed in {42..51}; do
  CUDA_VISIBLE_DEVICES=0 python -m mechanism_influence.race_model_measure \
    --ckpt "checkpoint/ckpt_ep79_seed${seed}_bs250_delay52_tau18_dL3.pt" \
    --checkpoint-seed "$seed" \
    --label baseline \
    --design-family scalar_rmi_v1 \
    --out "$RACE_MODEL_WORK/baseline/raw/baseline_seed${seed}.json" \
    --device cuda:0 \
    --run-seed 20260712 \
    --n-trials 1000 \
    --intensities 0.05 0.1 0.2 0.4 0.8 1.0 1.6
done

python -m mechanism_influence.race_model_analysis \
  "$RACE_MODEL_WORK"/baseline/raw/baseline_seed{42..51}.json \
  --baseline-label baseline \
  --out "$RACE_MODEL_WORK/baseline/family_analysis.json"
```

The acquisition is GPU work. The analysis command is CPU-only. The archived
reference baseline was acquired at revision `4da0644` and analyzed with the
tie-corrected revision `7cbe8d8`.

## Archived GABA timing diagnostics

The [compact timing summary](../results/race_model/gaba_timing_summary.json)
records the exact low-intensity non-recruitment result, the stronger-intensity
ordering table, hardware, scope, source hashes and limitations. These are
historical, checksummed RTX 5090 diagnostics, not part of the portable CPU
completed-protocol reanalysis. No semantically reviewed path-independent
trace-acquisition runner exists: the archived JSONs retain the bespoke runner
hashes, while those runners depended on historical absolute paths, an exact
GPU UUID and exact source-text instrumentation points. Do not treat this as a
fresh in-repository acquisition recipe.

The evidence is limited to seed 42, one 100-trial substream, reset-state
first-spike measurements at `I=.05`, `.2` and `1`. It establishes causal event
ordering for that endpoint, not across-checkpoint inference, later/TBW timing,
other intensities or biological irrelevance.

## Reanalyze the completed held-out protocol

Set `RACE_MODEL_ARCHIVE` to the external `race_model_2026-07-13` directory.
The command below consumes the archived acquisition ledger, immutable baseline
anchors, raw candidate records and saved reference analysis. It writes only a
compact summary and endpoint table.

```bash
: "${RACE_MODEL_ARCHIVE:?set RACE_MODEL_ARCHIVE to race_model_2026-07-13}"
RACE_MODEL_WORK=race_model_reanalysis
mkdir -p "$RACE_MODEL_WORK/heldout"

python -m mechanism_influence.race_model_completed_analysis heldout \
  --ledger "$RACE_MODEL_ARCHIVE/heldout/acquisition_ledger.json" \
  --raw-dir "$RACE_MODEL_ARCHIVE/heldout/raw" \
  --baseline-dir "$RACE_MODEL_ARCHIVE/baseline/raw" \
  --baseline-family "$RACE_MODEL_ARCHIVE/baseline/family_analysis.json" \
  --reference-analysis "$RACE_MODEL_ARCHIVE/heldout/confirmation_analysis.json" \
  --summary-out "$RACE_MODEL_WORK/heldout/heldout_confirmation_summary.json" \
  --endpoint-table-out "$RACE_MODEL_WORK/heldout/heldout_confirmation_endpoints.csv" \
  --overwrite
```

The completed held-out inference uses checkpoints 43–51 and excludes the
screening checkpoint 42. The expected portable outputs are represented by the
committed [summary](../results/race_model/heldout_confirmation_summary.json)
and [endpoint table](../results/race_model/heldout_confirmation_endpoints.csv).

## Reanalyze the triggered adaptation attribution

This command additionally verifies the seed-42 continuation gate and the exact
bridge from the historical compound adaptation cell to the isolated `aM` cell.

```bash
: "${RACE_MODEL_ARCHIVE:?set RACE_MODEL_ARCHIVE to race_model_2026-07-13}"
RACE_MODEL_WORK=race_model_reanalysis
mkdir -p "$RACE_MODEL_WORK/adaptation"

python -m mechanism_influence.race_model_completed_analysis adaptation \
  --ledger "$RACE_MODEL_ARCHIVE/adaptation/adaptation_attribution_ledger.json" \
  --raw-dir "$RACE_MODEL_ARCHIVE/adaptation/confirmation_raw" \
  --baseline-dir "$RACE_MODEL_ARCHIVE/baseline/raw" \
  --baseline-family "$RACE_MODEL_ARCHIVE/baseline/family_analysis.json" \
  --reference-analysis "$RACE_MODEL_ARCHIVE/adaptation/adaptation_attribution_analysis.json" \
  --summary-out "$RACE_MODEL_WORK/adaptation/adaptation_attribution_summary.json" \
  --endpoint-table-out "$RACE_MODEL_WORK/adaptation/adaptation_attribution_endpoints.csv" \
  --seed42-gate "$RACE_MODEL_ARCHIVE/adaptation/seed42_continuation_gate.json" \
  --trigger-analysis "$RACE_MODEL_ARCHIVE/heldout/confirmation_analysis.json" \
  --bridge-ledger "$RACE_MODEL_ARCHIVE/heldout/acquisition_ledger.json" \
  --bridge-raw-dir "$RACE_MODEL_ARCHIVE/heldout/raw" \
  --overwrite
```

The expected portable outputs are represented by the committed
[summary](../results/race_model/adaptation_attribution_summary.json) and
[endpoint table](../results/race_model/adaptation_attribution_endpoints.csv).
Verify the external archive before either reanalysis:

```bash
(cd "$RACE_MODEL_ARCHIVE" && sha256sum -c SHA256SUMS)
```

## Historical screen analyzer

The branch `phase2-v1-screen-analyzer-hotfix` and annotated tag
`phase2-v1-screen-analyzer-hotfix-v1` preserve the analysis-only hotfix for the
exploratory seed-42 Phase-2 v1 screen. They are historical lineage, not current
main, not the completed checkpoints-43–51 held-out analyzer above, and not
evidence of a completed 120-cell protocol. The archived seed-42 screen remains
selection evidence only.
