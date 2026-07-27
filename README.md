# From Spikes to Symptoms

This repository implements the spiking neural network used by Mohan & Rideaux
to study superior-colliculus-driven audiovisual integration and how
assay-local changes to NMDA conductance, feed-forward inhibition, and neuronal
adaptation alter model behaviour.

## Project status and manuscript anchor

The pretrained analysis path is validated and frozen for this release. The
final gate completed on 2026-07-27: seven GPU regression tests passed, all six
standalone assays and the full TBW/SBW pipeline exited successfully, and every
checkpoint hash was unchanged after evaluation.

- Manuscript: [Mohan & Rideaux, _From Spikes to Symptoms: Simulating
  SC-Driven Multisensory Deficits in Autism Spectrum
  Disorder_](manuscript_rev_2.pdf), revision 2, 57 pages.
- PDF SHA-256:
  `9d9f156e802462bac0c238c47f59749302204d8b5bc1b3c3424d4c040d6378eb`.
- PDF metadata timestamp: 2026-05-07 23:42:20 AEST.
- Validated executable baseline:
  `ecadfd4d472cead684c2d49d18657df4c1b6525b`.

The biological phenomenon is the primary acceptance target. Exact equality to
an earlier manuscript number is not required when the current result is valid
and stronger. TBW/SBW perturbations are judged by prespecified direction (or
stability), not by treating model widths as human clinical effect sizes. See
the [validation protocol](docs/validation_protocol.md) and
[research log](docs/research_log.md) for definitions and evidence.

## Architecture and repository map

```text
Training.py                         model and optional training entry point
    │
    └── checkpoint/
        └── msi_model_surr_10_00.pt … msi_model_surr_10_09.pt
            │                       one validated 10-checkpoint ensemble
            ├── assay scripts       E/I, localization, cue weighting,
            │                       inverse effectiveness, latency, Fano
            │       └── Saved_Images/
            ├── generate_all_fresh.py
            │       ├── cache/      pooled TBW/SBW metrics
            │       └── Saved_Images/ (five TBW + five SBW conditions)
            └── tests/              state, metric, timing, and routing gates
```

`Training.py` defines the SC-inspired auditory, visual, multisensory
excitatory, and inhibitory populations; Izhikevich-type dynamics; AMPA, NMDA,
and GABA currents; conduction delays; short-term synaptic dynamics; and
plasticity. The onboarding path uses the supplied checkpoints and does not
retrain them.

> **Run every command from the repository root.** Scripts use relative paths
> such as `checkpoint/`, `fonts/`, `cache/`, and `Saved_Images/`; launching
> them from another directory can produce missing-file errors or write output
> in the wrong place.

## Requirements and installation

- Python 3.10 or newer.
- PyTorch with a build appropriate for the selected CPU/CUDA environment.
- A CUDA-capable device for the validated fast path.

The unpinned dependency list below is derived from imports in the model,
canonical assays, plotting code, and regression tests:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy scipy matplotlib pandas pytest
```

No package versions are pinned in this archive. Install the PyTorch build that
matches the host CUDA driver when GPU execution is required.

### CUDA and headless execution

Choose the physical GPU visible to the process and force a non-interactive
plotting backend on SSH, CI, or other headless hosts:

```bash
export CUDA_VISIBLE_DEVICES=0  # replace 0 with the chosen physical GPU index
export MPLBACKEND=Agg
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Final validation used physical GPU 1, exposed as logical CUDA device 0, on an
NVIDIA RTX A6000. The assay code uses ordinary PyTorch CUDA device strings and
does not contain A6000-specific kernels. The current regression tests do
explicitly assert an A6000, so reproducing the exact seven-test hardware gate
requires one; other supported CUDA devices may run the assays but are outside
the recorded release matrix. CPU support is not uniform across command-line
entry points and was not part of the final validation.

## Quickstart: validated figures in about two minutes

With the environment active, a selected CUDA device, and the repository root
as the working directory, these three checkpoint-based assays took about two
minutes on the validated A6000. Their exact observed total was 2m08s
(1m01s + 35s + 32s); this is a reference measurement, not a runtime guarantee:

```bash
python run_ei_balance.py
python cue_reliability_test.py
python response_latency_test.py
```

Expected headline results are E/I `1.050 ± 0.004`, cue-weighting `R² = .969`,
and AV latency benefit `+7.5 ms` versus the mean unimodal latency (`0.0 ms`
versus the fastest auditory route).

These commands write figures. If generated-file hashes must remain clean, run
the repository in an isolated copy or disposable worktree.

## Canonical analysis commands

Runtimes are wall-clock observations from the final A6000 validation, not
performance guarantees.

| Assay | Canonical command | Coverage | Primary outputs | Validated runtime |
|---|---|---|---|---:|
| E/I balance | `python run_ei_balance.py` | 10 checkpoints; separated currents over a 125 ms evoked window | `Saved_Images/EI_balance.svg`, `.png` | 1m 01s |
| Localization | `python precision_hist_test.py` | 10 checkpoints; control and low-NMDA sensitivity/localization | `Saved_Images/Err_Hist.svg`, `sens_lowNMDA.svg` | 19m 16s |
| Cue reliability | `python cue_reliability_test.py` | 10 checkpoints; reliability combinations, 20 trials each by default | `Saved_Images/cue_rel.svg`; three `cue_reliability_*.png` diagnostics in the root | 35s |
| Inverse effectiveness | `python inverse_effectiveness_test.py` | 10 checkpoints × 6 intensities; fresh state per intensity | `Saved_Images/inv_eff.svg` | 15m 43s |
| Response latency | `python response_latency_test.py` | 10 checkpoints × A/V/AV; fresh state per modality | `Saved_Images/Latency.svg` | 32s |
| Fano dynamics | `python fano_factor_test.py` | 10 checkpoints × 32 seeded trials; physical 10 ms bins | `Saved_Images/fano.svg` | 3m 40s |
| TBW and SBW | `python generate_all_fresh.py` | 5 conditions × 10 checkpoints × 50 trials; 51 temporal offsets and 33 spatial separations | 20 SVG/PNG files in `Saved_Images/`; pooled `.npz` files in `cache/` | 26m 56s |

The full pipeline writes
`Saved_Images/{TBW,SBW}_{control,ff_inhibition,adaptation,nmda,nmda_increase}.{svg,png}`,
`cache/tbw_<condition>.npz`, and
`cache/sbw_<condition>_t1110.npz`.

## Regression gate

The recommended scoped regression command targets the public tests directory
so that historical diagnostic scripts under `debug_dt/` are not collected:

```bash
env CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
```

The final independent A6000 run reported `7 passed in 268.02s (0:04:28)`; its
literal command is preserved in the [research log](docs/research_log.md). The tests
cover canonical E/I routing and state preservation, physical-time Fano
measurement on one and ten checkpoints, inverse-effectiveness direction, and
latency semantics/reporting, plus presynaptic inhibitory-input STP release,
shape, finiteness, and resource bounds.

## Final validated metric matrix

| Phenomenon | Final observation | Primary acceptance summary (full gates in protocol) |
|---|---|---|
| E/I balance | E/I `1.050 ± .004` | Ratio is order one: `[0.5, 2.0]` and within `.15` of 1 |
| Localization precision | σ A/V/AV `8.99°/10.28°/8.26°`; positive AV sensitivity advantage (`+8.8–9.5%` across accepted summaries) | AV sigma below both unimodal sigmas **and** sensitivity improvement over the best unimodal condition ≥5% |
| Cue reliability | `R² = .969`, `MAE = .063`, `RMSE = .079` | `R² ≥ .7`, `MAE < .15`; stronger valid result retained |
| Inverse effectiveness | MEI `.914724 → .030694` from low to high intensity | Low-intensity MEI exceeds high-intensity MEI; no strict pointwise-monotonic requirement |
| Response latency | A/V/AV `26.7/41.7/26.7 ms`; `+7.5 ms` vs mean, `0.0 ms` vs fastest | Positive 5–10 ms mean-unimodal benefit and no faster-than-fastest claim |
| Fano dynamics | Primary active-ROI FF `1.123541 → .783592`; rate `.095922 → .121753`; quenching 10/10, rate rise 8/10 | Active-neuron FF falls by more than `.2`; ensemble rate rises |
| TBW perturbations | `110 ms` control; FF inhibition `158`; adaptation `229`; NMDA reduction `94`; NMDA increase `112` | Wider, wider, narrower, stable: 4/4 pass |
| SBW perturbations | `26.7°` control; FF inhibition `29.0`; adaptation `30.2`; NMDA reduction `13.2`; NMDA increase `30.6` | Wider, wider, narrower, wider: 4/4 pass |

The conventional all-neuron ROI Fano value near `.22` after onset is a
secondary diagnostic. It must not be substituted for the primary
active-neuron fixed ±2σ ROI result `.783592`.

## Reproducibility and state safety

All primary assays use the same ensemble:
`checkpoint/msi_model_surr_10_00.pt` through
`checkpoint/msi_model_surr_10_09.pt`. Evaluation follows
**fresh load → apply assay-local configuration → run → discard**. Configuration
changes, including the control evaluation calibration `gNMDA = 1.30`, are
in-memory only and are never written into checkpoints. Stationary assays also
disable plasticity and freeze feed-forward inhibitory adaptation where their
measurement contract requires it.

The full validation compared SHA-256 hashes for every checkpoint before and
after all commands; 12/12 files were identical. Some stochastic assays use
local seeds; the full TBW/SBW entry point does not pin a top-level NumPy seed,
so small rerun variation is possible while the directional acceptance rule
remains the release criterion.

Analysis commands overwrite or create files in `Saved_Images/`, `cache/`, and,
for cue reliability, the repository root. Use an isolated copy when the
existing generated assets must remain byte-for-byte unchanged.

## Which files are current?

- **Canonical:** `Training.py`, the seven commands in the analysis table,
  `generate_all_fresh.py`, and `tests/`.
- **Compatibility:** `EI_balance_test.py` retains probe APIs and delegates its
  command-line route to `run_ei_balance.py`. `TBW_test.py` and `SBW_test.py`
  provide component APIs used by the canonical five-condition pipeline.
- **Cache-only utilities:** `replot_all_cosmetic.py` and
  `replot_tbw_from_cache.py` only render matching pre-generated cache schemas;
  they do not simulate missing data. Use `generate_all_fresh.py` for a fresh
  canonical TBW/SBW run.
- **Legacy/special-case:** `run_nmda_increase.py` represents an older isolated
  condition and is not an onboarding command. `debug_dt/`, `task*_logs/`, and
  [CHANGES.md](CHANGES.md) are historical engineering records and may contain
  superseded parameters or results.

## Further documentation

- [Biological validation protocol](docs/validation_protocol.md)
- [Research, assumptions, results, and primary sources](docs/research_log.md)
- [Revision-2 manuscript](manuscript_rev_2.pdf)
- [Historical engineering record](CHANGES.md)

## Citation

If this code contributes to published work, cite the accompanying revision-2
manuscript linked above. Author and publication metadata should be updated to
the final bibliographic record if the manuscript is formally published.

Corresponding author: `reuben.rideaux@sydney.edu.au`.

## License

**No project-wide software license is present.** Absence of a license does not
grant permission to copy, modify, or redistribute the code. The files under
`fonts/` have their own license in `fonts/LICENSE.txt`.
