#!/usr/bin/env python3
"""TEST 3 — Cue-reliability weighting (MLE cue combination) on route-C dL3-ep79 (SPEC §TEST 3).

Biology: statistically optimal (inverse-variance) cue combination — the visual weight tracks
relative reliability, w_V* = σ_A² / (σ_A² + σ_V²). More reliable (smaller σ) modality gets more
weight. Manuscript context: equal reliability → w_V≈0.51; σ_V=20,σ_A=2 → w_V≈0.04; R²=0.71, MAE=0.13.

Method (distilled VERBATIM from cue_reliability_test.py: `make_dataset` :20 + `reliability_sweep_batched`
:64): σ-grid {2,4,…,20}²; A@80°, V@100°; constant-area Gaussian × gain (σ_ref/σ)^gain_exp (σ_ref=2);
present stim_frames=10; reset_state(B); per frame update_all_layers_batch(..., return_spike_sum=True),
spike_sum += sum_sM (FULL-FRAME MSI spikes); est = decode_msi_location(spike_sum, "com");
w_V = (est−80)/(100−80). Compare to MLE w_pred; R², MAE.

READOUT FIX (debugger P1 — stage_flat/DIAG_P1_IE_CUE.md): an earlier draft decoded net._latest_sMSI (the
LAST 1 of 100 sub-steps), which is empty at weak/wide stimuli ⇒ COM(zeros)=0° ⇒ w_V=(0−80)/20=−4.0 floor —
a READOUT ARTIFACT, not the net. The full-frame sum_sM is robust (60–753 spk/cell at every σ); the
single-variable swap _latest_sMSI→sum_sM rescued R² 0.03→0.67, MAE 1.86→0.26. sum_sM is the net's actual
full-frame response (NOT a tuned knob). A defensive guard sends any profile with <MIN_SPIKES total spikes
to NaN (excluded), so an empty profile can never masquerade as a spurious 0°.

Spec divergence (§1g.3): gain_exp=1 (manuscript Methods gain ∝ σ_ref/σ), NOT master default 2. The
stimulus WIDTHS are the swept reliabilities σ_A/σ_V (the manipulated variable) — net.sigma_in is not a
free knob here; it is recorded for provenance. Faithful as-built net (no neuron overrides).

Firewall: net.eval(), plasticity off, single torch.no_grad(); weight bit-identity before==after;
stage_flat TBW/SBW md5 before+after. Local GPU only.

Usage:
  python cue_reliability_routec.py --seed 42 --device cuda:0     # per-seed JSON + scorecard
  python cue_reliability_routec.py --aggregate                  # pool -> aggregate JSON + SVG
"""
import os, sys, json, argparse, time, glob, math, itertools
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import routec_net_io as N

METRIC = "cue_reliability"
SIGMA_LEVELS = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
AUD_DEG, VIS_DEG = 80.0, 100.0
STIM_FRAMES = 10
BASE_INTENSITY = 30.0
GAIN_EXP = 1                # paper-faithful (master default 2) — §1g.3
DEFAULT_NTRIALS = 20
MIN_SPIKES = 1.0            # guard: a profile with < this many TOTAL MSI spikes can't be localized -> NaN


def w_pred_mle(sigA, sigV):
    """inverse-variance visual weight: σ_A² / (σ_A² + σ_V²) = (1/σ_V²)/(1/σ_A² + 1/σ_V²)."""
    va, vv = sigA ** 2, sigV ** 2
    return (1.0 / vv) / (1.0 / va + 1.0 / vv)


@torch.no_grad()
def make_dataset(net, n_trials, device):
    combos = list(itertools.product(SIGMA_LEVELS, SIGMA_LEVELS))      # 100 combos
    B = len(combos) * n_trials
    T = STIM_FRAMES + 1
    n = net.n
    xA = torch.zeros(B, T, n, device=device)
    xV = torch.zeros_like(xA)
    meta = []
    sigma_ref = min(SIGMA_LEVELS)
    xs = torch.arange(n, dtype=torch.float32, device=device)
    idx_a = int(round(AUD_DEG * (n - 1) / (net.space_size - 1)))
    idx_v = int(round(VIS_DEG * (n - 1) / (net.space_size - 1)))
    for c_idx, (sigA, sigV) in enumerate(combos):
        normA = 1.0 / (sigA * math.sqrt(2 * math.pi))
        normV = 1.0 / (sigV * math.sqrt(2 * math.pi))
        gainA = (sigma_ref / sigA) ** GAIN_EXP
        gainV = (sigma_ref / sigV) ** GAIN_EXP
        gA = BASE_INTENSITY * gainA * normA * torch.exp(-0.5 * ((xs - idx_a) / sigA) ** 2)
        gV = BASE_INTENSITY * gainV * normV * torch.exp(-0.5 * ((xs - idx_v) / sigV) ** 2)
        sl = slice(c_idx * n_trials, (c_idx + 1) * n_trials)
        xA[sl, :STIM_FRAMES] = gA
        xV[sl, :STIM_FRAMES] = gV
        meta.extend([(sigA, sigV)] * n_trials)
    return xA, xV, meta, combos


@torch.no_grad()
def measure_cue(net, n_trials, device):
    xA, xV, meta, combos = make_dataset(net, n_trials, device)
    B, T, n = xA.shape
    net.reset_state(batch_size=B)
    spike_sum = torch.zeros(B, n, device=device)
    for t in range(T):
        *_, sSum = net.update_all_layers_batch(xA[:, t], xV[:, t], return_spike_sum=True)
        spike_sum += sSum                                            # FULL-FRAME MSI spikes (all sub-steps)
    total = spike_sum.sum(dim=1).detach().cpu().numpy()              # (B,) total spikes / trial
    est = N.T.decode_msi_location(spike_sum, space_size=net.space_size, method="com")
    est = est.detach().cpu().numpy() if torch.is_tensor(est) else np.asarray(est)
    wV = (est - AUD_DEG) / (VIS_DEG - AUD_DEG)
    # defensive guard (debugger P1): empty profiles (~0 spikes) -> NaN (excluded), NOT a spurious COM=0 -> w_V floor
    empty = total < MIN_SPIKES
    wV = np.where(empty, np.nan, wV)
    n_empty = int(empty.sum())
    meta_arr = np.array(meta, float)
    sa, sv, wmean, wsem, wpred = [], [], [], [], []
    for (sigA, sigV) in combos:
        mask = (meta_arr[:, 0] == sigA) & (meta_arr[:, 1] == sigV)
        vals = wV[mask]; vals = vals[~np.isnan(vals)]
        sa.append(sigA); sv.append(sigV)
        wmean.append(float(vals.mean()) if vals.size else float("nan"))
        wsem.append(float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
        wpred.append(float(w_pred_mle(sigA, sigV)))
    wmean = np.array(wmean); wpred = np.array(wpred)
    ok = ~np.isnan(wmean)
    r2 = float(np.corrcoef(wpred[ok], wmean[ok])[0, 1] ** 2) if ok.sum() > 1 else float("nan")
    mae = float(np.mean(np.abs(wmean[ok] - wpred[ok]))) if ok.any() else float("nan")
    rmse = float(np.sqrt(np.mean((wmean[ok] - wpred[ok]) ** 2))) if ok.any() else float("nan")
    return dict(sigma_a=sa, sigma_v=sv, w_emp=wmean.tolist(), w_sem=wsem, w_pred=wpred.tolist(),
                r2=r2, mae=mae, rmse=rmse, sigma_levels=list(SIGMA_LEVELS),
                gain_exp=GAIN_EXP, base_intensity=BASE_INTENSITY,
                read_source="sum_sM_full_frame", min_spikes=MIN_SPIKES,
                n_empty_trials=n_empty, n_total_trials=int(B), n_valid_cells=int(ok.sum()))


def _summ(rec):
    """equal-reliability w_V (diagonal mean), corner (σA=2,σV=20), monotonicity along σV (NaN-aware)."""
    sa = np.array(rec["sigma_a"]); sv = np.array(rec["sigma_v"]); we = np.array(rec["w_emp"])
    diag = we[sa == sv]
    eqrel = float(np.nanmean(diag)) if np.any(~np.isnan(diag)) else float("nan")
    cval = we[(sa == 2) & (sv == 20)]
    corner = float(cval[0]) if cval.size else float("nan")
    # monotone non-increasing in σV for each σA (visual down-weighted as it gets less reliable)
    mono = []
    for a in SIGMA_LEVELS:
        row = we[sa == a][np.argsort(sv[sa == a])]
        row = row[~np.isnan(row)]
        mono.append(bool(np.all(np.diff(row) <= 1e-9)) if row.size > 1 else False)
    return eqrel, corner, float(np.mean(mono))


def run_seed(args):
    device, seed = args.device, args.seed
    ckpt = args.ckpt or N.ckpt_path_for_seed(seed)
    out_json = args.out_json or os.path.join(N.OUT_DIR, f"{METRIC}_seed{seed}.json")
    os.makedirs(N.OUT_DIR, exist_ok=True)
    t0 = time.time()
    fw_before = N.assert_frozen_readouts("BEFORE")
    B = len(SIGMA_LEVELS) ** 2 * args.n_trials
    net, ld, epoch, g_rec, comment = N.load_ckpt(ckpt, seed, B, device)
    print(f"[{METRIC} s{seed}] dev={device} epoch={epoch} g_rec={g_rec} tau_nmda_inh={float(net.tau_nmda_inh)} "
          f"tau_gaba={float(net.tau_gaba)} sigma_in={float(net.sigma_in)} gain_exp={GAIN_EXP} B={B} "
          f"missing={list(ld.missing_keys)} unexpected={list(ld.unexpected_keys)} plast={net.plasticity_enabled}",
          flush=True)
    snap = N.snapshot_weights(net)
    fp_before = N.weight_fingerprint(net)
    with torch.no_grad():
        m = measure_cue(net, args.n_trials, device)
    fp_after = N.assert_weights_unchanged(net, snap, "AFTER")
    fw_after = N.assert_frozen_readouts("AFTER")
    assert fw_after == fw_before
    rec = dict(metric=METRIC, seed=seed, ckpt=os.path.basename(ckpt), epoch=epoch, g_rec=g_rec,
               device=device, build=N.BUILD, tau_nmda_inh=float(net.tau_nmda_inh),
               tau_gaba=float(net.tau_gaba), sigma_in=float(net.sigma_in), n_trials=args.n_trials,
               aud_deg=AUD_DEG, vis_deg=VIS_DEG, stim_frames=STIM_FRAMES,
               plasticity_enabled=bool(net.plasticity_enabled),
               missing_keys=list(ld.missing_keys), unexpected_keys=list(ld.unexpected_keys),
               weight_fp_before=fp_before, weight_fp_after=fp_after,
               weight_bit_identical=bool(fp_before == fp_after),
               md5_TBW=fw_after[0], md5_SBW=fw_after[1], **m)
    json.dump(rec, open(out_json, "w"), indent=1)
    eqrel, corner, mono = _summ(rec)
    print(f"\n[{METRIC} s{seed}] SCORECARD  ({time.time()-t0:.1f}s)")
    print(f"  R²={m['r2']:.3f}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}   (paper R²=0.71 MAE=0.13)")
    print(f"  w_V(equal-reliability diag)={eqrel:.3f}  w_V(σA=2,σV=20)={corner:.3f}  "
          f"monotone↓-in-σV frac={mono:.2f}")
    print(f"  read={m['read_source']}  empty_trials={m['n_empty_trials']}/{m['n_total_trials']}  "
          f"valid_cells={m['n_valid_cells']}/{len(SIGMA_LEVELS)**2}")
    print(f"  -> wrote {out_json}", flush=True)
    return rec


def aggregate():
    files = sorted(glob.glob(os.path.join(N.OUT_DIR, f"{METRIC}_seed*.json")))
    assert files, f"no {METRIC}_seed*.json in {N.OUT_DIR}"
    recs = [json.load(open(f)) for f in files]
    seeds = [r["seed"] for r in recs]
    sa = np.array(recs[0]["sigma_a"]); sv = np.array(recs[0]["sigma_v"])
    wpred = np.array(recs[0]["w_pred"])
    W = np.array([r["w_emp"] for r in recs], float)                  # (n_seeds, 100)
    w_m = np.nanmean(W, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w_sem = (np.nanstd(W, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(W), 0))) if W.shape[0] > 1 else np.zeros_like(w_m)
    okp = ~np.isnan(w_m)
    r2_pool = float(np.corrcoef(wpred[okp], w_m[okp])[0, 1] ** 2) if okp.sum() > 1 else float("nan")
    mae_pool = float(np.mean(np.abs(w_m[okp] - wpred[okp]))) if okp.any() else float("nan")
    rmse_pool = float(np.sqrt(np.mean((w_m[okp] - wpred[okp]) ** 2))) if okp.any() else float("nan")
    # per-seed scalar spreads
    r2s = np.array([r["r2"] for r in recs]); maes = np.array([r["mae"] for r in recs])
    n_empty_pool = int(sum(r.get("n_empty_trials", 0) for r in recs))
    n_valid_cells_min = int(min(r.get("n_valid_cells", len(SIGMA_LEVELS) ** 2) for r in recs))
    dvals = w_m[sa == sv]
    eqrel_pool = float(np.nanmean(dvals)) if np.any(~np.isnan(dvals)) else float("nan")
    cvals = w_m[(sa == 2) & (sv == 20)]
    corner_pool = float(cvals[0]) if cvals.size else float("nan")
    mono_rows = []
    for a in SIGMA_LEVELS:
        row = w_m[sa == a][np.argsort(sv[sa == a])]
        row = row[~np.isnan(row)]
        mono_rows.append(bool(np.all(np.diff(row) <= 1e-9)) if row.size > 1 else False)
    agg = dict(metric=METRIC, seeds=seeds, n_seeds=len(seeds), sigma_levels=list(SIGMA_LEVELS),
               read_source=recs[0].get("read_source", "?"), min_spikes=recs[0].get("min_spikes"),
               n_empty_trials_total=n_empty_pool, n_valid_cells_min=n_valid_cells_min,
               r2_pooled=r2_pool, mae_pooled=mae_pool, rmse_pooled=rmse_pool,
               r2_perseed_mean=float(np.nanmean(r2s)), r2_perseed_sem=float(np.nanstd(r2s, ddof=1) / np.sqrt(r2s.size)) if r2s.size > 1 else 0.0,
               mae_perseed_mean=float(np.nanmean(maes)), w_emp_pooled=w_m.tolist(), w_pred=wpred.tolist(),
               sigma_a=sa.tolist(), sigma_v=sv.tolist(),
               w_equal_reliability=eqrel_pool, w_corner_sA2_sV20=corner_pool,
               monotone_decr_in_sigV_frac=float(np.mean(mono_rows)),
               paper_context="w_V≈0.51 equal; ≈0.04 corner; R²=0.71 MAE=0.13 (direction is the bar)")
    out_json = os.path.join(N.OUT_DIR, f"{METRIC}_aggregate.json")
    json.dump(agg, open(out_json, "w"), indent=1)
    # ---- primary figure: clean scatter w_emp vs w_pred + y=x (house style) ----
    plt = N.setup_house_style(base_fontsize=15)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(wpred, w_m, yerr=w_sem, fmt="o", ms=5, capsize=2, color="C0", alpha=0.85,
                label=f"model w$_V$ (mean±SEM, n={len(seeds)})")
    ax.plot([0, 1], [0, 1], "--k", lw=1.2, label="MLE optimal (y=x)")
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02),
           xlabel="Predicted visual weight  w$_V^*$  = σ$_A^2$/(σ$_A^2$+σ$_V^2$)",
           ylabel="Model visual weight  w$_V$",
           title=f"Cue-reliability weighting — route-C dL3 ep79\nR²={r2_pool:.2f}  MAE={mae_pool:.2f}")
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    N.despine(ax)
    plt.tight_layout()
    svg = os.path.join(N.OUT_DIR, f"{METRIC}.svg")
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(svg.replace(".svg", ".png"), dpi=160, bbox_inches="tight")
    # ---- secondary: w_V heat-map (σA × σV) ----
    nlev = len(SIGMA_LEVELS)
    Z = np.full((nlev, nlev), np.nan)
    for k in range(len(sa)):
        i = SIGMA_LEVELS.index(int(sa[k])); j = SIGMA_LEVELS.index(int(sv[k]))
        Z[i, j] = w_m[k]
    fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
    im = ax2.imshow(Z, origin="lower", vmin=0, vmax=1, cmap="viridis")
    ax2.set_xticks(range(nlev)); ax2.set_yticks(range(nlev))
    ax2.set_xticklabels(SIGMA_LEVELS); ax2.set_yticklabels(SIGMA_LEVELS)
    ax2.set(xlabel="σ$_V$ (°)", ylabel="σ$_A$ (°)", title="Model visual weight w$_V$")
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).set_label("w$_V$")
    fig2.tight_layout()
    fig2.savefig(os.path.join(N.OUT_DIR, f"{METRIC}_heatmap.png"), dpi=160, bbox_inches="tight")
    print(f"\n================ {METRIC} AGGREGATE (n={len(seeds)} seeds {seeds}) ================")
    print(f"  read={agg['read_source']}  empty_trials_total={n_empty_pool}  min_valid_cells/seed={n_valid_cells_min}/{len(SIGMA_LEVELS)**2}")
    print(f"  R²(pooled)={r2_pool:.3f}  MAE(pooled)={mae_pool:.3f}  RMSE(pooled)={rmse_pool:.3f}   "
          f"(paper R²=0.71 MAE=0.13)")
    print(f"  per-seed R²={np.nanmean(r2s):.3f}±{(np.nanstd(r2s, ddof=1)/np.sqrt(r2s.size) if r2s.size>1 else 0):.3f}")
    print(f"  w_V(equal)={eqrel_pool:.3f}  w_V(σA=2,σV=20)={corner_pool:.3f}  "
          f"monotone↓-in-σV frac={np.mean(mono_rows):.2f}")
    print(f"  -> {out_json}\n  -> {svg}", flush=True)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_trials", type=int, default=DEFAULT_NTRIALS)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    else:
        assert args.seed is not None, "need --seed NN (or --aggregate)"
        run_seed(args)


if __name__ == "__main__":
    main()
