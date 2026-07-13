#!/usr/bin/env python3
"""TEST 4 — Response latency (multisensory facilitation) on route-C dL3-ep79 (SPEC §TEST 4).

Biology: bimodal first-spike latency is SHORTER than the mean unisensory latency
(descriptive Δlatency = (L_A+L_V)/2 − L_B > 0; not a formal Miller test).
Manuscript context: L_A=56±5.2, L_V=54±1.6, L_B=36±1.6 ms;
Δ≈19 ms ≈35% acceleration.

Method (distilled VERBATIM from response_latency_test.py: `measure_latency` :613 + `latency_profile_for_model`
:654): per modality A/V/B, Gaussian pulse @ centre_deg=90, pulse_frames=10, n_frames=40; reset_state;
step the net; first frame where _latest_sMSI.sum()>0 → latency = (t+1)*frame_ms (frame_ms = dt*substeps
= 10 ms); nan if silent. Δlatency = mean(L_A,L_V) − L_B.

Spec divergences honoured:
  §1g.1 — drive with net.sigma_in (=10.0), NOT master's hard-coded 5.0 (recorded in JSON).
  §1g.2 — DO NOT apply master's `net.aM,bM,cM,dM = 0.001,0.2,-60,0.1` Izhikevich override; measure the
          faithful as-built net (control arm only). (Validator asserts neuron params == build values.)
We vectorise n_trials per modality (identical pulse, independent net noise) and report the mean
first-spike latency — a stabler per-seed estimate than a single draw; direction is the bar.

Firewall: net.eval(), plasticity off, single torch.no_grad(); weight bit-identity before==after;
stage_flat TBW/SBW md5 before+after. Local GPU only.

Usage:
  python response_latency_routec.py --seed 42 --device cuda:0     # per-seed JSON + scorecard
  python response_latency_routec.py --aggregate                  # pool -> aggregate JSON + SVG
"""
import os, sys, json, argparse, time, glob
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import routec_net_io as N

METRIC = "response_latency"
CENTRE_DEG = 90.0
INTENSITY = 1.0          # paper-style strong pulse
PULSE_FRAMES = 10
N_FRAMES = 40
DEFAULT_NTRIALS = 16
MODS = ["A", "V", "B"]


@torch.no_grad()
def first_spike_latency(net, modality, sigma_in, n_trials, device, a_onset_frame=0):
    n = net.n
    frame_ms = float(net.dt) * int(net.n_substeps)                 # 0.1 * 100 = 10 ms
    idx_c = int(round(CENTRE_DEG * (n - 1) / (net.space_size - 1)))
    xs = torch.arange(n, dtype=torch.float32, device=device)
    gauss = torch.exp(-0.5 * ((xs - idx_c) / sigma_in) ** 2) * INTENSITY
    B = n_trials
    xA = torch.zeros(N_FRAMES, B, n, device=device)
    xV = torch.zeros_like(xA)
    # task#133 §B2: a_onset_frame delays the AUDITORY pulse so the VISUAL leads by a_onset_frame*frame_ms,
    # aligning A+V arrival at the MSI (A conduction 25ms < V 40ms => ~15ms V-lead is coincidence-optimal).
    # Default 0 = simultaneous onset (the #129 reference; byte-identical to the pre-edit behaviour).
    if modality in ("A", "B"):
        xA[a_onset_frame:a_onset_frame + PULSE_FRAMES] = gauss
    if modality in ("V", "B"):
        xV[:PULSE_FRAMES] = gauss
    net.reset_state(batch_size=B)
    net._fs_probe_on = True                                       # task#114 STEP-1: enable substep first-spike probe
    first = torch.full((B,), -1, dtype=torch.long, device=device)
    for t in range(N_FRAMES):
        net.update_all_layers_batch(xA[t], xV[t])
        s = net._latest_sMSI.sum(dim=1)                            # (B,)
        newly = (first < 0) & (s > 0)
        first = torch.where(newly, torch.full_like(first, t), first)
    # task#114 STEP-1: latency from the substep-resolution probe (0.1ms grid) instead of the 10ms-frame quantiser.
    # _first_spike_substep[b] = global substep idx of trial b's first MSI-population spike (-1 if silent);
    # the per-frame `first` above is kept only as a coarse cross-check. dt=0.1ms -> sub-frame dV/dt advances resolve.
    fs = net._first_spike_substep
    spiked = (fs >= 0)
    lat = (fs.float() + 1.0) * float(net.dt)                      # ms at dt=0.1 resolution (was frame_ms=10ms)
    lat = lat.cpu().numpy()
    lat[~spiked.cpu().numpy()] = np.nan
    mean_lat = float(np.nanmean(lat)) if spiked.any() else float("nan")
    return mean_lat, int(spiked.sum().item()), B, lat.tolist()


@torch.no_grad()
def measure_latency(net, sigma_in, n_trials, device, soa_lead_frames=0):
    out = {}
    for mod in MODS:
        a_onset = soa_lead_frames if mod == "B" else 0     # task#133 §B2: V leads by soa in the BIMODAL only
        ml, nspk, B, lats = first_spike_latency(net, mod, sigma_in, n_trials, device, a_onset_frame=a_onset)
        out[mod] = dict(latency_ms=ml, n_spiked=nspk, n_trials=B, per_trial_ms=lats)
    L_A, L_V, L_B = out["A"]["latency_ms"], out["V"]["latency_ms"], out["B"]["latency_ms"]
    uni_mean = float(np.nanmean([L_A, L_V]))
    delta = uni_mean - L_B
    speedup = (delta / uni_mean * 100.0) if (uni_mean == uni_mean and uni_mean > 0) else float("nan")
    return dict(L_A_ms=L_A, L_V_ms=L_V, L_B_ms=L_B, uni_mean_ms=uni_mean,
                delta_ms=delta, speedup_pct=speedup, per_modality=out,
                B_fastest=bool((L_B == L_B) and (L_B < L_A) and (L_B < L_V)))


@torch.no_grad()
def measure_latency_coincidence(net, sigma_in, n_trials, device, soa_max_frames=3):
    """Measure a coincidence-optimized descriptive condition-mean latency benefit.

    This is not Miller's distributional race-model inequality. The legacy
    ``delta_race_ms`` result key is retained for schema compatibility.

    L_A/L_V are unimodal (SOA-independent, measured once in the base). For the BIMODAL, scan V-lead SOA =
    0..soa_max frames (A delayed; V leads, compensating A's 25ms vs V's 40ms conduction so they arrive together
    at the MSI ~15ms V-lead) and pin the coincidence-optimal = argmin L_B (max facilitation). B_fastest is scored
    there, identically for both arms (single-variable in k). The descriptor is
    min(mean L_A, mean L_V) - mean L_B_opt. The full L_B(SOA)
    curve is reported so the bracket of the ~15ms (1.5-frame, between SOA 10 and 20ms) optimum is visible."""
    frame_ms = float(net.dt) * int(net.n_substeps)
    base = measure_latency(net, sigma_in, n_trials, device, soa_lead_frames=0)
    L_A, L_V = base["L_A_ms"], base["L_V_ms"]
    lb_by_soa = {0: base["L_B_ms"]}                                  # A/V measured once; only B scans SOA
    for s in range(1, int(soa_max_frames) + 1):
        ml, _, _, _ = first_spike_latency(net, "B", sigma_in, n_trials, device, a_onset_frame=s)
        lb_by_soa[s] = ml
    valid = {s: v for s, v in lb_by_soa.items() if v == v}
    s_opt = min(valid, key=valid.get) if valid else 0
    L_B_opt = lb_by_soa[s_opt]
    uni_min = min(L_A, L_V) if (L_A == L_A and L_V == L_V) else float("nan")
    b_fastest = bool((L_B_opt == L_B_opt) and (uni_min == uni_min) and (L_B_opt < L_A) and (L_B_opt < L_V))
    # Legacy schema name: descriptive condition-mean difference, not formal RMI.
    delta_race = (uni_min - L_B_opt) if (uni_min == uni_min and L_B_opt == L_B_opt) else float("nan")
    return dict(L_A_coin_ms=L_A, L_V_coin_ms=L_V, L_B_coincidence_ms=L_B_opt,
                soa_opt_frames=int(s_opt), soa_opt_ms=float(s_opt * frame_ms),
                L_B_by_soa_ms={float(s * frame_ms): v for s, v in lb_by_soa.items()},
                uni_min_ms=uni_min, delta_race_ms=delta_race, B_fastest_coincidence=b_fastest,
                soa_scan_max_ms=float(int(soa_max_frames) * frame_ms))


def run_seed(args):
    device, seed = args.device, args.seed
    ckpt = args.ckpt or N.ckpt_path_for_seed(seed)
    out_json = args.out_json or os.path.join(N.OUT_DIR, f"{METRIC}_seed{seed}.json")
    os.makedirs(N.OUT_DIR, exist_ok=True)
    t0 = time.time()
    fw_before = N.assert_frozen_readouts("BEFORE")
    net, ld, epoch, g_rec, comment = N.load_ckpt(ckpt, seed, args.n_trials, device)
    sigma_in = float(net.sigma_in)
    # record the faithful Izhikevich MSI params so the validator can assert NO override (§1g.2)
    izh = {k: float(getattr(net, k)) for k in ("aM", "bM", "cM", "dM") if hasattr(net, k)}
    print(f"[{METRIC} s{seed}] dev={device} epoch={epoch} g_rec={g_rec} tau_nmda_inh={float(net.tau_nmda_inh)} "
          f"tau_gaba={float(net.tau_gaba)} sigma_in={sigma_in} izh={izh} "
          f"missing={list(ld.missing_keys)} unexpected={list(ld.unexpected_keys)} plast={net.plasticity_enabled}",
          flush=True)
    snap = N.snapshot_weights(net)
    fp_before = N.weight_fingerprint(net)
    with torch.no_grad():
        m = measure_latency(net, sigma_in, args.n_trials, device)
        coin = measure_latency_coincidence(net, sigma_in, args.n_trials, device, args.soa_max_frames) \
            if getattr(args, "coincidence_scan", False) else None
    fp_after = N.assert_weights_unchanged(net, snap, "AFTER")
    fw_after = N.assert_frozen_readouts("AFTER")
    assert fw_after == fw_before
    rec = dict(metric=METRIC, seed=seed, ckpt=os.path.basename(ckpt), epoch=epoch, g_rec=g_rec,
               device=device, build=N.BUILD, tau_nmda_inh=float(net.tau_nmda_inh),
               tau_gaba=float(net.tau_gaba), sigma_in=sigma_in, n_trials=args.n_trials,
               intensity=INTENSITY, centre_deg=CENTRE_DEG, pulse_frames=PULSE_FRAMES, n_frames=N_FRAMES,
               izh_params=izh, plasticity_enabled=bool(net.plasticity_enabled),
               missing_keys=list(ld.missing_keys), unexpected_keys=list(ld.unexpected_keys),
               weight_fp_before=fp_before, weight_fp_after=fp_after,
               weight_bit_identical=bool(fp_before == fp_after),
               md5_TBW=fw_after[0], md5_SBW=fw_after[1], **m, **(coin or {}))
    json.dump(rec, open(out_json, "w"), indent=1)
    print(f"\n[{METRIC} s{seed}] SCORECARD  ({time.time()-t0:.1f}s)")
    print(f"  L_A={m['L_A_ms']:.1f}  L_V={m['L_V_ms']:.1f}  L_B={m['L_B_ms']:.1f} ms   "
          f"uni_mean={m['uni_mean_ms']:.1f}  Δ={m['delta_ms']:+.1f} ms ({m['speedup_pct']:+.1f}%)")
    print(f"  B_fastest={m['B_fastest']}  spiked A/V/B="
          f"{m['per_modality']['A']['n_spiked']}/{m['per_modality']['V']['n_spiked']}/{m['per_modality']['B']['n_spiked']}"
          f" of {args.n_trials}   (paper L_A=56 L_V=54 L_B=36 Δ=19)")
    if coin is not None:
        curve = "  ".join(f"{k:.0f}ms:{v:.1f}" for k, v in sorted(coin['L_B_by_soa_ms'].items()))
        print(f"  [COINCIDENCE §B2] L_B(V-lead SOA): {curve}")
        print(f"  coincidence-opt SOA={coin['soa_opt_ms']:.0f}ms  L_B*={coin['L_B_coincidence_ms']:.1f}  "
              f"min(mean L_A,mean L_V)={coin['uni_min_ms']:.1f}  fastest-mean benefit={coin['delta_race_ms']:+.1f}ms  "
              f"B_fastest_coincidence={coin['B_fastest_coincidence']}")
    print(f"  -> wrote {out_json}", flush=True)
    return rec


def _msem(vals):
    a = np.asarray(vals, float)
    return float(np.nanmean(a)), (float(np.nanstd(a, ddof=1) / np.sqrt(np.sum(~np.isnan(a)))) if np.sum(~np.isnan(a)) > 1 else 0.0)


def aggregate():
    files = sorted(glob.glob(os.path.join(N.OUT_DIR, f"{METRIC}_seed*.json")))
    assert files, f"no {METRIC}_seed*.json in {N.OUT_DIR}"
    recs = [json.load(open(f)) for f in files]
    seeds = [r["seed"] for r in recs]
    LA_m, LA_s = _msem([r["L_A_ms"] for r in recs])
    LV_m, LV_s = _msem([r["L_V_ms"] for r in recs])
    LB_m, LB_s = _msem([r["L_B_ms"] for r in recs])
    D_m, D_s = _msem([r["delta_ms"] for r in recs])
    SU_m, SU_s = _msem([r["speedup_pct"] for r in recs])
    n_silent = sum(1 for r in recs if not (r["L_A_ms"] == r["L_A_ms"] and r["L_V_ms"] == r["L_V_ms"] and r["L_B_ms"] == r["L_B_ms"]))
    agg = dict(metric=METRIC, seeds=seeds, n_seeds=len(seeds), sigma_in=recs[0]["sigma_in"],
               izh_params=recs[0].get("izh_params", {}),
               L_A_mean=LA_m, L_A_sem=LA_s, L_V_mean=LV_m, L_V_sem=LV_s, L_B_mean=LB_m, L_B_sem=LB_s,
               delta_mean=D_m, delta_sem=D_s, speedup_pct_mean=SU_m, speedup_pct_sem=SU_s,
               B_fastest_all=bool(LB_m < LA_m and LB_m < LV_m), n_silent_seeds=n_silent,
               paper_context="L_A=56 L_V=54 L_B=36 Δ=19ms/35% (direction is the bar)")
    out_json = os.path.join(N.OUT_DIR, f"{METRIC}_aggregate.json")
    json.dump(agg, open(out_json, "w"), indent=1)
    # ---- figure: 3-bar latency + Δ annotation (house style) ----
    plt = N.setup_house_style(base_fontsize=15)
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Auditory", "Visual", "Bimodal"]
    vals = [LA_m, LV_m, LB_m]; errs = [LA_s, LV_s, LB_s]
    cols = ["C0", "C0", "C1"]
    ax.bar(labels, vals, yerr=errs, capsize=4, color=cols, alpha=0.85, edgecolor="k", linewidth=0.8)
    ax.set(ylabel="First-spike latency (ms)",
           title=f"Multisensory latency facilitation — route-C dL3 ep79\nΔ={D_m:.1f}±{D_s:.1f} ms ({SU_m:.0f}% faster)")
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 0.5, f"{v:.0f}", ha="center", va="bottom", fontsize=12)
    N.despine(ax)
    plt.tight_layout()
    svg = os.path.join(N.OUT_DIR, f"{METRIC}.svg")
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(svg.replace(".svg", ".png"), dpi=160, bbox_inches="tight")
    print(f"\n================ {METRIC} AGGREGATE (n={len(seeds)} seeds {seeds}) ================")
    print(f"  L_A={LA_m:.1f}±{LA_s:.1f}  L_V={LV_m:.1f}±{LV_s:.1f}  L_B={LB_m:.1f}±{LB_s:.1f} ms")
    print(f"  Δ={D_m:+.1f}±{D_s:.1f} ms ({SU_m:+.0f}%)  B_fastest={agg['B_fastest_all']}  silent_seeds={n_silent}")
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
    ap.add_argument("--coincidence_scan", action="store_true",
                    help="task#133 §B2: scan V-lead SOA + score B_fastest at the coincidence-optimal (argmin L_B)")
    ap.add_argument("--soa_max_frames", type=int, default=3,
                    help="max V-lead SOA in frames (1 frame=10ms); scans 0..max, brackets the ~15ms optimum")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    else:
        assert args.seed is not None, "need --seed NN (or --aggregate)"
        run_seed(args)


if __name__ == "__main__":
    main()
