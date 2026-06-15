#!/usr/bin/env python
"""ei_routec_run.py — measure E/I balance on ONE route-c ep80 checkpoint.

ADAPTS repo/EI_balance_test.py (which targets the OLD single-net-current
lineage) to the route-c lineage. The old probe extracts
    exc = clamp(I_M, min=0);  inh = -clamp(I_M, max=0)
i.e. the positive/negative parts of a SINGLE net current I_M. In route-c the
net current is (I_M - I_M_gaba) where I_M holds ONLY excitation (AMPA+NMDA,
>=0) and inhibition lives in the SEPARATE slow state I_M_gaba (>=0). So the
old extraction reads inh ~ 0 and yields a garbage ratio. The route-c-correct
definition (per lead) is:
    E/I = <I_M> / <I_M_gaba>   time-averaged during the response.

Same stimulus construction / window as run_ei_probe_with_offset (Gaussian bump,
n_frames=20=200ms, pulse_frames=5), averaged across SOA offsets like
run_ei_probe_averaged. Also reports:
  * component-level E/I from the model's built-in start/stop_ei_recording
    (I_E = AMPA+NMDA clamped ; I_I = Recur+Lat GABA clamped) — secondary x-check.
  * v_rep = mean v_msi over the response window — the representative operating
    voltage used to drive-force-rescale gNMDA when Erev_nmda moves 20->0 mV.

ONE ckpt per process (Rule 6 parallel). Apparatus/Training.py untouched; the
Phase-2 corrected hparams are applied post-load via routec_overrides.
"""
import os, sys, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--eval_app", default="/scratch/eval_apparatus")
ap.add_argument("--code_dir", required=True, help="route-c lineage code dir (defines Training.py + routec_overrides.py)")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--tag", required=True, help="e.g. asymrc_m0")
ap.add_argument("--out_json", required=True)
ap.add_argument("--override_json", default="", help="JSON dict of corrected hparams; empty = baseline")
ap.add_argument("--intensity", type=float, default=1.0, help="stimulus intensity (old EI apparatus default 1.0)")
ap.add_argument("--n_frames", type=int, default=20, help="trial length in 10ms frames (200ms)")
ap.add_argument("--pulse_frames", type=int, default=5)
ap.add_argument("--centre_deg", type=float, default=90.0)
ap.add_argument("--offsets", default="-30,-20,-10,0,10,20,30", help="SOA offsets in frames (10ms each)")
args = ap.parse_args()

sys.path.insert(0, args.eval_app)
sys.path.insert(0, args.code_dir)
import torch
# torch>=2.6 defaults weights_only=True; restore pre-2.6 load for trusted research ckpts.
_orig = torch.load
def _compat(*a, **k):
    k.setdefault("weights_only", False)
    return _orig(*a, **k)
torch.load = _compat
from TBW_test import load_msi_model  # route-c loader (constructor_hparams + mutable_hparams + tau_nmda_inh=25)

device = "cuda" if torch.cuda.is_available() else "cpu"
offsets = [int(x) for x in args.offsets.split(",") if x.strip() != ""]
print("=== EI %s === code_dir=%s device=%s torch=%s offsets=%s" %
      (args.tag, args.code_dir, device, torch.__version__, offsets), flush=True)

net = load_msi_model(args.ckpt, device=device)

# ── Phase-2 inference override ──
override_applied = None
if args.override_json:
    with open(args.override_json) as f:
        cfg = json.load(f)
    from routec_overrides import apply_routec_overrides
    override_applied = apply_routec_overrides(net, cfg)
    print("[OVERRIDE] %s applied=%s" % (
        args.tag, {k: v for k, v in override_applied.items() if k != "live"}), flush=True)
    print("[OVERRIDE] %s LIVE=%s" % (args.tag, override_applied.get("live")), flush=True)


@torch.inference_mode()
def ei_probe_route_c(net, *, offset_steps, centre_deg, sigma_in, pulse_frames,
                     n_frames, intensity):
    """Route-c E/I probe at one SOA. Returns per-frame I_M / I_M_gaba / v_msi
    traces (means over neurons+batch), plus the built-in component recorder."""
    B, N = 1, net.n
    net.reset_state(batch_size=B)
    net.start_ei_recording()

    xA = torch.zeros(n_frames, N, device=net.device)
    xV = torch.zeros_like(xA)
    idx_c = int(round(centre_deg * (N - 1) / (net.space_size - 1)))
    xs = torch.arange(N, dtype=torch.float32, device=net.device)
    gauss = torch.exp(-0.5 * ((xs - idx_c) / sigma_in) ** 2) * intensity

    aud_start = 0 if offset_steps >= 0 else abs(offset_steps)
    vis_start = 0 if offset_steps <= 0 else offset_steps
    xA[aud_start:aud_start + pulse_frames] = gauss
    xV[vis_start:vis_start + pulse_frames] = gauss

    I_M_tr, I_Mg_tr, v_tr, vw_tr = [], [], [], []
    for t in range(n_frames):
        net.update_all_layers_batch(xA[t].unsqueeze(0), xV[t].unsqueeze(0), valid_mask=None)
        iM = net.I_M.detach().clamp(min=0)        # route-c: I_M is purely excitatory
        iMg = net.I_M_gaba.detach().clamp(min=0)  # inhibition magnitude (separate slow state)
        v = net.v_msi.detach()
        I_M_tr.append(iM.mean().item())
        I_Mg_tr.append(iMg.mean().item())
        v_tr.append(v.mean().item())
        # excitatory-current-weighted mean V: the "representative operating V" for
        # the NMDA driving force — weights V by where excitatory (AMPA+NMDA) current
        # actually flows, not the silent majority. Used to rescale gNMDA on Erev 20->0.
        wsum = iM.sum().item()
        vw_tr.append((iM * v).sum().item() / wsum if wsum > 1e-9 else float("nan"))

    rec = net.stop_ei_recording()
    return (np.asarray(I_M_tr), np.asarray(I_Mg_tr), np.asarray(v_tr), np.asarray(vw_tr), rec)


# Response window = onset of the later stimulus .. end (mirrors old apparatus
# which averages over the full n_frames; we also report a tighter post-onset window).
sigma_in = float(getattr(net, "sigma_in", 6.5))
per_off = []
exc_full, inh_full, vrep_vals, vrepw_vals = [], [], [], []
comp_IE, comp_II = [], []
for off in offsets:
    I_M_tr, I_Mg_tr, v_tr, vw_tr, rec = ei_probe_route_c(
        net, offset_steps=off, centre_deg=args.centre_deg, sigma_in=sigma_in,
        pulse_frames=args.pulse_frames, n_frames=args.n_frames, intensity=args.intensity)
    exc_m = float(I_M_tr.mean()); inh_m = float(I_Mg_tr.mean())
    ratio = exc_m / (inh_m + 1e-12)
    # representative operating V over the response window (post later-onset)
    onset = abs(off)
    vseg = v_tr[onset:] if onset < len(v_tr) else v_tr
    vwseg = vw_tr[onset:] if onset < len(vw_tr) else vw_tr
    vrep = float(np.mean(vseg))                       # plain all-neuron mean V
    vrepw = float(np.nanmean(vwseg))                  # excitatory-current-weighted mean V
    # component-level E/I (built-in recorder)
    ie = float(np.mean(rec["I_E_mean"])) if len(rec["I_E_mean"]) else float("nan")
    ii = float(np.mean(rec["I_I_mean"])) if len(rec["I_I_mean"]) else float("nan")
    per_off.append(dict(offset_steps=off, offset_ms=off * 10, exc=exc_m, inh=inh_m,
                        ei_ratio=ratio, v_rep=vrep, v_rep_weighted=vrepw,
                        comp_I_E=ie, comp_I_I=ii,
                        comp_ei_ratio=(ie / (ii + 1e-12)) if ii == ii else float("nan")))
    exc_full.append(exc_m); inh_full.append(inh_m)
    vrep_vals.append(vrep); vrepw_vals.append(vrepw)
    comp_IE.append(ie); comp_II.append(ii)

exc_full = np.asarray(exc_full); inh_full = np.asarray(inh_full)
ei_ratios = exc_full / (inh_full + 1e-12)
# headline: synchronous (SOA=0) ratio + offset-averaged ratio
j0 = offsets.index(0) if 0 in offsets else len(offsets) // 2
ei_sync = float(ei_ratios[j0])
ei_mean = float(ei_ratios.mean())
comp_IE = np.asarray(comp_IE); comp_II = np.asarray(comp_II)
comp_ei_mean = float(np.nanmean(comp_IE / (comp_II + 1e-12)))
v_rep_sync = float(vrep_vals[j0])
v_rep_mean = float(np.mean(vrep_vals))
v_rep_w_sync = float(vrepw_vals[j0])
v_rep_w_mean = float(np.nanmean(vrepw_vals))

out = dict(
    tag=args.tag, ckpt=args.ckpt, override_json=args.override_json, override=override_applied,
    definition="route-c E/I = <I_M> / <I_M_gaba> (exc state vs inhibition-magnitude state), time-avg over response window",
    intensity=args.intensity, n_frames=args.n_frames, pulse_frames=args.pulse_frames,
    centre_deg=args.centre_deg, offsets_frames=offsets,
    ei_ratio_sync=ei_sync, ei_ratio_offsetmean=ei_mean,
    comp_ei_ratio_offsetmean=comp_ei_mean,           # secondary: AMPA+NMDA vs Recur+Lat
    v_rep_sync=v_rep_sync, v_rep_offsetmean=v_rep_mean,
    v_rep_weighted_sync=v_rep_w_sync, v_rep_weighted_offsetmean=v_rep_w_mean,  # I_M-weighted (for gNMDA rescale)
    per_offset=per_off,
    wiring=dict(tau_gaba=float(net.tau_gaba), Erev_nmda=float(net.Erev_nmda),
                gNMDA=float(net.gNMDA), tau_nmda_inh=float(net.tau_nmda_inh),
                msi_inh2exc_ms=float(getattr(net, "conduction_delay_msi_inh2exc_ms", float("nan"))),
                sigma_in=sigma_in, n=int(net.n)),
)
json.dump(out, open(args.out_json, "w"), indent=1)
print("[EI] %s  E/I(sync)=%.3f  E/I(offmean)=%.3f  comp-E/I=%.3f  v_rep(sync)=%.1fmV  v_rep_w(sync)=%.1fmV -> %s" %
      (args.tag, ei_sync, ei_mean, comp_ei_mean, v_rep_sync, v_rep_w_sync, args.out_json), flush=True)
print("EI_PROBE_DONE", flush=True)
