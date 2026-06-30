#!/usr/bin/env python3
"""DEBUGGER route-C — #53 divisive-surround cheap-prove SCORECARD grader (lead's 3 asks).

Grades the RETRAINED #53 ep79 ckpt (gate#1 + mg_vhalf_inh=-30 + divisive surround-shunt) on:

  [1] POP-IE MONOTONIC?  — validator TEST1 (IE.measure_ie), whole-population MEI(I). Monotone-decreasing
      = GO (inverse-effectiveness biology: enhancement greatest at WEAK drive). A mid-peak (inverted-U)
      = the residual subtractive-threshold artifact we are trying to kill.

  [2] g_rec mid-peak SUBSUMED or PERSISTS? — re-measure TEST1 IE with net.g_rec at trained vs 0.0 (shunt
      ON in both). g_rec gates the recurrent MSI->MSI excitation (Training L3115 `if self.g_rec!=0.0`,
      scaled into I_M at L3133). Clean single-variable inference KO on co-adapted weights:
        both monotone                  => surround SUBSUMES g_rec's recruitment role (g_rec inert for shape)
        trained mid-peaks, 0 monotone  => g_rec still INDEPENDENTLY mid-peaks (NOT subsumed)
        trained monotone, 0 mid-peaks  => surround alone insufficient, g_rec is HELPING

  [3] E-I in [0.80,1.25]? — SHUNT-AWARE membrane ratio. The validator's stock primary (measure_ei ->
      ei_ratio_sync = <I_M>/<I_M_gaba>, ei_probe_route_c L210-211) is NOT shunt-aware: under
      gaba_shunt_surr=True the surround leaves I_M_gaba for the separate I_surr_shunt term
      (Training L3080/L3096-99: surround -> I_M_gaba_surr_sh, NOT I_M_gaba), so the stock denominator
      OMITS the surround inhibition and OVER-reports E/I. The membrane integrates
      dVM += I_M - I_M_gaba + I_surr_shunt  (L3082, I_surr_shunt<=0), so the faithful inhibition is
      (I_M_gaba - I_surr_shunt). I instrument I_surr_shunt (additive source-patch storing
      self._last_I_surr_shunt at L3080) and report BOTH stock and shunt-aware so the gap is visible.

Single-variable inference/instrumentation ONLY (no production edit; no frozen readout touched). Directions
are proven on the retrained co-adapted net; this scorecard is the cheap-prove read (the multi-seed/expensive
gate still binds the production retrain). Firewall: frozen TBW/SBW md5 + net weight-sums asserted BEFORE+AFTER.

Run:  GABA_SHUNT_SURR=1 K_SHUNT_SURR=0.026 E_GABA=-70 DEND_COUPLING_ALPHA=2 MG_VHALF=-48 MG_K=0.15 \
      MG_VHALF_INH=-30 CUDA_VISIBLE_DEVICES=0 python q5_53_scorecard.py  <CKPT_PATH>
"""
import os, sys, json, inspect, textwrap
# match #53 training env so a bare net __init__ reproduces the divisive config; ckpt mutable_hparams override.
os.environ.setdefault("GABA_SHUNT_SURR", "1"); os.environ.setdefault("K_SHUNT_SURR", "0.026")
os.environ.setdefault("E_GABA", "-70.0")
os.environ.setdefault("DEND_COUPLING_ALPHA", "2"); os.environ.setdefault("MG_VHALF", "-48")
os.environ.setdefault("MG_K", "0.15"); os.environ.setdefault("MG_VHALF_INH", "-30")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import routec_net_io as N
import inverse_effectiveness_routec as IE

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CKPT53", "")
DEV = "cuda:0"; SEED = 42
EI_BAND = (0.80, 1.25)


# ── IE shape classifier (verbatim logic from q5_50_decouple.shape) ──
def shape(mei):
    mei = np.array(mei, float)
    if np.all(np.isnan(mei)):
        return "ALL-NAN"
    am = int(np.nanargmax(mei))
    mono = am <= 1 and all((mei[i] >= mei[i+1] - 1e-6) or np.isnan(mei[i+1]) for i in range(am, len(mei)-1))
    return ("MONOTONE-dec" if mono else f"MID-PEAK@{IE.INTENSITIES[am]}")


def measure_ie_row(net):
    m = IE.measure_ie(net, float(net.sigma_in), IE.DEFAULT_NTRIALS, DEV)
    mei = np.array(m["mei"], float); RA = np.array(m["R_A"], float); RB = np.array(m["R_B"], float)
    return mei, RA, RB


def fmt(mei):
    return " ".join(f"{x:6.2f}" if np.isfinite(x) else "   nan" for x in mei)


# ── [3] shunt-aware E/I: additive source-patch storing I_surr_shunt at the dVM step ──
def patch_store_shunt(net):
    cls = net.__class__
    if getattr(cls, "_ss_patched", False):
        return
    src = textwrap.dedent(inspect.getsource(cls.update_all_layers_batch))
    anchor = "I_surr_shunt = self.k_shunt_surr * self.I_M_gaba_surr_sh * (self.E_gaba - self.v_msi)"
    lines = src.splitlines()
    hits = [ln for ln in lines if anchor in ln]
    assert len(hits) == 1, f"anchor not unique ({len(hits)})"
    line = hits[0]; indent = line[:len(line) - len(line.lstrip())]
    src = src.replace(line, line + "\n" + indent + "self._last_I_surr_shunt = I_surr_shunt.detach()", 1)
    g = cls.__init__.__globals__
    ns = {}
    exec(compile(src, "<ss_patch>", "exec"), g, ns)
    cls.update_all_layers_batch = ns["update_all_layers_batch"]
    cls._ss_patched = True


@torch.no_grad()
def ei_probe(net, offset_steps, sigma_in, pulse_frames, n_frames, intensity, centre_deg=90.0):
    """mirrors val36 ei_probe_route_c but ALSO captures self._last_I_surr_shunt -> stock + shunt-aware inh."""
    B, Nn = 1, net.n
    net.reset_state(batch_size=B)
    net._last_I_surr_shunt = torch.zeros((B, Nn), device=net.device)
    xA = torch.zeros(n_frames, Nn, device=net.device); xV = torch.zeros_like(xA)
    idx_c = int(round(centre_deg * (Nn - 1) / (net.space_size - 1)))
    xs = torch.arange(Nn, dtype=torch.float32, device=net.device)
    gauss = torch.exp(-0.5 * ((xs - idx_c) / sigma_in) ** 2) * intensity
    a0 = 0 if offset_steps >= 0 else abs(offset_steps); v0 = 0 if offset_steps <= 0 else offset_steps
    xA[a0:a0 + pulse_frames] = gauss; xV[v0:v0 + pulse_frames] = gauss
    E, I_stock, I_shunt = [], [], []
    for t in range(n_frames):
        net.update_all_layers_batch(xA[t].unsqueeze(0), xV[t].unsqueeze(0), valid_mask=None)
        iM = net.I_M.detach().clamp(min=0)
        iMg = net.I_M_gaba.detach()
        iss = net._last_I_surr_shunt.detach()                 # <= 0 (inhibitory); 0 when shunt OFF
        E.append(float(iM.mean()))
        I_stock.append(float(iMg.clamp(min=0).mean()))        # validator's stock denominator (omits surround under shunt)
        I_shunt.append(float((iMg - iss).clamp(min=0).mean()))  # faithful membrane inhibition: I_M_gaba + |I_surr_shunt|
    return np.array(E), np.array(I_stock), np.array(I_shunt)


@torch.no_grad()
def measure_ei_shuntaware(net):
    offsets = [-30, -20, -10, 0, 10, 20, 30]
    sig = float(getattr(net, "sigma_in", 10.0))
    e, istk, ish = [], [], []
    for off in offsets:
        E, Is, Ish = ei_probe(net, off, sig, pulse_frames=5, n_frames=20, intensity=1.0)
        e.append(E.mean()); istk.append(Is.mean()); ish.append(Ish.mean())
    e = np.array(e); istk = np.array(istk); ish = np.array(ish)
    r_stock = e / (istk + 1e-12); r_shunt = e / (ish + 1e-12)
    j0 = offsets.index(0)
    return dict(stock_sync=float(r_stock[j0]), stock_off=float(r_stock.mean()),
                shunt_sync=float(r_shunt[j0]), shunt_off=float(r_shunt.mean()),
                exc_off=float(e.mean()), inh_stock_off=float(istk.mean()), inh_shunt_off=float(ish.mean()))


def in_band(x):
    return EI_BAND[0] <= x <= EI_BAND[1]


@torch.no_grad()
def main():
    assert CKPT and os.path.exists(CKPT), f"CKPT not found: {CKPT!r}  (pass path as argv[1] or set CKPT53)"
    N.assert_frozen_readouts("BEFORE")

    # load once to read config + trained g_rec
    net0, res0, ep0, g_rec_tr, comment0 = N.load_ckpt(CKPT, SEED, 1, DEV)
    wsum0 = {k: float(getattr(net0, k).sum()) for k in ("W_MSI_inh", "W_MSI_exc", "W_inA", "W_inV")}
    cfg = dict(gaba_shunt_surr=bool(getattr(net0, "gaba_shunt_surr", False)),
               k_shunt_surr=float(getattr(net0, "k_shunt_surr", 0.0)),
               E_gaba=float(getattr(net0, "E_gaba", -70.0)),
               g_rec_trained=float(g_rec_tr), epoch=int(ep0))
    del net0; torch.cuda.empty_cache()

    print(f"\n===== #53 SCORECARD — divisive-surround cheap-prove (ep{cfg['epoch']}, seed{SEED}) =====")
    print(f"  ckpt: {CKPT}")
    print(f"  config: gaba_shunt_surr={cfg['gaba_shunt_surr']}  k_shunt_surr={cfg['k_shunt_surr']:.4f}  "
          f"E_gaba={cfg['E_gaba']:.1f}  g_rec(trained)={cfg['g_rec_trained']:.3f}")
    if not cfg["gaba_shunt_surr"]:
        print("  !! WARNING: gaba_shunt_surr is FALSE on this ckpt — not the divisive build; E/I shunt-aware == stock.")

    out = {"cfg": cfg, "ckpt": CKPT}

    # ── [1] POP-IE (headline) ──  (run BEFORE any patch so measure_ie is validator-pristine)
    print(f"\n[1] POP-IE (validator TEST1, whole-population MEI)  band: monotone-decreasing = GO")
    net, *_ = N.load_ckpt(CKPT, SEED, 1, DEV)
    mei, RA, RB = measure_ie_row(net)
    i16 = IE.INTENSITIES.index(1.6)
    print(f"    INTENSITIES {IE.INTENSITIES}")
    print(f"    MEI       = {fmt(mei)}   -> {shape(mei)}")
    print(f"    R_A@1.6={RA[i16]:.0f}   R_B@1.6={RB[i16]:.0f}")
    out["pop_ie"] = dict(mei=mei.tolist(), R_A=RA.tolist(), R_B=RB.tolist(), shape=shape(mei),
                         monotone=(shape(mei) == "MONOTONE-dec"))
    del net; torch.cuda.empty_cache()

    # ── [2] g_rec subsumption ──
    print(f"\n[2] g_rec subsumption (IE shape, shunt ON; single-variable g_rec KO):")
    g2 = {}
    for tag, gv in (("trained", cfg["g_rec_trained"]), ("KO_0", 0.0)):
        net, *_ = N.load_ckpt(CKPT, SEED, 1, DEV)
        net.g_rec = float(gv)
        mei_g, RA_g, RB_g = measure_ie_row(net)
        print(f"    g_rec={gv:<5.3f} ({tag:>7}): MEI = {fmt(mei_g)}  -> {shape(mei_g)}")
        g2[tag] = dict(g_rec=float(gv), mei=mei_g.tolist(), shape=shape(mei_g),
                       monotone=(shape(mei_g) == "MONOTONE-dec"))
        del net; torch.cuda.empty_cache()
    tr_mono, ko_mono = g2["trained"]["monotone"], g2["KO_0"]["monotone"]
    if tr_mono and ko_mono:
        verdict2 = "SUBSUMED — both monotone; surround carries the shape, g_rec inert for IE-shape"
    elif tr_mono and not ko_mono:
        verdict2 = "g_rec HELPING — trained monotone but KO mid-peaks; surround alone insufficient"
    elif (not tr_mono) and ko_mono:
        verdict2 = "g_rec PERSISTS — trained mid-peaks, KO monotone; g_rec independently re-introduces mid-peak"
    else:
        verdict2 = "BOTH mid-peak — neither surround nor g_rec achieves monotonicity"
    print(f"    -> {verdict2}")
    g2["verdict"] = verdict2
    out["g_rec_subsumption"] = g2

    # ── [3] E/I shunt-aware ──  (patch now; additive store cannot change numerics, but IE already done above)
    print(f"\n[3] E-I membrane ratio (offsets -30..30; band {EI_BAND}):")
    net, *_ = N.load_ckpt(CKPT, SEED, 1, DEV)
    patch_store_shunt(net)
    ei = measure_ei_shuntaware(net)
    print(f"    STOCK        <I_M>/<I_M_gaba>                : sync={ei['stock_sync']:.3f}  off={ei['stock_off']:.3f}"
          f"   {'IN' if in_band(ei['stock_sync']) else 'OUT'}-band  (NOT shunt-aware: omits surround)")
    print(f"    SHUNT-AWARE  <I_M>/<I_M_gaba - I_surr_shunt> : sync={ei['shunt_sync']:.3f}  off={ei['shunt_off']:.3f}"
          f"   {'IN' if in_band(ei['shunt_sync']) else 'OUT'}-band  <- FAITHFUL gate")
    print(f"    (exc_off={ei['exc_off']:.2f}  inh_stock={ei['inh_stock_off']:.2f}  inh_shuntaware={ei['inh_shunt_off']:.2f})")
    ei["stock_sync_in_band"] = in_band(ei["stock_sync"]); ei["shunt_sync_in_band"] = in_band(ei["shunt_sync"])
    out["ei"] = ei
    wsum1 = {k: float(getattr(net, k).sum()) for k in wsum0}
    out["net_intact"] = all(abs(wsum1[k] - wsum0[k]) < 1e-6 for k in wsum0)
    del net; torch.cuda.empty_cache()

    # ── scorecard summary ──
    print(f"\n================ #53 SCORECARD SUMMARY ================")
    print(f"  [1] POP-IE          : {out['pop_ie']['shape']:>14}   {'PASS' if out['pop_ie']['monotone'] else 'FAIL'}")
    print(f"  [2] g_rec           : {verdict2}")
    print(f"  [3] E/I shunt-aware : sync={ei['shunt_sync']:.3f}   {'PASS' if ei['shunt_sync_in_band'] else 'FAIL'} (band {EI_BAND})")
    print(f"      (stock E/I      : sync={ei['stock_sync']:.3f}  — inflated; omits surround inhibition)")

    N.assert_frozen_readouts("AFTER")
    op = os.path.join(N.OUT_DIR, "q5_53_scorecard.json")
    json.dump(out, open(op, "w"), indent=1)
    print(f"\n  -> {op}")
    print(f"  net_intact (weight-sums unchanged): {out['net_intact']}")
    print("[firewall OK] frozen-readout md5 unchanged; inference/instrumentation only; no production edit.")


if __name__ == "__main__":
    main()
