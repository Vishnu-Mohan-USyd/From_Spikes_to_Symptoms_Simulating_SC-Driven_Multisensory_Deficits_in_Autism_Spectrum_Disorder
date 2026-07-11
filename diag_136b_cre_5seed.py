#!/usr/bin/env python3
"""DEBUGGER task #136 — SC-standard cross-modal response enhancement (CRE / MSI index) vs A-V disparity,
straight from the FROZEN dL3 ep79 network's OWN spikes. NO decoder, NO p_common, NO tuned parameter.
THROWAWAY measurement driver. Pure inference; net NEVER mutated; frozen readouts asserted before+after.

This REPLACES the P(fusion) bump-counting comparator (the wrong comparator for an SC-neuron model).
We do NOT touch the P(fusion) ruler. We compute, for each A-V azimuth disparity d, the cross-modal
enhancement the way SC electrophysiology does:

    CRE(d) = (R_AV(d) - max(R_A(d), R_V(d))) / max(R_A(d), R_V(d)) * 100      [enhancement %]

with R_A(d)/R_V(d) the MATCHED-POSITION unimodal baselines (A-alone / V-alone at the identical trial
geometry). Response magnitude R is reported THREE ways, each labelled (the window depends on the choice):
    (1) total population activity  = sum over neurons of the time-collapsed population profile
    (2) peak population activity   = max over neurons
    (3) center-neuron analog       = the single neuron at the coincidence locus (midpoint of A & V)

Also reported: additivity class per d (R_AV vs R_A+R_V: super / additive / sub), mean +/- SD over the
60 trials/d with a t-test of CRE>0, the peak-enhancement disparity, the HALF-MAX half-width of the
positive enhancement lobe, and the zero-crossing.

DATA: capture_profiles() is copied VERBATIM from diag_135_decode.py (same frozen ckpt, SEED=42, NT=60,
SEPS=0..80 step5, INTENSITY=0.5, DURATION=20, @torch.inference_mode). It runs all three trial types
per separation: AV=run(gA,gV), AO=run(gA,z) [A-alone], VO=run(z,gV) [V-alone]; shape (17,60,180).
(Imported reuse is avoided because diag_135_decode.py opens its LOG at module import with mode "w",
which would truncate the #135 evidence log; the function body is identical.)

FIREWALL: frozen md5 TBW 80d33465 / SBW 73b7d136 asserted BEFORE + AFTER; 4 weight-sums asserted
unchanged BEFORE + AFTER. NO production edits, NO readout/weight edits, NO fixes, NO retrain, NO 5-seed.
cuda:1 (A6000) per CUDA_VISIBLE_DEVICES. Measurement only -- no interpretation, no fix proposals.
UNVERIFIED until run.
"""
import os, sys, json, time
# device: honour an externally-set CUDA_VISIBLE_DEVICES; do not force a GPU index (portability)
os.environ.setdefault("TAU_GABA", "10")
os.environ.setdefault("SIGMA_DL_FRAMES", "0.0")
os.environ.setdefault("AFFERENT_SHARPEN_FACTOR", "1.0")
os.environ.setdefault("SURROUND_BUDGET", "0.0")
os.environ.setdefault("VAL36_BUILD", "delayfix")
import numpy as np
import torch
from scipy import stats

# self-locating: flat layout - modules, frozen readouts and checkpoint/ all sit beside this file
THIS = os.path.dirname(os.path.abspath(__file__))
VAL = THIS  # flat
sys.path.insert(0, VAL)
import val36_traj_d52 as V
import harden_85 as H  # noqa: F401  (kept for parity with #135 env)

OUTD = os.path.join(THIS, "out") if os.path.isdir(os.path.join(THIS, "out")) else THIS
SEED = int(os.environ.get("SEED", "42"))
CKPT = os.path.join(THIS, "checkpoint", f"ckpt_ep79_seed{SEED}_bs250_delay52_tau10_dL3.pt")
TBW_MD5 = "80d33465c4bf55d6e85b5990acb92da7"
SBW_MD5 = "73b7d13626964d851cc090818b728311"
SMOKE = os.environ.get("SMOKE", "0") == "1"
NT = 4 if SMOKE else 60                              # trials per separation (#135 used 60)
SEPS = [0, 20, 40, 80] if SMOKE else list(range(0, 85, 5))   # disparities, deg
INTENSITY, DURATION = 0.5, 20                        # SAME stimulus as #135 capture / the frozen ruler
LOG = open(os.path.join(OUTD, f"diag_136b_seed{SEED}.log"), "w", buffering=1)


def P(*a):
    print(*a, flush=True)
    print(*a, file=LOG, flush=True)


def md5s():
    return (V.md5(os.path.join(V.TBW_DIR, "TBW_test.py")), V.md5(os.path.join(V.SBW_DIR, "SBW_test.py")))


# ----------------------------------------------------------------------------------------------------
# capture_profiles  -- VERBATIM copy from diag_135_decode.py (net NEVER mutated; @torch.inference_mode)
# ----------------------------------------------------------------------------------------------------
@torch.inference_mode()
def capture_profiles(net):
    N, S = net.n, net.space_size
    xs = torch.arange(N, device=net.device, dtype=torch.float32)
    init_gFF, init_sc = net.g_FFinh, net.step_counter
    rng = np.random.default_rng(SEED)

    def to_idx(deg):
        return torch.round(torch.as_tensor(deg, device=net.device, dtype=torch.float32) * (N - 1) / (S - 1)).long()

    def make_gauss(idx):
        return torch.exp(-0.5 * ((xs - idx[:, None]) / net.sigma_in) ** 2) * INTENSITY

    AV, AO, VO, locA_all, locV_all = [], [], [], [], []
    for sep in SEPS:
        net.g_FFinh, net.step_counter = init_gFF, init_sc
        base = rng.integers(0, S, size=NT)
        locA = base
        locV = V.SBW._second_stim_loc(base, sep, S, linear=False)   # (base+sep)%S, exactly the frozen ruler
        idxA, idxV = to_idx(locA), to_idx(locV)
        gA, gV, z = make_gauss(idxA), make_gauss(idxV), torch.zeros((NT, N), device=net.device)

        def run(sa, sv):
            net.g_FFinh, net.step_counter = init_gFF, init_sc
            net.reset_state(batch_size=NT)
            acc = torch.zeros(NT, N, device=net.device)
            for _ in range(DURATION):
                ret = net.update_all_layers_batch(sa, sv, return_spike_sum=True)
                acc += ret[-1]
            return acc.cpu().numpy()

        AV.append(run(gA, gV)); AO.append(run(gA, z)); VO.append(run(z, gV))
        locA_all.append(np.asarray(locA)); locV_all.append(np.asarray(locV))
    net.g_FFinh, net.step_counter = init_gFF, init_sc
    return (np.stack(AV), np.stack(AO), np.stack(VO),
            np.stack(locA_all), np.stack(locV_all), N, S)


# ----------------------------------------------------------------------------------------------------
def cre_block(RAV, RA, RV):
    """RAV/RA/RV each shape (nsep, NT). Returns the per-d CRE stats both ways + additivity + t-test."""
    best_pt = np.maximum(RA, RV)
    with np.errstate(divide="ignore", invalid="ignore"):
        cre_pt = np.where(best_pt > 1e-9, (RAV - best_pt) / best_pt * 100.0, np.nan)   # per-trial CRE %
    mRAV, mRA, mRV = RAV.mean(1), RA.mean(1), RV.mean(1)
    best_m = np.maximum(mRA, mRV)
    cre_rom = np.where(best_m > 1e-9, (mRAV - best_m) / best_m * 100.0, np.nan)        # ratio-of-means CRE %
    cre_mean = np.nanmean(cre_pt, 1)
    cre_sd = np.nanstd(cre_pt, 1, ddof=1)
    pvals = np.array([_ttest(cre_pt[k]) for k in range(cre_pt.shape[0])])
    add_ratio = np.where((mRA + mRV) > 1e-9, mRAV / (mRA + mRV), np.nan)               # R_AV / (R_A+R_V)
    return dict(mRAV=mRAV, mRA=mRA, mRV=mRV, cre_rom=cre_rom, cre_mean=cre_mean,
                cre_sd=cre_sd, pvals=pvals, add_ratio=add_ratio)


def _ttest(x):
    x = x[np.isfinite(x)]
    if x.size < 2 or np.allclose(x, x[0]):
        return float("nan")
    return float(stats.ttest_1samp(x, 0.0).pvalue)


def add_class(r):
    if not np.isfinite(r):
        return "n/a"
    if r > 1.05:
        return "super"
    if r < 0.95:
        return "sub"
    return "add"


def lobe_metrics(seps, cre):
    """Peak-enhancement disparity, HALF-MAX half-width of the positive lobe, and zero-crossing (deg).
    Linear interpolation on the grid; nan when no decaying positive lobe exists."""
    seps = np.asarray(seps, float)
    cre = np.asarray(cre, float)
    kmax = int(np.nanargmax(cre))
    peak_d, peak_v = float(seps[kmax]), float(cre[kmax])
    half = peak_v / 2.0
    hwhm = float("nan")
    if peak_v > 0:
        for i in range(kmax, len(seps) - 1):
            if cre[i] >= half >= cre[i + 1]:
                d_at_half = seps[i] + (seps[i + 1] - seps[i]) * (cre[i] - half) / (cre[i] - cre[i + 1])
                hwhm = float(d_at_half - peak_d)
                break
    zc = float("nan")
    for i in range(kmax, len(seps) - 1):
        if cre[i] >= 0 >= cre[i + 1]:
            zc = float(seps[i] + (seps[i + 1] - seps[i]) * (cre[i] - 0.0) / (cre[i] - cre[i + 1]))
            break
    return peak_d, peak_v, hwhm, zc


def print_table(tag, seps, B):
    P(f"\n--- CRE table [{tag}] :  CRE(d) = (R_AV - max(R_A,R_V)) / max(R_A,R_V) * 100 ---")
    P(f"{'d':>4}{'R_AV':>10}{'R_A':>10}{'R_V':>10}{'R_A+R_V':>10}{'AVrt':>7}{'class':>7}"
      f"{'CRE_rom%':>10}{'CRE_mn%':>9}{'+-SD':>8}{'p':>9}{'sig':>4}")
    for k, d in enumerate(seps):
        sig = "*" if (np.isfinite(B["pvals"][k]) and B["pvals"][k] < 0.05 and B["cre_mean"][k] > 0) else ""
        P(f"{d:>4}{B['mRAV'][k]:>10.2f}{B['mRA'][k]:>10.2f}{B['mRV'][k]:>10.2f}"
          f"{B['mRA'][k]+B['mRV'][k]:>10.2f}{B['add_ratio'][k]:>7.2f}{add_class(B['add_ratio'][k]):>7}"
          f"{B['cre_rom'][k]:>10.1f}{B['cre_mean'][k]:>9.1f}{B['cre_sd'][k]:>8.1f}"
          f"{B['pvals'][k]:>9.1e}{sig:>4}")
    pd, pv, hwhm, zc = lobe_metrics(seps, B["cre_rom"])
    P(f"  window[{tag}]: peak-enhancement disparity = {pd:.1f} deg (CRE={pv:.1f}%); "
      f"HALF-MAX half-width = {hwhm if hwhm!=hwhm else round(hwhm,1)} deg; "
      f"zero-crossing = {zc if zc!=zc else round(zc,1)} deg")
    return dict(peak_d=pd, peak_cre=pv, hwhm=hwhm, zero_cross=zc)


def main():
    t0 = time.time()
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mb = md5s()
    P(f"[#136] device={dev} BUILD={V.BUILD} netsrc={V.md5(V.NETSRC)} SMOKE={SMOKE} NT={NT} nseps={len(SEPS)}")
    P(f"[#136] frozen readouts BEFORE: TBW={mb[0]} SBW={mb[1]}")
    assert mb == (TBW_MD5, SBW_MD5), f"FATAL: frozen readout md5 drift BEFORE: {mb}"
    P(f"[#136] ckpt={os.path.basename(CKPT)}  exists={os.path.exists(CKPT)}")
    assert os.path.exists(CKPT)

    net, res, epoch, g_rec, comment = V.load_ckpt(CKPT, SEED, NT, device)
    assert not res.missing_keys and not res.unexpected_keys, f"non-strict load: {res}"
    assert epoch == 79, f"expected ep79, got {epoch}"
    N, S, sig = net.n, net.space_size, float(net.sigma_in)
    w_before = {k: float(getattr(net, k).sum()) for k in ("W_MSI_inh", "W_MSI_exc", "W_inA", "W_inV")}
    P(f"[#136] loaded ep{epoch} g_rec={g_rec} tau_gaba={float(net.tau_gaba)} n={N} space={S} sigma_in={sig}  "
      f"Wsums={ {k: round(v,3) for k,v in w_before.items()} }")

    ei = V.measure_ei(net)
    P(f"[#136] net-identity X-check  E/I: sync={float(ei['ei_ratio_sync']):.3f} off={float(ei['ei_ratio_offsetmean']):.3f}  "
      f"(dL3 baseline ~0.923 / ~0.677)")

    tcap = time.time()
    AV, AO, VO, locA, locV, N, S = capture_profiles(net)
    P(f"[#136] captured AV/AO/VO shape={AV.shape}  (nsep={len(SEPS)}, NT={NT}, N={N})  capture={time.time()-tcap:.1f}s")
    np.savez_compressed(os.path.join(OUTD, f"diag_136b_profiles_seed{SEED}.npz"),
                        AV=AV, AO=AO, VO=VO, locA=locA, locV=locV, seps=np.array(SEPS), N=N, S=S)

    # --- response magnitudes, three definitions ---
    RAV_tot, RA_tot, RV_tot = AV.sum(2), AO.sum(2), VO.sum(2)        # total population activity
    RAV_pk, RA_pk, RV_pk = AV.max(2), AO.max(2), VO.max(2)          # peak population activity
    mid = (np.rint(locA + (np.array(SEPS, float)[:, None] / 2.0)).astype(int)) % S    # coincidence-locus neuron
    RAV_ct = np.take_along_axis(AV, mid[:, :, None], axis=2)[:, :, 0]
    RA_ct = np.take_along_axis(AO, mid[:, :, None], axis=2)[:, :, 0]
    RV_ct = np.take_along_axis(VO, mid[:, :, None], axis=2)[:, :, 0]

    P(f"\n[#136] sanity -- unimodal baselines should be ~disparity-independent for sum/peak:")
    P(f"  R_A(tot) over d: min={RA_tot.mean(1).min():.1f} max={RA_tot.mean(1).max():.1f}  | "
      f"R_A(peak) min={RA_pk.mean(1).min():.2f} max={RA_pk.mean(1).max():.2f}")

    B_tot = cre_block(RAV_tot, RA_tot, RV_tot)
    B_pk = cre_block(RAV_pk, RA_pk, RV_pk)
    B_ct = cre_block(RAV_ct, RA_ct, RV_ct)

    w_tot = print_table("TOTAL-pop-sum", SEPS, B_tot)
    w_pk = print_table("PEAK-pop", SEPS, B_pk)
    w_ct = print_table("CENTER-neuron@coincidence-locus", SEPS, B_ct)

    # --- provisional SC placement (researcher to re-verify; NOT asserted as truth) ---
    P("\n[#136] PROVISIONAL placement vs SC neural enhancement numbers (flag for researcher re-verification, "
      "NOT ground truth):")
    P("  user-supplied: mouse SC Ito2021 ~ half-max +-20-25 deg, zero +-35-40 deg (weak-auditory regime); "
      "cat Meredith/Stein/Kadunce = RF-zone/effectiveness-based, broad, no clean degree cutoff.")
    P(f"  measured here: peak@{w_pk['peak_d']:.0f}deg | PEAK-pop half-max hw={w_pk['hwhm']:.1f} zero={w_pk['zero_cross']:.1f} | "
      f"TOTAL-pop half-max hw={w_tot['hwhm']:.1f} zero={w_tot['zero_cross']:.1f} | "
      f"CENTER half-max hw={w_ct['hwhm']:.1f} zero={w_ct['zero_cross']:.1f}  (R-definition dependence is explicit above)")

    w_after = {k: float(getattr(net, k).sum()) for k in ("W_MSI_inh", "W_MSI_exc", "W_inA", "W_inV")}
    net_intact = all(abs(w_after[k] - w_before[k]) < 1e-6 for k in w_before)
    P(f"\n[#136] net intact (weight sums unchanged): {net_intact}  before={ {k:round(v,3) for k,v in w_before.items()} } "
      f"after={ {k:round(v,3) for k,v in w_after.items()} }")
    ma = md5s()
    P(f"[#136] frozen readouts AFTER: TBW={ma[0]} SBW={ma[1]}  (unchanged: {ma==(TBW_MD5,SBW_MD5)})")
    assert ma == (TBW_MD5, SBW_MD5), "FATAL: frozen readout md5 drift AFTER run!"
    assert net_intact, "FATAL: net weights mutated!"

    def packB(B):
        return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in B.items()}

    out = dict(task="136_CRE", ckpt=os.path.basename(CKPT), epoch=int(epoch), seed=SEED, nt=NT, seps=SEPS,
               smoke=SMOKE, ei_sync=float(ei["ei_ratio_sync"]), ei_off=float(ei["ei_ratio_offsetmean"]),
               cre_total_pop_sum=packB(B_tot), cre_peak_pop=packB(B_pk), cre_center_neuron=packB(B_ct),
               window_total=w_tot, window_peak=w_pk, window_center=w_ct,
               net_intact=net_intact, w_before=w_before, w_after=w_after,
               md5_before=mb, md5_after=ma, wall_s=time.time() - t0)
    outp = os.path.join(OUTD, f"diag_136b_seed{SEED}.json")
    json.dump(out, open(outp, "w"), indent=1)
    P(f"\n[#136] total {time.time()-t0:.1f}s -> {outp}")
    LOG.close()


if __name__ == "__main__":
    main()
