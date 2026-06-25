#!/usr/bin/env python3
"""ONE clean SBW-as-AV-enhancement curve, 5-seed (frozen dL3 ep79) — the canonical per-neuron
best-aligned ME% (Meredith&Stein SC estimator), in the repo's SBW_test.py house style. SINGLE
estimator: no population-sum contrast, no tuned term, no center-neuron read, no inverse-effectiveness,
no negative surround.

PURE re-analysis of saved frozen-net spikes -> matplotlib. The net is NEVER loaded or run; we only
reduce the spike arrays already on disk (out/diag_136b_profiles_seed{42..46}.npz, written by the
validated 5-seed #136b run) with #141's exact estimator.

FIDELITY GATE (single variable = the estimator, DATA held constant): the estimator (take / cre_stats /
the b_aligned concat block, copied VERBATIM from diag_141_pedestal.py) is run on out/diag_136_profiles.npz
-- the SAME npz #141 used -- and must reproduce diag_141_pedestal.json's b_aligned.cre_rom_avg to
~0 (< 0.5 abs). That proves the math is faithfully reused on identical data; only THEN is it applied
to the 5 validated seeds. (Holding the data constant is the point: a cross-RUN spike wobble at
identical stimulus positions is a separate question and must NOT be folded into a math-fidelity test.)

FIREWALL: net not loaded -> weights trivially unchanged; frozen readout md5s (TBW 80d33465 /
SBW 73b7d136) asserted on disk BEFORE + AFTER. No model/readout/ckpt touched.
"""
import os, sys, json, hashlib, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning

THIS = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(THIS, "out")
SEEDS = [42, 43, 44, 45, 46]
NPZ_GATE = os.path.join(OUTD, "diag_136_profiles.npz")        # SAME npz #141 used (data held constant)
PEDESTAL = os.path.join(OUTD, "diag_141_pedestal.json")       # #141 reference (its b_aligned.cre_rom_avg)
_FSTS5 = os.path.dirname(os.path.dirname(os.path.dirname(THIS)))   # measopt -> delayfix -> perilog -> fsts_5
TBW = os.path.join(THIS, "TBW_test.py")
SBW = os.path.join(THIS, "SBW_test.py")
TBW_MD5 = "80d33465c4bf55d6e85b5990acb92da7"
SBW_MD5 = "73b7d13626964d851cc090818b728311"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def firewall(tag):
    t, s = md5(TBW), md5(SBW)
    print(f"[firewall {tag}] TBW={t}  SBW={s}")
    return [t, s]


# ===================== #141 estimator math, copied VERBATIM =====================
def cre_stats(RAV, RA, RV):
    """RAV/RA/RV: (T,) per-trial response magnitudes for one disparity. Returns rom, mean, sd."""
    mx = np.maximum(RA, RV)
    pt = (RAV - mx) / mx * 100.0
    rom = (RAV.mean() - max(RA.mean(), RV.mean())) / max(RA.mean(), RV.mean()) * 100.0
    return float(rom), float(pt.mean()), float(pt.std())


def hwhm(seps, cre):
    """half-max half-width: first disparity where cre falls to peak/2 (linear interp). peak at seps[0]."""
    seps = np.asarray(seps, float); cre = np.asarray(cre, float)
    pk = cre[0]; half = pk / 2.0
    for i in range(1, len(cre)):
        if cre[i] <= half:
            x0, x1, y0, y1 = seps[i-1], seps[i], cre[i-1], cre[i]
            return float(x0 + (half - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x0)
    return float("nan")


def take(M, idx):
    """M (S,T,N), idx (S,T) -> (S,T) gather along neuron axis."""
    return np.take_along_axis(M, idx[:, :, None], axis=2)[:, :, 0]
# ================================================================================


def b_aligned_avg(npz_path):
    """per-disparity per-neuron best-aligned ME% (cre_rom_avg) -- #141's symmetric aligned block, verbatim."""
    z = np.load(npz_path)
    AV, AO, VO = z["AV"], z["AO"], z["VO"]                 # (17,60,180)
    locA, locV = z["locA"], z["locV"]                      # (17,60)
    seps = z["seps"].astype(float); nSep = AV.shape[0]
    AV_A, AO_A, VO_A = take(AV, locA), take(AO, locA), take(VO, locA)
    AV_V, AO_V, VO_V = take(AV, locV), take(AO, locV), take(VO, locV)
    rom = []
    for i in range(nSep):
        RAV_b = np.concatenate([AV_A[i], AV_V[i]])
        Rown  = np.concatenate([AO_A[i], VO_V[i]])         # on-locus (own) modality alone
        Roth  = np.concatenate([VO_A[i], AO_V[i]])         # other (far) modality alone (-> ~0)
        rB, mB, sB = cre_stats(RAV_b, Rown, Roth)
        rom.append(rB)
    return seps, np.array(rom, float)


def sym(absd, folded):
    """mirror a folded 0..80 array to symmetric -80..+80 (0 not double-counted)."""
    absd = np.asarray(absd, float); folded = np.asarray(folded, float)
    return np.concatenate([-absd[::-1][:-1], absd]), np.concatenate([folded[::-1][:-1], folded])


def gaussian(x, base, amp, mu, sigma):
    return base + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_curve(x, y):
    with warnings.catch_warnings():                       # covariance (pcov) is unused -> ignore its warning
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, _ = curve_fit(gaussian, x, y, p0=[np.min(y), np.max(y) - np.min(y), 0.0, 18.0], maxfev=20000)
    xs = np.linspace(x.min(), x.max(), 600)
    return xs, gaussian(xs, *popt), popt


def msd(a):
    a = np.asarray(a, float)
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


# ============================== run ==============================
fw_before = firewall("BEFORE")
assert fw_before == [TBW_MD5, SBW_MD5], "FIREWALL FAIL (before)"

# ---- CORRECTED FIDELITY GATE: estimator on the SAME npz #141 used must reproduce #141 to ~0 ----
seps_g, gate = b_aligned_avg(NPZ_GATE)
ref = np.array(json.load(open(PEDESTAL))["b_aligned"]["cre_rom_avg"], float)
gate_diff = float(np.abs(gate - ref).max())
print(f"\n[FIDELITY GATE] estimator on diag_136_profiles.npz (data #141 used) vs diag_141 json:"
      f"  max-abs-diff = {gate_diff:.3e}  ({'PASS' if gate_diff < 0.5 else 'FAIL'}; threshold <0.5)")
if not (gate_diff < 0.5):
    print("[FIDELITY GATE] FAIL -> estimator math diverges from #141 on identical data. NOT plotting.")
    firewall("AFTER")
    sys.exit(1)

# ---- apply the proven estimator to the 5 VALIDATED seeds ----
M = []
for s in SEEDS:
    seps, rom = b_aligned_avg(os.path.join(OUTD, f"diag_136b_profiles_seed{s}.npz"))
    M.append(rom)
M = np.array(M, float)                                     # (5,17)
mean = M.mean(0)
sd = M.std(0, ddof=1)

# scalar reads per seed -> 5-seed mean +/- SD
far = seps >= 50
peak_m, peak_s = msd(M[:, 0])                              # peak @ 0 deg
hwhm_per = np.array([hwhm(seps, M[k]) for k in range(len(SEEDS))])
hwhm_m, hwhm_s = msd(hwhm_per)
floor_per = M[:, far].mean(1)                              # mean over d>=50 deg, per seed
floor_m, floor_s = msd(floor_per)

# honesty check: seed42 cross-RUN drift @ d=25 (136 vs 136b seed42) vs the 5-seed SD there
i25 = int(np.where(seps == 25)[0][0])
drift25 = float(abs(gate[i25] - M[0, i25]))               # |orig#136 - validator136b seed42| at d=25
sd25 = float(sd[i25])

# ---- single clean curve (house style; one estimator) ----
x, y = sym(seps, mean)
_, y_sd = sym(seps, sd)
xs, yfit, popt = fit_curve(x, y)
peak = float(mean[0])
level = peak / 2.0
inside = (yfit - level)[:-1] * (yfit - level)[1:] <= 0
cross = np.array([xs[i] + (level - yfit[i]) * (xs[i + 1] - xs[i]) / (yfit[i + 1] - yfit[i])
                  for i in np.where(inside)[0]])

fig, ax = plt.subplots(figsize=(7, 4))
y_bottom = -10.0
y_top = max(peak * 1.12, yfit.max() * 1.12)

# faint same-color +/-SD band = the error of THIS one curve (not a second estimator)
ax.fill_between(x, y - y_sd, y + y_sd, color="C0", alpha=0.15, lw=0, zorder=2)
# 5-seed mean markers + Gaussian fit
ax.plot(x, y, "o", ms=5, color="C0", zorder=4, label="per-neuron best-aligned ME%  (5-seed mean±SD)")
ax.plot(xs, yfit, color="C1", lw=2, zorder=3, label="Gaussian fit")
# half-max window under the fit + crossing-degree text
if len(cross) == 2:
    mask = (xs >= cross.min()) & (xs <= cross.max())
    ax.fill_between(xs, y_bottom, yfit, where=mask, color="C1", alpha=0.15, zorder=0)
ax.axhline(0, color="k", lw=0.8, zorder=1)
ax.axhline(level, ls=":", color="0.4", zorder=1)
for xc in cross:
    ax.vlines(xc, y_bottom, level, ls=":", color="0.4", zorder=1)
    ax.text(xc, level + 10, f"{abs(xc):.1f}°", ha="center", va="bottom", fontsize=8, color="0.25")

xticks = np.arange(-80, 85, 20)
ax.set_xticks(xticks)
ax.set_xticklabels([f"{abs(int(t)):d}" for t in xticks])
ax.set(xlabel="Spatial disparity (°)",
       ylabel="AV cross-modal enhancement (%)",
       title="Spatial binding window – AV cross-modal enhancement",
       ylim=(y_bottom, y_top), xlim=(xs.min(), xs.max()))
ax.legend(frameon=False, fontsize=8.0, loc="upper left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("outward", 5))
ax.spines["bottom"].set_position(("outward", 5))
ax.grid(False)
plt.tight_layout()
out = os.path.join(OUTD, "SBW_curve_5seed_dL3_ep79.png")
fig.savefig(out, dpi=160, bbox_inches="tight")

fw_after = firewall("AFTER")
assert fw_after == fw_before == [TBW_MD5, SBW_MD5], "FIREWALL FAIL (after)"

# ---- report (checkable against the npz/json) ----
halfwin = (cross.max() - cross.min()) / 2 if len(cross) == 2 else float("nan")
print("\n================ REPORT ================")
print(f"source: pure re-analysis of out/diag_136b_profiles_seed{{{','.join(map(str,SEEDS))}}}.npz  (net NEVER loaded)")
print(f"fidelity-gate max-abs-diff = {gate_diff:.3e}  (estimator faithfully reproduces #141 on identical data)")
print("\n5-seed MEAN curve  b_aligned.cre_rom_avg (%):")
print("   d(°):  " + " ".join(f"{int(s):>5}" for s in seps))
print("   mean:  " + " ".join(f"{v:>5.1f}" for v in mean))
print("    ±SD:  " + " ".join(f"{v:>5.1f}" for v in sd))
print(f"\npeak@0° = {peak_m:.1f} ± {peak_s:.1f} %   "
      f"HWHM = {hwhm_m:.1f} ± {hwhm_s:.1f} °   "
      f"far-floor(d≥50°) = {floor_m:.1f} ± {floor_s:.1f} %")
print(f"Gaussian-fit half-max window = ±{halfwin:.1f}°  (0.5-crossings {[round(c,1) for c in cross]})")
print(f"\n[honesty] seed42 cross-RUN drift @ d=25° = {drift25:.2f}  vs  5-seed SD @ d=25° = {sd25:.2f}"
      f"  ->  run-to-run wobble is {'WITHIN' if drift25 <= sd25 else 'ABOVE'} the seed-to-seed spread")
print(f"\nfirewall before==after==frozen: {fw_after == fw_before == [TBW_MD5, SBW_MD5]}  (net never loaded)")
print(f"SAVED  {out}")
