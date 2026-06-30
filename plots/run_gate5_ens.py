"""GATE 5 SBW (CRE, ENSEMBLE) — repo SBW double-sided GRAPHIC house-style (verbatim from frozen
SBW_test.plot_spatial_binding_pedestal), across-seed error bars. SBW = AV cross-modal ENHANCEMENT
(#136 CRE %), PEAK_pop R-def. Curve = per-offset mean ± SD over the ensemble (measure_ens_cre.py
cre_per_offset). The negative surround lobe is plotted AS MEASURED (interpretation HELD pending the
lead's biological-evidence relay — task #184); NOT removed. PCHIP C1 fit through the mean, lightcoral
suppressive flanks, zero-cross verticals, trough annotation (mean ± SD). Pure plot from gate5_cre.json;
frozen md5 before==after.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bridge as B
import numpy as np, matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.interpolate import PchipInterpolator

CRE = os.path.join(B.WORK, "results", "gate5_cre.json")
OUTENS = os.path.join(B.WORK, "figures")
os.makedirs(OUTENS, exist_ok=True)

B.firewall("gate5-ens before")
J = json.load(open(CRE))
n = J["n"]
seps = np.asarray(J["seps"], float)                                  # 0..80 deg
cre = np.asarray([d["mean"] for d in J["cre_per_offset"]], float)    # across-seed mean CRE %
cre_sd = np.asarray([d["sd"] for d in J["cre_per_offset"]], float)
ZC, HWHM = J["zero_cross"]["mean"], J["cen_hwhm"]["mean"]
TR, TR_sd, TR_D = J["surr_min"]["mean"], J["surr_min"]["sd"], J["surr_min_d"]["mean"]
print(f"[gate5-ens] n={n} seeds {J['seeds']}  peak={J['peak_cre']['mean']:.1f}±{J['peak_cre']['sd']:.1f}%  "
      f"zero-cross={ZC:.1f}±{J['zero_cross']['sd']:.1f}  HWHM={HWHM:.1f}±{J['cen_hwhm']['sd']:.1f}  "
      f"trough={TR:.1f}±{TR_sd:.1f}%@{TR_D:.1f}", flush=True)

# double-sided (SBW symmetric in disparity sign) — mirror 0..80 to -80..+80
x = np.concatenate([-seps[::-1][:-1], seps])
y = np.concatenate([cre[::-1][:-1], cre])
yerr = np.concatenate([cre_sd[::-1][:-1], cre_sd])
xs = np.linspace(x.min(), x.max(), 1200)
ys = PchipInterpolator(x, y)(xs)
s = np.sign(ys)
cr = xs[:-1][s[:-1] * s[1:] < 0]; cr = cr[np.argsort(np.abs(cr))]
inner = np.sort(np.abs(cr))[0] if len(cr) else ZC
outer_cand = [c for c in np.sort(np.abs(cr)) if c > inner + 1]
outer = outer_cand[0] if outer_cand else 49.0

font_manager.fontManager.addfont(os.path.join(B.REPO, "fonts", "Roboto-Regular.ttf"))
plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['svg.fonttype'] = 'none'
for k in ('font.size', 'xtick.labelsize', 'ytick.labelsize', 'axes.labelsize', 'legend.fontsize'):
    plt.rcParams[k] = 60
plt.rcParams['axes.titlesize'] = 46

level = 0.0
y_bottom = min(-12.0, (ys.min() - 2), float((y - yerr).min()) - 2)
y_top = max(85.0, ys.max() * 1.08, float((y + yerr).max()) + 2)
fig, ax = plt.subplots(figsize=(20, 14))
ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=6, markersize=11, elinewidth=2.5,
            capthick=2.0, markeredgewidth=0.8, color="C0",
            label=f"CRE, mean ± SD (n={n})", zorder=3)
ax.plot(xs, ys, color="C1", lw=5, label="PCHIP fit (mean)", zorder=2)
mask_c = (xs >= -inner) & (xs <= inner)
ax.fill_between(xs, level, ys, where=mask_c & (ys > level), color="C1", alpha=0.15, zorder=1)
mask_s = (np.abs(xs) >= inner) & (np.abs(xs) <= outer)
ax.fill_between(xs, level, ys, where=mask_s & (ys < level), color="lightcoral", alpha=0.18, zorder=0)
ax.axhline(level, ls=":", color="0.4", lw=2)
for xc in (-inner, inner):
    ax.vlines(xc, y_bottom, level, ls=":", color="C1", lw=2.5)
ax.annotate(f"zero-cross {inner:.1f}°", (inner + 1, 3), fontsize=34, color="0.25", ha="left", va="bottom")
ax.annotate(f"trough {TR:.1f} ± {TR_sd:.1f}% @ {TR_D:.0f}°", (TR_D + 1, TR - 0.5), fontsize=34,
            color="firebrick", ha="left", va="top")
xticks = np.arange(-80, 85, 20)
ax.set_xticks(xticks); ax.set_xticklabels([f"{abs(int(t))}" for t in xticks])
ax.set(xlabel="Spatial disparity (°)  —  RF-border-crossing proxy", ylabel="AV enhancement, CRE (%)",
       ylim=(y_bottom, y_top), xlim=(x.min(), x.max()))
ax.set_title("Spatial binding window — AV enhancement (ensemble)", fontsize=36, pad=18)
ax.yaxis.label.set_size(48)
ax.legend(frameon=False, loc="upper left", fontsize=40, handlelength=1.4,
          labelspacing=0.34, borderaxespad=0.6)
ax.tick_params(axis='both', which='major', length=20, width=2)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("outward", 5))
ax.spines["bottom"].set_position(("outward", 5))
ax.grid(False)
# RF-border-proxy + conservative-magnitude caveat (researcher #184 §2/§4/§6; lead-required framing)
caption = (
    "Cross-modal surround suppression is biologically real — excitatory RF centre / inhibitory surround "
    "(Meredith & Stein 1996; Kadunce et al. 1997, cat SC; Wallace et al. 1996, macaque).\n"
    "Spatial axis is an RF-border-crossing PROXY: zero-cross ~%.0f° / trough ~%.0f° mark ‘cue beyond its "
    "RF border’, NOT a validated degree-width.  The %.0f%% trough is CONSERVATIVE vs biology "
    "(mean depression ~46%%, up to ~100%%)." % (inner, TR_D, TR)
)
fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=24, color="0.30")
plt.tight_layout(rect=[0, 0.13, 1, 1])
svg = os.path.join(OUTENS, "gate5_sbw_ens.svg"); png = os.path.join(OUTENS, "gate5_sbw_ens.png")
plt.savefig(svg, format="svg", bbox_inches="tight")
plt.savefig(png, format="png", dpi=100, bbox_inches="tight")
plt.close("all")
print(f"[gate5-ens] saved: {svg} / {png}  (zero-cross {inner:.1f}°, neg-lobe end {outer:.1f}°)", flush=True)
B.firewall("gate5-ens after")
