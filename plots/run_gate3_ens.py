"""GATE 3 E/I (ENSEMBLE) — repo plot_ei_simple house style, lead's ask #1: collapse the per-neuron
cloud to ONE point = the network's AVERAGED (E,I), with error bars ACROSS the ensemble (both axes),
shown vs the 5 biology reference points.

Per seed we measured the shunt-aware sync (E,I) = (<I_M>, <I_M_gaba - I_surr_shunt>) at offset 0 (the
condition defining E/I=1.059) via q5_53.ei_probe (gates_main.json E_sync/I_sync). The ENSEMBLE point =
mean over seeds; error bars = SD over seeds in BOTH E and I. Bio refs normalised exactly as the repo
(scale so bio grand-mean exc == model exc), unity line, Roboto despined house style. Pure plot from the
measured JSON; frozen TBW/SBW md5 asserted before==after.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bridge as B
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

GATES = os.path.join(B.WORK, "results", "gates_main.json")
OUTENS = os.path.join(B.WORK, "figures")
os.makedirs(OUTENS, exist_ok=True)

# 5 biology reference points (exc, inh) — identical set to the single-seed gate3 (run_gate3_ei_perneuron)
BIO = {
    "Mouse V1 (Xue 2014)":    ([267.0], [251.0]),
    "Mouse V1 (Okun 2008)":   ([7.1], [6.9]),
    "Rat A1 (Wehr 2003)":     ([710.0], [650.0]),
    "Mouse S1 (Barral 2016)": ([4.2], [4.6]),
    "Cat V1 (Priebe 2005)":   ([29.0], [31.0]),
}

B.firewall("gate3-ens before")
J = json.load(open(GATES))
rows = J["rows"]
E = np.array([r["E_sync"] for r in rows], float)
I = np.array([r["I_sync"] for r in rows], float)
n = E.size
E_m, I_m = float(E.mean()), float(I.mean())
E_sd = float(E.std(ddof=1)) if n > 1 else 0.0
I_sd = float(I.std(ddof=1)) if n > 1 else 0.0
ratio = E_m / I_m
ei_sd = float(J["ei_sync"]["sd"])   # across-seed SD of the per-seed E/I ratio
print(f"[gate3-ens] n={n} seeds {J['seeds']}  (E,I)=({E_m:.3f}±{E_sd:.3f}, {I_m:.3f}±{I_sd:.3f})  "
      f"E/I={ratio:.3f}±{ei_sd:.3f}  (per-seed E/I_sync mean={np.mean([r['ei_sync'] for r in rows]):.3f})", flush=True)

# ---- repo plot_ei_simple house style, lead's averaged-point + error-bars modification ----
font_manager.fontManager.addfont(os.path.join(B.REPO, "fonts", "Roboto-Regular.ttf"))
plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['svg.fonttype'] = 'none'
for k in ('font.size', 'xtick.labelsize', 'ytick.labelsize', 'axes.labelsize'):
    plt.rcParams[k] = 80
plt.rcParams['legend.fontsize'] = 44
plt.rcParams['axes.titlesize'] = 50

fig, ax = plt.subplots(figsize=(25, 25))
# OUR ensemble-averaged (E,I) point with both-axes error bars (SD across seeds)
ax.errorbar(E_m, I_m, xerr=E_sd, yerr=I_sd, fmt="o", ms=42, color="C0",
            ecolor="C0", elinewidth=5, capsize=18, capthick=5, zorder=5,
            label=f"Model ensemble (n={n})\nE/I = {ratio:.2f}")

# bio overlay — normalised exactly as repo: scale so bio grand-mean exc == model exc
bio_exc_all = np.concatenate([np.asarray(v[0]) for v in BIO.values()])
scale = E_m / float(bio_exc_all.mean())
markers = ["^", "s", "d", "v", "P", "X"]
for i, (lbl, (ex, ih)) in enumerate(BIO.items()):
    ax.scatter(np.asarray(ex) * scale, np.asarray(ih) * scale, s=2000,
               marker=markers[i % len(markers)], edgecolor="k", linewidth=1.0,
               zorder=4, label=lbl)

lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.5, zorder=1)
ax.set(xlim=(0, lim * 1.05), ylim=(0, lim * 1.05),
       xlabel="Excitation (a.u.)", ylabel="Inhibition (a.u.)",
       title="E vs I (model ensemble vs biology)")
ax.legend(frameon=False, fontsize=38, loc="upper left")
ax.tick_params(axis='both', which='major', length=20, width=1)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# ---- zoomed inset (lower-right, below the diagonal = empty): make the across-seed error bars
#      (both axes, SD over the 10 seeds) VISIBLE — they are sub-pixel on the biology scale because
#      the network's E/I balance is highly reproducible across seeds (the tight cloud IS the result). ----
axin = ax.inset_axes([0.55, 0.09, 0.41, 0.41])
padx = max(3.0 * E_sd, 0.8); pady = max(3.0 * I_sd, 0.8)
lo = min(E_m - padx, I_m - pady); hi = max(E_m + padx, I_m + pady)
axin.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.5, zorder=1)
axin.scatter(E, I, s=160, color="C0", alpha=0.30, edgecolor="none", zorder=3, label="per-seed (n=%d)" % n)
axin.errorbar(E_m, I_m, xerr=E_sd, yerr=I_sd, fmt="o", ms=22, color="C0", ecolor="C0",
              elinewidth=4, capsize=14, capthick=4, zorder=5)
axin.set_xlim(E_m - padx, E_m + padx); axin.set_ylim(I_m - pady, I_m + pady)
axin.set_title("zoom — ensemble mean ± SD (both axes)\n"
               f"E={E_m:.1f}±{E_sd:.1f}  I={I_m:.1f}±{I_sd:.1f}   E/I={ratio:.2f}±{ei_sd:.2f}",
               fontsize=28)
axin.tick_params(axis='both', which='major', labelsize=30, length=10, width=1)
for sp in ("top", "right"):
    axin.spines[sp].set_visible(False)
try:
    ax.indicate_inset_zoom(axin, edgecolor="0.55", alpha=0.7, lw=2)
except Exception as _e:
    print("   [inset connector skipped]:", _e, flush=True)
plt.tight_layout()
svg = os.path.join(OUTENS, "gate3_ei_ens.svg"); png = os.path.join(OUTENS, "gate3_ei_ens.png")
plt.savefig(svg, format="svg", bbox_inches="tight")
plt.savefig(png, format="png", dpi=100, bbox_inches="tight")
plt.close("all")
print(f"[gate3-ens] saved: {svg} / {png}", flush=True)
B.firewall("gate3-ens after")
