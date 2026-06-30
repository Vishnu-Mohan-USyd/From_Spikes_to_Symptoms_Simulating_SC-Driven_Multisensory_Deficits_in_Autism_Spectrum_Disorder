"""GATE 4 IE (ENSEMBLE) — repo inverse_effectiveness_test broken-axis log-x inverted-U GRAPHIC, with
across-seed error bars. MEI per intensity = mean +/- SD over the ensemble (measure_ens_main.py
mei_per_intensity). Same broken-axis figure as the single-seed (figsize 25x20, Roboto 80, zig-zag
break, log-x), yerr now = ensemble SD. Pure plot from gates_main.json; frozen md5 before==after.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bridge as B
import numpy as np, matplotlib.pyplot as plt
from matplotlib import font_manager

GATES = os.path.join(B.WORK, "results", "gates_main.json")
OUTENS = os.path.join(B.WORK, "figures")
os.makedirs(OUTENS, exist_ok=True)

B.firewall("gate4-ens before")
J = json.load(open(GATES))
INTENSITIES = list(J["ie_intensities"])
mei_mean = np.array([d["mean"] for d in J["mei_per_intensity"]], float)
mei_sd = np.array([d["sd"] for d in J["mei_per_intensity"]], float)
n = J["n"]
print(f"[gate4-ens] n={n} seeds {J['seeds']}  MEI/intensity: " +
      "  ".join(f"{INTENSITIES[j]}:{mei_mean[j]:.2f}±{mei_sd[j]:.2f}" for j in range(len(INTENSITIES))),
      flush=True)

font_manager.fontManager.addfont(os.path.join(B.REPO, "fonts", "Roboto-Regular.ttf"))
plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['svg.fonttype'] = 'none'
for k in ('font.size', 'xtick.labelsize', 'ytick.labelsize', 'axes.labelsize', 'legend.fontsize'):
    plt.rcParams[k] = 80
plt.rcParams['axes.titlesize'] = 50

fig = plt.figure(figsize=(25, 20))
gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
ax = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax)
for a in (ax, ax2):
    a.errorbar(INTENSITIES, mei_mean, yerr=mei_sd, marker='o', ms=18, capsize=10,
               capthick=3, elinewidth=3, ls='-', lw=3, color="C0",
               label=f"MEI mean ± SD (n={n})")
y_lo, y_hi = 0.00, 0.50
ax.set_ylim(y_hi * 1.05, (mei_mean + mei_sd).max() * 1.10)
ax2.set_ylim(y_lo, y_lo + 1e-6)
ax.xaxis.set_visible(False)
ax2.set_yticks([])
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()
d = .015
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs); ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs['transform'] = ax2.transAxes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs); ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
ax2.set_xscale('log')
ax2.set_xlabel("Stimulus intensity (a.u., log scale)")
ax.set_ylabel("MEI")
ax.set_title("Inverse effectiveness (ensemble)")
ax2.axhline(0, color='k', lw=.8)
pk = int(mei_mean.argmax())
ax.annotate(f"peak {mei_mean[pk]:.2f} ± {mei_sd[pk]:.2f} @ {INTENSITIES[pk]}",
            xy=(INTENSITIES[pk], mei_mean[pk]), xytext=(INTENSITIES[pk] * 1.15, mei_mean[pk] * 0.92),
            fontsize=46, color="C3")
ax.legend(frameon=False, loc="upper right", fontsize=52)
ax.tick_params(axis='both', which='major', length=30, width=1)
ax2.tick_params(axis='both', which='major', length=30, width=1)
plt.tight_layout()
svg = os.path.join(OUTENS, "gate4_ie_ens.svg"); png = os.path.join(OUTENS, "gate4_ie_ens.png")
plt.savefig(svg, format="svg", bbox_inches="tight")
plt.savefig(png, format="png", dpi=100, bbox_inches="tight")
plt.close("all")
print(f"[gate4-ens] saved: {svg} / {png}", flush=True)
B.firewall("gate4-ens after")
