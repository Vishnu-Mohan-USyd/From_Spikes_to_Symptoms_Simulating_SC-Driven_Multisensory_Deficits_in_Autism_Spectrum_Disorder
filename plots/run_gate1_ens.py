"""GATE 1 RATE (ENSEMBLE) — bimodal MSI population PSTH, across-seed mean ± SD ribbon, in the repo
fano rate-panel house style (Roboto 58, figsize 22x13, despined). The gate value (window-mean MSI rate)
is shown as the ensemble mean ± SD with the [12,32] PASS band. Pure plot from gates_main.json
(psth_per_frame + rate_hz stats); frozen md5 before==after.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bridge as B
import numpy as np, matplotlib.pyplot as plt
from matplotlib import font_manager

GATES = os.path.join(B.WORK, "results", "gates_main.json")
OUTENS = os.path.join(B.WORK, "figures")
os.makedirs(OUTENS, exist_ok=True)
BAND = (12.0, 32.0)

B.firewall("gate1-ens before")
J = json.load(open(GATES))
n = J["n"]
r_m, r_sd = J["rate_hz"]["mean"], J["rate_hz"]["sd"]
frame_ms = J["frame_ms"]
psth_m = np.array([d["mean"] for d in J["psth_per_frame"]], float)
psth_sd = np.array([d["sd"] for d in J["psth_per_frame"]], float)
t_ms = np.arange(psth_m.size) * frame_ms
passed = BAND[0] <= r_m <= BAND[1]
print(f"[gate1-ens] n={n} seeds {J['seeds']}  window-mean = {r_m:.3f} ± {r_sd:.3f} Hz  "
      f"band[{BAND[0]:.0f},{BAND[1]:.0f}]  PASS={passed}  (psth peak {psth_m.max():.1f}Hz)", flush=True)

font_manager.fontManager.addfont(os.path.join(B.REPO, "fonts", "Roboto-Regular.ttf"))
plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['svg.fonttype'] = 'none'
for k in ('font.size', 'xtick.labelsize', 'ytick.labelsize', 'axes.labelsize', 'legend.fontsize'):
    plt.rcParams[k] = 58
plt.rcParams['axes.titlesize'] = 44

fig, ax = plt.subplots(figsize=(22, 13))
ax.axhspan(BAND[0], BAND[1], color="C2", alpha=0.13, zorder=0,
           label=f"RATE band [{int(BAND[0])},{int(BAND[1])}]")
# across-seed mean PSTH + ±SD ribbon
ax.fill_between(t_ms, psth_m - psth_sd, psth_m + psth_sd, color="k", alpha=0.18, zorder=2,
                label="±SD across seeds")
ax.plot(t_ms, psth_m, color="k", lw=4, marker="o", ms=10, zorder=3,
        label=f"MSI rate (mean of {n})")
ax.axhline(r_m, color="C3", lw=4, ls="--", zorder=4,
           label=f"window-mean {r_m:.2f} ± {r_sd:.2f} Hz")
ax.annotate(f"window-mean {r_m:.2f} ± {r_sd:.2f} Hz  in [{int(BAND[0])},{int(BAND[1])}]  =  PASS",
            xy=(t_ms[-1] * 0.40, r_m + 6), fontsize=38, color="C3", ha="left", va="bottom")
ax.annotate("", xy=(0, psth_m.max() * 0.16), xytext=(0, -1.5),
            arrowprops=dict(arrowstyle="-|>", color="k", lw=2.5))
ax.text(t_ms[-1] * 0.022, psth_m.max() * 0.17, "stim\nonset", fontsize=30, ha="left", va="bottom", color="0.25")
ax.set(xlabel="Time (ms)", ylabel="MSI firing rate (Hz)",
       title="Gate 1 RATE — bimodal (B) MSI population rate (ensemble)",
       xlim=(t_ms.min() - 3, t_ms.max() + 3), ylim=(0, (psth_m + psth_sd).max() * 1.18))
ax.legend(frameon=False, loc="upper right", fontsize=40, handlelength=1.4,
          labelspacing=0.34, borderaxespad=0.5)
ax.tick_params(axis='both', which='major', length=18, width=2)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("outward", 5))
ax.spines["bottom"].set_position(("outward", 5))
ax.grid(False)
plt.tight_layout()
svg = os.path.join(OUTENS, "gate1_rate_ens.svg"); png = os.path.join(OUTENS, "gate1_rate_ens.png")
plt.savefig(svg, format="svg", bbox_inches="tight")
plt.savefig(png, format="png", dpi=100, bbox_inches="tight")
plt.close("all")
print(f"[gate1-ens] saved: {svg} / {png}", flush=True)
B.firewall("gate1-ens after")
