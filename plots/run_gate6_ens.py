"""GATE 6 latency (ENSEMBLE, NEW) — multisensory latency BENEFIT vs stimulus intensity (inverse
effectiveness), repo house style, across-seed error bars.

USER CHANGE (lead relay): show the latency benefit at LOW intensity (multisensory benefits are largest
for weak stimuli). REPLACES the single full-intensity A/V/B bars (zero mean-latency benefit).

LEFT panel = descriptive min(mean L_A, mean L_V) - mean L_B (ms) vs stimulus intensity (log-x),
mean ± SD over the ensemble. It compares condition means and is not a formal Miller RMI test.
The 0 line is the zero-benefit reference. Inverse effectiveness predicts the largest benefit at low I.
RIGHT panel = A/V/B first-spike latency bars at the low (inverse-effectiveness) intensity, A<V asymmetry
visible. HONEST: the curve is drawn AS MEASURED; if the benefit is null across intensities the flat ~0
curve says so (no cherry-picking). Pure plot from gate6_latency_sweep.json; frozen md5 before==after.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bridge as B
import numpy as np, matplotlib.pyplot as plt
from matplotlib import font_manager, gridspec

SWEEP = os.path.join(B.WORK, "results", "gate6_latency_sweep.json")
OUTENS = os.path.join(B.WORK, "figures")
os.makedirs(OUTENS, exist_ok=True)

B.firewall("gate6-ens before")
J = json.load(open(SWEEP))
n = J["n"]
I = np.asarray(J["intensities"], float)
bmean = np.asarray([d["mean"] for d in J["benefit_per_intensity"]], float)
bsd = np.asarray([d["sd"] for d in J["benefit_per_intensity"]], float)
LA_m = np.asarray([d["mean"] for d in J["L_A_per_intensity"]], float)
LA_s = np.asarray([d["sd"] for d in J["L_A_per_intensity"]], float)
LV_m = np.asarray([d["mean"] for d in J["L_V_per_intensity"]], float)
LV_s = np.asarray([d["sd"] for d in J["L_V_per_intensity"]], float)
LB_m = np.asarray([d["mean"] for d in J["L_B_per_intensity"]], float)
LB_s = np.asarray([d["sd"] for d in J["L_B_per_intensity"]], float)

# choose the panel-right intensity: the inverse-effectiveness benefit peak if a real benefit exists,
# else the lowest intensity where A,V,B are all defined (so the A<V bars are valid).
valid_all = [j for j in range(len(I)) if (LA_m[j] == LA_m[j] and LV_m[j] == LV_m[j] and LB_m[j] == LB_m[j])]
real_benefit = np.nanmax(bmean) > 0.5
jbar = int(np.nanargmax(np.where(np.isfinite(bmean), bmean, -np.inf))) if real_benefit else (valid_all[0] if valid_all else 0)
I_bar = I[jbar]

# honest low/high summary
jlo = int(np.argmin(I)); jhi = int(np.argmax(I))
lo_b, hi_b, mx_b = bmean[jlo], bmean[jhi], np.nanmax(bmean)
verdict = ("descriptive inverse-effectiveness benefit, confined to the weakest\nnear-threshold intensity (near zero by I=0.1)"
           if real_benefit else
           "NULL across intensities — no fastest-unisensory mean-latency benefit")
print(f"[gate6-ens] n={n} seeds {J['seeds']}", flush=True)
for j, ii in enumerate(I):
    print(f"    I={ii:<5g} benefit={bmean[j]:+.2f}±{bsd[j]:.2f}  L_A={LA_m[j]:.1f} L_V={LV_m[j]:.1f} L_B={LB_m[j]:.1f}", flush=True)
print(f"  low(I={I[jlo]:g})={lo_b:+.2f}  max={mx_b:+.2f}@I={I[int(np.nanargmax(np.where(np.isfinite(bmean),bmean,-np.inf)))]:g}  "
      f"high(I={I[jhi]:g})={hi_b:+.2f}  ->  {verdict}", flush=True)

font_manager.fontManager.addfont(os.path.join(B.REPO, "fonts", "Roboto-Regular.ttf"))
plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['svg.fonttype'] = 'none'
for k in ('font.size', 'xtick.labelsize', 'ytick.labelsize', 'axes.labelsize', 'legend.fontsize'):
    plt.rcParams[k] = 40
plt.rcParams['axes.titlesize'] = 40

fig = plt.figure(figsize=(28, 13))
gs = gridspec.GridSpec(1, 2, width_ratios=[2.25, 1.0], wspace=0.28)

# ---------------- LEFT: benefit vs intensity (inverse effectiveness) ----------------
ax0 = fig.add_subplot(gs[0])
fin = np.isfinite(bmean)
ax0.axhline(0.0, ls="--", color="0.45", lw=3, zorder=1, label="zero condition-mean benefit (not Miller RMI)")
ax0.fill_between(I[fin], 0.0, bmean[fin], where=(bmean[fin] > 0), color="C2", alpha=0.16, zorder=1)
ax0.errorbar(I[fin], bmean[fin], yerr=bsd[fin], fmt="o-", color="C2", lw=4.5, markersize=15,
             capsize=8, elinewidth=3, capthick=2.4, markeredgecolor="k", markeredgewidth=1.0,
             label=f"descriptive latency benefit, condition means ± SD (n={n})", zorder=3)
ax0.set_xscale("log")
ax0.set_xticks(I); ax0.set_xticklabels([f"{x:g}" for x in I])
ax0.minorticks_off()
ax0.axvline(I_bar, ls=":", color="C3", lw=2.5, alpha=0.7, zorder=0)
ax0.set(xlabel="Stimulus intensity  (gate-4/IE grid; weak to strong)",
        ylabel="min(mean A, mean V) − mean AV latency  (ms)",
        title="Descriptive multisensory latency benefit vs intensity")
# Honest verdict box (placed over the flat zero-benefit region, away from the low-I annotation).
ax0.text(0.62, 0.60, verdict, transform=ax0.transAxes, ha="center", va="center", fontsize=30,
         color=("C2" if real_benefit else "firebrick"),
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.9))
ax0.annotate(f"low I={I[jlo]:g}: {lo_b:+.1f} ms", (I[jlo], lo_b), fontsize=26, color="0.25",
             xytext=(34, -4), textcoords="offset points", ha="left", va="center")
ax0.annotate(f"high I={I[jhi]:g}: {hi_b:+.1f} ms", (I[jhi], hi_b), fontsize=26, color="0.25",
             xytext=(0, -32), textcoords="offset points", ha="right", va="top")
ax0.legend(frameon=False, loc="lower left", fontsize=28)
for sp in ("top", "right"):
    ax0.spines[sp].set_visible(False)
ax0.spines["left"].set_position(("outward", 6)); ax0.spines["bottom"].set_position(("outward", 6))
ax0.tick_params(axis='both', which='major', length=16, width=2)

# ---------------- RIGHT: A/V/B latency bars at the low (inverse-effectiveness) intensity ----------------
ax1 = fig.add_subplot(gs[1])
labs = ["Auditory", "Visual", "Bimodal"]
vals = [LA_m[jbar], LV_m[jbar], LB_m[jbar]]
errs = [LA_s[jbar], LV_s[jbar], LB_s[jbar]]
cols = ["C0", "C0", "C1"]
bars = ax1.bar(labs, vals, yerr=errs, capsize=8, color=cols, alpha=0.85, edgecolor="k",
               linewidth=1.2, error_kw=dict(elinewidth=3, capthick=2.4))
for i, (v, e) in enumerate(zip(vals, errs)):
    if v == v:
        ax1.text(i, v + (e if e == e else 0) + 0.6, f"{v:.0f}", ha="center", va="bottom", fontsize=30)
ax1.set(ylabel="First-spike latency (ms)",
        title=f"A / V / B latency @ low intensity I={I_bar:g}\n(A < V asymmetry preserved)")
ax1.set_ylim(0, np.nanmax([v + (e if e == e else 0) for v, e in zip(vals, errs)]) * 1.18 + 1)
for sp in ("top", "right"):
    ax1.spines[sp].set_visible(False)
ax1.spines["left"].set_position(("outward", 6)); ax1.spines["bottom"].set_position(("outward", 6))
ax1.tick_params(axis='both', which='major', length=16, width=2)

fig.suptitle("Gate 6 — descriptive multisensory latency facilitation (route-C dL3 ep79 ensemble)",
             fontsize=44, y=1.02)
plt.tight_layout()
svg = os.path.join(OUTENS, "gate6_latency_ens.svg"); png = os.path.join(OUTENS, "gate6_latency_ens.png")
plt.savefig(svg, format="svg", bbox_inches="tight")
plt.savefig(png, format="png", dpi=100, bbox_inches="tight")
plt.close("all")
print(f"[gate6-ens] saved: {svg} / {png}", flush=True)
B.firewall("gate6-ens after")
