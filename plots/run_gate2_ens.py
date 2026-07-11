"""GATE 2 TBW (ENSEMBLE) — repo TBW_test psychometric GRAPHIC house-style (plot_psychometric_tbw_ax),
across-seed error bars. P(fusion) vs A-V onset = per-SOA mean ± SD over the ensemble (measure_ens_main.py
tbw_pfusion_per_offset, the MDC.tbw_fused_counts_jit readout that defines the validated 260 ms width).
The validated metric = RAW half-max-of-peak width; we shade it and annotate raw_fwhm = mean ± SD. The
frozen TBW_test.fit_psychometric_curve_improved is used RUN-ONLY (md5 unchanged) for the C1 pedestal
guide. Pure plot from gates_main.json; frozen md5 before==after.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bridge as B
import numpy as np, matplotlib.pyplot as plt
from matplotlib import font_manager

GATES = os.path.join(B.WORK, "results", "gates_main.json")
OUTENS = os.path.join(B.WORK, "figures")
os.makedirs(OUTENS, exist_ok=True)


def _crossings_half(xv, yv, level):
    out = []
    for i in range(len(xv) - 1):
        y0, y1 = yv[i], yv[i + 1]
        if (y0 - level) * (y1 - level) <= 0 and y0 != y1:
            t = (level - y0) / (y1 - y0)
            out.append(xv[i] + t * (xv[i + 1] - xv[i]))
    return out


B.firewall("gate2-ens before")
# frozen TBW_test, run-only, for the pedestal-fit guide (md5 stays 80d33465)
TBW = B.load_repo_mod("TBW_test_local", os.path.join(B.WORK, "TBW_test.py"))

J = json.load(open(GATES))
n = J["n"]
offs = np.asarray(J["tbw_offsets"], float)
pf_m = np.asarray([d["mean"] for d in J["tbw_pfusion_per_offset"]], float)
pf_sd = np.asarray([d["sd"] for d in J["tbw_pfusion_per_offset"]], float)
fwhm_m, fwhm_sd = J["tbw_raw_fwhm"]["mean"], J["tbw_raw_fwhm"]["sd"]
print(f"[gate2-ens] n={n} seeds {J['seeds']}  raw_fwhm = {fwhm_m:.0f} ± {fwhm_sd:.0f} ms  "
      f"peak P(fus)={pf_m.max():.3f}", flush=True)

# validated metric: raw half-max-of-peak width on the across-seed MEAN curve
peak = float(pf_m.max()); half = 0.5 * peak
cr = _crossings_half(offs, pf_m, half)
xL = min(cr) if cr else float(offs.min()); xR = max(cr) if cr else float(offs.max())

# C1 pedestal guide via frozen fit (run-only)
try:
    fit = TBW.fit_psychometric_curve_improved(offs, pf_m)
    xs, ys = np.asarray(fit["xs"], float), np.asarray(fit["ys"], float)
except Exception as e:
    print(f"[gate2-ens] pedestal-fit guide skipped: {e!r}", flush=True)
    xs, ys = offs, pf_m

font_manager.fontManager.addfont(os.path.join(B.REPO, "fonts", "Roboto-Regular.ttf"))
plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['svg.fonttype'] = 'none'
for k in ('font.size', 'xtick.labelsize', 'ytick.labelsize', 'axes.labelsize', 'legend.fontsize'):
    plt.rcParams[k] = 60
plt.rcParams['axes.titlesize'] = 50

fig, ax = plt.subplots(figsize=(25, 14))
ax.errorbar(offs, pf_m, yerr=pf_sd, fmt='o', ms=14, capsize=8, color='C0',
            label=f'mean ± SD (n={n})', zorder=3, elinewidth=2.0, capthick=2.0)
ax.plot(xs, ys, lw=4, color='C1', label='Pedestal fit (mean)', zorder=2)
mask = (xs >= xL) & (xs <= xR)
ax.fill_between(xs, 0, half, where=mask, color='C1', alpha=.18, zorder=1)
ax.vlines([xL, xR], 0, half, ls=':', lw=3, color='C1')
ax.axhline(half, ls=':', lw=2, color='.4')
ax.axvline(0, ls='--', lw=2, color='k', alpha=.3)
ax.annotate(f"raw half-max width = {fwhm_m:.0f} ± {fwhm_sd:.0f} ms",
            xy=(0, half + 0.06), fontsize=42, color="C1", ha="center", va="bottom")
ax.set(xlabel='Audio – Visual onset (ms)', ylabel='Fusion probability',
       ylim=(-.05, 1.05), xlim=(offs.min() - 20, offs.max() + 20),
       title='Temporal Binding Window (ensemble)')
ax.legend(frameon=False, loc='upper right')
ax.grid(False)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(axis='both', which='major', length=20, width=1)
plt.tight_layout()
svg = os.path.join(OUTENS, "gate2_tbw_ens.svg"); png = os.path.join(OUTENS, "gate2_tbw_ens.png")
plt.savefig(svg, format="svg", bbox_inches="tight")
plt.savefig(png, format="png", dpi=100, bbox_inches="tight")
plt.close("all")
print(f"[gate2-ens] saved: {svg} / {png}  (half-max crossings [{xL:.0f},{xR:.0f}] ms)", flush=True)
B.firewall("gate2-ens after")
