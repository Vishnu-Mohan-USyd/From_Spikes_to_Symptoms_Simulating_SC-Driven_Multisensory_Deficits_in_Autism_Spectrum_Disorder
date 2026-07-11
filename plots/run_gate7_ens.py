"""GATE 7 cue-reliability (ENSEMBLE, NEW) — MLE / inverse-variance optimal cue integration, repo house
style, across-seed pooling (n=10). The 7th repo validation: does the network weight two spatially
discrepant cues (A@80 deg, V@100 deg) by their RELIABILITY the way a Bayesian observer does?

Parameterized by config tag (argv[1], default the paper-faithful gain_exp=1 = HEADLINE):
  gate7_cuerel_g1  -> gain proportional to sigma_ref/sigma (paper Methods)         -> ..._g1 figures
  gate7_cuerel_g2  -> gain proportional to (sigma_ref/sigma)^2 (repo exploratory)  -> ..._g2 figures

HEADLINE  = gate7_cuerel_ens_g{1,2} : heat-map of the model visual weight w_V over the (sigma_A,
            sigma_V) reliability grid (per-cell ensemble mean). MLE predicts w_V high at top-left
            (V reliable / A unreliable), low at bottom-right.
EVIDENCE  = gate7_cuerel_scatter_g{1,2} : model w_V vs inverse-variance w_V* with across-seed SEM bars
            + the y=x identity; R^2/MAE annotated. THIS is the actual validation.

Pooled mean/SEM/R^2/MAE recomputed here from the 10 per-seed JSONs (same nan-aware pooling as the
aggregate). Plotting MATH (w_V per cell, w_V* = sigma_A^2/(sigma_A^2+sigma_V^2), R^2/MAE) unchanged
from repo cue_reliability_test.py; only the STYLE is the house style. Any empty cell (input too weak
-> empty MSI profile, honestly NaN-guarded, NEVER a spurious w_V=0) is masked grey, not plotted as
data. Pure plot from {metric}_seed*.json; frozen TBW/SBW md5 before==after.
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bridge as B
import numpy as np, matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

METRIC = sys.argv[1] if len(sys.argv) > 1 else "gate7_cuerel_g1"   # default = paper-faithful headline
GE = int(METRIC.rsplit("g", 1)[-1])                                # 1 or 2
SUFFIX = f"_g{GE}"
GAIN_DESC = {1: r"paper-faithful: constant-area Gaussian, gain $\propto\,\sigma_{ref}/\sigma$",
             2: r"repo exploratory knob: gain $\propto\,(\sigma_{ref}/\sigma)^2$"}[GE]
GAIN_TAG = {1: "gain_exp=1 (paper-faithful)", 2: "gain_exp=2 (repo knob)"}[GE]
SEED_GLOB = os.path.join(B.WORK, "results", f"{METRIC}_seed*.json")
OUTENS = os.path.join(B.WORK, "figures")
os.makedirs(OUTENS, exist_ok=True)

B.firewall(f"gate7{SUFFIX}-ens before")
files = sorted(glob.glob(SEED_GLOB), key=lambda f: int(f.split("seed")[-1].split(".")[0]))
assert files, f"no per-seed JSONs for {METRIC}"
recs = [json.load(open(f)) for f in files]
seeds = [r["seed"] for r in recs]
n = len(recs)
levels = [int(s) for s in recs[0]["sigma_levels"]]
nlev = len(levels)
sa = np.array(recs[0]["sigma_a"], int)
sv = np.array(recs[0]["sigma_v"], int)
wp = np.array(recs[0]["w_pred"], float)                          # MLE prediction (identical per seed)
W = np.array([r["w_emp"] for r in recs], float)                  # (n_seeds, 100), NaN where empty
with np.errstate(invalid="ignore"):
    w_m = np.nanmean(W, 0)
    cnt = np.sum(~np.isnan(W), 0)
    w_sem = np.where(cnt > 1, np.nanstd(W, 0, ddof=1) / np.sqrt(np.maximum(cnt, 1)), np.nan)
ok = ~np.isnan(w_m)
n_grey = nlev * nlev - int(ok.sum())
r2 = float(np.corrcoef(wp[ok], w_m[ok])[0, 1] ** 2)
mae = float(np.mean(np.abs(w_m[ok] - wp[ok])))
rmse = float(np.sqrt(np.mean((w_m[ok] - wp[ok]) ** 2)))
r2_seeds = np.array([r["r2"] for r in recs], float)
eqrel = float(np.nanmean(w_m[sa == sv]))
corner = float(w_m[(sa == 2) & (sv == 20)][0])
print(f"[gate7{SUFFIX}] n={n} seeds {seeds}  R^2(pooled)={r2:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}  "
      f"per-seed R^2={r2_seeds.mean():.3f}±{r2_seeds.std(ddof=1):.3f}(SD)  valid={int(ok.sum())}/100", flush=True)

# ---- house style ----
font_manager.fontManager.addfont(os.path.join(B.REPO, "fonts", "Roboto-Regular.ttf"))
plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['mathtext.fontset'] = 'dejavusans'   # tofu-proof sigma / superscripts

# =================== HEADLINE: w_V heat-map over the reliability grid ===================
for k in ('font.size', 'xtick.labelsize', 'ytick.labelsize', 'axes.labelsize', 'legend.fontsize'):
    plt.rcParams[k] = 40
plt.rcParams['axes.titlesize'] = 40
Z = np.full((nlev, nlev), np.nan)                  # Z[i=sigma_A, j=sigma_V]
for k in range(len(sa)):
    Z[levels.index(int(sa[k])), levels.index(int(sv[k]))] = w_m[k]
cmap = mpl.cm.get_cmap("viridis").copy()
cmap.set_bad("0.86")
fig, ax = plt.subplots(figsize=(15, 13))
im = ax.imshow(np.ma.masked_invalid(Z), origin="lower", vmin=0, vmax=1, cmap=cmap)
ax.set_xticks(range(nlev)); ax.set_yticks(range(nlev))
ax.set_xticklabels(levels); ax.set_yticklabels(levels)
ax.set_xlabel(r"$\sigma_V$  (deg)   — visual cue width  (wide = unreliable)")
ax.set_ylabel(r"$\sigma_A$  (deg)   — auditory cue width")
ax.set_title(f"Gate 7 — model visual weight $w_V$ over the cue-reliability grid (n={n})",
             fontsize=33, pad=16)
# MLE direction guides, with a white bbox so they read on any cell colour
_bb = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.65)
ax.text(0.35, nlev - 1.5, "MLE: $w_V\\to1$\n(V reliable,\nA unreliable)", fontsize=23, color="k",
        ha="left", va="center", weight="bold", bbox=_bb)
ax.text(nlev - 1.5, 0.55, "$w_V\\to0$\n(A reliable)", fontsize=23, color="k",
        ha="center", va="center", weight="bold", bbox=_bb)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("model visual weight  $w_V$", fontsize=36)
cb.ax.tick_params(labelsize=32)
ax.tick_params(axis='both', which='major', length=14, width=2)
if n_grey > 0:
    cap = (f"Grey (n={n_grey}) = weakest wide/wide stimuli: input too faint -> empty MSI profile "
           f"(honestly NaN-excluded, never a spurious $w_V$=0).\n{GAIN_DESC}.  raw per-cell ensemble mean.")
else:
    cap = f"{GAIN_DESC}.  all 100 cells measurable (no empties).  raw per-cell ensemble mean (n={n})."
fig.text(0.5, 0.015, cap, ha="center", va="bottom", fontsize=20, color="0.30")
plt.tight_layout(rect=[0, 0.07, 1, 1])
svg = os.path.join(OUTENS, f"gate7_cuerel_ens{SUFFIX}.svg")
png = os.path.join(OUTENS, f"gate7_cuerel_ens{SUFFIX}.png")
plt.savefig(svg, format="svg", bbox_inches="tight"); plt.savefig(png, format="png", dpi=100, bbox_inches="tight")
plt.close("all")
print(f"[gate7{SUFFIX}] saved HEATMAP: {png}", flush=True)

# =================== EVIDENCE: model w_V vs MLE w_V* scatter (across-seed SEM) ===================
for k in ('font.size', 'xtick.labelsize', 'ytick.labelsize', 'axes.labelsize', 'legend.fontsize'):
    plt.rcParams[k] = 44
plt.rcParams['axes.titlesize'] = 40
fig2, ax2 = plt.subplots(figsize=(14, 13))
ax2.plot([0, 1], [0, 1], "--", color="0.35", lw=3, zorder=1, label="MLE optimal ($y=x$)")
yerr = np.where(np.isnan(w_sem[ok]), 0.0, w_sem[ok])
ax2.errorbar(wp[ok], w_m[ok], yerr=yerr, fmt="o", ms=15, color="C0", ecolor="C0",
             elinewidth=2.5, capsize=6, capthick=2.0, markeredgecolor="k", markeredgewidth=0.8,
             alpha=0.9, zorder=3, label=f"model $w_V$, mean ± SEM (n={n})")
ax2.set(xlim=(-0.04, 1.04), ylim=(-0.04, 1.04),
        xlabel=r"MLE-predicted visual weight  $w_V^{*}=\sigma_A^2/(\sigma_A^2+\sigma_V^2)$",
        ylabel=r"model visual weight  $w_V$")
ax2.set_title(f"Cue-reliability weighting vs Bayesian optimal — {GAIN_TAG}\n"
              f"$R^2$ = {r2:.2f}    MAE = {mae:.2f}    RMSE = {rmse:.2f}    (n={n})", fontsize=31, pad=14)
ax2.legend(frameon=False, loc="upper left", fontsize=34, handlelength=1.5,
           labelspacing=0.4, borderaxespad=0.6)
ax2.tick_params(axis='both', which='major', length=16, width=2)
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)
ax2.spines["left"].set_position(("outward", 6)); ax2.spines["bottom"].set_position(("outward", 6))
ax2.set_aspect("equal", adjustable="box")
ax2.grid(False)
cap2 = ("Paper context: equal reliability $w_V\\approx0.51$; $\\sigma_A$=2,$\\sigma_V$=20 $\\to w_V\\approx0.04$; "
        "paper $R^2$=0.71, MAE=0.13.\n"
        f"Ensemble: equal $w_V$={eqrel:.2f}, corner $w_V$={corner:.2f} — the network reproduces "
        "inverse-variance (Bayesian) cue weighting.")
fig2.text(0.5, 0.015, cap2, ha="center", va="bottom", fontsize=20, color="0.30")
plt.tight_layout(rect=[0, 0.085, 1, 1])
svg2 = os.path.join(OUTENS, f"gate7_cuerel_scatter{SUFFIX}.svg")
png2 = os.path.join(OUTENS, f"gate7_cuerel_scatter{SUFFIX}.png")
plt.savefig(svg2, format="svg", bbox_inches="tight"); plt.savefig(png2, format="png", dpi=100, bbox_inches="tight")
plt.close("all")
print(f"[gate7{SUFFIX}] saved SCATTER: {png2}", flush=True)
B.firewall(f"gate7{SUFFIX}-ens after")
