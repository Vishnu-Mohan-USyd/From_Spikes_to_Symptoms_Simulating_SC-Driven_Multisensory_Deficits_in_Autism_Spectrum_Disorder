"""Task #136-V1 SBW recovery — bypass broken fit_sbw_pedestal.

Loads cache/sbw_{cond}_t10.npz (5 conds, 10 ckpts pooled), measures
HW directly via half-max-width on the pooled mean_prob curve using
absolute 0.5 threshold (paper convention). Outputs:
  - Verbatim per-cond HW + raw curve characterization
  - Overlay plot of all 5 curves at task136_logs/sbw_recovery_overlay.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
OUT = Path(__file__).parent

CONDS = ["control", "ff_inhibition", "adaptation", "nmda", "nmda_increase"]
COLORS = {"control": "#39b54a", "ff_inhibition": "#b469a3", "adaptation": "#f7941d",
          "nmda": "#1f77b4", "nmda_increase": "#d62728"}


def _find_crossings(xs, ys, threshold):
    """Find x where ys crosses threshold (linear interpolation)."""
    crossings = []
    for i in range(len(ys) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (y0 - threshold) * (y1 - threshold) < 0:
            # linear interp
            t = (threshold - y0) / (y1 - y0)
            x_cross = xs[i] + t * (xs[i + 1] - xs[i])
            crossings.append(x_cross)
        elif y0 == threshold:
            crossings.append(xs[i])
    return crossings


def measure_hw(separations, p_fusion, threshold=0.5):
    """HW = (right_cross - left_cross) / 2 at absolute threshold.

    If no crossings (curve never drops below threshold over the
    measured range), HW is reported as off-scale (> half range).
    """
    cross = _find_crossings(separations, p_fusion, threshold)
    if len(cross) >= 2:
        return float((cross[-1] - cross[0]) / 2), "fit"
    if len(cross) == 1:
        # one-sided cross — use distance from center to cross
        return float(abs(cross[0])), "one-sided"
    # zero crossings → saturated or always-below
    if p_fusion.min() >= threshold:
        return float("inf"), "saturated_above"
    return 0.0, "saturated_below"


def main():
    print("=" * 78)
    print("Task #136-V1  SBW HW recovery from cache (10 ckpts/cond × v2 Training.py)")
    print("=" * 78)
    print(f"{'cond':18s} {'min':>6s} {'max':>6s} {'p(sep=0)':>9s} {'p(±80)':>9s}  "
          f"{'HW (deg)':>10s}  method")
    print("-" * 78)

    fig, ax = plt.subplots(figsize=(7, 5))
    hw_table = {}

    for cond in CONDS:
        path = CACHE / f"sbw_{cond}_t10.npz"
        d = np.load(path)
        seps = d["separations_deg"]
        p = d["mean_prob"]
        sem = d["sem_prob"]
        # Center: separation closest to 0
        ic = int(np.argmin(np.abs(seps)))
        hw, method = measure_hw(seps, p, threshold=0.5)
        hw_str = f"{hw:.1f}" if np.isfinite(hw) else ">80 (sat)"
        print(f"{cond:18s} {p.min():6.3f} {p.max():6.3f} {p[ic]:9.3f} "
              f"{p[0]:.3f}/{p[-1]:.3f}  {hw_str:>10s}  {method}")
        hw_table[cond] = hw

        ax.errorbar(seps, p, yerr=sem, marker="o", ms=4, lw=1.5,
                    color=COLORS[cond], label=f"{cond} (HW={hw_str})")

    ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Spatial separation (deg)")
    ax.set_ylabel("P(fusion) — enhancement threshold=10")
    ax.set_title("SBW recovery — pooled across 10 surr_10 ckpts (v2 Training.py)")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="lower center", fontsize=8)
    fig.tight_layout()
    plot_path = OUT / "sbw_recovery_overlay.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot: {plot_path}")
    print(f"\nVerdict: {sum(1 for h in hw_table.values() if np.isfinite(h))}/5 conds yield finite HW.")
    if not all(np.isfinite(h) for h in hw_table.values()):
        print("WARNING: ≥1 cond's pooled P(fusion) never crosses 0.5 → SBW HW undefined "
              "(curve saturated above 0.5 across full ±80° range).")


if __name__ == "__main__":
    main()
