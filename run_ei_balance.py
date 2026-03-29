"""E/I balance figure — separated synaptic currents, evoked window only.

Uses existing run_ei_probe_separated() to record full traces, then slices
to the evoked response window (stimulus + 100ms tail, Haider 2013) before
computing E/I ratio.

Evoked window: pulse_frames=5 (50ms) + 10 frames (100ms) = 15 frames.
At 100 substeps/frame = substeps 0:1500.
"""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager

from EI_balance_test import run_ei_probe_separated
from TBW_test import load_msi_model

BASE = Path("checkpoint")
MODELS = [BASE / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
SAVE = Path("Saved_Images")

# Color scheme
CLR_DATA = "#939598"
CLR_MAIN = "#b469a3"

PULSE_FRAMES = 5       # 50ms stimulus
N_FRAMES = 20          # 200ms total simulation
N_SUBSTEPS = 100       # per frame
EVOKED_FRAMES = 12.5   # stimulus (5) + 75ms tail (7.5) = 125ms
EVOKED_SUBSTEPS = int(EVOKED_FRAMES * N_SUBSTEPS)  # 1250


def run_ei_evoked(model_paths, device="cuda"):
    """Run E/I probe on all models, extract evoked-window statistics."""
    results = []

    for i, p in enumerate(model_paths):
        net = load_msi_model(p, device=device)
        res = run_ei_probe_separated(
            net, centre_deg=90.0, pulse_frames=PULSE_FRAMES,
            n_frames=N_FRAMES, intensity=1.0,
        )
        traces = res["traces"]

        # Slice to evoked window
        I_E_evoked = traces["I_E_mean"][:EVOKED_SUBSTEPS]
        I_I_evoked = traces["I_I_mean"][:EVOKED_SUBSTEPS]
        Q_E_evoked = traces["Q_E"][:EVOKED_SUBSTEPS]
        Q_I_evoked = traces["Q_I"][:EVOKED_SUBSTEPS]

        exc_mean = I_E_evoked.mean()
        inh_mean = I_I_evoked.mean()
        ei_ratio = exc_mean / (inh_mean + 1e-12)

        # Component breakdown (evoked window)
        ampa_evoked = traces["AMPA"][:EVOKED_SUBSTEPS].mean()
        nmda_evoked = traces["NMDA"][:EVOKED_SUBSTEPS].mean()
        ff_evoked = traces["FFInh"][:EVOKED_SUBSTEPS].mean()
        recur_evoked = traces["RecurInh"][:EVOKED_SUBSTEPS].mean()
        lat_evoked = traces["LatInh"][:EVOKED_SUBSTEPS].mean()

        r = {
            "exc_mean": float(exc_mean),
            "inh_mean": float(inh_mean),
            "ei_ratio": float(ei_ratio),
            "Q_E_total": float(Q_E_evoked.sum()),
            "Q_I_total": float(Q_I_evoked.sum()),
            "AMPA_mean": float(ampa_evoked),
            "NMDA_mean": float(nmda_evoked),
            "FFInh_mean": float(ff_evoked),
            "RecurInh_mean": float(recur_evoked),
            "LatInh_mean": float(lat_evoked),
        }
        results.append(r)
        print(f"  M{i:02d}: E={exc_mean:.4f} I={inh_mean:.4f} E/I={ei_ratio:.3f}  "
              f"AMPA={ampa_evoked:.4f} NMDA={nmda_evoked:.4f} "
              f"FF={ff_evoked:.4f} Rec={recur_evoked:.4f} Lat={lat_evoked:.4f}")

        del net; torch.cuda.empty_cache()

    # Aggregate
    keys = ["exc_mean", "inh_mean", "ei_ratio", "Q_E_total", "Q_I_total",
            "AMPA_mean", "NMDA_mean", "FFInh_mean", "RecurInh_mean", "LatInh_mean"]
    arrays = {k: np.array([r[k] for r in results]) for k in keys}
    n = len(results)
    summary = {}
    for k, arr in arrays.items():
        summary[k] = arr
        summary[f"{k}_mean"] = arr.mean()
        summary[f"{k}_sem"] = arr.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return summary


def plot_ei_scatter(summary, *, out_path_base=None):
    """Scatter plot: Excitation vs Inhibition, evoked window.

    Matches the original plot_ei_simple() style from the published code,
    with updated data (separated currents) and curated bio references.
    """
    exc = summary["exc_mean"]
    inh = summary["inh_mean"]

    # ── Bio references (current-based only) ──
    # Format: label -> ([exc_values], [inh_values])
    bio_data = {
        "Rat A1 (Wehr 2003)":     ([710], [650]),
        "Mouse V1 (Xue 2014)":    ([267], [251]),
        "Mouse V1 (Okun 2008)":   ([7.1], [6.9]),
    }

    # ── Font setup (match original: 80pt base) ──
    try:
        font_manager.fontManager.addfont('./fonts/Roboto-Regular.ttf')
        plt.rcParams['font.family'] = 'Roboto'
    except Exception:
        pass
    plt.rcParams['font.size'] = 80
    plt.rcParams['xtick.labelsize'] = 80
    plt.rcParams['ytick.labelsize'] = 80
    plt.rcParams['axes.titlesize'] = 50
    plt.rcParams['axes.labelsize'] = 80
    plt.rcParams['legend.fontsize'] = 80
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42

    fig, ax = plt.subplots(figsize=(25, 25))

    # ── Model data points — grey (#939598) ──
    ax.scatter(exc, inh, linewidth=0.1,
               s=2000, color=CLR_DATA, edgecolor="none", alpha=0.9,
               label="Model (n=10)")

    # ── Biological overlay (normalised to model scale) ──
    # Scale factor: set bio grand-mean E = model grand-mean E
    model_scale = exc.mean()
    bio_exc_all = np.concatenate([np.asarray(v[0]) for v in bio_data.values()])
    scale_factor = model_scale / bio_exc_all.mean()

    markers = ["^", "s", "d"]  # triangle, square, diamond
    for i, (lbl, (exc_bio, inh_bio)) in enumerate(bio_data.items()):
        exc_bio_s = np.asarray(exc_bio) * scale_factor
        inh_bio_s = np.asarray(inh_bio) * scale_factor
        ax.scatter(exc_bio_s, inh_bio_s,
                   s=2000, marker=markers[i % len(markers)],
                   edgecolor="k", linewidth=0.1,
                   label=lbl)

    # ── Model mean marker — purple (#b469a3) ──
    e_m, i_m = summary["exc_mean_mean"], summary["inh_mean_mean"]
    e_s, i_s = summary["exc_mean_sem"], summary["inh_mean_sem"]
    ax.errorbar(e_m, i_m, xerr=e_s, yerr=i_s,
                fmt='o', ms=25, color=CLR_MAIN, elinewidth=3, capsize=8,
                zorder=4, label=f"Mean E/I = {summary['ei_ratio_mean']:.2f}")

    # ── Unity line & cosmetics (match original) ──
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.5)
    ax.set(xlim=(0, lim * 1.05), ylim=(0, lim * 1.05),
           xlabel="Excitation (a.u.)", ylabel="Inhibition (a.u.)",
           title="E vs I (model vs biology)")
    ax.legend(frameon=False, fontsize=70)
    ax.tick_params(axis='both', which='major', length=20, width=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    base = out_path_base or str(SAVE / "EI_balance")
    fig.savefig(f"{base}.svg", format='svg')
    fig.savefig(f"{base}.png", format='png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {base}.svg and {base}.png")


if __name__ == "__main__":
    print("E/I BALANCE — separated currents, evoked window (125ms)")
    print(f"Evoked window: {PULSE_FRAMES} frames stim + 7.5 frames tail = {EVOKED_FRAMES} frames ({EVOKED_FRAMES*10:.0f}ms)\n")

    summary = run_ei_evoked(MODELS, device="cuda")

    print(f"\n{'='*60}")
    print("E/I BALANCE RESULTS (evoked window)")
    print(f"{'='*60}")
    print(f"  E/I ratio: {summary['ei_ratio_mean']:.3f} ± {summary['ei_ratio_sem']:.3f}")
    print(f"  Exc mean:  {summary['exc_mean_mean']:.4f} ± {summary['exc_mean_sem']:.4f}")
    print(f"  Inh mean:  {summary['inh_mean_mean']:.4f} ± {summary['inh_mean_sem']:.4f}")

    # Component breakdown
    total_E = summary['AMPA_mean_mean'] + summary['NMDA_mean_mean']
    total_I = summary['FFInh_mean_mean'] + summary['RecurInh_mean_mean'] + summary['LatInh_mean_mean']
    print(f"\n  Excitatory breakdown:")
    print(f"    AMPA:     {summary['AMPA_mean_mean']:.4f} ({100*summary['AMPA_mean_mean']/total_E:.1f}%)")
    print(f"    NMDA:     {summary['NMDA_mean_mean']:.4f} ({100*summary['NMDA_mean_mean']/total_E:.1f}%)")
    print(f"  Inhibitory breakdown:")
    print(f"    FFInh:    {summary['FFInh_mean_mean']:.4f} ({100*summary['FFInh_mean_mean']/total_I:.1f}%)")
    print(f"    RecurInh: {summary['RecurInh_mean_mean']:.4f} ({100*summary['RecurInh_mean_mean']/total_I:.1f}%)")
    print(f"    LatInh:   {summary['LatInh_mean_mean']:.4f} ({100*summary['LatInh_mean_mean']/total_I:.1f}%)")

    plot_ei_scatter(summary, out_path_base=str(SAVE / "EI_balance"))
