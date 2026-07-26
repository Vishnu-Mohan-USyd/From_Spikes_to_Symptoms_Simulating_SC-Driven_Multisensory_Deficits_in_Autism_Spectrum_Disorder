"""E/I balance probes for ``MultiBatchAudVisMSINetworkTime``.

Provides the **separated synaptic** E/I probe used by ``run_ei_balance.py``:
``run_ei_probe_separated`` records each excitatory and inhibitory
component independently via ``net.start_ei_recording()`` /
``net.stop_ei_recording()``:

    Excitatory: AMPA, NMDA
    Inhibitory: FFInh (feed-forward), RecurInh (MSI->MSI),
                LatInh (lateral / surround).

This decomposition lets us report the E/I ratio plus fractional
contributions matching Wehr & Zador (2003), Xue et al. (2014) etc.

(Earlier sign-based probes that pooled all positive / negative inputs
into "E" / "I" have been removed.)

Units / shapes
--------------
Synaptic currents are stored in arbitrary model units (see
``Training.py`` for the conductance-based dynamics).  Traces returned by
``net.stop_ei_recording()`` are 1-D arrays over substeps
(``n_frames * n_substeps``, default 100 substeps/frame -> 1 ms / substep
when n_frames ticks at 10 ms each).  Summary scalars are population
means over MSI excit neurons and substeps.

Randomness
----------
The probes are deterministic given the network weights and stimulus.

Side effects
------------
``run_ei_probe_separated`` saves and restores ``plasticity_enabled``
and ``g_FFinh`` so that probing does not perturb adaptation state.
"""
from Training import *
from matplotlib import font_manager


def load_msi_model(ckpt_path: Path, *, device="cuda"):
    """
    Restore a MultiBatchAudVisMSINetworkTime exactly as it was saved.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
    net.load_state_dict(ckpt["model_state"])
    for k, v in ckpt["mutable_hparams"].items():
        setattr(net, k, v)

    net.to(device).eval()
    return net


# ============================================================================
#  NEW: Separated E/I probe using individual synaptic current components
# ============================================================================

def run_ei_probe_separated(
        net,
        *,
        centre_deg: float = 90.0,
        sigma_in: float = 5.0,
        pulse_frames: int = 5,
        n_frames: int = 20,
        intensity: float = 1.0,
) -> dict:
    """Run a synchronous AV pulse and record SEPARATE synaptic current components.

    Uses net.start_ei_recording() / net.stop_ei_recording() to capture:
      Excitatory: AMPA, NMDA
      Inhibitory: FFInh, RecurInh, LatInh

    Parameters
    ----------
    net : MultiBatchAudVisMSINetworkTime
        Network model (already on device).
    centre_deg : float
        Spatial centre of the stimulus in degrees.
    sigma_in : float
        Gaussian width of the stimulus.
    pulse_frames : int
        Duration of the stimulus pulse in external frames (10 ms each).
    n_frames : int
        Total simulation duration in external frames.
    intensity : float
        Peak stimulus intensity.

    Returns
    -------
    dict with keys:
        exc_mean, inh_mean, ei_ratio,
        Q_E_total, Q_I_total, charge_ratio,
        AMPA_mean, NMDA_mean, FFInh_mean, RecurInh_mean, LatInh_mean,
        traces (raw per-substep arrays)
    """
    B, N = 1, net.n
    net.reset_state(batch_size=B)

    # Save and disable plasticity
    orig_plasticity = getattr(net, 'plasticity_enabled', True)
    orig_g_FFinh = net.g_FFinh
    net.plasticity_enabled = False

    # Build synchronous AV stimulus (offset=0)
    xA = torch.zeros(n_frames, N, device=net.device)
    xV = torch.zeros_like(xA)

    idx_c = int(round(centre_deg * (N - 1) / (net.space_size - 1)))
    xs = torch.arange(N, dtype=torch.float32, device=net.device)
    gauss = torch.exp(-0.5 * ((xs - idx_c) / sigma_in) ** 2) * intensity

    xA[:pulse_frames] = gauss
    xV[:pulse_frames] = gauss

    # Start recording
    net.start_ei_recording()

    # Run simulation
    for t in range(n_frames):
        net.update_all_layers_batch(xA[t].unsqueeze(0), xV[t].unsqueeze(0))

    # Stop recording
    traces = net.stop_ei_recording()

    # Restore
    net.plasticity_enabled = orig_plasticity
    net.g_FFinh = orig_g_FFinh

    # Compute summary statistics
    exc_mean = traces["I_E_mean"].mean()
    inh_mean = traces["I_I_mean"].mean()
    Q_E_total = traces["Q_E"].sum()
    Q_I_total = traces["Q_I"].sum()

    return {
        "exc_mean": float(exc_mean),
        "inh_mean": float(inh_mean),
        "ei_ratio": float(exc_mean / (inh_mean + 1e-12)),
        "Q_E_total": float(Q_E_total),
        "Q_I_total": float(Q_I_total),
        "charge_ratio": float(Q_E_total / (Q_I_total + 1e-12)),
        "AMPA_mean": float(traces["AMPA"].mean()),
        "NMDA_mean": float(traces["NMDA"].mean()),
        "FFInh_mean": float(traces["FFInh"].mean()),
        "RecurInh_mean": float(traces["RecurInh"].mean()),
        "LatInh_mean": float(traces["LatInh"].mean()),
        "traces": traces,
    }


def pool_ei_separated(
        model_paths,
        *,
        device: str = "cuda",
        modify_net=None,
        **probe_kw,
) -> dict:
    """Run separated E/I probe across all model replicas.

    Returns dict with per-model arrays and summary statistics.
    """
    all_results = []

    for i, p in enumerate(model_paths):
        ckpt = torch.load(p, map_location=device)
        net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
        net.load_state_dict(ckpt["model_state"])
        for k, v in ckpt["mutable_hparams"].items():
            setattr(net, k, v)
        net.to(device).eval()

        if callable(modify_net):
            modify_net(net)

        res = run_ei_probe_separated(net, **probe_kw)
        all_results.append(res)
        print(f"  Model {i}: E={res['exc_mean']:.4f} I={res['inh_mean']:.4f} "
              f"E/I={res['ei_ratio']:.3f}  "
              f"AMPA={res['AMPA_mean']:.4f} NMDA={res['NMDA_mean']:.4f} "
              f"FF={res['FFInh_mean']:.4f} Rec={res['RecurInh_mean']:.4f} Lat={res['LatInh_mean']:.4f}")

        del net
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Aggregate
    keys = ["exc_mean", "inh_mean", "ei_ratio", "Q_E_total", "Q_I_total",
            "charge_ratio", "AMPA_mean", "NMDA_mean", "FFInh_mean",
            "RecurInh_mean", "LatInh_mean"]
    arrays = {k: np.array([r[k] for r in all_results]) for k in keys}
    n = len(all_results)

    summary = {}
    for k, arr in arrays.items():
        summary[k] = arr
        summary[f"{k}_mean"] = arr.mean()
        summary[f"{k}_sem"] = arr.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0

    return summary


def plot_ei_scatter_separated(summary: dict, *, out_path_base: str = None):
    """Scatter plot of Excitation vs Inhibition from separated components.

    Each point = one model replica. Diagonal = E=I reference.
    """
    exc = summary["exc_mean"]
    inh = summary["inh_mean"]

    try:
        font_path = './fonts/Roboto-Regular.ttf'
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Roboto'
    except Exception:
        pass
    plt.rcParams['font.size'] = 60
    plt.rcParams['xtick.labelsize'] = 60
    plt.rcParams['ytick.labelsize'] = 60
    plt.rcParams['axes.titlesize'] = 50
    plt.rcParams['axes.labelsize'] = 60
    plt.rcParams['legend.fontsize'] = 50
    plt.rcParams['svg.fonttype'] = 'none'

    fig, ax = plt.subplots(figsize=(25, 25))

    # Scatter: one point per model
    ax.scatter(exc, inh, s=2000, color="C0", edgecolor="none", alpha=0.9,
               zorder=3, label="Model replicas (n=10)")

    # E=I diagonal
    lim = max(exc.max(), inh.max()) * 1.15
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.5, lw=2,
            label="E = I")

    # Mean crosshairs
    e_m, i_m = summary["exc_mean_mean"], summary["inh_mean_mean"]
    e_s, i_s = summary["exc_mean_sem"], summary["inh_mean_sem"]
    ax.errorbar(e_m, i_m, xerr=e_s, yerr=i_s,
                fmt='o', ms=25, color='C1', elinewidth=3, capsize=8,
                zorder=4, label=f"Mean E/I = {summary['ei_ratio_mean']:.2f}")

    ax.set(xlim=(0, lim), ylim=(0, lim),
           xlabel="Excitatory current (a.u.)",
           ylabel="Inhibitory current (a.u.)",
           title="E/I Balance (separated synaptic currents)")
    ax.legend(frameon=False, loc='upper left')
    ax.tick_params(axis='both', which='major', length=20, width=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    base = out_path_base or './Saved_Images/EI_balance'
    fig.savefig(f"{base}.svg", format='svg')
    fig.savefig(f"{base}.png", format='png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {base}.svg and {base}.png")


def main_separated():
    """Run the corrected E/I balance analysis using separated synaptic currents."""
    base = Path("checkpoint")
    model_paths = [base / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]

    print("=" * 60)
    print("E/I BALANCE — separated synaptic currents")
    print("=" * 60)

    summary = pool_ei_separated(model_paths, device="cuda")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Excitation:  {summary['exc_mean_mean']:.4f} ± {summary['exc_mean_sem']:.4f}")
    print(f"  Inhibition:  {summary['inh_mean_mean']:.4f} ± {summary['inh_mean_sem']:.4f}")
    print(f"  E/I ratio:   {summary['ei_ratio_mean']:.3f} ± {summary['ei_ratio_sem']:.3f}")
    print(f"  Charge E:    {summary['Q_E_total_mean']:.6f} ± {summary['Q_E_total_sem']:.6f}")
    print(f"  Charge I:    {summary['Q_I_total_mean']:.6f} ± {summary['Q_I_total_sem']:.6f}")
    print(f"  Charge E/I:  {summary['charge_ratio_mean']:.3f} ± {summary['charge_ratio_sem']:.3f}")
    print(f"\n  Component breakdown (mean across models):")
    print(f"    AMPA:      {summary['AMPA_mean_mean']:.4f}")
    print(f"    NMDA:      {summary['NMDA_mean_mean']:.4f}")
    print(f"    FFInh:     {summary['FFInh_mean_mean']:.4f}")
    print(f"    RecurInh:  {summary['RecurInh_mean_mean']:.4f}")
    print(f"    LatInh:    {summary['LatInh_mean_mean']:.4f}")

    plot_ei_scatter_separated(summary)

    return summary


if __name__ == "__main__":
    # Avoid loading this probe module a second time when the canonical runner
    # imports ``EI_balance_test.run_ei_probe_separated``.
    import sys
    sys.modules.setdefault("EI_balance_test", sys.modules[__name__])

    from run_ei_balance import main as _run_canonical_ei_balance

    _run_canonical_ei_balance()
    # base = Path("checkpoint")
    # check_ei_single_model(base / "msi_model_surr_2_00.pt

#
# # ───────────────────────── runner script ─────────────────────────
# def main():
#     base_dir = Path("checkpoint")
#
#     # CONTROL
#     pooled_ctrl = run_temporal_integration_across_models(model_paths, offsets, device="cuda")
#     xs_ref, ys_ref, params_ref = tbw_gaussian_fit_curve(pooled_ctrl)
#
#     # (optional) plot control alone
#     plot_temporal_binding_summary(pooled_ctrl, reference_fit=None)
#
#     def bump_nmda(net):
#
#         # net.gNMDA = 0.5
#         # net.u_a.fill_(0.05)
#         # net.u_v.fill_(0.05)
#         # net.tau_rec = 50.0
#         # # MSI excit
#         # # MSI inh (fast spiking)
#         # # Out
#         # net.g_GABA = 0
#         net.pv_scale = 0.4
#         # net.conduction_delay_v2msi=460
#
#     pooled_mod = run_temporal_integration_across_models(
#         model_paths, offsets, device="cuda",
#         modify_net=bump_nmda,
#     )
#
#     plot_temporal_binding_summary(
#         pooled_mod,
#     )

