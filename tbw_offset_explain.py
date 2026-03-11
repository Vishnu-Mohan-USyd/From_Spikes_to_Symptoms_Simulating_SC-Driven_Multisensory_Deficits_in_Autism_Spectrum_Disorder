"""
GPU-only explainer for a large-offset TBW case.

This script reproduces the current TBW readout for a single checkpoint,
highlights one selected offset (for example +400 ms), and plots the
corresponding MSI timecourse under:
  1) the current last-substep-per-frame readout
  2) the full-frame spike-sum readout

The figure is intended to explain why the current TBW curve can still assign a
nonzero/high "fusion probability" at large temporal offsets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from TBW_test import calculate_fusion_from_integration, load_msi_model
from Training import generate_av_batch_tensor, generate_flash_sound_batch


def _set_style() -> None:
    from matplotlib import font_manager

    font_path = Path("fonts/Roboto-Regular.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "Roboto"
    plt.rcParams["font.size"] = 18
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["legend.fontsize"] = 14


@torch.no_grad()
def compute_tbw_readouts(
    ckpt_path: Path,
    *,
    offsets_steps: list[int],
    loc: float,
    t_steps: int,
    pulse_steps: int,
    extra_steps: int,
    intensity: float,
    bg_lambda: float,
    device: str,
) -> dict[str, object]:
    net = load_msi_model(ckpt_path, device=device)

    loc_seqs, mod_seqs, off_flags, seq_lens = generate_flash_sound_batch(
        offsets_steps,
        loc=loc,
        T=t_steps,
        D=pulse_steps,
        space_size=net.space_size,
    )
    max_len = max(seq_lens)
    x_a, x_v, mask = generate_av_batch_tensor(
        loc_seqs,
        mod_seqs,
        off_flags,
        n=net.n,
        space_size=net.space_size,
        sigma_in=net.sigma_in,
        noise_std=0.0,
        device=net.device,
        max_len=max_len,
        stimulus_intensity=intensity,
    )

    if bg_lambda > 0.0:
        x_a = x_a + torch.poisson(torch.full_like(x_a, float(bg_lambda)))
        x_v = x_v + torch.poisson(torch.full_like(x_v, float(bg_lambda)))

    net.reset_state(len(offsets_steps))
    rast_last = torch.zeros((max_len, len(offsets_steps)), device=net.device)
    rast_full = torch.zeros((max_len, len(offsets_steps)), device=net.device)

    for t in range(max_len):
        _, _, s_m_last, _, sum_s_m = net.update_all_layers_batch(
            x_a[:, t],
            x_v[:, t],
            mask[:, t],
            return_spike_sum=True,
        )
        rast_last[t] = s_m_last.sum(dim=1)
        rast_full[t] = sum_s_m.sum(dim=1)

    int_last = []
    int_full = []
    windows = []
    for idx, off in enumerate(offsets_steps):
        later_onset = abs(int(off))
        win_start = later_onset
        win_stop = min(win_start + pulse_steps + extra_steps, rast_last.size(0))
        int_last.append(rast_last[win_start:win_stop, idx].sum().item())
        int_full.append(rast_full[win_start:win_stop, idx].sum().item())
        windows.append((win_start, win_stop))

    offsets_ms = [int(o) * 10 for o in offsets_steps]
    pooled_last = {"offsets_ms": offsets_ms, "mean_int_spikes": np.asarray(int_last, dtype=float)}
    pooled_full = {"offsets_ms": offsets_ms, "mean_int_spikes": np.asarray(int_full, dtype=float)}
    fusion_last = calculate_fusion_from_integration(pooled_last, method="preserve_peak")
    fusion_full = calculate_fusion_from_integration(pooled_full, method="preserve_peak")

    return {
        "offsets_steps": np.asarray(offsets_steps, dtype=int),
        "offsets_ms": np.asarray(offsets_ms, dtype=int),
        "rast_last": rast_last.detach().cpu().numpy(),
        "rast_full": rast_full.detach().cpu().numpy(),
        "int_last": np.asarray(int_last, dtype=float),
        "int_full": np.asarray(int_full, dtype=float),
        "fusion_last": np.asarray(fusion_last, dtype=float),
        "fusion_full": np.asarray(fusion_full, dtype=float),
        "windows": np.asarray(windows, dtype=int),
        "t_steps": int(t_steps),
        "pulse_steps": int(pulse_steps),
        "extra_steps": int(extra_steps),
        "ckpt": str(ckpt_path),
    }


def plot_explainer(
    data: dict[str, object],
    *,
    offset_ms: int,
    out_path: Path,
) -> None:
    _set_style()

    offsets_ms = np.asarray(data["offsets_ms"], dtype=int)
    select_idx = int(np.where(offsets_ms == int(offset_ms))[0][0])
    windows = np.asarray(data["windows"], dtype=int)
    win_start, win_stop = windows[select_idx]
    offset_steps = int(data["offsets_steps"][select_idx])

    last_tc = np.asarray(data["rast_last"], dtype=float)[:, select_idx]
    full_tc = np.asarray(data["rast_full"], dtype=float)[:, select_idx]
    xs_ms = np.arange(int(data["t_steps"])) * 10

    aud_on = 0 if offset_steps >= 0 else abs(offset_steps)
    vis_on = 0 if offset_steps <= 0 else offset_steps
    aud_on_ms = aud_on * 10
    vis_on_ms = vis_on * 10
    win_start_ms = int(win_start) * 10
    win_stop_ms = int(win_stop) * 10

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)

    ax_curve = axes[0]
    ax_curve.plot(offsets_ms, np.asarray(data["fusion_last"], dtype=float), marker="o", ms=4, lw=1.8, label="Current TBW metric")
    ax_curve.plot(offsets_ms, np.asarray(data["fusion_full"], dtype=float), marker="o", ms=3, lw=1.5, ls="--", label="Full-frame readout")
    ax_curve.scatter(
        [offset_ms],
        [float(data["fusion_last"][select_idx])],
        color="tab:red",
        s=70,
        zorder=5,
        label=f"Selected offset ({offset_ms:+d} ms)",
    )
    ax_curve.axvline(offset_ms, color="tab:red", ls=":", lw=1.2)
    ax_curve.set(
        xlabel="Audio – Visual onset (ms)",
        ylabel="Fusion probability",
        ylim=(-0.05, 1.05),
        title="TBW readout at large temporal offset",
    )
    ax_curve.legend(frameon=False, loc="upper right")
    ax_curve.grid(False)
    for side in ("top", "right"):
        ax_curve.spines[side].set_visible(False)

    def _plot_timecourse(ax, timecourse, title, color, int_value, fus_value):
        norm = timecourse / timecourse.max() if np.max(timecourse) > 0 else timecourse
        ax.plot(xs_ms, norm, color=color, lw=2.2)
        ax.axvspan(win_start_ms, win_stop_ms, color="tab:orange", alpha=0.14, label="Integration window")
        ax.axvspan(aud_on_ms, aud_on_ms + int(data["pulse_steps"]) * 10, color="tab:red", alpha=0.12, label="Audio stimulus")
        ax.axvspan(vis_on_ms, vis_on_ms + int(data["pulse_steps"]) * 10, color="tab:green", alpha=0.12, label="Visual stimulus")
        ax.axvline(aud_on_ms, color="tab:red", ls="--", lw=1.2)
        ax.axvline(vis_on_ms, color="tab:green", ls="--", lw=1.2)
        ax.set(
            xlabel="Time (ms)",
            ylabel="Normalised MSI spikes",
            xlim=(xs_ms.min(), xs_ms.max()),
            ylim=(0, 1.05),
            title=title,
        )
        ax.text(
            0.015,
            0.95,
            f"Integrated spikes = {int_value:.1f}\nFusion score = {fus_value:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=13,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.8",
                "alpha": 0.95,
            },
        )
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    _plot_timecourse(
        axes[1],
        last_tc,
        "Current TBW metric timecourse (last sub-step only)",
        "C0",
        float(data["int_last"][select_idx]),
        float(data["fusion_last"][select_idx]),
    )
    _plot_timecourse(
        axes[2],
        full_tc,
        "Full-frame timecourse (sum over all sub-steps)",
        "C1",
        float(data["int_full"][select_idx]),
        float(data["fusion_full"][select_idx]),
    )
    axes[2].legend(frameon=False, loc="upper right")

    fig.suptitle(
        f"TBW large-offset explainer at {offset_ms:+d} ms | "
        f"window = [{win_start_ms}, {win_stop_ms}] ms | ckpt={Path(str(data['ckpt'])).name}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format=out_path.suffix.lstrip("."))
    print(f"[saved] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoint/msi_model_surr_10_00.pt"))
    ap.add_argument("--offset-ms", type=int, default=400)
    ap.add_argument("--bg-lambda", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument("--t-steps", type=int, default=60)
    ap.add_argument("--pulse-steps", type=int, default=5)
    ap.add_argument("--extra-steps", type=int, default=5)
    ap.add_argument("--loc", type=float, default=90.0)
    ap.add_argument("--out", type=Path, default=Path("Saved_Images/tbw_offset400_explain.svg"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this helper is GPU-only.")
    if args.offset_ms % 10 != 0:
        raise SystemExit("--offset-ms must be a multiple of 10 ms.")

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    offsets_steps = list(range(-50, 51, 2))
    if int(args.offset_ms // 10) not in offsets_steps:
        raise SystemExit("Requested offset is outside the current TBW grid.")

    data = compute_tbw_readouts(
        args.ckpt,
        offsets_steps=offsets_steps,
        loc=float(args.loc),
        t_steps=int(args.t_steps),
        pulse_steps=int(args.pulse_steps),
        extra_steps=int(args.extra_steps),
        intensity=float(args.intensity),
        bg_lambda=float(args.bg_lambda),
        device="cuda",
    )

    idx = int(np.where(np.asarray(data["offsets_ms"]) == int(args.offset_ms))[0][0])
    print(
        f"[offset {args.offset_ms:+d} ms] current_fusion={float(data['fusion_last'][idx]):.4f} "
        f"full_fusion={float(data['fusion_full'][idx]):.4f} "
        f"current_int={float(data['int_last'][idx]):.1f} full_int={float(data['int_full'][idx]):.1f}"
    )
    plot_explainer(data, offset_ms=int(args.offset_ms), out_path=args.out)


if __name__ == "__main__":
    main()
