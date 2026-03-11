"""
GPU-only helper: capture an SBW example trial at a chosen spatial disparity.

This script supports searching for a representative trial that is classified as
fused by the *current* SBW decision rule and saves enough metadata to explain
why that trial counted as fused.

Saved outputs include:
  - per-frame MSI raster sampled at the last 0.1 ms sub-step of each 10 ms frame
    (the quantity used by the current SBW analysis in `SBW_test.py`)
  - per-frame MSI raster integrated over all sub-steps within each 10 ms frame
  - summed MSI profiles for both methods
  - smoothed profiles, detected peaks, and valley-ratio diagnostics for both
    methods, so the plotting script can annotate the decision rule directly

Example:
  python sbw_gpu_trace_disparity.py \
      --ckpt checkpoint/msi_model_surr_10_00.pt \
      --disparity-deg 80 \
      --bg-lambda 1e-5 \
      --intensity 1.0 \
      --target two_peak_shallow_last \
      --out /tmp/sbw_trace_disp80_fused.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from Training import MultiBatchAudVisMSINetworkTime


def load_msi_model(
    ckpt_path: Path,
    *,
    device: torch.device,
) -> MultiBatchAudVisMSINetworkTime:
    ckpt = torch.load(ckpt_path, map_location=device)
    net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
    net.load_state_dict(ckpt["model_state"])
    for key, value in ckpt.get("mutable_hparams", {}).items():
        setattr(net, key, value)
    net.to(device).eval()
    net.device = device
    return net


def classify_profile(profile: np.ndarray) -> dict[str, object]:
    """
    Apply the same SBW fusion rule used in `SBW_test.py` and expose the
    intermediate quantities needed for an explanatory inset.
    """
    n = int(profile.size)
    smoothed = gaussian_filter1d(np.asarray(profile, dtype=float), sigma=2, mode="wrap")
    if smoothed.max() < 1e-6:
        return {
            "fused": True,
            "label": "silent",
            "smoothed": smoothed,
            "peaks": np.empty(0, dtype=int),
            "peak_heights": np.empty(0, dtype=float),
            "top_peaks": np.array([-1, -1], dtype=int),
            "top_peak_heights": np.array([np.nan, np.nan], dtype=float),
            "valley_idx": -1,
            "valley_value": np.nan,
            "valley_ratio": np.nan,
            "valley_path_indices": np.empty(0, dtype=int),
        }

    smoothed = smoothed / (smoothed.max() + 1e-12)
    peaks, props = find_peaks(smoothed, height=0.2, distance=10)
    if len(peaks) <= 1:
        top_peak = int(peaks[0]) if len(peaks) == 1 else -1
        top_height = float(props["peak_heights"][0]) if len(peaks) == 1 else np.nan
        return {
            "fused": True,
            "label": "single_peak",
            "smoothed": smoothed,
            "peaks": peaks.astype(int),
            "peak_heights": np.asarray(props["peak_heights"], dtype=float),
            "top_peaks": np.array([top_peak, -1], dtype=int),
            "top_peak_heights": np.array([top_height, np.nan], dtype=float),
            "valley_idx": -1,
            "valley_value": np.nan,
            "valley_ratio": np.nan,
            "valley_path_indices": np.empty(0, dtype=int),
        }

    top_order = np.argsort(props["peak_heights"])[::-1][:2]
    top_peaks = peaks[top_order].astype(int)
    top_peak_heights = np.asarray(props["peak_heights"], dtype=float)[top_order]
    peak_a, peak_b = int(top_peaks[0]), int(top_peaks[1])

    def valley_path(idx_a: int, idx_b: int) -> np.ndarray:
        direct = abs(idx_b - idx_a)
        if direct <= n - direct:
            lo, hi = min(idx_a, idx_b), max(idx_a, idx_b)
            return np.arange(lo, hi + 1, dtype=int)
        hi, lo = max(idx_a, idx_b), min(idx_a, idx_b)
        return np.r_[np.arange(hi, n, dtype=int), np.arange(0, lo + 1, dtype=int)]

    valley_path_indices = valley_path(peak_a, peak_b)
    valley_local_idx = int(np.argmin(smoothed[valley_path_indices]))
    valley_idx = int(valley_path_indices[valley_local_idx])
    valley_value = float(smoothed[valley_idx])
    valley_ratio = valley_value / (min(smoothed[peak_a], smoothed[peak_b]) + 1e-12)
    fused = bool(valley_ratio > 0.6)

    return {
        "fused": fused,
        "label": "two_peak_shallow" if fused else "two_peak_deep",
        "smoothed": smoothed,
        "peaks": peaks.astype(int),
        "peak_heights": np.asarray(props["peak_heights"], dtype=float),
        "top_peaks": top_peaks,
        "top_peak_heights": top_peak_heights.astype(float),
        "valley_idx": valley_idx,
        "valley_value": valley_value,
        "valley_ratio": float(valley_ratio),
        "valley_path_indices": valley_path_indices.astype(int),
    }


def circular_distance(idx_a: int, idx_b: int, n: int) -> int:
    direct = abs(int(idx_a) - int(idx_b))
    return int(min(direct, n - direct))


def valley_path(idx_a: int, idx_b: int, n: int) -> np.ndarray:
    direct = abs(int(idx_b) - int(idx_a))
    if direct <= n - direct:
        lo, hi = min(int(idx_a), int(idx_b)), max(int(idx_a), int(idx_b))
        return np.arange(lo, hi + 1, dtype=int)
    hi, lo = max(int(idx_a), int(idx_b)), min(int(idx_a), int(idx_b))
    return np.r_[np.arange(hi, n, dtype=int), np.arange(0, lo + 1, dtype=int)]


def input_pair_diagnostics(
    *,
    peaks: np.ndarray,
    peak_heights: np.ndarray,
    smoothed: np.ndarray,
    audio_idx: int,
    visual_idx: int,
) -> dict[str, object]:
    """
    Find the detected peak nearest the audio input and the detected peak nearest
    the visual input, then compute the valley ratio between that pair. This is
    not the classifier's rule; it is an explanatory diagnostic for plotting.
    """
    peaks = np.asarray(peaks, dtype=int)
    peak_heights = np.asarray(peak_heights, dtype=float)
    if peaks.size == 0:
        return {
            "input_peaks": np.array([-1, -1], dtype=int),
            "input_peak_heights": np.array([np.nan, np.nan], dtype=float),
            "input_peak_labels": np.array(["", ""], dtype="<U1"),
            "input_pair_distinct": False,
            "input_valley_idx": -1,
            "input_valley_value": np.nan,
            "input_valley_ratio": np.nan,
            "input_valley_path_indices": np.empty(0, dtype=int),
        }

    n = int(smoothed.size)
    audio_pick = int(peaks[np.argmin([circular_distance(p, audio_idx, n) for p in peaks])])
    visual_pick = int(peaks[np.argmin([circular_distance(p, visual_idx, n) for p in peaks])])
    height_map = {int(p): float(h) for p, h in zip(peaks, peak_heights)}

    if audio_pick == visual_pick:
        return {
            "input_peaks": np.array([audio_pick, visual_pick], dtype=int),
            "input_peak_heights": np.array([height_map[audio_pick], height_map[visual_pick]], dtype=float),
            "input_peak_labels": np.array(["A", "V"], dtype="<U1"),
            "input_pair_distinct": False,
            "input_valley_idx": -1,
            "input_valley_value": np.nan,
            "input_valley_ratio": np.nan,
            "input_valley_path_indices": np.empty(0, dtype=int),
        }

    path_indices = valley_path(audio_pick, visual_pick, n)
    valley_idx = int(path_indices[int(np.argmin(smoothed[path_indices]))])
    valley_value = float(smoothed[valley_idx])
    valley_ratio = valley_value / (min(height_map[audio_pick], height_map[visual_pick]) + 1e-12)
    return {
        "input_peaks": np.array([audio_pick, visual_pick], dtype=int),
        "input_peak_heights": np.array([height_map[audio_pick], height_map[visual_pick]], dtype=float),
        "input_peak_labels": np.array(["A", "V"], dtype="<U1"),
        "input_pair_distinct": True,
        "input_valley_idx": valley_idx,
        "input_valley_value": valley_value,
        "input_valley_ratio": float(valley_ratio),
        "input_valley_path_indices": path_indices.astype(int),
    }


@torch.no_grad()
def run_trial(
    net: MultiBatchAudVisMSINetworkTime,
    *,
    centre_deg: float,
    disparity_deg: float,
    duration: int,
    intensity: float,
    bg_lambda: float,
) -> dict[str, object]:
    n, space_deg, device = net.n, net.space_size, net.device

    def to_idx(deg: float) -> int:
        return int(round(deg * (n - 1) / (space_deg - 1))) % n

    audio_deg = float(centre_deg % space_deg)
    visual_deg = float((centre_deg + disparity_deg) % space_deg)
    audio_idx, visual_idx = to_idx(audio_deg), to_idx(visual_deg)

    xs = torch.arange(n, dtype=torch.float32, device=device)
    gauss = lambda idx: torch.exp(-0.5 * ((xs - float(idx)) / net.sigma_in) ** 2) * float(intensity)
    x_a_step = gauss(audio_idx).unsqueeze(0)
    x_v_step = gauss(visual_idx).unsqueeze(0)

    net.reset_state(batch_size=1)
    msi_raster_last = torch.zeros(duration, n, device=device)
    msi_raster_full = torch.zeros(duration, n, device=device)
    for t in range(duration):
        x_a = x_a_step
        x_v = x_v_step
        if bg_lambda > 0.0:
            lam = float(bg_lambda)
            x_a = x_a + torch.poisson(torch.full_like(x_a, lam))
            x_v = x_v + torch.poisson(torch.full_like(x_v, lam))
        _, _, s_m_last, _, sum_s_m = net.update_all_layers_batch(
            x_a,
            x_v,
            return_spike_sum=True,
        )
        msi_raster_last[t] = s_m_last[0]
        msi_raster_full[t] = sum_s_m[0]

    profile_last = msi_raster_last.sum(0)
    profile_full = msi_raster_full.sum(0)
    return {
        "msi_raster_last": msi_raster_last,
        "msi_raster_full": msi_raster_full,
        "msi_profile_last": profile_last,
        "msi_profile_full": profile_full,
        "audio_deg": audio_deg,
        "visual_deg": visual_deg,
        "audio_idx": audio_idx,
        "visual_idx": visual_idx,
        "bg_lambda": float(bg_lambda),
    }


def matches_target(
    target: str,
    *,
    cls_last: dict[str, object],
    cls_full: dict[str, object],
) -> bool:
    if target == "any":
        return True
    if target == "fused_last":
        return bool(cls_last["fused"])
    if target == "fused_full":
        return bool(cls_full["fused"])
    if target == "fused_last_only":
        return bool(cls_last["fused"]) and (not bool(cls_full["fused"]))
    if target == "two_peak_shallow_last":
        return str(cls_last["label"]) == "two_peak_shallow"
    if target == "two_peak_shallow_full":
        return str(cls_full["label"]) == "two_peak_shallow"
    raise ValueError(f"Unknown target: {target}")


def score_candidate(
    trial: dict[str, object],
    *,
    cls_last: dict[str, object],
    cls_full: dict[str, object],
    preferred_metric: str,
) -> tuple[float, float]:
    """
    Prefer high-spike, visually interpretable examples. The tuple ordering keeps
    the search deterministic.
    """
    profile_last = trial["msi_profile_last"].detach().float().cpu().numpy()
    profile_full = trial["msi_profile_full"].detach().float().cpu().numpy()
    if preferred_metric == "full":
        primary = float(profile_full.sum())
        secondary = float(cls_full["valley_ratio"]) if np.isfinite(cls_full["valley_ratio"]) else -np.inf
    else:
        primary = float(profile_last.sum())
        secondary = float(cls_last["valley_ratio"]) if np.isfinite(cls_last["valley_ratio"]) else -np.inf
    return primary, secondary


def pack_classification(prefix: str, cls: dict[str, object]) -> dict[str, object]:
    return {
        f"{prefix}_fused": bool(cls["fused"]),
        f"{prefix}_label": str(cls["label"]),
        f"{prefix}_smoothed": np.asarray(cls["smoothed"], dtype=float),
        f"{prefix}_peaks": np.asarray(cls["peaks"], dtype=int),
        f"{prefix}_peak_heights": np.asarray(cls["peak_heights"], dtype=float),
        f"{prefix}_top_peaks": np.asarray(cls["top_peaks"], dtype=int),
        f"{prefix}_top_peak_heights": np.asarray(cls["top_peak_heights"], dtype=float),
        f"{prefix}_valley_idx": int(cls["valley_idx"]),
        f"{prefix}_valley_value": float(cls["valley_value"]),
        f"{prefix}_valley_ratio": float(cls["valley_ratio"]),
        f"{prefix}_valley_path_indices": np.asarray(cls["valley_path_indices"], dtype=int),
    }


def pack_input_pair(prefix: str, diag: dict[str, object]) -> dict[str, object]:
    return {
        f"{prefix}_input_peaks": np.asarray(diag["input_peaks"], dtype=int),
        f"{prefix}_input_peak_heights": np.asarray(diag["input_peak_heights"], dtype=float),
        f"{prefix}_input_peak_labels": np.asarray(diag["input_peak_labels"]),
        f"{prefix}_input_pair_distinct": bool(diag["input_pair_distinct"]),
        f"{prefix}_input_valley_idx": int(diag["input_valley_idx"]),
        f"{prefix}_input_valley_value": float(diag["input_valley_value"]),
        f"{prefix}_input_valley_ratio": float(diag["input_valley_ratio"]),
        f"{prefix}_input_valley_path_indices": np.asarray(diag["input_valley_path_indices"], dtype=int),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoint/msi_model_surr_10_00.pt"))
    ap.add_argument("--disparity-deg", type=float, default=80.0)
    ap.add_argument("--duration", type=int, default=20)
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument("--bg-lambda", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tries", type=int, default=200)
    ap.add_argument(
        "--target",
        choices=(
            "any",
            "fused_last",
            "fused_full",
            "fused_last_only",
            "two_peak_shallow_last",
            "two_peak_shallow_full",
        ),
        default="fused_last",
        help="Which kind of example to search for.",
    )
    ap.add_argument(
        "--preferred-metric",
        choices=("last", "full"),
        default="last",
        help="Which spike-count metric to maximize when several trials match.",
    )
    ap.add_argument("--out", type=Path, default=Path("sbw_trace_large_disparity.npz"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available (GPU-only helper).")

    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)
    net = load_msi_model(args.ckpt, device=device)

    best = None
    matched = False
    for attempt in range(1, args.max_tries + 1):
        centre = float(rng.integers(30, 150))
        trial = run_trial(
            net,
            centre_deg=centre,
            disparity_deg=float(args.disparity_deg),
            duration=int(args.duration),
            intensity=float(args.intensity),
            bg_lambda=float(args.bg_lambda),
        )
        profile_last = trial["msi_profile_last"].detach().float().cpu().numpy()
        profile_full = trial["msi_profile_full"].detach().float().cpu().numpy()
        cls_last = classify_profile(profile_last)
        cls_full = classify_profile(profile_full)
        input_last = input_pair_diagnostics(
            peaks=np.asarray(cls_last["peaks"], dtype=int),
            peak_heights=np.asarray(cls_last["peak_heights"], dtype=float),
            smoothed=np.asarray(cls_last["smoothed"], dtype=float),
            audio_idx=int(trial["audio_idx"]),
            visual_idx=int(trial["visual_idx"]),
        )
        input_full = input_pair_diagnostics(
            peaks=np.asarray(cls_full["peaks"], dtype=int),
            peak_heights=np.asarray(cls_full["peak_heights"], dtype=float),
            smoothed=np.asarray(cls_full["smoothed"], dtype=float),
            audio_idx=int(trial["audio_idx"]),
            visual_idx=int(trial["visual_idx"]),
        )
        score = score_candidate(
            trial,
            cls_last=cls_last,
            cls_full=cls_full,
            preferred_metric=str(args.preferred_metric),
        )

        if (best is None) or (score > best["score"]):
            best = {
                "trial": trial,
                "cls_last": cls_last,
                "cls_full": cls_full,
                "input_last": input_last,
                "input_full": input_full,
                "score": score,
                "attempt": attempt,
                "centre_deg": centre,
            }

        if matches_target(str(args.target), cls_last=cls_last, cls_full=cls_full):
            best = {
                "trial": trial,
                "cls_last": cls_last,
                "cls_full": cls_full,
                "input_last": input_last,
                "input_full": input_full,
                "score": score,
                "attempt": attempt,
                "centre_deg": centre,
            }
            matched = True
            break

    assert best is not None
    if not matched and args.target != "any":
        raise SystemExit(
            f"No trial matched target={args.target!r} within {args.max_tries} tries. "
            "Increase --max-tries or loosen --target."
        )

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trial = best["trial"]
    msi_raster_last = trial["msi_raster_last"].detach().float().cpu().numpy()
    msi_raster_full = trial["msi_raster_full"].detach().float().cpu().numpy()
    msi_profile_last = trial["msi_profile_last"].detach().float().cpu().numpy()
    msi_profile_full = trial["msi_profile_full"].detach().float().cpu().numpy()

    np.savez_compressed(
        out_path,
        ckpt=str(args.ckpt),
        seed=int(args.seed),
        attempt=int(best["attempt"]),
        centre_deg=float(best["centre_deg"]),
        disparity_deg=float(args.disparity_deg),
        target=str(args.target),
        preferred_metric=str(args.preferred_metric),
        audio_deg=float(trial["audio_deg"]),
        visual_deg=float(trial["visual_deg"]),
        audio_idx=int(trial["audio_idx"]),
        visual_idx=int(trial["visual_idx"]),
        duration=int(args.duration),
        intensity=float(args.intensity),
        bg_lambda=float(trial["bg_lambda"]),
        total_spikes_last=float(msi_profile_last.sum()),
        total_spikes_full=float(msi_profile_full.sum()),
        msi_raster_last=msi_raster_last,
        msi_raster_full=msi_raster_full,
        msi_profile_last=msi_profile_last,
        msi_profile_full=msi_profile_full,
        **pack_classification("last", best["cls_last"]),
        **pack_classification("full", best["cls_full"]),
        **pack_input_pair("last", best["input_last"]),
        **pack_input_pair("full", best["input_full"]),
    )

    print(
        f"[saved] {out_path} | disparity={args.disparity_deg:g}° | "
        f"centre={best['centre_deg']:.2f}° (A={trial['audio_deg']:.2f}°, V={trial['visual_deg']:.2f}°) | "
        f"target={args.target} | "
        f"last={best['cls_last']['label']} full={best['cls_full']['label']} | "
        f"spikes_last={msi_profile_last.sum():.0f} spikes_full={msi_profile_full.sum():.0f} | "
        f"tries={best['attempt']}"
    )


if __name__ == "__main__":
    main()
