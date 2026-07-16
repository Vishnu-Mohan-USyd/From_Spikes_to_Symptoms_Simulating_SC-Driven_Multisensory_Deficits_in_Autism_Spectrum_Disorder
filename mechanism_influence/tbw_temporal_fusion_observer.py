"""Absolute, sham-calibrated observer for Gate-2 temporal fusion.

The scientific endpoint remains the established Gate-2 probability that an
audio-visual trial produces one temporally fused MSI response.  This module
only corrects the observer's proven failure mode: a weak surviving response
from one modality must not be called fusion merely because it has one peak.

Inputs are raw 60-frame MSI population spike-count rasters (10 ms/frame).  No
trial-wise or condition-wise amplitude normalisation is performed.  The
single-peak amplitude floor and latency geometry are frozen from the shipped
seed-42 sham calibration and are shared unchanged by every lesion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


Outcome = Literal[
    "fused_one",
    "fused_two",
    "separate",
    "dropout_ambiguous",
    "no_response",
]
ObserverMode = Literal["corrected", "legacy_regression"]


@dataclass(frozen=True)
class TemporalFusionCalibration:
    """Frozen Gate-2 observer constants.

    All time-valued fields are in 10-ms external simulation frames.  The
    ``one_peak_amplitude_floor_spikes`` value is the common sham-derived
    absolute height threshold after Gaussian smoothing; it is never
    recalculated for a perturbation condition or an individual trial.
    """

    n_frames: int = 60
    smoothing_sigma_frames: float = 2.0
    valley_threshold: float = 0.4
    minimum_peak_separation_frames: int = 3
    minimum_total_spikes: float = 10.0
    one_peak_amplitude_floor_spikes: float = 98.83377430574426
    auditory_latency_frames: int = 5
    visual_latency_frames: int = 7
    maximum_merged_latency_separation_frames: int = 6
    peak_interval_margin_frames: int = 1

    def __post_init__(self) -> None:
        if self.n_frames != 60:
            raise ValueError("Gate-2 temporal profiles must contain exactly 60 frames")
        if not np.isfinite(self.smoothing_sigma_frames) or self.smoothing_sigma_frames <= 0:
            raise ValueError("smoothing_sigma_frames must be finite and positive")
        if not 0.0 <= self.valley_threshold <= 1.0:
            raise ValueError("valley_threshold must lie in [0, 1]")
        if self.minimum_peak_separation_frames < 1:
            raise ValueError("minimum_peak_separation_frames must be >= 1")
        if not np.isfinite(self.minimum_total_spikes) or self.minimum_total_spikes < 0:
            raise ValueError("minimum_total_spikes must be finite and non-negative")
        if (
            not np.isfinite(self.one_peak_amplitude_floor_spikes)
            or self.one_peak_amplitude_floor_spikes <= 0
        ):
            raise ValueError("one_peak_amplitude_floor_spikes must be finite and positive")
        if self.maximum_merged_latency_separation_frames < 0:
            raise ValueError("maximum_merged_latency_separation_frames must be non-negative")
        if self.peak_interval_margin_frames < 0:
            raise ValueError("peak_interval_margin_frames must be non-negative")


DEFAULT_CALIBRATION = TemporalFusionCalibration()


@dataclass(frozen=True)
class TemporalFusionDecision:
    """Auditable outcome for one AV trial."""

    outcome: Outcome
    n_peaks: int
    peak_frames: tuple[int, ...]
    peak_heights_spikes: tuple[float, ...]
    total_spikes: float
    effective_offset_frames: int
    predicted_auditory_peak_frame: int
    predicted_visual_peak_frame: int
    valley_ratio: float | None = None

    @property
    def fused(self) -> bool:
        return self.outcome in {"fused_one", "fused_two"}


def _validated_profile(
    temporal_profile: np.ndarray,
    calibration: TemporalFusionCalibration,
) -> np.ndarray:
    profile = np.asarray(temporal_profile, dtype=float)
    if profile.ndim != 1 or profile.shape[0] != calibration.n_frames:
        raise ValueError(
            f"temporal_profile must have shape ({calibration.n_frames},), got {profile.shape}"
        )
    if not np.all(np.isfinite(profile)):
        raise ValueError("temporal_profile contains non-finite values")
    if np.any(profile < 0.0):
        raise ValueError("temporal_profile contains negative spike counts")
    return profile


def _absolute_positive_peaks(
    smoothed: np.ndarray,
    minimum_separation_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect every absolute positive peak without amplitude normalisation."""
    pad = minimum_separation_frames + 1
    padded = np.pad(smoothed, pad, mode="constant", constant_values=0.0)
    peak_frames, properties = find_peaks(
        padded,
        height=np.nextafter(0.0, 1.0),
        distance=minimum_separation_frames,
    )
    peak_frames = peak_frames - pad
    keep = (peak_frames >= 0) & (peak_frames < smoothed.size)
    return peak_frames[keep], properties["peak_heights"][keep]


def _predicted_peak_frames(
    effective_offset_frames: int,
    calibration: TemporalFusionCalibration,
) -> tuple[int, int]:
    offset = int(effective_offset_frames)
    auditory_onset = 0 if offset >= 0 else -offset
    visual_onset = 0 if offset <= 0 else offset
    return (
        auditory_onset + calibration.auditory_latency_frames,
        visual_onset + calibration.visual_latency_frames,
    )


def classify_temporal_profile(
    temporal_profile: np.ndarray,
    effective_offset_frames: int,
    *,
    calibration: TemporalFusionCalibration = DEFAULT_CALIBRATION,
) -> TemporalFusionDecision:
    """Classify one raw MSI temporal profile with the corrected observer.

    A two-peak response retains the established valley rule.  A one-peak
    response is fused only when it clears the frozen absolute sham threshold
    and is geometrically compatible with merger of the predicted auditory and
    visual response latencies.  Every other one-peak response is an auditable
    dropout/ambiguous trial and remains in the P(fusion) denominator.
    """
    profile = _validated_profile(temporal_profile, calibration)
    smoothed = gaussian_filter1d(profile, sigma=calibration.smoothing_sigma_frames)
    total_spikes = float(smoothed.sum())
    predicted_a, predicted_v = _predicted_peak_frames(effective_offset_frames, calibration)

    if total_spikes < calibration.minimum_total_spikes:
        return TemporalFusionDecision(
            outcome="no_response",
            n_peaks=0,
            peak_frames=(),
            peak_heights_spikes=(),
            total_spikes=total_spikes,
            effective_offset_frames=int(effective_offset_frames),
            predicted_auditory_peak_frame=predicted_a,
            predicted_visual_peak_frame=predicted_v,
        )

    peaks, heights = _absolute_positive_peaks(
        smoothed, calibration.minimum_peak_separation_frames
    )
    if peaks.size == 0:
        return TemporalFusionDecision(
            outcome="no_response",
            n_peaks=0,
            peak_frames=(),
            peak_heights_spikes=(),
            total_spikes=total_spikes,
            effective_offset_frames=int(effective_offset_frames),
            predicted_auditory_peak_frame=predicted_a,
            predicted_visual_peak_frame=predicted_v,
        )

    if peaks.size >= 2:
        top_two = np.argsort(heights)[-2:]
        p1, p2 = sorted(int(value) for value in peaks[top_two])
        smaller_peak = min(float(smoothed[p1]), float(smoothed[p2]))
        valley = float(smoothed[p1:p2 + 1].min())
        valley_ratio = valley / smaller_peak
        outcome: Outcome = (
            "fused_two"
            if valley_ratio > calibration.valley_threshold
            else "separate"
        )
        return TemporalFusionDecision(
            outcome=outcome,
            n_peaks=int(peaks.size),
            peak_frames=tuple(int(value) for value in peaks),
            peak_heights_spikes=tuple(float(value) for value in heights),
            total_spikes=total_spikes,
            effective_offset_frames=int(effective_offset_frames),
            predicted_auditory_peak_frame=predicted_a,
            predicted_visual_peak_frame=predicted_v,
            valley_ratio=float(valley_ratio),
        )

    peak_frame = int(peaks[0])
    peak_height = float(heights[0])
    separation = abs(predicted_a - predicted_v)
    lower = min(predicted_a, predicted_v) - calibration.peak_interval_margin_frames
    upper = max(predicted_a, predicted_v) + calibration.peak_interval_margin_frames
    geometry_matches = (
        separation <= calibration.maximum_merged_latency_separation_frames
        and lower <= peak_frame <= upper
    )
    outcome = (
        "fused_one"
        if (
            peak_height >= calibration.one_peak_amplitude_floor_spikes
            and geometry_matches
        )
        else "dropout_ambiguous"
    )
    return TemporalFusionDecision(
        outcome=outcome,
        n_peaks=1,
        peak_frames=(peak_frame,),
        peak_heights_spikes=(peak_height,),
        total_spikes=total_spikes,
        effective_offset_frames=int(effective_offset_frames),
        predicted_auditory_peak_frame=predicted_a,
        predicted_visual_peak_frame=predicted_v,
    )


def classify_temporal_profile_legacy_regression(
    temporal_profile: np.ndarray,
    effective_offset_frames: int = 0,
    *,
    calibration: TemporalFusionCalibration = DEFAULT_CALIBRATION,
    minimum_relative_peak_height: float = 0.2,
) -> TemporalFusionDecision:
    """Reproduce the prior max-normalised observer for regression only.

    This function is deliberately named and separated so it cannot become the
    default scientific observer accidentally.  In particular, its historical
    ``<=1 peak => fused`` behavior is retained solely to prove the corrected
    observer's suppression-not-fusion regression.
    """
    profile = _validated_profile(temporal_profile, calibration)
    smoothed = gaussian_filter1d(profile, sigma=calibration.smoothing_sigma_frames)
    total_spikes = float(smoothed.sum())
    predicted_a, predicted_v = _predicted_peak_frames(effective_offset_frames, calibration)
    if float(smoothed.max()) < 1e-6 or total_spikes < calibration.minimum_total_spikes:
        return TemporalFusionDecision(
            "no_response", 0, (), (), total_spikes, int(effective_offset_frames),
            predicted_a, predicted_v,
        )

    normalised = smoothed / smoothed.max()
    pad = calibration.minimum_peak_separation_frames + 1
    padded = np.pad(normalised, pad, mode="constant", constant_values=0.0)
    peaks, properties = find_peaks(
        padded,
        height=minimum_relative_peak_height,
        distance=calibration.minimum_peak_separation_frames,
    )
    peaks = peaks - pad
    keep = (peaks >= 0) & (peaks < normalised.size)
    peaks = peaks[keep]
    heights = properties["peak_heights"][keep]
    if peaks.size <= 1:
        peak_frames = tuple(int(value) for value in peaks)
        peak_heights = tuple(float(value) for value in heights)
        return TemporalFusionDecision(
            "fused_one", int(peaks.size), peak_frames, peak_heights,
            total_spikes, int(effective_offset_frames), predicted_a, predicted_v,
        )

    top_two = np.argsort(heights)[-2:]
    p1, p2 = sorted(int(value) for value in peaks[top_two])
    smaller_peak = min(float(normalised[p1]), float(normalised[p2]))
    valley_ratio = float(normalised[p1:p2 + 1].min()) / smaller_peak
    outcome: Outcome = (
        "fused_two" if valley_ratio > calibration.valley_threshold else "separate"
    )
    return TemporalFusionDecision(
        outcome,
        int(peaks.size),
        tuple(int(value) for value in peaks),
        tuple(float(value) for value in heights),
        total_spikes,
        int(effective_offset_frames),
        predicted_a,
        predicted_v,
        valley_ratio,
    )


def half_peak_width(
    offsets_ms: np.ndarray,
    p_fusion: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    """Return the unchanged sampled half-peak TBW span."""
    offsets = np.asarray(offsets_ms, dtype=float)
    probability = np.asarray(p_fusion, dtype=float)
    if offsets.ndim != 1 or probability.shape != offsets.shape:
        raise ValueError("offsets_ms and p_fusion must be same-length 1D arrays")
    if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(probability)):
        raise ValueError("offsets_ms and p_fusion must be finite")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("p_fusion must lie in [0, 1]")
    peak = float(probability.max())
    if peak <= 0.0:
        return None, None, None
    above = offsets[probability >= 0.5 * peak]
    if above.size < 2:
        return None, None, None
    left = float(above.min())
    right = float(above.max())
    return right - left, left, right


def classify_temporal_rasters(
    raster_msi: np.ndarray,
    offsets_ms: np.ndarray,
    effective_offsets_frames: np.ndarray,
    *,
    calibration: TemporalFusionCalibration = DEFAULT_CALIBRATION,
    observer: ObserverMode = "corrected",
) -> dict[str, object]:
    """Classify a ``(60, n_SOA, n_trial)`` raw MSI raster cube.

    P(fusion) always uses every AV trial as its denominator.  Separate,
    dropout/ambiguous and no-response counts are returned independently so a
    lesion-induced response loss cannot masquerade as fusion.
    """
    rasters = np.asarray(raster_msi, dtype=float)
    offsets = np.asarray(offsets_ms, dtype=float)
    effective = np.asarray(effective_offsets_frames)
    if rasters.ndim != 3 or rasters.shape[0] != calibration.n_frames:
        raise ValueError(
            f"raster_msi must have shape ({calibration.n_frames}, n_SOA, n_trial), "
            f"got {rasters.shape}"
        )
    n_soas, n_trials = rasters.shape[1:]
    if offsets.shape != (n_soas,):
        raise ValueError(f"offsets_ms must have shape ({n_soas},), got {offsets.shape}")
    if effective.shape != (n_soas, n_trials):
        raise ValueError(
            f"effective_offsets_frames must have shape ({n_soas}, {n_trials}), "
            f"got {effective.shape}"
        )
    if not np.all(np.isfinite(rasters)) or np.any(rasters < 0.0):
        raise ValueError("raster_msi must contain finite non-negative spike counts")
    if observer == "corrected":
        classifier = classify_temporal_profile
    elif observer == "legacy_regression":
        classifier = classify_temporal_profile_legacy_regression
    else:
        raise ValueError(f"unknown observer mode {observer!r}")

    outcomes: tuple[Outcome, ...] = (
        "fused_one", "fused_two", "separate", "dropout_ambiguous", "no_response"
    )
    counts = {outcome: np.zeros(n_soas, dtype=np.int64) for outcome in outcomes}
    for soa_index in range(n_soas):
        for trial_index in range(n_trials):
            decision = classifier(
                rasters[:, soa_index, trial_index],
                int(effective[soa_index, trial_index]),
                calibration=calibration,
            )
            counts[decision.outcome][soa_index] += 1

    fused_counts = counts["fused_one"] + counts["fused_two"]
    dropout_counts = counts["dropout_ambiguous"] + counts["no_response"]
    accounted = fused_counts + counts["separate"] + dropout_counts
    if not np.array_equal(accounted, np.full(n_soas, n_trials, dtype=np.int64)):
        raise AssertionError("observer outcomes do not account for every AV trial")
    p_fusion = fused_counts.astype(float) / float(n_trials)
    width, left, right = half_peak_width(offsets, p_fusion)
    return {
        "observer": observer,
        "offsets_ms": offsets.tolist(),
        "n_trials": np.full(n_soas, n_trials, dtype=np.int64).tolist(),
        "p_fusion": p_fusion.tolist(),
        "fused_counts": fused_counts.tolist(),
        "fused_one_counts": counts["fused_one"].tolist(),
        "fused_two_counts": counts["fused_two"].tolist(),
        "separate_counts": counts["separate"].tolist(),
        "dropout_ambiguous_counts": counts["dropout_ambiguous"].tolist(),
        "no_response_counts": counts["no_response"].tolist(),
        "dropout_counts": dropout_counts.tolist(),
        "peak_p_fusion": float(p_fusion.max()),
        "half_peak_width_ms": width,
        "half_peak_left_ms": left,
        "half_peak_right_ms": right,
        "overall_dropout_fraction": float(dropout_counts.sum() / (n_soas * n_trials)),
        "overall_separate_fraction": float(counts["separate"].sum() / (n_soas * n_trials)),
    }
