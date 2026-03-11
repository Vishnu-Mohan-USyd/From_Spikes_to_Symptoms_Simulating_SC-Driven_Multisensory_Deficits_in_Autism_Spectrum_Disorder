"""
Empirically measure baseline (spontaneous) spiking of a trained MSI network.

This script:
  - requires CUDA and runs on cuda:0
  - loads a checkpoint using the same pattern as SBW_test.py:180 (load_msi_model)
  - drives the network with zero external input (xA=xV=0) by default, or with
    low independent Poisson background if --bg-lambda > 0
  - reports baseline firing rates (Hz) for A, V, MSI-excitatory populations

Notes on confounds:
  Training.py::update_all_layers_batch can update internal gains/homeostasis even in eval().
  Here we minimize known confounds by setting lr_uni/lr_msi=0 and
  allow_inhib_plasticity=False, and by keeping the background drive low.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch

from Training import MultiBatchAudVisMSINetworkTime


def load_msi_model(ckpt_path: Path, *, device: str) -> MultiBatchAudVisMSINetworkTime:
    """Checkpoint loader equivalent to SBW_test.py:180 but device-explicit."""
    ckpt = torch.load(ckpt_path, map_location=device)
    net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
    net.load_state_dict(ckpt["model_state"])
    for k, v in ckpt["mutable_hparams"].items():
        setattr(net, k, v)
    net.to(device).eval()
    net.device = torch.device(device)  # make sure helpers pick this up
    return net


@torch.no_grad()
def measure_baseline_spiking(
    net: MultiBatchAudVisMSINetworkTime,
    *,
    n_frames: int,
    bg_lambda: float = 0.0,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Returns:
      spike_counts_per_neuron: dict of tensors
        MSI_exc: (n,)
      sim_time_s: total simulated time in seconds
    """
    device = net.device
    n = int(net.n)

    # One batch element
    net.reset_state(batch_size=1)
    net._dbg_spk_A = 0.0
    net._dbg_spk_V = 0.0
    net._dbg_spk_MSI = 0.0

    # External input: either exactly zero, or ongoing low Poisson background.
    xA0 = torch.zeros((1, n), device=device, dtype=torch.float32)
    xV0 = torch.zeros_like(xA0)
    lam = None
    if bg_lambda and bg_lambda > 0.0:
        lam = torch.full((1, n), float(bg_lambda), device=device, dtype=torch.float32)

    total_M = torch.zeros((n,), device=device)

    for _ in range(n_frames):
        if lam is None:
            xA, xV = xA0, xV0
        else:
            xA = torch.poisson(lam)
            xV = torch.poisson(lam)

        (
            _sA_last,
            _sV_last,
            _sM_last,
            _sO_last,
            sum_sM,
        ) = net.update_all_layers_batch(
            xA,
            xV,
            valid_mask=None,
            return_spike_sum=True,
        )

        total_M += sum_sM[0]

    sim_time_s = float(n_frames * int(net.n_substeps) * float(net.dt) / 1000.0)
    return (
        {
            "MSI_exc": total_M,
        },
        sim_time_s,
    )


def _summarize(pop: str, spikes: torch.Tensor, sim_time_s: float) -> str:
    spikes_cpu = spikes.detach().float().cpu()
    rates_hz = spikes_cpu / sim_time_s
    total_spikes = float(spikes_cpu.sum().item())
    n_neurons = int(spikes_cpu.numel())
    mean_hz = float(rates_hz.mean().item())
    active = int((spikes_cpu > 0).sum().item())
    return (
        f"{pop:8s}  n={n_neurons:4d}  total_spikes={total_spikes:12.0f}  "
        f"mean_rate={mean_hz:9.4f} Hz  active_neurons={active:4d}/{n_neurons}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        type=Path,
        default=Path("checkpoint/msi_model_surr_10_00.pt"),
        help="Path to checkpoint .pt file",
    )
    p.add_argument("--frames", type=int, default=1000, help="External frames to simulate")
    p.add_argument(
        "--bg-lambda",
        type=float,
        default=0.0,
        help="If >0, drive with independent Poisson background per neuron/frame (same units as SBW/TBW bg_lambda).",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available (torch.cuda.is_available() is False).")

    device = "cuda:0"
    net = load_msi_model(args.ckpt, device=device)

    # Minimize known confounds (even if plasticity=False is used).
    net.lr_uni = 0.0
    net.lr_msi = 0.0
    net.allow_inhib_plasticity = False
    if hasattr(net, "_probe"):
        net._probe = None

    ms_per_frame = float(net.dt) * float(net.n_substeps)
    print(f"device={device}  dt_ms={net.dt}  n_substeps={net.n_substeps}  ms_per_frame={ms_per_frame}")
    print(f"ckpt={args.ckpt}")
    print(f"frames={args.frames}  sim_time_s={args.frames * ms_per_frame / 1000.0:.3f}")
    print(f"bg_lambda={float(args.bg_lambda)}")
    print(f"lr_uni={net.lr_uni}  lr_msi={net.lr_msi}  allow_inhib_plasticity={net.allow_inhib_plasticity}")

    spike_counts, sim_time_s = measure_baseline_spiking(
        net,
        n_frames=args.frames,
        bg_lambda=float(args.bg_lambda),
    )

    print("\n--- Baseline spiking summary ---")
    print(f"A        mean_rate={net._dbg_spk_A / (net.n * sim_time_s):9.4f} Hz")
    print(f"V        mean_rate={net._dbg_spk_V / (net.n * sim_time_s):9.4f} Hz")
    print(f"MSI_exc  mean_rate={net._dbg_spk_MSI / (net.n * sim_time_s):9.4f} Hz")
    print(_summarize("MSI_exc", spike_counts["MSI_exc"], sim_time_s))

    total_all = sum(float(v.detach().sum().item()) for v in spike_counts.values())
    print(f"\nTOTAL (all pops) spikes={total_all:.0f}")
    if total_all == 0.0:
        print("Baseline is exactly silent (no spikes observed).")


if __name__ == "__main__":
    main()
