#!/usr/bin/env python3
"""TASK #76 STEP 2 — MSI firing-rate proxy A/B (L8-full vs certified L6), READ-ONLY.

The L8-full retrain logged its MSI firing-rate proxy `_g_out_sMSI.mean()` declining ~36x over
training (1.11e-2 @ep0 -> 3.1e-4 @ep79); the certified L6 run did NOT log it. This resolves
whether that decline is normal trained-model sparsity or an L8-full artifact: load BOTH ep79
ckpts into the SAME certified measurement net (delayfix forward — the #45/#70 apparatus), drive
the SAME fixed multimodal (sync, centred) stimulus at the SAME seed, and read the final-substep
MSI spike fraction on each. Report both raw values + ratio (L8full / L6). Benign if ratio >= 0.5.

WHY `_latest_sMSI` (not literally `_g_out_sMSI`):
  `_g_out_sMSI` is a graphdf-ONLY static-address capture buffer
  (Training_graphdf.py:3547 `self._g_out_sMSI.copy_(sM)`), populated ONLY under the L4 graph
  path; the graphdf build then ALIASES `self._latest_sMSI = self._g_out_sMSI`
  (Training_graphdf.py:3835/4049). In the certified delayfix MEASUREMENT net the identical
  final-substep MSI spike vector is `self._latest_sMSI = sM` (Training_*.py:3443, shared
  update_all_layers_batch). So `_latest_sMSI` IS the build-agnostic equivalent of the logged
  `_g_out_sMSI`; an eager (non-graph) measurement net never allocates the graph buffers, so we
  read the alias through the SAME certified forward STEP 1 uses (no second, non-certified forward,
  no graph-capture confound). The ABSOLUTE value differs from the training-time proxy by design
  (different stimulus / batch / eval mode / plasticity off); STEP 2 is the RELATIVE ratio
  L8full/L6 under one identical measurement drive.
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import val36_traj as V  # sets up Training(delayfix)/TBW/SBW; provides build_net + load_ckpt + md5/NETSRC/BUILD


@torch.no_grad()
def msi_proxy(net, *, seed, B, centre_deg=90.0, sigma_in=None,
              pulse_frames=5, n_frames=20, intensity=1.0):
    """Drive ONE fixed sync (offset-0) centred bimodal pulse; return final-substep MSI spike
    fraction per external step (mean over the (B,n) MSI population). Same seed -> common-mode noise."""
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    N = net.n
    if sigma_in is None:
        sigma_in = float(getattr(net, "sigma_in", 10.0))
    net.reset_state(batch_size=B)
    xs = torch.arange(N, dtype=torch.float32, device=net.device)
    idx_c = int(round(centre_deg * (N - 1) / (net.space_size - 1)))
    gauss = torch.exp(-0.5 * ((xs - idx_c) / sigma_in) ** 2) * intensity      # (N,)
    drive = gauss.unsqueeze(0).expand(B, N).contiguous()                       # (B,N) identical rows
    zero = torch.zeros(B, N, device=net.device)
    spk = []
    for t in range(n_frames):
        on = (t < pulse_frames)
        net.update_all_layers_batch(drive if on else zero, drive if on else zero, valid_mask=None)
        spk.append(float(net._latest_sMSI.detach().mean().item()))
    spk = np.asarray(spk, float)
    return dict(proxy_mean=float(spk.mean()),                # primary: mean over all frames
                proxy_on=float(spk[:pulse_frames].mean()),   # cross-check: pulse-on window only
                proxy_peak=float(spk.max()),
                per_step=spk.tolist(),
                has_g_out_sMSI=bool(hasattr(net, "_g_out_sMSI")),
                B=B, n_frames=n_frames, pulse_frames=pulse_frames,
                centre_deg=centre_deg, sigma_in=sigma_in, intensity=intensity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l8full", required=True)
    ap.add_argument("--l6", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epoch", type=int, default=79)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--benign_ratio", type=float, default=0.5)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[msi_proxy] device={device} BUILD={V.BUILD} netsrc.md5={V.md5(V.NETSRC)} "
          f"seed={args.seed} ep={args.epoch} B={args.batch}", flush=True)
    assert V.md5(V.NETSRC) == "5e7d6d20538592b5fc92165b371949d7", \
        "netsrc md5 != 5e7d6d20 — not the certified delayfix build"

    out = {}
    # L6 first (denominator), then L8full — SAME net recipe, SAME seed, SAME stimulus.
    for tag, ck in [("L6", args.l6), ("L8full", args.l8full)]:
        net, ldres, ck_epoch, g_rec, comment = V.load_ckpt(ck, args.seed, args.batch, device)
        if args.epoch is not None:
            net.g_rec = 0.1 if int(args.epoch) > 25 else 0.0
        assert not ldres.missing_keys and not ldres.unexpected_keys, \
            f"{tag} strict-load mismatch: missing={list(ldres.missing_keys)} unexpected={list(ldres.unexpected_keys)}"
        print(f"[msi_proxy:{tag}] loaded {os.path.basename(ck)} ep={ck_epoch} "
              f"missing={list(ldres.missing_keys)} unexpected={list(ldres.unexpected_keys)} "
              f"tau_nmda_inh={float(net.tau_nmda_inh)} g_rec={float(net.g_rec)} "
              f"plast={net.plasticity_enabled} comment={comment!r}", flush=True)
        r = msi_proxy(net, seed=args.seed, B=args.batch)
        r.update(ckpt=ck, comment=comment, epoch=ck_epoch)
        out[tag] = r
        print(f"[msi_proxy:{tag}] proxy_mean={r['proxy_mean']:.6e} proxy_on={r['proxy_on']:.6e} "
              f"proxy_peak={r['proxy_peak']:.6e} has_g_out_sMSI={r['has_g_out_sMSI']}", flush=True)

    l8, l6 = out["L8full"]["proxy_mean"], out["L6"]["proxy_mean"]
    on8, on6 = out["L8full"]["proxy_on"], out["L6"]["proxy_on"]
    ratio = (l8 / l6) if l6 > 0 else (float("inf") if l8 > 0 else float("nan"))
    ratio_on = (on8 / on6) if on6 > 0 else (float("inf") if on8 > 0 else float("nan"))
    degenerate = (l8 <= 0.0 and l6 <= 0.0)  # neither model spiked -> ratio meaningless
    benign = None if degenerate else (ratio == ratio and ratio >= args.benign_ratio)
    out["summary"] = dict(
        l8full_proxy=l8, l6_proxy=l6, ratio_l8full_over_l6=ratio,
        l8full_proxy_on=on8, l6_proxy_on=on6, ratio_on_l8full_over_l6=ratio_on,
        benign_ratio_threshold=args.benign_ratio, degenerate_both_zero=bool(degenerate),
        msi_benign=benign,
        proxy_attr="_latest_sMSI.mean() (certified delayfix forward; build-agnostic alias of graphdf _g_out_sMSI)",
        note="Relative ratio under one identical sync-bimodal drive; absolute differs from training-time proxy by design.")
    print("=" * 84)
    print(f"MSI PROXY  L8full={l8:.6e}  L6={l6:.6e}  ratio(L8full/L6)={ratio:.4f}  "
          f"(pulse-on ratio={ratio_on:.4f})  benign(>= {args.benign_ratio})? {benign}")
    print("=" * 84)
    json.dump(out, open(args.out_json, "w"), indent=1)
    print(f"[msi_proxy] wrote {args.out_json}", flush=True)
    if degenerate:
        sys.exit(3)            # undecidable — escalate, do NOT silently pass
    sys.exit(0 if benign else 2)


if __name__ == "__main__":
    main()
