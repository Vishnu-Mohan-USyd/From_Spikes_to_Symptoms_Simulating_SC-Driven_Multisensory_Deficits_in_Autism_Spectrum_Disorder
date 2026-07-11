#!/usr/bin/env python3
"""VALIDATOR #394 STAGE-2 SHIM — run the OFFICIAL ensemble gate scripts VERBATIM on the dm10 (tau18)
ensemble, without editing the bundle or the frozen readouts.

WHY a shim: the stock measure/ drivers hard-set TAU_GABA=10 + GNMDA=0.50 and resolve ckpts via
N.ckpt_path_for_seed -> the committed *tau10* checkpoint/ models. dm10 is a tau18 operating point
(aM=0.02, dM=10, tau_gaba=18, gNMDA=0.51). load_ckpt (val36_traj_d52.py L124-129 & L146-148) asserts
env==ckpt, so the stock scripts FATAL-crash on dm10. This shim imports the chosen official module
(so its measurement code + frozen-md5 firewall run byte-identical), then:
  (a) PRE-FLIGHT (CPU-only, no GPU): reads EVERY seed's ckpt mutable_hparams, asserts each is exactly
      the dm10 point, and DERIVES the tau_gaba/gNMDA env overrides FROM the ckpts (per-seed, lead #394
      reinforcement) — a drifted seed FATAL-aborts by name BEFORE any measurement (independent check on
      the coder's config fidelity);
  (b) overrides TAU_GABA/GNMDA in os.environ from the pre-flight-derived (verified-uniform) values —
      both are re-read by load_ckpt at RUNTIME, so an after-import override is honored;
  (c) monkeypatches routec_net_io.ckpt_path_for_seed -> the dm10 ensemble paths;
  (d) wraps routec_net_io.load_ckpt with a GUARD that re-asserts the RESTORED net is the dm10 point
      (aM==0.02, dM==10, tau_gaba==18, gNMDA==0.51, u_stp_a==0.2) — per-seed runtime tripwire;
  (e) routes OUT via OUT_JSON_OVERRIDE into /tmp/val394_dm10/.
THREE independent per-seed layers guard the operating point: pre-flight mutable_hparams assert (b),
load_ckpt's own env==ckpt assert, and the restored-net GUARD (d). Frozen TBW/SBW readouts are
IMPORT-ONLY and never touched; each official script asserts their md5 BEFORE==AFTER itself.

ONE gate per process (avoids cross-gate ENV contamination), mirroring run_all.sh / the #373 per-seed runs.

Usage (Stage 2, on lead signal + ensemble down):
  # FIRST (no GPU needed): config-fidelity pre-flight over all 10 seeds —
  python val394_dm10_stage2.py --gate preflight --seeds 42 43 44 45 46 47 48 49 50 51
  # seed42 faithfulness anchor, then the sweeps (GPU):
  CUDA_VISIBLE_DEVICES=0 python val394_dm10_stage2.py --gate main --seeds 42
  CUDA_VISIBLE_DEVICES=0 python val394_dm10_stage2.py --gate main --seeds 42 43 44 45 46 47 48 49 50 51
  CUDA_VISIBLE_DEVICES=0 python val394_dm10_stage2.py --gate cre  --seeds 42 ... 51
  CUDA_VISIBLE_DEVICES=0 python val394_dm10_stage2.py --gate lat  --seeds 42 ... 51
  CUDA_VISIBLE_DEVICES=0 python val394_dm10_stage2.py --gate g7   --seeds 42 ... 51 --gain_exp 1
  python val394_dm10_stage2.py --gate g7agg --gain_exp 1
"""
import os, sys, argparse

BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # bundle root (this shim lives in measure/)
MEASURE = os.path.join(BUNDLE, "measure")
OUTDIR = os.environ.get("VAL394_OUT", "/tmp/val394_dm10")
# dm10 ensemble layout in the committed bundle: ALL 10 models (seeds 42-51, ep79, tau18) live in
# checkpoint/. DM10_ENS/DM10_S42 stay env-overridable for an out-of-tree (e.g. /tmp) ensemble.
DM10_ENS = os.environ.get("DM10_ENS", os.path.join(BUNDLE, "checkpoint"))
DM10_S42 = os.environ.get("DM10_S42", os.path.join(BUNDLE, "checkpoint"))
TMPL = "ckpt_ep79_seed{seed}_bs250_delay52_tau18_dL3.pt"
# dm10 trained operating point (authoritative = ckpt mutable_hparams) — the per-seed tripwire
EXPECT = {"aM": 0.02, "dM": 10.0, "tau_gaba": 18.0, "gNMDA": 0.51, "u_stp_a": 0.2}


def dm10_ckpt(seed):
    """Resolve a seed's dm10 ckpt robustly across the orchestrator's possible layouts (flat, or a
    per-seed subdir as the aM014 ensemble used), returning the first that exists; else the flat path
    (so pre-flight's existence assert FATALs clearly by name if a seed is genuinely absent)."""
    seed = int(seed)
    fname = TMPL.format(seed=seed)
    cands = [
        os.path.join(DM10_ENS, fname),                       # flat: /tmp/dm10_ensemble/ckpt_...pt
        os.path.join(DM10_ENS, f"dm10_s{seed}", fname),      # per-seed subdir (aM014-style)
        os.path.join(DM10_ENS, f"s{seed}", fname),
        os.path.join(DM10_ENS, f"seed{seed}", fname),
    ]
    if seed == 42:
        cands.append(os.path.join(DM10_S42, fname))          # the existing 10th (dbg_sfa_isolate)
    for p in cands:
        if os.path.exists(p):
            return p
    return cands[0]


def preflight(seeds):
    """CPU-only per-seed config-fidelity check; returns env overrides DERIVED from the ckpts.
    FATAL-aborts by name if any seed drifted off the dm10 point (lead #394 independent cross-check)."""
    import torch
    seen = {"tau_gaba": set(), "gNMDA": set()}
    print(f"[preflight] reading mutable_hparams of {len(seeds)} dm10 ckpts (CPU, no GPU) ...", flush=True)
    for s in seeds:
        ck = dm10_ckpt(s)
        assert os.path.exists(ck), f"PREFLIGHT FATAL: missing dm10 ckpt seed {s}: {ck}"
        mh = torch.load(ck, map_location="cpu", weights_only=False).get("mutable_hparams", {})
        missing = [k for k in EXPECT if k not in mh]
        assert not missing, f"PREFLIGHT FATAL s{s}: ckpt missing mutable_hparams {missing}"
        got = {k: float(mh[k]) for k in EXPECT}
        bad = [(k, got[k], EXPECT[k]) for k in EXPECT if abs(got[k] - EXPECT[k]) > 1e-9]
        status = "OK" if not bad else f"*** CONFIG DRIFT {bad} ***"
        print("  [preflight s%d] %s  ckpt=%s  %s" %
              (s, " ".join(f"{k}={got[k]:g}" for k in EXPECT), os.path.basename(ck), status), flush=True)
        assert not bad, (f"PREFLIGHT FATAL s{s}: config drift {bad} — NOT the dm10 operating point. "
                         f"Coder config-fidelity FAIL; refusing to measure a mismatched ensemble.")
        seen["tau_gaba"].add(got["tau_gaba"]); seen["gNMDA"].add(got["gNMDA"])
    for k in ("tau_gaba", "gNMDA"):
        assert len(seen[k]) == 1, f"PREFLIGHT FATAL: non-uniform {k} across seeds: {seen[k]}"
    derived = {"TAU_GABA": f"{next(iter(seen['tau_gaba'])):g}", "GNMDA": f"{next(iter(seen['gNMDA'])):g}"}
    print(f"[preflight] PASS — all {len(seeds)} seeds at the dm10 point; "
          f"env DERIVED from ckpts: {derived}", flush=True)
    return derived


def _install_patches(N, env_override):
    """Redirect ckpt paths + guard the operating point on the shared routec_net_io module object."""
    for k, v in env_override.items():
        os.environ[k] = v                    # re-read by load_ckpt at runtime (L126/L146)
    N.ckpt_path_for_seed = dm10_ckpt
    _orig_load = N.load_ckpt

    def _guard_load(ck, seed, nt, dev, *a, **k):
        out = _orig_load(ck, seed, nt, dev, *a, **k)
        net = out[0]
        for attr, exp in EXPECT.items():
            got = float(getattr(net, attr))
            assert abs(got - exp) < 1e-9, (
                f"DM10 GUARD FAIL: net.{attr}={got} != {exp} for ckpt {ck} — clamp leak / wrong ckpt / "
                f"mismeasured operating point. ABORTING (would be an invalid dm10 read).")
        return out

    N.load_ckpt = _guard_load
    print(f"[val394-shim] patches installed: ckpt->({DM10_ENS} | s42:{DM10_S42}), "
          f"env {env_override}, GUARD aM/dM/tau_gaba/gNMDA/u_stp_a", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, choices=("preflight", "main", "cre", "lat", "g7", "g7agg"))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(42, 52)))
    ap.add_argument("--gain_exp", type=int, default=1, choices=(1, 2))
    ap.add_argument("--device", default="cuda:0")
    A = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    if A.gate == "preflight":
        preflight(A.seeds)                          # CPU-only; raises (FATAL) on any drift
        print("[val394-shim] preflight-only complete.", flush=True)
        return
    if A.gate == "g7agg":
        sys.path.insert(0, MEASURE); sys.path.insert(0, BUNDLE)
        import run_gate7_cuerel as mod
        _install_patches(mod.CR.N, {})              # aggregate reads JSONs; no load, no env needed
        mod.CR.N.OUT_DIR = OUTDIR
        sys.argv = ["run_gate7_cuerel", "--aggregate", "--gain_exp", str(A.gain_exp)]
        mod.main()
        return

    # measurement gates: PRE-FLIGHT first (derive env per-seed from ckpts + catch drift), then run
    env_override = preflight(A.seeds)
    sys.path.insert(0, MEASURE); sys.path.insert(0, BUNDLE)
    if A.gate == "main":
        os.environ["OUT_JSON_OVERRIDE"] = os.path.join(OUTDIR, "gates_main.json")
        import measure_ens_main as mod
        _install_patches(mod.N, env_override)
        sys.argv = ["measure_ens_main"] + [str(s) for s in A.seeds]
        mod.main()
    elif A.gate == "cre":
        os.environ["OUT_JSON_OVERRIDE"] = os.path.join(OUTDIR, "gate5_cre.json")
        import measure_ens_cre as mod
        _install_patches(mod.N, env_override)
        sys.argv = ["measure_ens_cre"] + [str(s) for s in A.seeds]
        mod.main()
    elif A.gate == "lat":
        os.environ["OUT_JSON_OVERRIDE"] = os.path.join(OUTDIR, "gate6_latency_sweep.json")
        import measure_ens_latency_sweep as mod
        _install_patches(mod.N, env_override)
        sys.argv = ["measure_ens_latency_sweep"] + [str(s) for s in A.seeds]
        mod.main()
    elif A.gate == "g7":
        import run_gate7_cuerel as mod
        _install_patches(mod.CR.N, env_override)
        mod.CR.N.OUT_DIR = OUTDIR
        for s in A.seeds:
            sys.argv = ["run_gate7_cuerel", "--seed", str(s), "--gain_exp", str(A.gain_exp),
                        "--device", A.device]
            mod.main()


if __name__ == "__main__":
    main()
