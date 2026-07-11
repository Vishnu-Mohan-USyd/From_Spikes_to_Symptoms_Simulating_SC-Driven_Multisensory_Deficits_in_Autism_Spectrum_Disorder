#!/usr/bin/env python3
"""Combine per-GPU-lane measurement jsons into the final ensemble aggregates.

The parallel launcher runs each GPU lane over a seed-subset, writing gates_main_g{N}.json and
gate5_cre_g{N}.json (per-seed rows). This merges all lanes' rows (dedup by seed, sorted) and recomputes
mean/SD/SEM identically to measure_ens_main / measure_ens_cre, writing the canonical gates_main.json +
gate5_cre.json that the plot scripts consume. Pure numpy; no GPU, no model load.
"""
import os, sys, json, glob
import numpy as np
ENS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")  # <root>/results


def _stats(vals):
    a = np.asarray([v for v in vals if v is not None], float); a = a[~np.isnan(a)]
    n = a.size
    if n == 0:
        return dict(mean=float("nan"), sd=float("nan"), sem=float("nan"), n=0)
    sd = float(np.std(a, ddof=1)) if n > 1 else 0.0
    return dict(mean=float(np.mean(a)), sd=sd, sem=(sd / np.sqrt(n) if n > 1 else 0.0), n=int(n))


def _load_rows(pattern):
    rows, env = {}, None
    for f in sorted(glob.glob(os.path.join(ENS, pattern))):
        J = json.load(open(f))
        env = env or J.get("env")
        for r in J["rows"]:
            rows[r["seed"]] = r          # dedup by seed (last wins); seeds are disjoint across lanes
    out = [rows[s] for s in sorted(rows)]
    assert out, f"no rows found for {pattern}"
    return out, env


def combine_main():
    rows, env = _load_rows("gates_main_g*.json")
    seeds = [r["seed"] for r in rows]
    agg = {"seeds": seeds, "n": len(seeds), "rows": rows, "env": env,
           "rate_hz": _stats([r["msi_hz"] for r in rows]),
           "tbw_raw_fwhm": _stats([r["tbw_raw_fwhm"] for r in rows]),
           "ei_sync": _stats([r["ei_sync"] for r in rows]),
           "E_sync": _stats([r["E_sync"] for r in rows]),
           "I_sync": _stats([r["I_sync"] for r in rows])}
    mei_mat = np.array([r["mei"] for r in rows], float)
    agg["mei_per_intensity"] = [_stats(mei_mat[:, j]) for j in range(mei_mat.shape[1])]
    agg["ie_intensities"] = rows[0]["ie_intensities"]
    offs0 = rows[0]["tbw_offsets"]
    if all(r["tbw_offsets"] == offs0 for r in rows):
        pf = np.array([r["tbw_pfusion"] for r in rows], float)
        agg["tbw_offsets"] = offs0
        agg["tbw_pfusion_per_offset"] = [_stats(pf[:, j]) for j in range(pf.shape[1])]
    if len({len(r["psth_hz"]) for r in rows}) == 1:
        ps = np.array([r["psth_hz"] for r in rows], float)
        agg["frame_ms"] = rows[0]["frame_ms"]
        agg["psth_per_frame"] = [_stats(ps[:, j]) for j in range(ps.shape[1])]
    json.dump(agg, open(os.path.join(ENS, "gates_main.json"), "w"), indent=1)
    print(f"[combine] gates_main: n={agg['n']} seeds {seeds}  "
          f"rate={agg['rate_hz']['mean']:.2f}±{agg['rate_hz']['sd']:.2f}  "
          f"TBW={agg['tbw_raw_fwhm']['mean']:.0f}±{agg['tbw_raw_fwhm']['sd']:.0f}  "
          f"E/I={agg['ei_sync']['mean']:.3f}±{agg['ei_sync']['sd']:.3f}", flush=True)


def combine_sweep():
    if not glob.glob(os.path.join(ENS, "gate6_latency_sweep_g*.json")):
        print("[combine] gate6_sweep: no lane jsons, skipping", flush=True)
        return
    rows, env = _load_rows("gate6_latency_sweep_g*.json")
    seeds = [r["seed"] for r in rows]
    I = rows[0]["intensities"]; nI = len(I)
    ben = np.array([r["benefit"] for r in rows], float)
    LA = np.array([r["L_A"] for r in rows], float)
    LV = np.array([r["L_V"] for r in rows], float)
    LB = np.array([r["L_B"] for r in rows], float)
    agg = {"seeds": seeds, "n": len(seeds), "intensities": I, "rows": rows, "env": env,
           "benefit_per_intensity": [_stats(ben[:, j]) for j in range(nI)],
           "L_A_per_intensity": [_stats(LA[:, j]) for j in range(nI)],
           "L_V_per_intensity": [_stats(LV[:, j]) for j in range(nI)],
           "L_B_per_intensity": [_stats(LB[:, j]) for j in range(nI)]}
    json.dump(agg, open(os.path.join(ENS, "gate6_latency_sweep.json"), "w"), indent=1)
    bmeans = [agg["benefit_per_intensity"][j]["mean"] for j in range(nI)]
    print(f"[combine] gate6_sweep: n={agg['n']} seeds {seeds}  benefit min(A,V)-B/I = " +
          "  ".join(f"{I[j]:g}:{bmeans[j]:+.1f}" for j in range(nI)), flush=True)


def combine_cre():
    rows, env = _load_rows("gate5_cre_g*.json")
    seeds = [r["seed"] for r in rows]
    seps0 = rows[0]["seps"]
    cre = np.array([r["cre_rom"] for r in rows], float)
    agg = {"seeds": seeds, "n": len(seeds), "seps": seps0, "rows": rows, "env": env,
           "cre_per_offset": [_stats(cre[:, j]) for j in range(cre.shape[1])],
           "peak_cre": _stats([r["peak_cre"] for r in rows]),
           "zero_cross": _stats([r["zero_cross"] for r in rows]),
           "cen_hwhm": _stats([r["cen_hwhm"] for r in rows]),
           "surr_min": _stats([r["surr_min"] for r in rows]),
           "surr_min_d": _stats([r["surr_min_d"] for r in rows])}
    json.dump(agg, open(os.path.join(ENS, "gate5_cre.json"), "w"), indent=1)
    print(f"[combine] gate5_cre: n={agg['n']} seeds {seeds}  "
          f"peak={agg['peak_cre']['mean']:.1f}±{agg['peak_cre']['sd']:.1f}%  "
          f"zc={agg['zero_cross']['mean']:.1f}±{agg['zero_cross']['sd']:.1f}  "
          f"trough={agg['surr_min']['mean']:.1f}±{agg['surr_min']['sd']:.1f}%", flush=True)


if __name__ == "__main__":
    combine_main()
    combine_cre()
    combine_sweep()
