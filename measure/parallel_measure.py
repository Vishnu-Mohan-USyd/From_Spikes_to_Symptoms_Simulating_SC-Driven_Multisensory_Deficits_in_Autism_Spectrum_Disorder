#!/usr/bin/env python3
"""PARALLEL tau40 ensemble measurement across BOTH local GPUs (standing directive: 5-seed runs go
parallel across the 5090 + A6000, never serialized on one GPU). Each GPU lane measures its seed-subset
through gates 1-4 (measure_ens_main) + gate5 CRE (measure_ens_cre) + gate6 latency (response_latency),
writing per-lane / per-seed jsons. Then: combine -> gates_main.json + gate5_cre.json, gate6 --aggregate,
and the 6 repo-house-style ensemble figures with across-seed error bars.

seed42 is PINNED to the 5090 (cuda:0) so its row exactly reproduces the validated baseline
(17.194 / 260 / 1.059 / MEI / 77.2 / 33.7-48.6); other seeds split across both GPUs (the ~0.7%
cross-device FP diff is far inside across-seed spread). Usage: python parallel_measure.py [SEED ...]
"""
import os, sys, time, json, glob, threading, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # bundle root
MEAS = os.path.join(ROOT, "measure")               # measure_ens_*.py + combine_ens.py (run from here)
PLOTS = os.path.join(ROOT, "plots")                # run_gate*_ens.py
RESULTS = os.path.join(ROOT, "results")            # per-seed + per-lane + canonical JSONs + lane logs
ENS = RESULTS                                      # alias: all per-lane/canonical JSONs live in results/
OUTENS = os.path.join(ROOT, "figures")             # rendered figures
OUTDIR = os.path.join(ROOT, "out")                 # routec_net_io OUT_DIR (runtime scratch)
PY = sys.executable
os.makedirs(OUTENS, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)

SUBENV = dict(DEND_COUPLING_ALPHA="2", MG_VHALF="-48", MG_VHALF_INH="-30", MG_K="0.15",
              GABA_SHUNT_SURR="1", K_SHUNT_SURR="0.026", E_GABA="-70.0", TAU_GABA="10",
              SIGMA_DL_FRAMES="3", GNMDA="0.50", TAU_NMDA="40", G_REC="0.03", EXP_G_REC="0.03",
              K_DVDT="0.0", TAU_DVDT="3.0", V_THRESH_FLOOR="20.0", DVDT_CAP="50.0", MPLBACKEND="Agg")


def ckpt_for(s):
    return os.path.join(ROOT, "checkpoint", f"ckpt_ep79_seed{int(s)}_bs250_delay52_tau10_dL3.pt")


def sh(cmd, gpu, extra_env=None, log=None):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if extra_env:
        env.update(extra_env)
    fo = open(log, "a") if log else None
    rc = subprocess.call(cmd, cwd=MEAS, env=env, stdout=fo, stderr=subprocess.STDOUT)
    if fo:
        fo.close()
    return rc


def lane(gpu, seeds, status):
    log = os.path.join(ENS, f"parallel_lane_gpu{gpu}.out")
    open(log, "w").close()
    t0 = time.time()
    print(f"[lane gpu{gpu}] seeds {seeds} START", flush=True)
    # gates 1-4
    rc1 = sh([PY, "measure_ens_main.py", *map(str, seeds)], gpu,
             {"OUT_JSON_OVERRIDE": os.path.join(ENS, f"gates_main_g{gpu}.json")}, log)
    # gate5 CRE
    rc5 = sh([PY, "measure_ens_cre.py", *map(str, seeds)], gpu,
             {"OUT_JSON_OVERRIDE": os.path.join(ENS, f"gate5_cre_g{gpu}.json")}, log)
    # gate6 latency-BENEFIT vs intensity sweep (inverse effectiveness); self-configures substrate env
    rc6 = sh([PY, "measure_ens_latency_sweep.py", *map(str, seeds)], gpu,
             {"OUT_JSON_OVERRIDE": os.path.join(ENS, f"gate6_latency_sweep_g{gpu}.json")}, log)
    status[gpu] = dict(seeds=seeds, rc_main=rc1, rc_cre=rc5, rc_lat=rc6, wall_min=(time.time() - t0) / 60)
    print(f"[lane gpu{gpu}] DONE  main={rc1} cre={rc5} lat={rc6}  wall={status[gpu]['wall_min']:.1f}min", flush=True)


def main():
    req = [int(x) for x in sys.argv[1:]] or [42] + list(range(43, 52))
    seeds = [s for s in req if os.path.exists(ckpt_for(s))]
    assert 42 in seeds, "seed42 (canonical anchor) ckpt missing"
    others = [s for s in seeds if s != 42]
    # 5090 (gpu0) is faster -> seed42 (anchor) + a slight majority of the rest; A6000 (gpu1) the remainder
    k = (len(others) + 1) // 2                                   # gpu0 share of 'others' (ceil half)
    g0 = [42] + others[:k]
    g1 = others[k:]
    print(f"=== parallel measure: gpu0(5090)={g0}  gpu1(A6000)={g1}  ({len(seeds)} models) ===", flush=True)

    # clean stale per-lane jsons so combine pools only this run
    for f in (glob.glob(os.path.join(ENS, "gates_main_g*.json"))
              + glob.glob(os.path.join(ENS, "gate5_cre_g*.json"))
              + glob.glob(os.path.join(ENS, "gate6_latency_sweep_g*.json"))):
        os.remove(f)

    status = {}
    threads = [threading.Thread(target=lane, args=(g, s, status)) for g, s in ((0, g0), (1, g1)) if s]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"=== lanes done, wall={(time.time()-t0)/60:.1f}min -> combine + aggregate + plots ===", flush=True)

    # combine -> gates_main.json + gate5_cre.json + gate6_latency_sweep.json
    subprocess.call([PY, "combine_ens.py"], cwd=MEAS)
    # the 6 ensemble figures (gate6 = latency-benefit vs intensity, inverse effectiveness)
    for g in (1, 2, 3, 4, 5, 6):
        subprocess.call([PY, f"run_gate{g}_ens.py"], cwd=PLOTS, env=dict(os.environ, MPLBACKEND="Agg"))

    json.dump(dict(seeds=seeds, g0=g0, g1=g1, status=status), open(os.path.join(ENS, "parallel_summary.json"), "w"), indent=1)
    print("=== ENSEMBLE MEASUREMENT + PLOTS COMPLETE ===", flush=True)
    for f in sorted(glob.glob(os.path.join(OUTENS, "*.png"))):
        print("  fig:", f, flush=True)


if __name__ == "__main__":
    main()
