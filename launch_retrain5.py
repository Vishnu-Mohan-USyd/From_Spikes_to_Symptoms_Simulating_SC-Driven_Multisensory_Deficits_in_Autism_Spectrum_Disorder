#!/usr/bin/env python3
"""task #105 — orchestrator for the 5-seed jitter VALIDATION retrain (ΔL σ=30ms baked-in, ep0->79).

Runs seeds 42-46 as ONE-SUBPROCESS-PER-SEED (one L6-graphed net per process -> dodges the #55/#68
two-graph segfault), split across the two LOCAL GPUs and run as two PARALLEL lanes (each lane sequential
internally). Per the project compute rule: 5090 (cuda:0) takes the heavier 3 seeds, A6000 (cuda:1) the 2.

  [#105 corrected-gate RE-RUN — only the 3 spuriously-killed seeds; 45/46 already reached ep79, ckpts reused as-is]
  lane cuda:0 (RTX 5090) : seeds 42, 43         (~13 min/seed -> ~26 min)
  lane cuda:1 (RTX A6000): seed  44             (~13 min)

HALT rule (Lead): if >=3 seeds FAIL (nonzero exit = an inline KILL or non-finite end state), stop
launching new seeds and escalate -> lead -> debugger. Each seed's own inline kill battery + ckpts
(ep5/30/79) + vitals log live in retrain5_jitter.py; this launcher only schedules + tallies exit codes.
Then HOLD: the validator runs the independent ep79 TBW/SBW/EI measurement (frozen readout md5 80d33465).

Usage:  python launch_retrain5.py        (foreground; intended to be run in the background by the coder)
"""
import os, sys, json, time, threading, subprocess

THIS    = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(THIS, "checkpoint")
os.makedirs(RUN_DIR, exist_ok=True)
DRIVER  = os.path.join(THIS, "retrain5_jitter.py")

LANES = {"0": [42, 43], "1": [44]}     # [#105 corrected-gate RE-RUN] only 42/43/44 (45/46 ep79 reused as-is); 42,43->5090 / 44->A6000
ENV_BASE = dict(SIGMA_DL_FRAMES="3", TAU_GABA="10")
HALT_AT_FAILS = 3

_lock = threading.Lock()
results = {}                                   # seed -> dict(rc, gpu, wall_s, status)
failures = []                                  # seeds that failed
halt = threading.Event()


def log(*a):
    print(*a, flush=True)


def run_seed(gpu, seed):
    env = dict(os.environ); env.update(ENV_BASE); env["CUDA_VISIBLE_DEVICES"] = gpu
    out_path = os.path.join(RUN_DIR, f"launch_seed{seed}_gpu{gpu}.out")
    t0 = time.time()
    log(f"[launch] seed {seed} -> cuda:{gpu}  ({time.strftime('%H:%M:%S')})")
    with open(out_path, "w", buffering=1) as fo:
        rc = subprocess.call([sys.executable, DRIVER, str(seed)], cwd=THIS, env=env,
                             stdout=fo, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    status = "PASS" if rc == 0 else "FAIL"
    with _lock:
        results[seed] = dict(rc=rc, gpu=gpu, wall_s=wall, status=status)
        if rc != 0:
            failures.append(seed)
            if len(failures) >= HALT_AT_FAILS:
                halt.set()
    log(f"[done  ] seed {seed} cuda:{gpu}  rc={rc} {status}  wall={wall/60:.1f}min  "
        f"(fails so far: {sorted(failures)})")


def lane(gpu, seeds):
    for seed in seeds:
        if halt.is_set():
            with _lock:
                results[seed] = dict(rc=None, gpu=gpu, wall_s=0.0, status="SKIPPED(halt)")
            log(f"[skip  ] seed {seed} cuda:{gpu} — HALT active (>= {HALT_AT_FAILS} failures)")
            continue
        run_seed(gpu, seed)


def main():
    log(f"=== 5-SEED JITTER RETRAIN (#105) — ΔL σ=30ms baked-in, ep0->79 ===")
    log(f"driver={DRIVER}")
    log(f"lanes: " + "  ".join(f"cuda:{g}->{s}" for g, s in LANES.items()))
    log(f"HALT if >= {HALT_AT_FAILS} seeds fail. RUN_DIR={RUN_DIR}")
    t0 = time.time()
    threads = [threading.Thread(target=lane, args=(g, s), daemon=False) for g, s in LANES.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    n_pass = sum(1 for r in results.values() if r["status"] == "PASS")
    n_fail = len(failures)
    halted = halt.is_set()
    summary = dict(seeds=sorted(results), results=results, n_pass=n_pass, n_fail=n_fail,
                   failures=sorted(failures), halted=halted, wall_s=wall, run_dir=RUN_DIR,
                   halt_at_fails=HALT_AT_FAILS)
    json.dump(summary, open(os.path.join(RUN_DIR, "launch_retrain5_summary.json"), "w"), indent=1)

    log("\n" + "=" * 72)
    log(f"5-SEED RETRAIN COMPLETE — wall={wall/60:.1f}min   PASS={n_pass}/5  FAIL={n_fail}")
    for seed in sorted(results):
        r = results[seed]
        log(f"  seed {seed}: {r['status']:>14}  cuda:{r['gpu']}  "
            f"{('wall=%.1fmin' % (r['wall_s']/60)) if r['wall_s'] else ''}")
    if halted:
        log(f"!!! HALTED: >= {HALT_AT_FAILS} seeds failed -> ESCALATE to lead -> debugger. NO post-hoc; HOLD.")
    elif n_fail:
        log(f"NOTE: {n_fail} seed(s) failed (<{HALT_AT_FAILS}); surviving seeds usable. Report to lead.")
    else:
        log(f"ALL 5 PASS. Next: ep30 post-hoc MSI/n_active vs no-jitter baseline (coder), then HOLD for validator ep79 TBW/SBW/EI.")
    log("=" * 72)
    sys.exit(2 if halted else (1 if n_fail else 0))


if __name__ == "__main__":
    main()
