#!/usr/bin/env python3
"""GATE 7 — cue-reliability weighting (MLE / inverse-variance optimal cue integration) on the full-10
tau40 ensemble (seed42 fallback_5of6_tau40 + seeds 43-51 tau40_ensemble).

Lead task: run the repo cue_reliability_test.py on the ensemble, adapting ONLY model-loading + ckpt
paths + substrate config; keep the measurement (sigma grid {2..20}^2, A@80/V@100, base_intensity=30,
gain_exp=2, decoder method='com', wV=(est-80)/20, MLE wV*=sigmaV^-2/(sigmaA^-2+sigmaV^-2)) and the
plotting math unchanged. This is a RUN, not a re-derivation.

Implementation: REUSE the validated + debugger-P1-fixed route-C measurement (cue_reliability_routec.py)
VERBATIM — its N.load_ckpt substrate loader (so models are NOT E/I-collapsed), its full-frame sum_sM
readout (NOT _latest_sMSI; the P1 readout fix), its empty-profile NaN guard, and its frozen-md5 +
weight-bit-identity firewall (run_seed asserts BEFORE==AFTER). We override ONLY:
  (a) the 16-var tau40 substrate ENV (set before import; = the gates 1-6 operating point),
  (b) GAIN_EXP 1 -> 2  (lead's explicit spec + repo default; see FLAG below),
  (c) the metric namespace (separate output files), and
  (d) the ensemble ckpt paths (seed42 in fallback_5of6_tau40, 43-51 in tau40_ensemble).
No measurement code is duplicated => the frozen measurement stays byte-identical.

FLAG to lead: cue_reliability_routec.py ships GAIN_EXP=1 (cited "manuscript Methods gain prop sigma_ref/sigma,
sec 1g.3"); the repo cue_reliability_test.py default AND the lead's task both say gain_exp=2. This RUN
uses 2 as instructed; the gain_exp=1 paper-faithful variant exists if the lead wants it.

Usage:
  python run_gate7_cuerel.py --seed 42 --device cuda:0   # per-seed JSON + scorecard (seed42 SANITY)
  python run_gate7_cuerel.py --aggregate                 # pool -> aggregate JSON
"""
import os, sys, argparse
# ---- canonical tau40 substrate env (SIGMA_DL=3) = gates 1-6 operating point, BEFORE imports ----
ENV = dict(DEND_COUPLING_ALPHA="2", MG_VHALF="-48", MG_VHALF_INH="-30", MG_K="0.15",
           GABA_SHUNT_SURR="1", K_SHUNT_SURR="0.026", E_GABA="-70.0", TAU_GABA="10",
           SIGMA_DL_FRAMES="3", GNMDA="0.50", TAU_NMDA="40", G_REC="0.03", EXP_G_REC="0.03",
           K_DVDT="0.0", TAU_DVDT="3.0", V_THRESH_FLOOR="20.0", DVDT_CAP="50.0")
for _k, _v in ENV.items():
    os.environ[_k] = _v
os.environ.pop("TAU_NMDA_V", None)
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # bundle root (this file is in <root>/measure/)
sys.path.insert(0, ROOT)
import cue_reliability_routec as CR     # triggers `import routec_net_io as N` with the env set
# write per-seed + aggregate JSONs into <root>/results (where plots/run_gate7_ens.py reads), not <root>/out
CR.N.OUT_DIR = os.path.join(ROOT, "results")
os.makedirs(CR.N.OUT_DIR, exist_ok=True)

# GAIN_EXP + the output namespace are set per-run in main() from --gain_exp.
# Lead decision (2026-06-29): gain_exp=1 is the PAPER-FAITHFUL headline (gain prop sigma_ref/sigma, the
# paper Methods, = this route-C measurement's native default); gain_exp=2 is the repo's exploratory
# "set 1.0 or 2.0" knob, kept only for a transparency note. Default below = 1.

def ckpt_for(seed):
    return CR.N.ckpt_path_for_seed(int(seed))       # <root>/checkpoint/ckpt_ep79_seed{seed}_bs250_delay52_tau10_dL3.pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_trials", type=int, default=CR.DEFAULT_NTRIALS)
    ap.add_argument("--gain_exp", type=int, default=1, choices=(1, 2),
                    help="1 = paper-faithful headline (default); 2 = repo exploratory knob")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    CR.GAIN_EXP = args.gain_exp
    CR.METRIC = f"gate7_cuerel_g{args.gain_exp}"
    if args.aggregate:
        CR.aggregate()
    else:
        assert args.seed is not None, "need --seed NN (or --aggregate)"
        a = argparse.Namespace(seed=args.seed, device=args.device, n_trials=args.n_trials,
                               ckpt=ckpt_for(args.seed), out_json=None)
        CR.run_seed(a)


if __name__ == "__main__":
    main()
