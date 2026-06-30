#!/usr/bin/env python3
"""Thin shared-utils wrapper for the 5 route-C validation tests (task #8/#10).

Per VALIDATION_SPEC.md §1 + lead correction (2026-06-25): this is a THIN wrapper that
`import val36_traj_d52 as V` — the idiomatic flat-repo reuse, exactly what
`measure_107_convergence.py`, `measure_develop_check.py`, `diag_136b_cre_5seed.py` already do.
It does NOT re-implement `build_net`/`load_ckpt`; it re-exports `V.build_net` / `V.load_ckpt` /
`V.T` and adds only the firewall + plotting helpers the 5 fresh tests share.

Re-exported from val36 (the recipe that TRAINED the dL3-ep79 ckpts, so the measuring build is
byte-faithful to the training build):
  build_net = V.build_net      load_ckpt = V.load_ckpt      T = V.T   (delayfix Training module)
`V.load_ckpt` forces `tau_nmda_inh=21.6`, reads `tau_gaba` (=10) from the ckpt, sets `g_rec=0.1`@ep79.
Importing val36 also loads the TBW/SBW apparatus (harmless — we use none of its classifiers; it
provides `V.md5` for the firewall and `V.T.*` primitives).

Firewall helpers:
  - assert_frozen_readouts(): V.md5-assert the TWO stage_flat frozen readouts (TBW 80d33465 /
    SBW 73b7d136), unchanged before+after each run. Never imported as templates, never edited.
  - snapshot_weights()/assert_weights_unchanged()/weight_fingerprint(): prove pure inference —
    every state_dict tensor bit-identical before == after.
Plotting helpers (lazy mpl import): setup_house_style() (Roboto + editable-SVG text), despine().
"""
import os, sys
import numpy as np
import torch

# ── idiomatic flat-repo reuse (spec §1) ──
os.environ.setdefault("VAL36_BUILD", "delayfix")      # faithful eager build (default anyway)
os.environ.setdefault("TAU_GABA", "10")               # optional; ckpt value (10) wins regardless
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                               # flat: siblings importable
import val36_traj_d52 as V        # -> V.build_net, V.load_ckpt, V.md5, V.T, V.TBW_DIR, V.SBW_DIR, V.BUILD

# ── re-export the canonical build recipe (spec §1: do NOT re-copy build_net/load_ckpt) ──
build_net = V.build_net
load_ckpt = V.load_ckpt
T = V.T                                                # delayfix Training module (all primitives live here)
BUILD = V.BUILD

# ── paths / constants (resolve relative to this flat-repo file) ──
CKPT_DIR  = os.path.join(HERE, "checkpoint")
OUT_DIR   = os.path.join(HERE, "out")
FONT_PATH = os.path.join(HERE, "fonts", "Roboto-Regular.ttf")
CKPT_TMPL = "ckpt_ep79_seed{seed}_bs250_delay52_tau10_dL3.pt"
SEEDS = [42, 43, 44, 45, 46]

TBW_PATH = os.path.join(V.TBW_DIR, "TBW_test.py")
SBW_PATH = os.path.join(V.SBW_DIR, "SBW_test.py")
TBW_MD5 = "80d33465c4bf55d6e85b5990acb92da7"
SBW_MD5 = "73b7d13626964d851cc090818b728311"


def ckpt_path_for_seed(seed):
    return os.path.join(CKPT_DIR, CKPT_TMPL.format(seed=int(seed)))


def assert_frozen_readouts(tag=""):
    """md5-assert (via V.md5) the TWO stage_flat frozen readouts; return [tbw, sbw]. Never edits them."""
    t, s = V.md5(TBW_PATH), V.md5(SBW_PATH)
    print(f"[firewall {tag}] TBW={t}  SBW={s}", flush=True)
    assert t == TBW_MD5, f"FIREWALL FAIL ({tag}): TBW_test.py md5 {t} != {TBW_MD5}"
    assert s == SBW_MD5, f"FIREWALL FAIL ({tag}): SBW_test.py md5 {s} != {SBW_MD5}"
    return [t, s]


# ── weight firewall: snapshot every state_dict tensor; assert bit-identical after inference ──
@torch.no_grad()
def snapshot_weights(net):
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


@torch.no_grad()
def assert_weights_unchanged(net, snap, tag=""):
    sd = net.state_dict()
    changed = [k for k, v0 in snap.items() if not torch.equal(v0, sd[k])]
    assert not changed, f"WEIGHT FIREWALL FAIL ({tag}): mutated tensors {changed}"
    return weight_fingerprint(net)


def weight_fingerprint(net):
    """compact float64 sum over all state_dict tensors (recorded in JSON for cross-check)."""
    return float(sum(float(v.double().sum().item()) for v in net.state_dict().values()))


# ── plotting house style (Roboto + editable-SVG text + despined outward spines); lazy mpl import ──
def setup_house_style(base_fontsize=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if os.path.exists(FONT_PATH):
        font_manager.fontManager.addfont(FONT_PATH)
        plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.size'] = base_fontsize
    plt.rcParams['xtick.labelsize'] = base_fontsize
    plt.rcParams['ytick.labelsize'] = base_fontsize
    plt.rcParams['axes.titlesize'] = base_fontsize
    plt.rcParams['axes.labelsize'] = base_fontsize
    plt.rcParams['legend.fontsize'] = base_fontsize * 0.8
    plt.rcParams['svg.fonttype'] = 'none'        # editable text in the SVG (repo convention)
    return plt


def despine(ax, outward=5):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_position(("outward", outward))
    ax.spines["bottom"].set_position(("outward", outward))
    ax.grid(False)
