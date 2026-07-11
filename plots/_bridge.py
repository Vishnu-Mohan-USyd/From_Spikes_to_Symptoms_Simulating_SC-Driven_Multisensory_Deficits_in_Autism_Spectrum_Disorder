"""Shared bridge: faithful current-model net (val36 delayfix) -> repo measurement+plot fns.
Registers sys.modules["Training"]=delayfix via routec/val36 import, so each repo script's
`from Training import *` binds to the faithful build. Captures any plt.savefig to /tmp (both
svg+png); repo dir stays pristine (read-only style/measurement ref). NO retrain — inference only.
"""
import os, sys, importlib.util, contextlib

_THIS  = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(_THIS)                    # bundle root (this file lives in <root>/plots/)
WORK   = ROOT                                      # bundle root: modules + checkpoint/ fonts/ results/ figures/
REPO   = ROOT                                      # fonts live at <root>/fonts -> os.path.join(B.REPO,"fonts",..) resolves
OUT    = os.path.join(ROOT, "figures")             # captured / saved figures
STAGE  = os.path.join(ROOT, "checkpoint")          # CANONICAL ckpt dir: 12c70662 = formA_k0 seed42 + 43-51
RUNDIR = os.path.join(ROOT, "out")                 # runtime scratch CWD for repo-style scripts: ./fonts + ./checkpoint
SEEDS = [42]                                       # canonical model is SINGLE-SEED (no tau40 ensemble exists)
CKPT_TMPL = "ckpt_ep79_seed{seed}_bs250_delay52_tau10_dL3.pt"

os.environ.setdefault("MPLBACKEND", "Agg")

# --- CANONICAL SUBSTRATE ENV (debugger #178 proof) -------------------------------------------------
# Root cause of the E/I=0.135 collapse was NOT g_rec: it was a MISSING substrate env. MG_VHALF (NMDA
# Mg-block half-V) is NOT stored in the ckpt; with no env it defaults to -35 instead of the canonical
# -48, collapsing NMDA-driven excitation ~37x. This exact set (= grade_vadvance.sh line 17) reproduces
# grade_5of6_tau40.out on ckpt 12c70662 seed42 (5090): E/I_sync 1.059, TBW 260ms, MSI 17.19Hz.
# MUST be set BEFORE importing routec_net_io (substrate params are read at net-build time).
_SUBSTRATE_ENV = {
    "DEND_COUPLING_ALPHA": "2", "MG_VHALF": "-48", "MG_VHALF_INH": "-30", "MG_K": "0.15",
    "GABA_SHUNT_SURR": "1", "K_SHUNT_SURR": "0.026", "E_GABA": "-70.0", "TAU_GABA": "10",
    "SIGMA_DL_FRAMES": "3", "GNMDA": "0.50", "TAU_NMDA": "40", "G_REC": "0.03",
    "K_DVDT": "0.0", "TAU_DVDT": "3.0", "V_THRESH_FLOOR": "20.0", "DVDT_CAP": "50.0",
}
for _k, _v in _SUBSTRATE_ENV.items():
    os.environ[_k] = _v                            # force (override any inherited wrong value)
os.environ.pop("TAU_NMDA_V", None)                 # MUST be UNSET -> code == grade's val36 (800c6f59)

sys.path.insert(0, WORK)
import routec_net_io as RIO          # imports val36 -> sys.modules["Training"]=delayfix Training
V = RIO.V                            # val36 module: V.build_net, V.load_ckpt, V.T
import numpy as np
import torch


def firewall(tag):
    return RIO.assert_frozen_readouts(tag)   # md5-assert frozen TBW/SBW (80d33465 / 73b7d136)


def load_repo_mod(name, path):
    """Import a repo script by path; its `from Training import *` binds to the delayfix class
    already registered by val36. Pre-inject the typing/pathlib/mpl names the repo scripts expected
    from their ORIGINAL `from Training import *` (the delayfix Training doesn't re-export them)."""
    import typing, pathlib
    from matplotlib import font_manager
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    for n in ("Sequence", "Optional", "Callable", "Union", "List", "Dict",
              "Tuple", "Any", "Iterable", "Mapping", "Set"):
        m.__dict__[n] = getattr(typing, n)
    m.__dict__["Path"] = pathlib.Path
    m.__dict__["font_manager"] = font_manager
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def faithful_net(seed, B, device="cuda:0"):
    """Build the faithful delayfix net for `seed` and load its staged ckpt (pure inference)."""
    ck = os.path.join(STAGE, CKPT_TMPL.format(seed=seed))
    out = V.load_ckpt(ck, seed=seed, B=B, device=device)   # -> (net, res, epoch, g_rec, comment)
    net = out[0]                                            # load_ckpt already did .to/.eval/g_rec
    net.device = torch.device(device)
    net.eval()
    if hasattr(net, "plasticity_enabled"):
        net.plasticity_enabled = False
    return net


import re


def setup_rundir():
    """A neutral CWD for repo scripts: ./fonts (Roboto) + ./checkpoint (5 staged seeds renamed to the
    repo's msi_model_surr_10_{i:02d}.pt glob). No writes to repo or production."""
    os.makedirs(RUNDIR, exist_ok=True)
    fonts = os.path.join(RUNDIR, "fonts")
    if not os.path.exists(fonts):
        os.symlink(os.path.join(REPO, "fonts"), fonts)
    ck = os.path.join(RUNDIR, "checkpoint")
    os.makedirs(ck, exist_ok=True)
    for i, seed in enumerate(SEEDS):
        link = os.path.join(ck, f"msi_model_surr_10_{i:02d}.pt")
        tgt = os.path.join(STAGE, CKPT_TMPL.format(seed=seed))
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(tgt, link)
    os.makedirs(os.path.join(RUNDIR, "Saved_Images"), exist_ok=True)
    return RUNDIR


def faithful_load_msi_model(ckpt_path, device="cuda:0", **kw):
    """Drop-in for a repo script's load_msi_model: resolve the symlink -> real seedNN ckpt -> faithful
    val36 delayfix net (forces tau_nmda_inh=21.6 / scheduled g_rec). Pure inference."""
    real = os.path.realpath(str(ckpt_path))
    m = re.search(r"seed(\d+)", real)
    if not m:
        raise ValueError(f"cannot parse seed from {real}")
    dev = "cuda:0" if str(device).startswith("cuda") else str(device)
    return faithful_net(int(m.group(1)), B=1, device=dev)


@contextlib.contextmanager
def capture(stem, cwd=None):
    """Run the repo plot fn (which uses './fonts' + savefig('./Saved_Images/..')) with CWD=RUNDIR
    so its Roboto addfont + ./checkpoint resolve, but redirect every savefig to OUT/<stem>.{svg,png}."""
    import matplotlib.pyplot as plt
    cwd0 = os.getcwd()
    os.chdir(cwd or RUNDIR)                           # so repo's relative './fonts/..' + './checkpoint' resolve
    orig_savefig, orig_show = plt.savefig, plt.show
    saved = {}

    def _cap_savefig(*a, **k):
        plt.rcParams["svg.fonttype"] = "none"
        svg = os.path.join(OUT, stem + ".svg")
        png = os.path.join(OUT, stem + ".png")
        orig_savefig(svg, format="svg", bbox_inches="tight")
        orig_savefig(png, format="png", dpi=100, bbox_inches="tight")
        saved["svg"], saved["png"] = svg, png

    plt.savefig = _cap_savefig
    plt.show = lambda *a, **k: None
    try:
        yield saved
    finally:
        plt.savefig, plt.show = orig_savefig, orig_show
        # fallback for plot fns that end in plt.show() with no savefig (e.g. SBW plot_spatial_binding_*)
        if not saved and plt.get_fignums():
            plt.rcParams["svg.fonttype"] = "none"
            svg = os.path.join(OUT, stem + ".svg")
            png = os.path.join(OUT, stem + ".png")
            plt.figure(plt.get_fignums()[-1])
            orig_savefig(svg, format="svg", bbox_inches="tight")
            orig_savefig(png, format="png", dpi=100, bbox_inches="tight")
            saved["svg"], saved["png"] = svg, png
        plt.close("all")
        os.chdir(cwd0)
