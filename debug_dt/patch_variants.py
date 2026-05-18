"""Monkey-patch variants for hypothesis testing of dt-dependence.

Each `apply_*` function wraps Training.MultiBatchAudVisMSINetworkTime.
update_all_layers_batch with a source-rewritten replacement.  All
replacements are pure additions / multiplications and leave every other
line of the function untouched.

The original is preserved in ORIG_UPDATE.  Always call `restore_original()`
between experiments to avoid cross-contamination.
"""

from __future__ import annotations
import inspect
import textwrap
import torch

import Training as TRN


ORIG_UPDATE = TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch


def restore_original():
    TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch = ORIG_UPDATE


def _source_of_original():
    src = inspect.getsource(ORIG_UPDATE)
    return textwrap.dedent(src)


def _install(src: str, fn_name: str = "_patched_update_all_layers_batch"):
    """Compile and install a rewritten method body."""
    ns: dict = {}
    glb = TRN.__dict__.copy()
    exec(src, glb, ns)
    new_fn = ns[fn_name]
    TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch = new_fn


def _rename_def(src: str, new_name: str) -> str:
    return src.replace("def update_all_layers_batch(", f"def {new_name}(", 1)


# ──────────────────────────────────────────────────────────────────────
# Variant A: scale I_M.add_(I_nmda) by (self.dt / 0.1).
#   At dt=0.1 unchanged.  At dt=0.05 halved.
# ──────────────────────────────────────────────────────────────────────
def apply_scale_nmda_to_IM(ref_dt: float = 0.1):
    src = _source_of_original()
    src = _rename_def(src, "_p1")
    old = "self.I_M.add_(I_nmda)"
    new = f"self.I_M.add_(I_nmda * (self.dt / {ref_dt!r}))"
    assert src.count(old) == 1
    src = src.replace(old, new)
    _install(src, fn_name="_p1")


# ──────────────────────────────────────────────────────────────────────
# Variant B: scale I_M.add_(I_AMPA_curr + b_msi) by (self.dt / 0.1).
# ──────────────────────────────────────────────────────────────────────
def apply_scale_ampa_to_IM(ref_dt: float = 0.1):
    src = _source_of_original()
    src = _rename_def(src, "_p2")
    old = "self.I_M.add_(I_AMPA_curr + self.b_msi)"
    new = f"self.I_M.add_((I_AMPA_curr + self.b_msi) * (self.dt / {ref_dt!r}))"
    assert src.count(old) == 1
    src = src.replace(old, new)
    _install(src, fn_name="_p2")


# ──────────────────────────────────────────────────────────────────────
# Variant C: scale ALL per-substep I_M additions by (self.dt / 0.1).
#   Targets:   I_M += I_AMPA_curr + b_msi
#              I_M += I_nmda
#              I_M -= g_FFinh*(I_inA_inh + I_inV_inh)
#              I_M -= I_M_inh2exc
#              I_M -= g_GABA*I_latM
# ──────────────────────────────────────────────────────────────────────
def apply_scale_all_IM(ref_dt: float = 0.1):
    src = _source_of_original()
    src = _rename_def(src, "_p3")
    pairs = [
        ("self.I_M.add_(I_AMPA_curr + self.b_msi)",
         f"self.I_M.add_((I_AMPA_curr + self.b_msi) * (self.dt / {ref_dt!r}))"),
        ("self.I_M.add_(I_nmda)",
         f"self.I_M.add_(I_nmda * (self.dt / {ref_dt!r}))"),
        ("self.I_M.sub_(self.g_FFinh\n                      * (I_inA_inh + I_inV_inh))",
         f"self.I_M.sub_(self.g_FFinh * (I_inA_inh + I_inV_inh) * (self.dt / {ref_dt!r}))"),
        ("self.I_M.sub_(I_M_inh2exc)",
         f"self.I_M.sub_(I_M_inh2exc * (self.dt / {ref_dt!r}))"),
        ("self.I_M.sub_(self.g_GABA * I_latM)",
         f"self.I_M.sub_(self.g_GABA * I_latM * (self.dt / {ref_dt!r}))"),
    ]
    for old, new in pairs:
        assert src.count(old) == 1, f"Couldn't find unique occurrence:\n{old}"
        src = src.replace(old, new)
    _install(src, fn_name="_p3")


# ──────────────────────────────────────────────────────────────────────
# Variant D: only the spike-driven inhibition adds (FF, recur, lateral)
#   No NMDA scaling, no AMPA scaling — only the inhibition.
# ──────────────────────────────────────────────────────────────────────
def apply_scale_inh_only(ref_dt: float = 0.1):
    src = _source_of_original()
    src = _rename_def(src, "_p4")
    pairs = [
        ("self.I_M.sub_(self.g_FFinh\n                      * (I_inA_inh + I_inV_inh))",
         f"self.I_M.sub_(self.g_FFinh * (I_inA_inh + I_inV_inh) * (self.dt / {ref_dt!r}))"),
        ("self.I_M.sub_(I_M_inh2exc)",
         f"self.I_M.sub_(I_M_inh2exc * (self.dt / {ref_dt!r}))"),
        ("self.I_M.sub_(self.g_GABA * I_latM)",
         f"self.I_M.sub_(self.g_GABA * I_latM * (self.dt / {ref_dt!r}))"),
    ]
    for old, new in pairs:
        assert src.count(old) == 1, f"Couldn't find unique occurrence:\n{old}"
        src = src.replace(old, new)
    _install(src, fn_name="_p4")


# ──────────────────────────────────────────────────────────────────────
# Variant E: replace linear decay with exact exp(-dt/tau)
# ──────────────────────────────────────────────────────────────────────
def apply_exp_decay():
    src = _source_of_original()
    src = _rename_def(src, "_p5")
    pairs = [
        ("decay_factor = 1.0 - self.dt / self.tau_syn",
         "decay_factor = float(torch.exp(torch.tensor(-self.dt / self.tau_syn)).item())"),
        ("ampa_decay = 1.0 - self.dt / self.tau_ampa_lp",
         "ampa_decay = float(torch.exp(torch.tensor(-self.dt / self.tau_ampa_lp)).item())"),
        ("nmda_decay = 1.0 - self.dt / self.tau_nmda",
         "nmda_decay = float(torch.exp(torch.tensor(-self.dt / self.tau_nmda)).item())"),
    ]
    for old, new in pairs:
        assert src.count(old) == 1, f"Couldn't find: {old}"
        src = src.replace(old, new)
    _install(src, fn_name="_p5")


# ──────────────────────────────────────────────────────────────────────
# Variant F: Izhikevich half-stepping for v_msi, v_msi_inh, v_uniA/V
# ──────────────────────────────────────────────────────────────────────
def apply_izhi_half():
    src = _source_of_original()
    src = _rename_def(src, "_p6")
    blocks = [
        ("        dVA = (0.04 * self.v_uniA.pow(2) + 5.0 * self.v_uniA + 140.0\n"
         "               - self.u_uniA + self.I_A)\n"
         "        self.v_uniA += self.dt * dVA\n"
         "        self.u_uniA += self.dt * (self.aA * (self.bA * self.v_uniA - self.u_uniA))",
         "        for _hs in range(2):\n"
         "            _dt_h = self.dt / 2.0\n"
         "            dVA = (0.04 * self.v_uniA.pow(2) + 5.0 * self.v_uniA + 140.0\n"
         "                   - self.u_uniA + self.I_A)\n"
         "            self.v_uniA += _dt_h * dVA\n"
         "            self.u_uniA += _dt_h * (self.aA * (self.bA * self.v_uniA - self.u_uniA))"),
        ("        dVV = (0.04 * self.v_uniV.pow(2) + 5.0 * self.v_uniV + 140.0\n"
         "               - self.u_uniV + self.I_V)\n"
         "        self.v_uniV += self.dt * dVV\n"
         "        self.u_uniV += self.dt * (self.aV * (self.bV * self.v_uniV - self.u_uniV))",
         "        for _hs in range(2):\n"
         "            _dt_h = self.dt / 2.0\n"
         "            dVV = (0.04 * self.v_uniV.pow(2) + 5.0 * self.v_uniV + 140.0\n"
         "                   - self.u_uniV + self.I_V)\n"
         "            self.v_uniV += _dt_h * dVV\n"
         "            self.u_uniV += _dt_h * (self.aV * (self.bV * self.v_uniV - self.u_uniV))"),
        ("        dVM = (0.04 * self.v_msi.pow(2) + 5.0 * self.v_msi + 140.0 - self.u_msi + self.I_M)\n"
         "        self.v_msi += self.dt * dVM\n"
         "        self.u_msi += self.dt * (self.aM * (self.bM * self.v_msi - self.u_msi))",
         "        for _hs in range(2):\n"
         "            _dt_h = self.dt / 2.0\n"
         "            dVM = (0.04 * self.v_msi.pow(2) + 5.0 * self.v_msi + 140.0 - self.u_msi + self.I_M)\n"
         "            self.v_msi += _dt_h * dVM\n"
         "            self.u_msi += _dt_h * (self.aM * (self.bM * self.v_msi - self.u_msi))"),
        ("        dVMi = (0.04 * self.v_msi_inh.pow(2) + 5.0 * self.v_msi_inh + 140.0 - self.u_msi_inh + self.I_M_inh)\n"
         "        self.v_msi_inh += self.dt * dVMi\n"
         "        self.u_msi_inh += self.dt * (self.aMi * (self.bMi * self.v_msi_inh - self.u_msi_inh))",
         "        for _hs in range(2):\n"
         "            _dt_h = self.dt / 2.0\n"
         "            dVMi = (0.04 * self.v_msi_inh.pow(2) + 5.0 * self.v_msi_inh + 140.0 - self.u_msi_inh + self.I_M_inh)\n"
         "            self.v_msi_inh += _dt_h * dVMi\n"
         "            self.u_msi_inh += _dt_h * (self.aMi * (self.bMi * self.v_msi_inh - self.u_msi_inh))"),
    ]
    for old, new in blocks:
        assert src.count(old) == 1, f"Couldn't find Izhi block:\n{old}"
        src = src.replace(old, new)
    _install(src, fn_name="_p6")


# ──────────────────────────────────────────────────────────────────────
# Variant G: scale ALL leaky integrator additions including bias by dt
# (broadest dt-fix attempt; includes I_A and I_V external input flows)
# ──────────────────────────────────────────────────────────────────────
def apply_scale_all_leaky(ref_dt: float = 0.1):
    src = _source_of_original()
    src = _rename_def(src, "_p7")
    pairs = [
        ("self.I_M.add_(I_AMPA_curr + self.b_msi)",
         f"self.I_M.add_((I_AMPA_curr + self.b_msi) * (self.dt / {ref_dt!r}))"),
        ("self.I_M.add_(I_nmda)",
         f"self.I_M.add_(I_nmda * (self.dt / {ref_dt!r}))"),
        ("self.I_M.sub_(self.g_FFinh\n                      * (I_inA_inh + I_inV_inh))",
         f"self.I_M.sub_(self.g_FFinh * (I_inA_inh + I_inV_inh) * (self.dt / {ref_dt!r}))"),
        ("self.I_M.sub_(I_M_inh2exc)",
         f"self.I_M.sub_(I_M_inh2exc * (self.dt / {ref_dt!r}))"),
        ("self.I_M.sub_(self.g_GABA * I_latM)",
         f"self.I_M.sub_(self.g_GABA * I_latM * (self.dt / {ref_dt!r}))"),
        # MSI-inh excitation
        ("self.I_M_inh.add_(I_Mi_a_AMPA + I_Mi_v_AMPA + self.b_msi_inh)",
         f"self.I_M_inh.add_((I_Mi_a_AMPA + I_Mi_v_AMPA + self.b_msi_inh) * (self.dt / {ref_dt!r}))"),
        ("self.I_M_inh.add_(I_nmda_inh)",
         f"self.I_M_inh.add_(I_nmda_inh * (self.dt / {ref_dt!r}))"),
        # Out path
        ("self.I_O.add_((I_O_msi + self.b_out))",
         f"self.I_O.add_((I_O_msi + self.b_out) * (self.dt / {ref_dt!r}))"),
        # I_A, I_V lateral inhibition (uni-modal)
        ("self.I_A.sub_(self.g_latA * I_latA)",
         f"self.I_A.sub_(self.g_latA * I_latA * (self.dt / {ref_dt!r}))"),
        ("self.I_V.sub_(self.g_latV * I_latV)",
         f"self.I_V.sub_(self.g_latV * I_latV * (self.dt / {ref_dt!r}))"),
    ]
    for old, new in pairs:
        assert src.count(old) == 1, f"Couldn't find:\n{old}"
        src = src.replace(old, new)
    _install(src, fn_name="_p7")
