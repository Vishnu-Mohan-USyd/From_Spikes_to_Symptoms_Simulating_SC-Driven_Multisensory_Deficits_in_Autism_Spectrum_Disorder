#!/usr/bin/env python
"""routec_overrides.py — Phase-2 inference-screen parameter overrides.

Applies the Phase-1 biology-corrected hyper-parameters to an ALREADY-LOADED
route-c ``MultiBatchAudVisMSINetworkTime`` at INFERENCE, with NO edit to the
md5-gated Training.py and NO retraining.  The override is a post-load setattr,
which is provably correct for every corrected parameter because:

  * ``tau_nmda_inh`` -> ``nmda_decay_inh = 1 - dt/tau_nmda_inh`` is recomputed
    LIVE every forward step (Training.py L2234).  setattr is read next step.
  * ``tau_gaba``     -> ``gaba_decay = 1 - dt/tau_gaba`` recomputed LIVE
    (Training.py L2236).
  * ``Erev_nmda`` / ``gNMDA`` are read LIVE in the NMDA-current expression
    every step (Training.py L2407/2508).
  * ``conduction_delay_msi_inh2exc`` is consumed via the *physical-ms* field
    ``conduction_delay_msi_inh2exc_ms`` because ``dt_correct_nmda is True``
    (Training.py L2292/2297).  The ring buffer ``buffer_msi_inh2exc`` is
    re-allocated to the corrected depth by ``_reset_delay_buffers()`` — which
    ``reset_state()`` calls on EVERY measurement (Training.py L2117).  We also
    call it here so the buffer is correct (and verifiable) immediately.

Baseline (saved in the asymrc ep80 ckpts, verified across all 5 seeds):
  gNMDA=0.7, Erev_nmda=20.0 mV, tau_gaba=50.0 ms,
  conduction_delay_msi_inh2exc=450 substeps (dt=0.1 -> 45 ms),
  tau_nmda_inh=25.0 ms (NOT saved; set fresh from ctor L1507 every load).

Corrected config (Phase-1 primary-source values; cite in logs):
  msi_inh2exc 45->5 ms (450->50 substeps)  [Whyland & Bickford 2018]
  tau_gaba    50->20 ms                     [Kirischuk 2005]
  tau_nmda_inh 25->90 ms                    [Booker et al. 2021 eNeuro]
  Erev_nmda   20->0 mV                      [Jahr & Stevens 1990]
  gNMDA       0.7 unchanged  AND  driving-force-rescaled (see rescale_gnmda)
"""
from __future__ import annotations


# Ground-truth baseline values (verified by inspecting the asymrc ep80 ckpts).
BASELINE = {
    "tau_nmda_inh": 25.0,        # ms (from ctor, not saved)
    "tau_gaba": 50.0,            # ms
    "Erev_nmda": 20.0,           # mV
    "gNMDA": 0.7,
    "msi_inh2exc_ms": 45.0,      # ms (450 substeps @ dt=0.1)
}

# Headline joint-corrected config (Erev variant set separately; gNMDA handled
# by the two-variant logic in the orchestration).
JOINT_CORRECTED = {
    "tau_nmda_inh": 90.0,        # ms
    "tau_gaba": 20.0,            # ms
    "Erev_nmda": 0.0,            # mV
    "msi_inh2exc_ms": 5.0,       # ms (50 substeps @ dt=0.1)
    # gNMDA: orchestration runs both {0.7 unchanged} and {rescaled}.
}


def rescale_gnmda(g0: float, erev_old: float, erev_new: float, v_rep: float) -> float:
    """Driving-force-preserving NMDA conductance rescale.

    Holds the NMDA current  I_nmda = g * (Erev - V)  constant at a
    representative operating voltage ``v_rep`` when the reversal potential
    moves from ``erev_old`` to ``erev_new``:

        g1 * (erev_new - v_rep) == g0 * (erev_old - v_rep)
        => g1 = g0 * (erev_old - v_rep) / (erev_new - v_rep)

    Example: g0=0.7, erev_old=20, erev_new=0, v_rep=-55 -> 0.954.
    """
    denom = (erev_new - v_rep)
    if abs(denom) < 1e-9:
        raise ValueError(f"rescale_gnmda: degenerate driving force (erev_new={erev_new}, v_rep={v_rep})")
    return float(g0) * (float(erev_old) - float(v_rep)) / denom


def apply_routec_overrides(net, cfg: dict | None) -> dict:
    """Apply corrected hparams to a loaded net in-place. Returns what was applied.

    ``cfg`` keys (all optional; None/absent => leave at the ckpt value):
        tau_nmda_inh   : float ms
        tau_gaba       : float ms
        Erev_nmda      : float mV
        gNMDA          : float
        msi_inh2exc_ms : float ms   (delay of the inh->exc disynaptic IPSC leg)

    A ``cfg`` of None or {} is a true no-op (baseline reproduce).
    """
    applied: dict = {}
    if not cfg:
        applied["_mode"] = "baseline_noop"
        # still record the realized live values for the log
        applied["live"] = _live_snapshot(net)
        return applied

    if cfg.get("tau_nmda_inh") is not None:
        net.tau_nmda_inh = float(cfg["tau_nmda_inh"])
        applied["tau_nmda_inh"] = net.tau_nmda_inh

    if cfg.get("tau_gaba") is not None:
        net.tau_gaba = float(cfg["tau_gaba"])
        applied["tau_gaba"] = net.tau_gaba

    if cfg.get("Erev_nmda") is not None:
        net.Erev_nmda = float(cfg["Erev_nmda"])
        applied["Erev_nmda"] = net.Erev_nmda

    if cfg.get("gNMDA") is not None:
        net.gNMDA = float(cfg["gNMDA"])
        applied["gNMDA"] = net.gNMDA

    if cfg.get("msi_inh2exc_ms") is not None:
        ms = float(cfg["msi_inh2exc_ms"])
        net.conduction_delay_msi_inh2exc_ms = ms
        # keep the raw-substep field consistent (used only if dt_correct_nmda were False)
        net.conduction_delay_msi_inh2exc = int(round(ms / float(net.dt)))
        # rebuild the ring buffer at the corrected depth NOW (reset_state also does
        # this on every measurement; doing it here makes the depth immediately
        # verifiable and correct even before the first reset).
        net._reset_delay_buffers()
        applied["msi_inh2exc_ms"] = ms
        applied["msi_inh2exc_substeps"] = net.conduction_delay_msi_inh2exc

    applied["live"] = _live_snapshot(net)
    return applied


def _live_snapshot(net) -> dict:
    """The values the forward pass will actually use, for log verification."""
    snap = {
        "tau_nmda_inh": float(getattr(net, "tau_nmda_inh", float("nan"))),
        "tau_gaba": float(getattr(net, "tau_gaba", float("nan"))),
        "Erev_nmda": float(getattr(net, "Erev_nmda", float("nan"))),
        "gNMDA": float(getattr(net, "gNMDA", float("nan"))),
        "tau_nmda": float(getattr(net, "tau_nmda", float("nan"))),  # exc pool (untouched)
        "dt": float(getattr(net, "dt", float("nan"))),
    }
    # realized live delay (substeps) for the inh->exc leg
    try:
        if getattr(net, "dt_correct_nmda", False) and hasattr(net, "conduction_delay_msi_inh2exc_ms"):
            snap["msi_inh2exc_substeps_live"] = int(
                net._delay_substeps_from_ms(net.conduction_delay_msi_inh2exc_ms)
            )
            snap["msi_inh2exc_ms"] = float(net.conduction_delay_msi_inh2exc_ms)
        else:
            snap["msi_inh2exc_substeps_live"] = int(getattr(net, "conduction_delay_msi_inh2exc", -1))
    except Exception as e:  # pragma: no cover
        snap["msi_inh2exc_substeps_live"] = f"ERR:{e!r}"
    # actual allocated buffer depth (proves _reset_delay_buffers resized it)
    try:
        buf = getattr(net, "buffer_msi_inh2exc", None)
        snap["buffer_msi_inh2exc_depth"] = int(buf.shape[0]) if buf is not None else None
    except Exception as e:  # pragma: no cover
        snap["buffer_msi_inh2exc_depth"] = f"ERR:{e!r}"
    return snap
