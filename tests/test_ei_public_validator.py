"""Public-routing and state-preservation regression for canonical E/I validation."""

import hashlib
from pathlib import Path

import numpy as np
import torch

import run_ei_balance


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of ``path`` without modifying the checkpoint."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bitwise_value(value) -> tuple[str, bytes]:
    """Return a dtype-tagged byte representation for scalar state comparison."""
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return array.dtype.str, array.tobytes()


def test_ei_public_validator(monkeypatch) -> None:
    """The public E/I route must be canonical, balanced, and state preserving on M00."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = readme.split("## Quickstart", 1)[1].split("```bash", 1)[1].split("```", 1)[0]
    assert "python run_ei_balance.py" in quickstart
    assert "python EI_balance_test.py" not in quickstart

    protocol = (ROOT / "docs" / "validation_protocol.md").read_text(encoding="utf-8")
    assert "**Test:** `run_ei_balance.py`" in protocol
    assert "from run_ei_balance import run_ei_evoked" in protocol
    assert "pool_ei_fast" not in protocol

    legacy_source = (ROOT / "EI_balance_test.py").read_text(encoding="utf-8")
    assert "from run_ei_balance import main as _run_canonical_ei_balance" in legacy_source
    assert "_run_canonical_ei_balance()" in legacy_source

    assert torch.cuda.is_available(), "This regression requires the A6000 GPU"
    device_name = torch.cuda.get_device_name(0)
    assert "A6000" in device_name, f"Expected A6000, got {device_name}"
    print(f"CUDA device: {device_name}")

    original_probe = run_ei_balance.run_ei_probe_separated
    observed_probe_calls = 0

    def state_checked_probe(net, **probe_kwargs):
        """Wrap the canonical probe and assert exact model-state preservation."""
        nonlocal observed_probe_calls
        observed_probe_calls += 1
        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in net.named_parameters()
        }
        g_nmda_before = _bitwise_value(net.gNMDA)
        g_ffinh_before = _bitwise_value(net.g_FFinh)
        plasticity_before = net.plasticity_enabled

        result = original_probe(net, **probe_kwargs)

        parameters_after = dict(net.named_parameters())
        assert parameters_after.keys() == parameters_before.keys()
        for name, parameter_before in parameters_before.items():
            assert torch.equal(parameter_before, parameters_after[name]), (
                f"Parameter changed during canonical E/I evaluation: {name}"
            )
        assert _bitwise_value(net.gNMDA) == g_nmda_before
        assert _bitwise_value(net.g_FFinh) == g_ffinh_before
        assert net.plasticity_enabled is plasticity_before
        return result

    monkeypatch.setattr(run_ei_balance, "run_ei_probe_separated", state_checked_probe)
    checkpoint_before = _sha256(CHECKPOINT)
    summary = run_ei_balance.run_ei_evoked([CHECKPOINT], device="cuda")
    checkpoint_after = _sha256(CHECKPOINT)

    ratio = float(summary["ei_ratio_mean"])
    print(f"M00 canonical E/I ratio: {ratio:.6f}")
    assert observed_probe_calls == 1
    assert checkpoint_after == checkpoint_before
    assert 0.5 <= ratio <= 2.0
    assert abs(ratio - 1.0) < 0.15
