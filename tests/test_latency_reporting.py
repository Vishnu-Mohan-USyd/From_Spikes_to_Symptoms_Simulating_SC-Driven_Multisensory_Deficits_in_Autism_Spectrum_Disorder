"""Regression coverage for latency comparison metrics and reporting semantics."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import response_latency_test as latency


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_TEMPLATE = str(
    ROOT / "checkpoint" / "msi_model_surr_10_{:02d}.pt"
)


def _bitwise_value(value) -> tuple[str, bytes]:
    """Return a dtype-tagged byte representation for scalar state comparison."""
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return array.dtype.str, array.tobytes()


def test_latency_reporting_m00(monkeypatch, capsys) -> None:
    """M00 AV must tie the fastest route and retain the mean-latency benefit."""
    assert torch.cuda.is_available(), "This regression requires the A6000 GPU"
    device_name = torch.cuda.get_device_name(0)
    assert "A6000" in device_name, f"Expected A6000, got {device_name}"

    original_loader = latency.load_msi_model
    state_records: list[dict[str, object]] = []

    def state_capturing_loader(checkpoint: Path, *, device: str):
        """Record each fresh modality network before evaluation configuration."""
        net = original_loader(checkpoint, device=device)
        state_records.append({
            "net": net,
            "parameters": {
                name: parameter.detach().clone()
                for name, parameter in net.named_parameters()
            },
            "g_FFinh": _bitwise_value(net.g_FFinh),
        })
        return net

    monkeypatch.setattr(latency, "load_msi_model", state_capturing_loader)
    result = latency.run_latency_test(
        ckpt_template=CHECKPOINT_TEMPLATE,
        n_models=1,
        device="cuda",
    )
    assert len(state_records) == 3
    row = result.iloc[0]
    dt_ms = float(state_records[0]["net"].dt)

    assert row["B_ms"] <= row["MeanUni_ms"]
    assert abs(row["B_ms"] - min(row["A_ms"], row["V_ms"])) <= dt_ms
    assert row["MeanUniBenefit_ms"] > 0.0
    assert abs(row["FastestUniBenefit_ms"]) <= dt_ms
    assert row["A_ms"] <= row["V_ms"]
    assert row["FastestUni_ms"] == row["A_ms"]
    assert row["UniMean_ms"] == row["MeanUni_ms"]
    assert row["ΔLatency_ms"] == row["MeanUniBenefit_ms"]
    assert 5.0 <= row["MeanUniBenefit_ms"] <= 10.0

    for modality, record in zip(("A", "V", "B"), state_records):
        net = record["net"]
        assert net.plasticity_enabled is False
        assert net.freeze_g_FFinh is True
        parameters_after = dict(net.named_parameters())
        parameters_before = record["parameters"]
        assert parameters_after.keys() == parameters_before.keys()
        for name, parameter_before in parameters_before.items():
            assert torch.equal(parameter_before, parameters_after[name]), (
                f"{modality} parameter changed during frozen latency evaluation: {name}"
            )
        assert _bitwise_value(net.g_FFinh) == record["g_FFinh"]

    report = pd.concat([result, result]).copy()
    report.index = ["M00_a", "M00_b"]
    saved_paths: list[str] = []
    monkeypatch.setattr(
        latency.plt,
        "savefig",
        lambda path, **_kwargs: saved_paths.append(str(path)),
    )
    monkeypatch.setattr(latency.plt, "show", lambda: None)
    latency.summarise(report)
    console = capsys.readouterr().out

    assert "MeanUni_ms" in console
    assert "MeanUniBenefit_ms" in console
    assert "FastestUni_ms" in console
    assert "FastestUniBenefit_ms" in console
    assert "positive; approximately 7.5 ms" in console
    assert "zero within dt resolution" in console
    assert "AV ties the fastest auditory route" in console
    assert saved_paths == ["./Saved_Images/Latency.svg"]

    axis = latency.plt.gca()
    tick_labels = " ".join(label.get_text() for label in axis.get_xticklabels())
    figure_text = " ".join(text.get_text() for text in axis.figure.texts)
    assert "Auditory\n(fastest)" in tick_labels
    assert "AV\n(ties auditory)" in tick_labels
    assert axis.get_title() == "AV ties the fastest auditory route"
    assert "Benefit vs mean unimodal" in figure_text
    assert "vs fastest unimodal" in figure_text
    latency.plt.close(axis.figure)

    source = (ROOT / "response_latency_test.py").read_text(encoding="utf-8")
    assert "faster than both" not in source
    assert "convergence-threshold acceleration" not in source

    print(f"CUDA device: {device_name}")
    print(
        "M00 latency: "
        f"A={row['A_ms']:.3f} ms, V={row['V_ms']:.3f} ms, "
        f"AV={row['B_ms']:.3f} ms, "
        f"mean benefit={row['MeanUniBenefit_ms']:.3f} ms, "
        f"fastest benefit={row['FastestUniBenefit_ms']:.3f} ms"
    )
