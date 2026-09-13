from __future__ import annotations

from types import SimpleNamespace

import pytest

from mliport.install.hardware import detect_gpus


@pytest.mark.parametrize(
    "visibility, expected",
    [
        (None, ["GPU-a", "GPU-b"]),
        ("", []),
        ("1", ["GPU-b"]),
        ("GPU-b,GPU-a", ["GPU-b", "GPU-a"]),
    ],
)
def test_probe_uses_visible_inventory(monkeypatch, visibility, expected):
    if visibility is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    monkeypatch.setattr(
        "mliport.install.hardware.subprocess.run",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout=("V100,7.0,580.1,16384,0,GPU-a\n4090,8.9,580.1,24576,1,GPU-b\n"),
        ),
    )
    gpus = detect_gpus() or []
    assert [g.uuid for g in gpus] == expected
    assert [g.logical_index for g in gpus] == list(range(len(gpus)))


def test_missing_smi_returns_no_inventory(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr("mliport.install.hardware.subprocess.run", missing)
    assert detect_gpus() is None
