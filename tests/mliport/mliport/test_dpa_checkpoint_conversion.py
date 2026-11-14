"""Trust-gated DPA checkpoint conversion regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.convert_dpa_checkpoint import convert_checkpoint


def _checkpoint(path: Path):
    original = {
        "model": {
            "layer.weight": torch.tensor([[1.0, 2.25]], dtype=torch.float64),
            "_extra_state": {
                "model_params": {"type_map": ["Li", "S"], "precision": "float64"},
                "train_infos": {"lr": np.float64(0.0125)},
            },
        },
        "optimizer": {"param_groups": [{"lr": np.float64(0.0125)}]},
    }
    torch.save(original, path)
    return original


def test_trusted_conversion_preserves_double_tensor_and_model_metadata(tmp_path):
    source, target = tmp_path / "training.pt", tmp_path / "inference.pt"
    original = _checkpoint(source)
    source_bytes = source.read_bytes()
    record = convert_checkpoint(source, target, trusted=True)
    safe = torch.load(target, map_location="cpu", weights_only=True)
    assert set(safe) == {"model"}
    assert safe["model"]["_extra_state"]["train_infos"]["lr"] == 0.0125
    assert (
        safe["model"]["_extra_state"]["model_params"]
        == original["model"]["_extra_state"]["model_params"]
    )
    assert torch.equal(safe["model"]["layer.weight"], original["model"]["layer.weight"])
    assert safe["model"]["layer.weight"].dtype == torch.float64
    assert source.read_bytes() == source_bytes
    assert record["tensor_dtypes"] == ["torch.float64"]
    assert record["tensor_count"] == 1
    assert "optimizer" in record["dropped_top_level_keys"]
    assert target.with_suffix(".conversion.json").is_file()


def test_untrusted_conversion_never_unpickles_or_writes(tmp_path):
    source, target = tmp_path / "training.pt", tmp_path / "inference.pt"
    source.write_bytes(b"not a checkpoint")
    with pytest.raises(PermissionError, match="explicit trust"):
        convert_checkpoint(source, target, trusted=False)
    assert not target.exists()


def test_original_and_existing_target_are_protected(tmp_path):
    source, target = tmp_path / "training.pt", tmp_path / "inference.pt"
    _checkpoint(source)
    with pytest.raises(ValueError, match="Never overwrite"):
        convert_checkpoint(source, source, trusted=True)
    target.write_bytes(b"existing output")
    with pytest.raises(FileExistsError):
        convert_checkpoint(source, target, trusted=True)
    assert target.read_bytes() == b"existing output"


def test_unknown_model_extra_state_fails_closed_without_writing(tmp_path):
    source, target = tmp_path / "training.pt", tmp_path / "inference.pt"
    data = _checkpoint(source)
    data["model"]["_extra_state"]["unhandled"] = np.array([1, 2])
    torch.save(data, source)
    with pytest.raises(TypeError, match="Unsupported value"):
        convert_checkpoint(source, target, trusted=True)
    assert not target.exists()
