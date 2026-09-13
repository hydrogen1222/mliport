"""A failed builder setup must never publish a partially configured calculator."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mliport.calculators.grace_calc import GRACECalculatorWrapper


@pytest.mark.parametrize("stage", ["constructor", "builder", "cache"])
def test_failed_initialization_can_rebuild(tmp_path, monkeypatch, stage):
    wrapper = GRACECalculatorWrapper(tmp_path, device="cpu", neighbor_cache=True)
    monkeypatch.setattr(wrapper, "_apply_device_env", lambda: None)
    good_builder = SimpleNamespace(extract_from_ase_atoms=lambda atoms: {})
    good = SimpleNamespace(data_builders=[good_builder])
    if stage == "constructor":
        factory = MagicMock(side_effect=[RuntimeError("constructor"), good])
    else:
        bad = SimpleNamespace(
            data_builders=[] if stage == "builder" else [good_builder]
        )
        factory = MagicMock(side_effect=[bad, good])
    monkeypatch.setitem(sys.modules, "tensorpotential", MagicMock())
    monkeypatch.setitem(
        sys.modules, "tensorpotential.calculator", SimpleNamespace(TPCalculator=factory)
    )
    with pytest.raises(RuntimeError):
        wrapper.get_calculator()
    assert wrapper._calculator is None
    assert wrapper._neighbor_cache_inst is None
    wrapper._neighbor_cache = False
    assert wrapper.get_calculator() is good
    assert wrapper.get_calculator() is good
    assert factory.call_count == 2


def test_first_inference_failure_invalidates_exposed_calculator(tmp_path):
    wrapper = GRACECalculatorWrapper(tmp_path)

    def fail():
        raise ValueError("first inference")

    candidate = SimpleNamespace(calculate=fail, results={"energy": 1})
    wrapper._guard_inference(candidate)
    wrapper._calculator = candidate
    with pytest.raises(ValueError, match="first inference"):
        candidate.calculate()
    assert wrapper._calculator is None
    assert candidate.results == {}
    with pytest.raises(RuntimeError, match="fresh wrapper"):
        wrapper.get_calculator()
    with pytest.raises(RuntimeError, match="failed inference state"):
        candidate.calculate()


class _DType:
    def __init__(self, name: str):
        self.name = name


def _fake_tensorflow(monkeypatch, reader):
    fake_tf = SimpleNamespace(
        train=SimpleNamespace(load_checkpoint=lambda path: reader)
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)


def _checkpoint_fixture(tmp_path, monkeypatch, reader):
    (tmp_path / "variables").mkdir()
    (tmp_path / "variables" / "variables").write_bytes(b"")
    _fake_tensorflow(monkeypatch, reader)
    return GRACECalculatorWrapper(tmp_path, device="cpu", neighbor_cache=False)


def test_checkpoint_precision_reports_weight_dtype(tmp_path, monkeypatch):
    """Stored weight dtypes are the precision evidence (plan A-09)."""
    reader = SimpleNamespace(
        get_variable_to_dtype_map=lambda: {
            "weights": _DType("float32"),
            "bias": _DType("float32"),
            "meta": _DType("string"),
            "index": _DType("int32"),
        }
    )
    wrapper = _checkpoint_fixture(tmp_path, monkeypatch, reader)
    assert wrapper._checkpoint_precision() == ["float32"]


def test_checkpoint_precision_survives_unreadable_checkpoint(tmp_path, monkeypatch):
    def boom(path):
        raise ValueError("no checkpoint")

    wrapper = _checkpoint_fixture(
        tmp_path, monkeypatch, SimpleNamespace(load_checkpoint=boom)
    )
    assert wrapper._checkpoint_precision() is None


def test_info_reports_checkpoint_weight_precision(tmp_path, monkeypatch):
    """info() must not label a float32 model float64 via output tensors."""
    reader = SimpleNamespace(
        get_variable_to_dtype_map=lambda: {"weights": _DType("float32")}
    )
    wrapper = _checkpoint_fixture(tmp_path, monkeypatch, reader)
    monkeypatch.setattr(wrapper, "_apply_device_env", lambda: None)
    calc = SimpleNamespace(
        data_builders=[SimpleNamespace(extract_from_ase_atoms=lambda atoms: {})],
        implemented_properties=["energy", "free_energy", "forces"],
    )
    factory = MagicMock(return_value=calc)
    monkeypatch.setitem(sys.modules, "tensorpotential", MagicMock())
    monkeypatch.setitem(
        sys.modules,
        "tensorpotential.calculator",
        SimpleNamespace(TPCalculator=factory),
    )
    assert wrapper.info()["model_precision"] == ["float32"]
