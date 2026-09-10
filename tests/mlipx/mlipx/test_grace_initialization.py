"""A failed builder setup must never publish a partially configured calculator."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mlipx.calculators.grace_calc import GRACECalculatorWrapper


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
