"""Tests for actionable MACE environment validation."""

from __future__ import annotations

import pytest

from mliport.calculators.mace_calc import MACECalculatorWrapper


def test_mace_rejects_incompatible_e3nn_before_loading_model(tmp_path, monkeypatch):
    model = tmp_path / "model.model"
    model.touch()
    monkeypatch.setattr(
        "mliport.doctor._installed_dependency_conflicts",
        lambda names: ["mace-torch requires e3nn==0.4.4, but 0.6.0 is installed"],
    )

    wrapper = MACECalculatorWrapper(model)

    with pytest.raises(RuntimeError, match=r"\.venv-mace/bin/mliport tui"):
        wrapper.get_calculator()
