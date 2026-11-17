"""`MDRunner` public constructor compatibility (Issue #102 / E8).

The seven NPT-only arguments (``pressure_gpa``, ``npt_tdamp``, ``npt_pdamp``,
``npt_tchain``, ``npt_pchain``, ``npt_tloop``, ``npt_ploop``) were added in the
middle of ``MDRunner.__init__`` - between ``thermostat`` and ``friction`` - so
an external caller that passed the historical arguments positionally had them
silently re-bound to the new names: ``friction`` became ``pressure_gpa``,
``bussi_tau`` became ``npt_tdamp``, and so on. Nothing raised, and the run used
a different thermostat configuration than the caller asked for.

These tests pin the *pre-NPT* positional order (the order documented before the
NPT round) and require the NPT arguments to be keyword-only, so the mistake
cannot come back. They do not test NPT physics; NPT behaviour is covered by the
NPT suites.
"""

from __future__ import annotations

import inspect

import pytest
from ase import units

from mliport.runners.md import MDRunner

# The historical positional prefix, in its original order: ensemble,
# temperature, timestep, steps, equilibration_steps, thermostat, friction,
# bussi_tau, nhc_tdamp, nhc_tchain, nhc_tloop, save_interval, output_dir.
HISTORICAL_POSITIONAL = (
    "NVT",
    300.0,
    1.0,
    500,
    10,
    "NHC",
    0.5,
    800.0,
    120.0,
    4,
    2,
    20,
)

NPT_KEYWORD_ONLY = (
    "pressure_gpa",
    "npt_tdamp",
    "npt_pdamp",
    "npt_tchain",
    "npt_pchain",
    "npt_tloop",
    "npt_ploop",
)

BOUND_ATTRIBUTES = (
    "ensemble",
    "temperature",
    "timestep",
    "steps",
    "equilibration_steps",
    "thermostat",
    "friction",
    "bussi_tau",
    "nhc_tdamp",
    "nhc_tchain",
    "nhc_tloop",
    "save_interval",
    "pressure_gpa",
    "npt_tdamp",
    "npt_pdamp",
    "npt_tchain",
    "npt_pchain",
    "npt_tloop",
    "npt_ploop",
)


def _calc() -> object:
    """A calculator stand-in: the constructor never touches it."""
    return object()


def _historical_positional_runner(tmp_path) -> MDRunner:
    return MDRunner(
        _calc(),
        *HISTORICAL_POSITIONAL,
        tmp_path,
        pre_relax=False,
        verbose=False,
    )


def _equivalent_keyword_runner(tmp_path) -> MDRunner:
    return MDRunner(
        _calc(),
        ensemble="NVT",
        temperature=300.0,
        timestep=1.0,
        steps=500,
        equilibration_steps=10,
        thermostat="NHC",
        friction=0.5,
        bussi_tau=800.0,
        nhc_tdamp=120.0,
        nhc_tchain=4,
        nhc_tloop=2,
        save_interval=20,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
    )


def test_historical_positional_call_binds_the_documented_parameters(tmp_path) -> None:
    """E8: the 8th positional argument is still ``friction``, not pressure."""
    runner = _historical_positional_runner(tmp_path)

    assert runner.thermostat == "nhc"
    assert runner.friction == pytest.approx(0.5 / units.fs)
    assert runner.bussi_tau == pytest.approx(800.0 * units.fs)
    assert runner.nhc_tdamp == pytest.approx(120.0 * units.fs)
    assert runner.nhc_tchain == 4
    assert runner.nhc_tloop == 2
    assert runner.save_interval == 20
    assert runner.output_dir == tmp_path
    # The NPT block keeps its defaults when it was never asked for.
    assert runner.pressure_gpa == pytest.approx(0.0)
    assert (runner.npt_tchain, runner.npt_pchain) == (3, 3)
    assert (runner.npt_tloop, runner.npt_ploop) == (1, 1)


def test_positional_and_keyword_forms_describe_the_same_run(tmp_path) -> None:
    """Metamorphic check: both spellings must produce identical settings."""
    positional = _historical_positional_runner(tmp_path / "a")
    keyword = _equivalent_keyword_runner(tmp_path / "b")

    for name in BOUND_ATTRIBUTES:
        assert getattr(positional, name) == getattr(
            keyword, name
        ), f"positional and keyword forms disagree on {name!r}"


def test_npt_arguments_are_keyword_only() -> None:
    """NPT keys must not be reachable positionally, so they cannot shift again."""
    parameters = inspect.signature(MDRunner.__init__).parameters
    for name in NPT_KEYWORD_ONLY:
        assert (
            parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        ), f"{name} must be keyword-only (Issue #102)"
    # ...and the historical parameters keep their original positions.
    ordered = [
        name
        for name, parameter in parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert ordered[:14] == [
        "self",
        "calculator",
        "ensemble",
        "temperature",
        "timestep",
        "steps",
        "equilibration_steps",
        "thermostat",
        "friction",
        "bussi_tau",
        "nhc_tdamp",
        "nhc_tchain",
        "nhc_tloop",
        "save_interval",
    ]


def test_npt_keyword_construction_still_validates(tmp_path) -> None:
    """The NPT path keeps working by keyword, including its fail-closed checks."""
    runner = MDRunner(
        _calc(),
        ensemble="NPT",
        temperature=300.0,
        thermostat="LANGEVIN",
        pressure_gpa=1.5,
        npt_tdamp=100.0,
        npt_pdamp=1000.0,
        npt_tchain=3,
        npt_pchain=3,
        npt_tloop=1,
        npt_ploop=1,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
    )
    assert runner.ensemble == "npt"
    assert runner.pressure_gpa == pytest.approx(1.5)
    assert runner.friction == pytest.approx(0.001 / units.fs)  # untouched default

    with pytest.raises(ValueError, match="NPT requires lossless"):
        MDRunner(
            _calc(),
            ensemble="NPT",
            pressure_gpa=1.0,
            output_dir=tmp_path,
            pre_relax=False,
            write_trajectory=False,
        )
