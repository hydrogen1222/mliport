"""PR-F acceptance tests: one explicit drift definition per backend.

For a multi-element framework the arithmetic-mean and mass-weighted
center-of-mass corrections are *not* equivalent.  Both backends must record
``drift_mode`` and ``drift_reference``, and a cross-backend comparison may
only be claimed when the two definitions match (task book section 8).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mliport.analysis.dataset import TrajectoryDataset
from mliport.analysis.drift import (
    drift_definitions_match,
    drift_displacement,
    drift_semantics,
    reference_indices,
)
from mliport.analysis.msd import calculate_msd
from mliport.analysis.transport import _production_positions_with_drift

_MASSES = np.asarray([22.99, 35.45, 39.10])  # Na, Cl, K(mobile)
_SYMBOLS = ("Na", "Cl", "K")


def _dataset(*, na_dx=0.1, cl_dx=0.3, translation=(0.0, 0.0, 0.0)):
    """Na(0) + Cl(1) framework drift plus a mobile K(2)."""
    n_frames = 4
    positions = np.zeros((n_frames, 3, 3), dtype=float)
    base = np.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    for frame in range(n_frames):
        positions[frame] = base
        positions[frame, 0, 0] += na_dx * frame
        positions[frame, 1, 0] += cl_dx * frame
        positions[frame] += np.asarray(translation, dtype=float) * frame
    cells = np.broadcast_to(np.diag([10.0, 10.0, 10.0]), (n_frames, 3, 3)).copy()
    return TrajectoryDataset(
        run_dir=Path("."),
        source_path=Path(".") / "trajectory.traj",
        positions=positions,
        cells=cells,
        pbc=np.ones(3, dtype=bool),
        symbols=_SYMBOLS,
        masses=_MASSES,
        times_fs=np.arange(n_frames, dtype=float) * 1000.0,
        steps=None,
        positions_convention="unwrapped",
        frame_interval_fs=1000.0,
    )


def _continuous(dataset):
    return dataset.positions


def test_uniform_translation_agrees_for_both_modes():
    dataset = _dataset(na_dx=0.0, cl_dx=0.0, translation=(0.2, -0.1, 0.0))
    reference = reference_indices(
        dataset,
        mobile=dataset.select("K"),
        drift_reference="indices",
        drift_indices=[0, 1],
    )
    arithmetic = drift_displacement(
        _continuous(dataset),
        reference=reference,
        masses=dataset.masses,
        drift_mode="arithmetic_mean",
    )
    mass_weighted = drift_displacement(
        _continuous(dataset),
        reference=reference,
        masses=dataset.masses,
        drift_mode="mass_weighted_com",
    )
    np.testing.assert_allclose(arithmetic, mass_weighted, atol=1e-12)
    np.testing.assert_allclose(arithmetic[1], [0.2, -0.1, 0.0], atol=1e-12)


def test_differential_framework_displacement_has_known_difference():
    dataset = _dataset(na_dx=0.1, cl_dx=0.3)
    reference = np.asarray([0, 1])
    arithmetic = drift_displacement(
        _continuous(dataset),
        reference=reference,
        masses=dataset.masses,
        drift_mode="arithmetic_mean",
    )[1]
    mass_weighted = drift_displacement(
        _continuous(dataset),
        reference=reference,
        masses=dataset.masses,
        drift_mode="mass_weighted_com",
    )[1]
    assert arithmetic[0] == pytest.approx((0.1 + 0.3) / 2)
    expected_com = (22.99 * 0.1 + 35.45 * 0.3) / (22.99 + 35.45)
    assert mass_weighted[0] == pytest.approx(expected_com)
    assert abs(float(mass_weighted[0] - arithmetic[0])) > 0.01
    # same physics, but only if the caller compares matching definitions
    assert not drift_definitions_match(
        {"drift_mode": "arithmetic_mean", "drift_reference": "nonmobile"},
        {"drift_mode": "mass_weighted_com", "drift_reference": "nonmobile"},
    )


def test_drift_definitions_match_requires_both_fields():
    arithmetic = {"drift_mode": "arithmetic_mean", "drift_reference": "nonmobile"}
    assert drift_definitions_match(arithmetic, dict(arithmetic))
    assert not drift_definitions_match(
        arithmetic, {"drift_mode": "mass_weighted_com", "drift_reference": "nonmobile"}
    )
    assert not drift_definitions_match(
        arithmetic, {"drift_mode": "arithmetic_mean", "drift_reference": "all"}
    )
    assert not drift_definitions_match(arithmetic, {})
    assert not drift_definitions_match(None, arithmetic)


def test_native_msd_records_the_mass_weighted_definition():
    dataset = _dataset()
    native = calculate_msd(
        dataset,
        mobile_species="K",
        drift_reference="nonmobile",
        drift_mode="mass_weighted_com",
    )
    semantics = native["drift_semantics"]
    assert semantics["drift_mode"] == "mass_weighted_com"
    assert semantics["drift_reference"] == "nonmobile"
    assert "mass-weighted" in semantics["center_definition"]


def test_transport_drift_mode_changes_corrected_positions():
    dataset = _dataset()
    mobile = dataset.select("K")
    arithmetic, _ = _production_positions_with_drift(
        dataset,
        mobile=mobile,
        drift_reference="nonmobile",
        drift_indices=None,
        drift_mode="arithmetic_mean",
    )
    mass_weighted, _ = _production_positions_with_drift(
        dataset,
        mobile=mobile,
        drift_reference="nonmobile",
        drift_indices=None,
        drift_mode="mass_weighted_com",
    )
    difference = arithmetic[:, mobile] - mass_weighted[:, mobile]
    expected = float((22.99 * 0.1 + 35.45 * 0.3) / (22.99 + 35.45) - (0.1 + 0.3) / 2)
    np.testing.assert_allclose(difference[:, 0, 0], expected * np.arange(4), atol=1e-12)
    np.testing.assert_allclose(difference[:, 0, 1:], 0.0, atol=1e-12)


def test_reference_all_is_explicit_and_changes_the_definition():
    dataset = _dataset()
    mobile = dataset.select("K")
    nonmobile = reference_indices(dataset, mobile=mobile, drift_reference="nonmobile")
    all_atoms = reference_indices(dataset, mobile=mobile, drift_reference="all")
    assert nonmobile.tolist() == [0, 1]
    assert all_atoms.tolist() == [0, 1, 2]
    semantics_nonmobile = drift_semantics(
        drift_mode="arithmetic_mean",
        drift_reference="nonmobile",
        reference=nonmobile,
        symbols=dataset.symbols,
    )
    semantics_all = drift_semantics(
        drift_mode="arithmetic_mean",
        drift_reference="all",
        reference=all_atoms,
        symbols=dataset.symbols,
    )
    assert not drift_definitions_match(semantics_nonmobile, semantics_all)


def test_cross_backend_comparability_flag_is_not_backend_agreement():
    import sys

    repo = Path(__file__).resolve().parents[4]
    scripts = repo / "validation" / "science" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import analysis_suite

    native = {
        "drift_semantics": {
            "drift_mode": "mass_weighted_com",
            "drift_reference": "none",
        }
    }
    kinisi = {
        "drift_semantics": {"drift_mode": "arithmetic_mean", "drift_reference": "none"}
    }
    mismatch = analysis_suite._drift_comparability(native, kinisi)
    assert mismatch["definitions_match"] is False
    assert "definitions_match is true" in mismatch["note"]

    matched = analysis_suite._drift_comparability(
        native,
        {
            "drift_semantics": {
                "drift_mode": "mass_weighted_com",
                "drift_reference": "none",
            }
        },
    )
    assert matched["definitions_match"] is True
