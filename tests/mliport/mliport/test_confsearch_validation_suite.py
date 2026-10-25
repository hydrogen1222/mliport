"""CI-checkable helpers of the owner-run confsearch validation campaign.

The campaign itself needs a real backend and a GPU (spec section 13.3), so CI
only checks that the system builder produces a legal confsearch input and that
the report helpers are honest about what the campaign claims.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "validation" / "science" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import confsearch_suite as suite  # noqa: E402


def test_build_slab_is_a_legal_confsearch_slab():
    from ase.constraints import FixAtoms

    slab = suite.build_slab()
    assert len(slab) > 8
    cell = np.asarray(slab.cell, dtype=float)
    # Surface normal is +z and the cell is a valid 3D periodic cell.
    assert abs(cell[0, 2]) < 1e-8
    assert abs(cell[1, 2]) < 1e-8
    assert cell[2, 2] > 0
    assert np.linalg.det(cell) > 0
    # One contiguous slab with its vacuum across the c boundary.
    z = np.sort(np.asarray(slab.get_positions(), dtype=float)[:, 2])
    periodic_gap = z[0] + cell[2, 2] - z[-1]
    assert periodic_gap > np.max(np.diff(z))
    # The slab is fixed, which enables the geometric identity rule G.
    assert len(slab.constraints) == 1
    assert isinstance(slab.constraints[0], FixAtoms)
    assert len(slab.constraints[0].get_indices()) == len(slab)


def test_build_molecule_is_cartesian_and_non_periodic():
    molecule = suite.build_molecule()
    assert molecule.get_chemical_formula() == "CH4O"
    assert not np.asarray(molecule.pbc, dtype=bool).any()
    assert np.isfinite(np.asarray(molecule.get_positions(), dtype=float)).all()


def test_monotone_unique_counts_helper():
    rows = [
        {"unique_valid": "1"},
        {"unique_valid": "1"},
        {"unique_valid": "3"},
    ]
    assert suite._monotone_unique_counts(rows) is True
    rows.append({"unique_valid": "2"})
    assert suite._monotone_unique_counts(rows) is False


def test_markdown_report_never_claims_dft_accuracy():
    report = {
        "system": "test",
        "engine": "mace",
        "device": "cpu",
        "n_starts": 1,
        "cycles": 1,
        "baseline": {
            "status": "completed",
            "best_energy_eV": -1.0,
            "n_rankable": 1,
            "n_quenches": 1,
        },
        "search": {
            "status": "completed",
            "best_energy_eV": -1.2,
            "n_rankable": 2,
            "n_quenches": 3,
        },
        "comparison": {
            "search_improved_on_baseline": True,
            "discovery_rows": 3,
            "discovery_unique_counts_monotone": True,
        },
    }
    text = suite._markdown(report)
    assert "not an accuracy" in text
    assert "re-ranked with DFT" in text


def test_campaign_cli_contract():
    with pytest.raises(SystemExit) as help_exit:
        suite.main(["--help"])
    assert help_exit.value.code == 0
    # --engine, --model and --output are mandatory.
    with pytest.raises(SystemExit) as missing_exit:
        suite.main([])
    assert missing_exit.value.code == 2
    with pytest.raises(SystemExit) as bad_engine:
        suite.main(["--engine", "nope", "--model", "m.pt", "--output", "out"])
    assert bad_engine.value.code == 2
