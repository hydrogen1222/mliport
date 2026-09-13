"""PR-C acceptance tests: GEMDAT failures must not become zero jumps.

The product adapter must distinguish a legitimately empty transition
collection (``no_events``, zero jumps is physical) from a backend failure
(``backend_error``, metrics unknown) and from invalid site geometry
(``invalid_input``).  A backend exception must never be rendered as a zero
jump matrix (task book section 5).
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mliport.analysis import electrolyte
from mliport.analysis.dataset import TrajectoryDataset
from mliport.analysis.validation import OptionalDependencyError


def _dataset(n_frames: int = 12, *, mobile_dx: float = 0.0, mobile_x=None):
    """Small production dataset; mobile Li either sits still or hops."""
    positions = np.zeros((n_frames, 2, 3))
    if mobile_x is not None:
        positions[:, 0, 0] = np.asarray(mobile_x, dtype=float)
    else:
        positions[:, 0, 0] = 1.0 + mobile_dx * np.arange(n_frames)
    positions[:, 0, 1] = 1.0
    positions[:, 0, 2] = 1.0
    positions[:, 1, :] = [8.0, 8.0, 8.0]
    frames = [
        Atoms("LiS", positions=frame, cell=[10.0, 10.0, 10.0], pbc=True)
        for frame in positions
    ]
    phases = ["production"] * n_frames
    return TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(n_frames) * 100.0,
        positions_convention="unwrapped",
        phases=phases,
    )


class _FakeVolume:
    def __init__(self, sites, discovery_failure=None):
        self._sites = sites
        self._discovery_failure = discovery_failure
        self.data = np.zeros((2, 2, 2))

    def probability(self):
        return np.zeros((2, 2, 2))

    def find_peaks(self):
        return np.empty((0, 3), dtype=int)

    def to_structure(self, **_kwargs):
        if self._discovery_failure is not None:
            raise self._discovery_failure
        return self._sites

    def get_free_energy(self, _temperature):
        volume = self

        class _FreeEnergy:
            data = volume.data

        return _FreeEnergy()


class _FakeTrajectory:
    def __init__(self, sites, transition, discovery_failure=None):
        self._sites = sites
        self._transition = transition
        self._discovery_failure = discovery_failure
        self.calls = 0

    def filter(self, _species):
        return self

    def to_volume(self, resolution=None):  # noqa: ARG002 - interface parity
        return _FakeVolume(self._sites, self._discovery_failure)

    def get_structure(self, _index):
        return self._sites

    def transitions_between_sites(self, sites, species, site_radius=None):
        self.calls += 1
        return self._transition(sites, species, site_radius)


def _install_fakes(
    monkeypatch,
    *,
    transition,
    sites_cell=10.0,
    sites_coords=((1.0, 1.0, 1.0),),
    discovery_failure=None,
):
    from pymatgen.core import Structure

    sites = Structure(
        lattice=[[sites_cell, 0, 0], [0, sites_cell, 0], [0, 0, sites_cell]],
        species=["Li"] * len(sites_coords),
        coords=[list(coord) for coord in sites_coords],
    )
    trajectory = _FakeTrajectory(sites, transition, discovery_failure)
    monkeypatch.setattr(
        electrolyte,
        "_require_gemdat",
        lambda: (object, object, Structure, "1.7.3"),
    )
    monkeypatch.setattr(electrolyte, "_gemdat_trajectory", lambda *a, **k: trajectory)
    return trajectory, sites


def _run(dataset):
    return electrolyte.gemdat_electrolyte(
        dataset,
        mobile_species="Li",
        discover_sites_from_density=True,
        temperature_K=600.0,
    )


def _raise(kind, message):
    def _transition(_sites, _species, _radius):
        raise kind(message)

    return _transition


def test_gemdat_empty_event_collection_is_no_events(monkeypatch):
    _install_fakes(
        monkeypatch,
        transition=_raise(ValueError, "need at least one array to stack"),
    )
    result = _run(_dataset(mobile_dx=0.0))
    assert result.summary["gemdat_transition_status"] == "no_events"
    assert result.summary["number_of_jumps"] == 0
    assert result.summary["transition_events"] == 0
    assert result.summary["gemdat_transition_error"]["type"] == "ValueError"
    np.testing.assert_array_equal(
        result.arrays["jump_matrix"], np.zeros((1, 1), dtype=int)
    )


def test_gemdat_backend_error_is_not_zero_jumps(monkeypatch):
    _install_fakes(
        monkeypatch,
        transition=_raise(ValueError, "shape mismatch in transition stack"),
    )
    result = _run(_dataset())
    assert result.summary["gemdat_transition_status"] == "backend_error"
    assert result.summary["number_of_jumps"] is None
    assert result.summary["transition_events"] is None
    assert result.summary["gemdat_transition_error"]["message"]
    # no physical zero matrices may be published for a backend failure
    assert "jump_matrix" not in result.arrays
    assert "transition_matrix" not in result.arrays
    assert "jumps" not in result.tables
    assert any("backend_error" in warning for warning in result.warnings)


def test_gemdat_empty_collection_with_real_site_change_is_backend_error(monkeypatch):
    """A 'lucky' empty stack while atoms change site is not a physical zero."""
    # Li hops exactly one site distance between frames: nearest-site indices
    # change even though GEMDAT claimed an empty collection.
    dataset = _dataset(n_frames=6, mobile_x=np.linspace(1.0, 5.0, 6))
    _install_fakes(
        monkeypatch,
        transition=_raise(ValueError, "need at least one array to stack"),
        sites_coords=((1.0, 1.0, 1.0), (5.0, 1.0, 1.0)),
    )
    result = _run(dataset)
    assert result.summary["gemdat_transition_status"] == "backend_error"
    assert result.summary["number_of_jumps"] is None
    assert "jump_matrix" not in result.arrays


def test_gemdat_api_drift_is_backend_error(monkeypatch):
    _install_fakes(
        monkeypatch,
        transition=_raise(
            AttributeError, "'Transitions' object has no attribute 'matrix'"
        ),
    )
    result = _run(_dataset())
    assert result.summary["gemdat_transition_status"] == "backend_error"
    assert result.summary["number_of_jumps"] is None


def test_gemdat_invalid_site_geometry_is_invalid_input(monkeypatch):
    # site lattice 20 A does not match the trajectory cell 10 A
    _install_fakes(
        monkeypatch,
        transition=lambda *a, **k: pytest.fail("must not reach GEMDAT"),
        sites_cell=20.0,
    )
    result = _run(_dataset())
    assert result.summary["gemdat_transition_status"] == "invalid_input"
    assert result.summary["number_of_jumps"] is None
    assert (
        "invalid site geometry" in result.summary["gemdat_transition_error"]["message"]
    )


def test_gemdat_missing_dependency_is_unsupported(monkeypatch):
    def _missing():
        raise OptionalDependencyError("GEMDAT not installed")

    monkeypatch.setattr(electrolyte, "_require_gemdat", _missing)
    with pytest.raises(OptionalDependencyError):
        _run(_dataset())


def test_t7_gemdat_diagnostic_refuses_backend_error(monkeypatch):
    import importlib
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    scripts = repo / "validation" / "science" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    analysis_suite = importlib.import_module("analysis_suite")

    class _Result:
        summary = {  # noqa: RUF012 - plain data holder
            "gemdat_transition_status": "backend_error",
            "gemdat_transition_error": {
                "type": "ValueError",
                "message": "shape mismatch",
            },
        }
        tables = {}  # noqa: RUF012
        warnings = []  # noqa: RUF012
        paths = {}  # noqa: RUF012

    monkeypatch.setattr(electrolyte, "gemdat_electrolyte", lambda *a, **k: _Result())
    from types import SimpleNamespace

    with pytest.raises(RuntimeError, match="backend_error"):
        analysis_suite._gemdat_diagnostic(
            _dataset(), SimpleNamespace(sites="sites.cif"), 600.0
        )
    # and the suite maps that RuntimeError to "fail", not to a zero-jump pass
    status, reason = analysis_suite._classify_gemdat_failure(
        RuntimeError("GEMDAT transition detection did not produce a physical result")
    )
    assert status == "fail"
    assert reason == {}


def test_gemdat_site_discovery_failure_is_backend_error(monkeypatch):
    """A failed site segmentation must not become a physical no-jump result."""
    _install_fakes(
        monkeypatch,
        transition=lambda *a, **k: pytest.fail("must not reach transitions"),
        discovery_failure=RuntimeError("segmentation exploded"),
    )
    result = _run(_dataset())
    assert result.summary["gemdat_transition_status"] == "backend_error"
    assert result.summary["number_of_jumps"] is None
    assert result.summary["transition_events"] is None
    assert result.summary["gemdat_transition_error"]["type"] == "RuntimeError"
    assert "jump_matrix" not in result.arrays
