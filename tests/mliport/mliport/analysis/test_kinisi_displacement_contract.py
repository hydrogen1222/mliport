"""PR-E acceptance tests: kinisi displacement contract.

Three input classes must stay distinct (task book section 7):

* ``exact_unwrapped`` -- exact stored coordinates; a step larger than half a
  cell is physical and must be consumed directly by the exact adapter;
* ``reconstructable_wrapped`` -- MIC reconstruction is safe;
* ``insufficient_trajectory_information`` -- the wrapped interval cannot be
  uniquely unwrapped and transport must refuse.

Production transport must never silently fall back to kinisi's unverified
``from_ase`` reconstruction; diagnostic fallback marks
``backend_displacement_verified=false`` / ``publication_grade=false``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mliport.analysis import transport as transport_module
from mliport.analysis.dataset import TrajectoryDataset
from mliport.analysis.transport import (
    _build_kinisi_analyzer,
    _exact_displacement_parser,
    _production_positions_with_drift,
    displacement_input_class,
    kinisi_transport,
)
from mliport.analysis.validation import (
    InsufficientTrajectoryInformationError,
    UnsupportedAnalysisError,
)

_FRAME_INTERVAL_FS = 1000.0


def _dataset(positions, cells, *, convention: str):
    n_frames = len(positions)
    return TrajectoryDataset(
        run_dir=Path("."),
        source_path=Path(".") / "trajectory.traj",
        positions=np.asarray(positions, dtype=float),
        cells=np.asarray(cells, dtype=float),
        pbc=np.ones(3, dtype=bool),
        symbols=("Li",),
        masses=np.asarray([7.0]),
        times_fs=np.arange(n_frames, dtype=float) * _FRAME_INTERVAL_FS,
        steps=None,
        positions_convention=convention,
        frame_interval_fs=_FRAME_INTERVAL_FS,
    )


def _wrapped(continuous, cell):
    """Wrap an exact continuous path into the given cell."""
    inverse = np.linalg.inv(cell)
    fractional = np.asarray(continuous, dtype=float) @ inverse
    fractional -= np.floor(fractional)
    return fractional @ cell


def _rotated_cell(angle_deg: float, size: float = 10.0) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rotation @ (np.eye(3) * size)


def _continuous_path(values, axis: int = 0):
    positions = np.zeros((len(values), 1, 3), dtype=float)
    positions[:, 0, axis] = values
    positions[:, 0, 1] += 5.0
    positions[:, 0, 2] += 5.0
    return positions


def _view(dataset):
    return dataset.analysis_view(include_equilibration=False)


def _continuous_from_dataset(dataset):
    view = _view(dataset)
    continuous, _ = _production_positions_with_drift(
        view,
        mobile=view.select("Li"),
        drift_reference="none",
        drift_indices=None,
    )
    return view, continuous


def test_orthorhombic_wrapped_crossing_reconstructs_exactly():
    values = np.asarray([8.8, 9.2, 9.6, 10.0, 10.4])
    continuous = _continuous_path(values)
    cell = np.diag([10.0, 10.0, 10.0])
    wrapped = np.stack([_wrapped(frame, cell) for frame in continuous])
    dataset = _dataset(
        wrapped, np.broadcast_to(cell, (len(wrapped), 3, 3)), convention="wrapped"
    )
    view, recovered = _continuous_from_dataset(dataset)
    assert (
        displacement_input_class(view, {"unwrap_safety_ratio": 0.1})
        == "reconstructable_wrapped"
    )
    np.testing.assert_allclose(recovered, continuous, atol=1e-9)
    # the wrapped representation really did cross a boundary
    assert np.any(np.abs(np.diff(wrapped[:, 0, 0])) > 5.0)


def test_skew_cell_exact_input_is_not_misfolded():
    """R05: a (0, 1.8, 0) A step in a skew cell must not become (4, 1.8, 0) A."""

    pytest.importorskip("kinisi")
    from kinisi.ase import ASEParser

    cell = np.asarray([[4.0, 0.0, 0.0], [12.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    values = np.asarray([1.8, 3.6, 5.4, 7.2])
    continuous = _continuous_path(values, axis=1)
    dataset = _dataset(
        continuous,
        np.broadcast_to(cell, (len(continuous), 3, 3)),
        convention="unwrapped",
    )
    view, recovered = _continuous_from_dataset(dataset)
    assert (
        displacement_input_class(view, {"unwrap_safety_ratio": 1.0})
        == "exact_unwrapped"
    )
    np.testing.assert_allclose(recovered, continuous, atol=1e-12)
    sc = transport_module._require_kinisi()[0]
    frames = [Atoms("Li", positions=frame, cell=cell, pbc=True) for frame in recovered]
    parser = _exact_displacement_parser(
        sc=sc,
        ASEParser=ASEParser,
        frames=frames,
        continuous_positions_A=recovered,
        local_mobile=np.asarray([0]),
        time_step_fs=_FRAME_INTERVAL_FS,
        step_skip=1,
        dt_values_fs=None,
        dimensions="xyz",
    )
    used = np.asarray(parser.displacements.to(unit="angstrom").values, dtype=float)
    np.testing.assert_allclose(used, recovered - recovered[0], atol=1e-9)
    np.testing.assert_allclose(np.diff(used[:, 0, 1]), 1.8, atol=1e-9)
    assert not np.allclose(np.diff(used[:, 0, 0]), 4.0, atol=1e-6)


def test_skew_cell_wrapped_input_is_flagged_insufficient():
    """Without exact image history the skew geometry is not recoverable."""

    cell = np.asarray([[4.0, 0.0, 0.0], [12.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    values = np.asarray([1.8, 3.6, 5.4, 7.2])
    continuous = _continuous_path(values, axis=1)
    wrapped = np.stack([_wrapped(frame, cell) for frame in continuous])
    dataset = _dataset(
        wrapped, np.broadcast_to(cell, (len(wrapped), 3, 3)), convention="wrapped"
    )
    view = _view(dataset)
    _, diagnostics = transport_module.unwrap_positions(view)
    assert displacement_input_class(view, diagnostics) == (
        "insufficient_trajectory_information"
    )


def test_rotated_cell_wrapped_crossing_reconstructs_exactly():
    cell = _rotated_cell(30.0)
    values = np.asarray([8.6, 9.0, 9.4, 9.8, 10.2])
    continuous = _continuous_path(values)
    wrapped = np.stack([_wrapped(frame, cell) for frame in continuous])
    dataset = _dataset(
        wrapped, np.broadcast_to(cell, (len(wrapped), 3, 3)), convention="wrapped"
    )
    view, recovered = _continuous_from_dataset(dataset)
    assert (
        displacement_input_class(view, {"unwrap_safety_ratio": 0.1})
        == "reconstructable_wrapped"
    )
    np.testing.assert_allclose(recovered, continuous, atol=1e-9)


def test_exact_unwrapped_single_step_beyond_half_cell_is_consumed():
    pytest.importorskip("kinisi")
    values = np.asarray([1.0, 7.0, 13.0, 19.0])
    continuous = _continuous_path(values)
    cell = np.diag([10.0, 10.0, 10.0])
    dataset = _dataset(
        continuous,
        np.broadcast_to(cell, (len(continuous), 3, 3)),
        convention="unwrapped",
    )
    view = _view(dataset)
    semantics = transport_module._validate_kinisi_periodic_reconstruction(
        view, {"unwrap_safety_ratio": 1.0}
    )
    assert semantics["displacement_input_class"] == "exact_unwrapped"
    assert semantics["exact_unwrapped_intervals_beyond_mic"] == 3
    _, recovered = _continuous_from_dataset(dataset)
    np.testing.assert_allclose(recovered, continuous, atol=1e-12)


def test_exact_unwrapped_multiple_cell_crossing_is_consumed():
    pytest.importorskip("kinisi")
    values = np.asarray([0.0, 25.0, 50.0, 75.0])  # 2.5 cells per saved step
    continuous = _continuous_path(values)
    cell = np.diag([10.0, 10.0, 10.0])
    dataset = _dataset(
        continuous,
        np.broadcast_to(cell, (len(continuous), 3, 3)),
        convention="unwrapped",
    )
    view, recovered = _continuous_from_dataset(dataset)
    semantics = transport_module._validate_kinisi_periodic_reconstruction(
        view, {"unwrap_safety_ratio": 1.0}
    )
    assert semantics["exact_unwrapped_preserved_directly"] is True
    assert semantics["exact_unwrapped_intervals_beyond_mic"] == 3
    np.testing.assert_allclose(recovered, continuous, atol=1e-12)
    from kinisi.ase import ASEParser as RealASEParser

    parser = _exact_displacement_parser(
        sc=transport_module._require_kinisi()[0],
        ASEParser=RealASEParser,
        frames=[
            Atoms("Li", positions=[frame[0]], cell=cell, pbc=True)
            for frame in continuous
        ],
        continuous_positions_A=recovered[:, [0], :],
        local_mobile=np.asarray([0]),
        time_step_fs=_FRAME_INTERVAL_FS,
        step_skip=1,
        dt_values_fs=None,
        dimensions="xyz",
    )
    used = np.asarray(parser.displacements.to(unit="angstrom").values, dtype=float)
    np.testing.assert_allclose(used, continuous - continuous[0], atol=1e-9)


def test_wrapped_insufficient_information_refuses_with_status():
    cell = np.diag([10.0, 10.0, 10.0])
    positions = np.zeros((4, 1, 3), dtype=float)
    positions[:, 0, 0] = np.asarray([0.0, 4.9, 0.2, 5.1])  # >half cell ambiguity
    dataset = _dataset(
        positions, np.broadcast_to(cell, (len(positions), 3, 3)), convention="wrapped"
    )
    view = _view(dataset)
    _, diagnostics = transport_module.unwrap_positions(view)
    assert displacement_input_class(view, diagnostics) == (
        "insufficient_trajectory_information"
    )
    with pytest.raises(InsufficientTrajectoryInformationError) as excinfo:
        _production_positions_with_drift(
            view,
            mobile=view.select("Li"),
            drift_reference="none",
            drift_indices=None,
        )
    assert excinfo.value.status == "insufficient_trajectory_information"


def test_adapter_unavailable_fails_closed_in_production():
    calls: list[dict] = []

    class FakeAnalyzer:
        @classmethod
        def from_ase(cls, **kwargs):
            calls.append(kwargs)
            return cls()

    kwargs = {
        "sc": None,
        "analyzer_cls": FakeAnalyzer,
        "frames": [],
        "continuous_positions_A": np.zeros((1, 1, 3)),
        "local_mobile": np.asarray([0]),
        "time_step_fs": 1.0,
        "step_skip": 1,
        "dt_values_fs": None,
        "dimensions": "xyz",
        "from_ase_kwargs": {},
        "allow_exact_adapter": True,
    }
    with pytest.raises(UnsupportedAnalysisError, match="Exact-displacement kinisi"):
        _build_kinisi_analyzer(**kwargs, production=True)
    assert calls == [], "production must not call from_ase"
    # the explicit diagnostic mode may fall back, but never claims exactness
    _analyzer, audit = _build_kinisi_analyzer(**kwargs, production=False)
    assert len(calls) == 1
    assert audit["exact_displacement_adapter"] is False
    assert audit["backend_displacement_verified"] is False
    assert audit["publication_grade"] is False


def test_kinisi_minor_api_drift_is_detected_before_fit(monkeypatch):
    pytest.importorskip("kinisi")
    from kinisi.ase import ASEParser

    sc = transport_module._require_kinisi()[0]
    frames = [
        Atoms("Li", positions=[[1.0 + index, 1.0, 1.0]], cell=[10, 10, 10], pbc=True)
        for index in range(3)
    ]
    continuous = np.asarray([[[1.0, 1.0, 1.0]], [[2.0, 1.0, 1.0]], [[3.0, 1.0, 1.0]]])

    original_init = ASEParser.__init__

    def drifting_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # simulate a kinisi minor version that stops honouring the injected
        # displacement array: the backend now exposes zeros.
        self.displacements = self.displacements * 0.0

    monkeypatch.setattr(ASEParser, "__init__", drifting_init)
    with pytest.raises(UnsupportedAnalysisError, match="not consuming the exact"):
        _exact_displacement_parser(
            sc=sc,
            ASEParser=ASEParser,
            frames=frames,
            continuous_positions_A=continuous,
            local_mobile=np.asarray([0]),
            time_step_fs=1000.0,
            step_skip=1,
            dt_values_fs=None,
            dimensions="xyz",
        )
    # sanity: the untouched API is callable and would have passed
    monkeypatch.setattr(ASEParser, "__init__", original_init)
    parser = _exact_displacement_parser(
        sc=sc,
        ASEParser=ASEParser,
        frames=frames,
        continuous_positions_A=continuous,
        local_mobile=np.asarray([0]),
        time_step_fs=1000.0,
        step_skip=1,
        dt_values_fs=None,
        dimensions="xyz",
    )
    assert parser._mliport_exact_displacement_audit["backend_displacement_verified"]


def test_backend_consumed_displacements_match_stored_step_by_step():
    pytest.importorskip("kinisi")
    from kinisi.ase import ASEParser

    sc = transport_module._require_kinisi()[0]
    continuous = np.asarray(
        [
            [[1.0, 1.0, 1.0]],
            [[3.5, 1.0, 1.0]],
            [[4.5, 1.0, 1.0]],
            [[7.5, 1.0, 1.0]],
        ]
    )
    frames = [
        Atoms("Li", positions=frame, cell=[10, 10, 10], pbc=True)
        for frame in continuous
    ]
    parser = _exact_displacement_parser(
        sc=sc,
        ASEParser=ASEParser,
        frames=frames,
        continuous_positions_A=continuous,
        local_mobile=np.asarray([0]),
        time_step_fs=1000.0,
        step_skip=1,
        dt_values_fs=None,
        dimensions="xyz",
    )
    used = np.asarray(parser.displacements.to(unit="angstrom").values, dtype=float)
    expected = continuous - continuous[0]
    np.testing.assert_allclose(used, expected, atol=1e-12)
    assert (
        parser._mliport_exact_displacement_audit[
            "backend_displacement_max_abs_difference_A"
        ]
        <= 1.0e-12
    )


def test_diagnostic_fallback_marks_result_not_publication_grade(monkeypatch):
    """A fake analyzer must be isolated from the publication-grade path."""
    sc = pytest.importorskip("scipp")

    class FakeDiffusionAnalyzer:
        @classmethod
        def from_ase(cls, **kwargs):
            analyzer = cls()
            analyzer.dt = kwargs.get("dt")
            if analyzer.dt is None:
                analyzer.dt = sc.array(
                    dims=["time interval"],
                    values=np.asarray([1.0, 2.0, 3.0]),
                    unit="ps",
                )
            return analyzer

        def diffusion(self, *_args, **_kwargs):
            n_points = self.dt.shape[0]
            self.msd = sc.array(
                dims=["time interval"],
                values=np.arange(1, n_points + 1, dtype=float),
                variances=np.ones(n_points),
                unit="angstrom^2",
            )
            self.D = sc.array(dims=["sample"], values=np.full(8, 1.0e-10), unit="m^2/s")

    monkeypatch.setattr(
        transport_module,
        "_require_kinisi",
        lambda: (sc, FakeDiffusionAnalyzer, FakeDiffusionAnalyzer, "2.1.0"),
    )
    dataset = _dataset(
        _continuous_path(np.asarray([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])),
        np.broadcast_to(np.diag([10.0, 10.0, 10.0]), (6, 3, 3)),
        convention="unwrapped",
    )
    result = kinisi_transport(
        dataset,
        mobile_species="Li",
        ionic_charge_e=1.0,
        fit_start_ps=0.0,
        lag_step_ps=1.0,
        lag_stop_ps=3.0,
        temperature_K=600.0,
        allow_reconstructed_fallback=True,
    )
    assert result["publication_grade"] is False
    assert result["kinisi_position_semantics"]["backend_displacement_verified"] is False
