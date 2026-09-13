"""PR-G acceptance tests: collective_system_particles > 1 is experimental.

Formal publication-grade collective conductivity is restricted to
``collective_system_particles == 1``.  Larger index-ordered groupings are
allowed only as explicitly flagged experimental diagnostics: they are *not*
independent MD replicas and cross-group correlations are not guaranteed to
be preserved (task book section 9).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mliport.analysis import transport as transport_module
from mliport.analysis.dataset import TrajectoryDataset
from mliport.analysis.transport import kinisi_transport

_WARNING = "cross-group correlations are not guaranteed to be preserved"


def _dataset():
    n_frames = 8
    positions = np.zeros((n_frames, 2, 3), dtype=float)
    positions[:, 0, 0] = np.arange(n_frames, dtype=float) * 0.1
    positions[:, 1, 0] = 5.0
    cells = np.broadcast_to(np.diag([20.0, 20.0, 20.0]), (n_frames, 3, 3)).copy()
    return TrajectoryDataset(
        run_dir=Path("."),
        source_path=Path(".") / "trajectory.traj",
        positions=positions,
        cells=cells,
        pbc=np.ones(3, dtype=bool),
        symbols=("Li", "S"),
        masses=np.asarray([7.0, 32.0]),
        times_fs=np.arange(n_frames, dtype=float) * 1000.0,
        steps=None,
        positions_convention="unwrapped",
        frame_interval_fs=1000.0,
    )


def _install_fakes(monkeypatch):
    sc = pytest.importorskip("scipp")

    class FakeDiffusionAnalyzer:
        @classmethod
        def from_ase(cls, **kwargs):
            analyzer = cls()
            analyzer.dt = kwargs.get("dt")
            if analyzer.dt is None:
                analyzer.dt = sc.array(
                    dims=["time interval"],
                    values=np.arange(1, 5, dtype=float),
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

    class FakeConductivityAnalyzer:
        @classmethod
        def from_ase(cls, **kwargs):
            analyzer = cls()
            analyzer.dt = kwargs.get("dt")
            if analyzer.dt is None:
                analyzer.dt = sc.array(
                    dims=["time interval"],
                    values=np.arange(1, 5, dtype=float),
                    unit="ps",
                )
            return analyzer

        def conductivity(self, *_args, **_kwargs):
            self.sigma = sc.array(
                dims=["sample"], values=np.full(8, 1.0e-3), unit="mS/cm"
            )
            self.mscd = sc.array(
                dims=["time interval"],
                values=np.arange(1, 5, dtype=float),
                variances=np.ones(4),
                unit="C^2*m^2",
            )

    class FakeJumpAnalyzer:
        @classmethod
        def from_ase(cls, **kwargs):
            analyzer = cls()
            analyzer.dt = kwargs.get("dt")
            if analyzer.dt is None:
                analyzer.dt = sc.array(
                    dims=["time interval"],
                    values=np.arange(1, 5, dtype=float),
                    unit="ps",
                )
            return analyzer

        def jump_diffusion(self, *_args, **_kwargs):
            self.D_J = sc.array(
                dims=["sample"], values=np.full(8, 1.0e-10), unit="m^2/s"
            )
            self.mstd = sc.array(
                dims=["time interval"],
                values=np.arange(1, 5, dtype=float),
                variances=np.ones(4),
                unit="angstrom^2",
            )

    monkeypatch.setattr(
        transport_module,
        "_require_kinisi",
        lambda: (
            sc,
            FakeDiffusionAnalyzer,
            FakeConductivityAnalyzer,
            FakeJumpAnalyzer,
            "2.1.0",
        ),
    )
    return sc


def _run(monkeypatch, *, system_particles, jump=False):
    _install_fakes(monkeypatch)
    return kinisi_transport(
        _dataset(),
        mobile_species="Li",
        ionic_charge_e=1.0,
        fit_start_ps=0.0,
        lag_step_ps=1.0,
        lag_stop_ps=4.0,
        temperature_K=600.0,
        collective_conductivity=True,
        collective_system_particles=system_particles,
        jump_diffusion=jump,
        allow_reconstructed_fallback=True,
    )


def test_single_group_is_not_experimental(monkeypatch):
    result = _run(monkeypatch, system_particles=1)
    collective = result["collective_conductivity"]
    assert collective["system_particles"] == 1
    assert collective["experimental"] is False
    assert collective["publication_grade"] is True
    assert collective["warning"] is None
    assert "replicas" in collective["system_particles_semantics"]


def test_grouping_greater_than_one_is_experimental(monkeypatch):
    result = _run(monkeypatch, system_particles=4)
    collective = result["collective_conductivity"]
    assert collective["system_particles"] == 4
    assert collective["experimental"] is True
    assert collective["publication_grade"] is False
    assert collective["warning"] == _WARNING
    # grouping is a statistical device, never an independent MD replica claim
    assert "not independent MD replicas" in collective["system_particles_semantics"]
    assert result["publication_grade"] is False
    assert result["experimental_components"] == ["collective_conductivity"]
    assert any(_WARNING in warning for warning in result["warnings"])


def test_jump_grouping_is_flagged_too(monkeypatch):
    result = _run(monkeypatch, system_particles=3, jump=True)
    jump = result["jump_diffusion"]
    assert jump["experimental"] is True
    assert jump["publication_grade"] is False
    assert jump["warning"] == _WARNING
    assert result["publication_grade"] is False
    assert set(result["experimental_components"]) == {
        "collective_conductivity",
        "jump_diffusion",
    }


def test_cli_help_does_not_call_groups_replicates():
    source = (
        Path(__file__).resolve().parents[4] / "mliport" / "mliport" / "cli.py"
    ).read_text(encoding="utf-8")
    assert "not independent MD replicas" in source
    assert "index-ordered statistical groups" in source
