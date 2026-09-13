from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.io.trajectory import Trajectory

import mliport.analysis.runner as runner_module
import mliport.analysis.transport as transport_module
from mliport.analysis import TrajectoryDataset
from mliport.analysis.runner import run_analysis, source_fingerprint
from mliport.analysis.schema import AnalysisRequest
from mliport.analysis.transport import (
    DEFAULT_MAX_NATIVE_KINISI_LAG_POINTS,
    _exact_displacement_parser,
    _kinisi_frames_and_indices,
    _kinisi_parser_peak_bytes,
    _resolve_kinisi_lag_grid,
    _validate_kinisi_periodic_reconstruction,
    kinisi_transport,
)
from mliport.analysis.validation import UnsupportedAnalysisError
from mliport.cli import main


def _write_short_run(path) -> None:
    raw = path / "raw"
    raw.mkdir(parents=True)
    positions = [9.4, 9.7, 0.0, 0.3, 0.6, 0.9]
    with Trajectory(raw / "trajectory.traj", "w") as writer:
        for index, x in enumerate(positions):
            atoms = Atoms(
                "LiS",
                positions=[[x, 1, 1], [5, 5, 5]],
                cell=[10, 10, 10],
                pbc=True,
            )
            atoms.info["mliport_step"] = index
            atoms.info["mliport_time_fs"] = float(index * 2)
            atoms.info["mliport_phase"] = "equilibration" if index < 2 else "production"
            writer.write(atoms)
    with (raw / "md.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "time_fs",
                "phase",
                "temperature_K",
                "potential_energy_eV",
                "kinetic_energy_eV",
                "total_energy_eV",
                "volume_A3",
            ]
        )
        for index in range(6):
            writer.writerow(
                [
                    index,
                    index * 2,
                    "equilibration" if index < 2 else "production",
                    600,
                    -2,
                    0.1,
                    -1.9,
                    1000,
                ]
            )
    (path / "artifacts.json").write_text(
        json.dumps(
            {
                "schema": "mliport.md-artifacts/2",
                "status": "completed",
                "trajectory": {
                    "md_timestep_fs": 1.0,
                    "frame_stride_steps": 2,
                    "frame_interval_fs": 2.0,
                    "positions_convention": "wrapped",
                    "production_start_step": 2,
                },
            }
        )
    )
    (path / "resolved_config.json").write_text(
        json.dumps(
            {
                "run_options": {
                    "ensemble": "NVE",
                    "temperature": 600,
                }
            }
        )
    )


def test_analysis_runner_writes_provenance_results_and_reuses_id(tmp_path) -> None:
    run = tmp_path / "short-run"
    _write_short_run(run)
    first = run_analysis(AnalysisRequest("validate", str(run)))
    assert first["status"] == "success"
    assert first["results"]["production_frames"] == 4
    output = run / "analysis" / "validate" / first["analysis_id"]
    assert (output / "request.json").is_file()
    assert (output / "provenance.json").is_file()
    assert (output / "results.json").is_file()
    assert (run / "analysis" / "index.json").is_file()

    second = run_analysis(AnalysisRequest("validate", str(run)))
    assert second["analysis_id"] == first["analysis_id"]
    assert second["reused"] is True


def test_cli_analyze_validate_short_run(tmp_path, capsys) -> None:
    run = tmp_path / "short-run"
    _write_short_run(run)
    assert main(["analyze", str(run), "validate"]) == 0
    output = capsys.readouterr().out
    assert "Analysis success" in output
    assert "Production frames: 4" in output
    assert "MSD/transport eligible: True" in output


def test_analysis_runner_msd_defaults_to_production(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    run = tmp_path / "short-run"
    _write_short_run(run)
    outcome = run_analysis(
        AnalysisRequest(
            "msd",
            str(run),
            parameters={
                "mobile_species": "Li",
                "axes": "x,y,z,xy,xyz",
                "drift_reference": "none",
            },
        )
    )
    assert outcome["status"] == "success"
    assert len(outcome["results"]["lag_time_ps"]) == 4
    output = run / "analysis" / "msd" / outcome["analysis_id"]
    request = json.loads((output / "request.json").read_text(encoding="utf-8"))
    assert request["task_output_revision"] == 7
    assert (output / "msd.csv").is_file()
    assert (output / "msd.png").is_file()
    assert (output / "msd.svg").is_file()
    assert (output / "alpha.png").is_file()
    assert (output / "alpha.svg").is_file()
    assert not (output / "diffusion_fits.csv").exists()
    with (output / "msd.csv").open(newline="", encoding="utf-8") as handle:
        columns = next(csv.reader(handle))
    for axes in ("x", "y", "z", "xy", "xyz"):
        assert f"alpha_{axes}" in columns
    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert {"alpha.png", "alpha.svg"} <= set(payload["artifacts"])
    assert "diffusion_fits.csv" not in payload["artifacts"]
    assert payload["results"]["fit_window_ps"] is None
    assert payload["results"]["diagnostic_linear_diffusion_fits"] == {}


def test_transport_analysis_id_includes_lag_parameters_and_revision(
    tmp_path, monkeypatch
) -> None:
    run = tmp_path / "short-run"
    _write_short_run(run)

    def fake_dispatch(_request, _output_dir):
        return (
            {
                "tracer_diffusion": {
                    "fit_start_ps": 40.0,
                    "fit_stop_ps": 200.0,
                    "lag_grid": {
                        "mode": "custom",
                        "requested_step_ps": 1.0,
                        "requested_stop_ps": 200.0,
                    },
                },
                "kinisi_position_semantics": {
                    "source_positions_convention": "unwrapped",
                    "backend_reconstruction": "kinisi periodic displacement reconstruction",
                },
            },
            [],
        )

    monkeypatch.setattr(runner_module, "_dispatch", fake_dispatch)
    first = run_analysis(
        AnalysisRequest(
            "transport",
            str(run),
            parameters={"lag_step_ps": 1.0, "lag_stop_ps": 200.0},
        )
    )
    second = run_analysis(
        AnalysisRequest(
            "transport",
            str(run),
            parameters={"lag_step_ps": 2.0, "lag_stop_ps": 200.0},
        )
    )
    assert first["analysis_id"] != second["analysis_id"]
    first_output = run / "analysis" / "transport" / first["analysis_id"]
    request = json.loads((first_output / "request.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (first_output / "provenance.json").read_text(encoding="utf-8")
    )
    assert request["task_output_revision"] == 6
    assert provenance["parameters"]["lag_step_ps"] == 1.0
    assert provenance["transport"]["lag_grid"]["requested_step_ps"] == 1.0
    reused = run_analysis(
        AnalysisRequest(
            "transport",
            str(run),
            parameters={"lag_step_ps": 1.0, "lag_stop_ps": 200.0},
        )
    )
    assert reused["analysis_id"] == first["analysis_id"]
    assert reused["reused"] is True


@pytest.mark.parametrize(
    ("lag_step_ps", "expected_points"),
    [(1.0, 200), (0.5, 400), (2.0, 100)],
)
def test_resolve_custom_kinisi_lag_grid(lag_step_ps, expected_points) -> None:
    result = _resolve_kinisi_lag_grid(
        frame_interval_fs=10.0,
        total_duration_ps=400.0,
        fit_start_ps=40.0,
        lag_step_ps=lag_step_ps,
        lag_stop_ps=200.0,
    )
    assert result["mode"] == "custom"
    assert result["n_lag_points"] == expected_points
    assert result["lag_frame_indices"][0] == int(lag_step_ps * 100)
    assert result["lag_frame_indices"][-1] == 20000
    assert 4000 in result["lag_frame_indices"]
    assert result["actual_stop_ps"] == pytest.approx(200.0)


def test_resolve_custom_grid_inserts_fit_start() -> None:
    result = _resolve_kinisi_lag_grid(
        frame_interval_fs=10.0,
        total_duration_ps=400.0,
        fit_start_ps=40.0,
        lag_step_ps=3.0,
        lag_stop_ps=198.0,
    )
    assert 4000 in result["lag_frame_indices"]
    assert result["lag_frame_indices"][-1] == 19800
    assert result["nominal_step_ps"] == pytest.approx(3.0)
    assert result["actual_step_ps"] is None
    assert result["fit_start_inserted"] is True
    assert result["is_uniform_grid"] is False


@pytest.mark.parametrize(
    ("lag_step_ps", "lag_stop_ps", "message"),
    [
        (1.0, None, "must be provided together"),
        (None, 200.0, "must be provided together"),
        (0.0, 200.0, "lag_step_ps must be finite and positive"),
        (-1.0, 200.0, "lag_step_ps must be finite and positive"),
        (1.0, 40.0, "lag_stop_ps must be greater than fit_start_ps"),
        (1.0, 401.0, "exceeds production duration"),
        (0.015, 200.0, "incompatible with the trajectory frame interval"),
        (1.0, 200.005, "incompatible with the trajectory frame interval"),
    ],
)
def test_resolve_custom_grid_rejects_invalid_parameters(
    lag_step_ps, lag_stop_ps, message
) -> None:
    with pytest.raises(ValueError, match=message):
        _resolve_kinisi_lag_grid(
            frame_interval_fs=10.0,
            total_duration_ps=400.0,
            fit_start_ps=40.0,
            lag_step_ps=lag_step_ps,
            lag_stop_ps=lag_stop_ps,
        )


def test_resolve_native_grid_guard() -> None:
    with pytest.raises(ValueError, match=r"default lag grid.*40000"):
        _resolve_kinisi_lag_grid(
            frame_interval_fs=10.0,
            total_duration_ps=400.0,
            fit_start_ps=40.0,
        )

    result = _resolve_kinisi_lag_grid(
        frame_interval_fs=10.0,
        total_duration_ps=10.0,
        fit_start_ps=1.0,
        native_lag_guard=DEFAULT_MAX_NATIVE_KINISI_LAG_POINTS,
    )
    assert result["mode"] == "kinisi_default"
    assert result["lag_times_fs"] is None
    assert result["estimated_n_lag_points"] == 1000


def _dense_dataset(tmp_path) -> TrajectoryDataset:
    n_frames = 40001
    cells = np.broadcast_to(np.diag([10.0, 10.0, 10.0]), (n_frames, 3, 3)).copy()
    positions = np.zeros((n_frames, 2, 3), dtype=float)
    positions[:, 0, 0] = np.arange(n_frames, dtype=float) * 0.001
    return TrajectoryDataset(
        run_dir=tmp_path,
        source_path=tmp_path / "trajectory.traj",
        positions=positions,
        cells=cells,
        pbc=np.ones(3, dtype=bool),
        symbols=("Li", "S"),
        masses=np.asarray([7.0, 32.0]),
        times_fs=np.arange(n_frames, dtype=float) * 10.0,
        steps=None,
        positions_convention="unwrapped",
        frame_interval_fs=10.0,
    )


def test_dense_native_guard_runs_before_kinisi(tmp_path, monkeypatch) -> None:
    dataset = _dense_dataset(tmp_path)
    monkeypatch.setattr(
        transport_module,
        "_require_kinisi",
        lambda: pytest.fail("kinisi must not be imported for a guarded grid"),
    )
    with pytest.raises(ValueError, match=r"default lag grid.*40000"):
        kinisi_transport(
            dataset,
            mobile_species="Li",
            ionic_charge_e=1,
            fit_start_ps=40.0,
            temperature_K=700.0,
        )


def test_kinisi_parser_memory_estimate_and_guard(monkeypatch) -> None:
    assert _kinisi_parser_peak_bytes(nframes=100, natoms=10, triclinic=False) == 128000
    assert _kinisi_parser_peak_bytes(nframes=100, natoms=10, triclinic=True) == 512000
    dataset = _synthetic_transport_dataset()
    monkeypatch.setattr(
        transport_module,
        "_require_kinisi",
        lambda: pytest.fail("kinisi must not be imported after a memory guard"),
    )
    with pytest.raises(UnsupportedAnalysisError, match="parser peak memory"):
        kinisi_transport(
            dataset,
            mobile_species="Li",
            ionic_charge_e=1,
            fit_start_ps=0.07,
            lag_step_ps=0.02,
            lag_stop_ps=0.2,
            temperature_K=600.0,
            parser_memory_limit_gib=1.0e-9,
        )


def _synthetic_transport_dataset() -> TrajectoryDataset:
    n_frames = 140
    positions = np.zeros((n_frames, 2, 3), dtype=float)
    positions[:, 0, 0] = np.arange(n_frames, dtype=float) * 0.01
    positions[:, 0, 1] = np.arange(n_frames, dtype=float) * 0.005
    frames = [
        Atoms("LiS", positions=frame, cell=[100, 100, 100], pbc=True)
        for frame in positions
    ]
    return TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(n_frames, dtype=float) * 2.0,
        positions_convention="unwrapped",
        md_timestep_fs=0.5,
        frame_stride_steps=4,
    )


def test_transport_tracer_and_collective_share_custom_dt(monkeypatch) -> None:
    sc = pytest.importorskip("scipp")
    dataset = _synthetic_transport_dataset()

    class FakeDiffusionAnalyzer:
        calls: ClassVar[list[dict[str, object]]] = []

        @classmethod
        def from_ase(cls, **kwargs):
            cls.calls.append(kwargs)
            analyzer = cls()
            analyzer.dt = kwargs.get("dt")
            if analyzer.dt is None:
                analyzer.dt = sc.array(
                    dims=["time interval"],
                    values=np.arange(1, 140, dtype=float) * 2.0,
                    unit="fs",
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
            self.D = sc.array(
                dims=["sample"],
                values=np.full(16, 1.0e-10),
                unit="m^2/s",
            )

    class FakeConductivityAnalyzer:
        calls: ClassVar[list[dict[str, object]]] = []

        @classmethod
        def from_ase(cls, **kwargs):
            cls.calls.append(kwargs)
            analyzer = cls()
            analyzer.dt = kwargs.get("dt")
            return analyzer

        def conductivity(self, *_args, **_kwargs):
            self.sigma = sc.array(
                dims=["sample"],
                values=np.full(16, 1.0e-3),
                unit="mS/cm",
            )

    monkeypatch.setattr(
        transport_module,
        "_require_kinisi",
        lambda: (
            sc,
            FakeDiffusionAnalyzer,
            FakeConductivityAnalyzer,
            "2.1.0",
        ),
    )
    result = kinisi_transport(
        dataset,
        mobile_species="Li",
        ionic_charge_e=1,
        fit_start_ps=0.07,
        lag_step_ps=0.02,
        lag_stop_ps=0.2,
        temperature_K=600.0,
        collective_conductivity=True,
        n_samples=10,
        n_walkers=16,
        n_burn=10,
        n_thin=1,
        allow_reconstructed_fallback=True,
    )
    assert result["publication_grade"] is False
    assert result["kinisi_position_semantics"]["backend_displacement_verified"] is False
    tracer_dt = FakeDiffusionAnalyzer.calls[0]["dt"]
    collective_dt = FakeConductivityAnalyzer.calls[0]["dt"]
    assert tracer_dt is collective_dt
    np.testing.assert_array_equal(
        tracer_dt.values, np.asarray([20, 40, 60, 70, 80, 100, 120, 140, 160, 180, 200])
    )
    assert result["tracer_diffusion"]["fit_stop_ps"] == pytest.approx(0.2)
    assert result["tracer_diffusion"]["lag_grid"]["n_lag_points_total"] == 11
    assert result["tracer_diffusion"]["lag_grid"]["n_lag_points_in_fit"] == 8
    assert (
        result["kinisi_resource_diagnostics"][
            "single_float64_square_matrix_lower_bound_bytes"
        ]
        == 8 * 8**2
    )


def test_kinisi_adapter_matches_official_ase_api() -> None:
    sc = pytest.importorskip("scipp")
    kinisi = pytest.importorskip("kinisi.analyze")
    rng = np.random.default_rng(4)
    n_frames = 140
    n_particles = 10
    walk = np.concatenate(
        (
            np.zeros((1, n_particles, 3)),
            np.cumsum(
                rng.normal(scale=0.1, size=(n_frames - 1, n_particles, 3)),
                axis=0,
            ),
        ),
        axis=0,
    )
    frames = [
        Atoms(
            f"Li{n_particles}",
            positions=positions,
            cell=[100, 100, 100],
            pbc=True,
        )
        for positions in walk
    ]
    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(n_frames) * 2.0,
        positions_convention="unwrapped",
        md_timestep_fs=0.5,
        frame_stride_steps=4,
    )
    adapter = kinisi_transport(
        dataset,
        mobile_species="Li",
        ionic_charge_e=1,
        fit_start_ps=0.07,
        temperature_K=600,
        random_seed=5,
        n_samples=10,
        n_walkers=16,
        n_burn=10,
        n_thin=1,
    )
    official = kinisi.DiffusionAnalyzer.from_ase(
        trajectory=frames,
        specie="Li",
        time_step=sc.scalar(0.5, unit="fs"),
        step_skip=sc.scalar(4, unit="dimensionless"),
        dimension="xyz",
        progress=False,
    )
    official.diffusion(
        sc.scalar(70, unit="fs"),
        n_samples=10,
        n_walkers=16,
        n_burn=10,
        n_thin=1,
        progress=False,
        random_state=np.random.RandomState(5),
    )
    np.testing.assert_allclose(adapter["lag_time_ps"], official.dt.to(unit="ps").values)
    np.testing.assert_allclose(
        adapter["kinisi_msd_A2"], official.msd.to(unit="angstrom^2").values
    )
    assert adapter["kinisi_time_mapping"]["resulting_frame_interval_fs"] == 2.0
    assert adapter["tracer_diffusion"]["D_posterior_m2_s"]["mean"] == pytest.approx(
        float(np.mean(official.D.to(unit="m^2/s").values))
    )


def test_mobile_only_precorrected_parser_matches_kinisi_framework_drift() -> None:
    """The lower-memory adapter must preserve kinisi's drift-corrected MSD."""

    sc = pytest.importorskip("scipp")
    kinisi = pytest.importorskip("kinisi.analyze")
    rng = np.random.default_rng(23)
    n_frames = 70
    cell = np.asarray([[12.0, 0.0, 0.0], [2.0, 11.0, 0.0], [1.0, 0.8, 10.0]])
    initial_fractional = np.asarray(
        [
            [0.20, 0.20, 0.20],
            [0.35, 0.30, 0.25],
            [0.45, 0.55, 0.40],
            [0.65, 0.45, 0.60],
            [0.25, 0.75, 0.70],
            [0.75, 0.70, 0.30],
        ]
    )
    framework_drift = np.arange(n_frames)[:, None, None] * np.asarray(
        [0.003, -0.002, 0.001]
    )
    li_walk = np.concatenate(
        (
            np.zeros((1, 4, 3)),
            np.cumsum(rng.normal(scale=0.003, size=(n_frames - 1, 4, 3)), axis=0),
        ),
        axis=0,
    )
    fractional = np.broadcast_to(initial_fractional, (n_frames, 6, 3)).copy()
    fractional += framework_drift
    fractional[:, :4] += li_walk
    frames = [
        Atoms("Li4S2", scaled_positions=frame, cell=cell, pbc=True)
        for frame in fractional
    ]
    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(n_frames, dtype=float),
        positions_convention="unwrapped",
    )
    mobile = dataset.select("Li")
    kinisi_frames = _kinisi_frames_and_indices(
        dataset,
        mobile=mobile,
        drift_reference="nonmobile",
        drift_indices=None,
    )
    compact_frames = kinisi_frames.frames
    local_mobile = kinisi_frames.local_mobile
    reference = kinisi_frames.reference
    common = {
        "time_step": sc.scalar(1.0, unit="fs"),
        "step_skip": sc.scalar(1, unit="dimensionless"),
        "dimension": "xyz",
        "progress": False,
    }
    official = kinisi.DiffusionAnalyzer.from_ase(
        trajectory=frames,
        specie="Li",
        **common,
    )
    compact = kinisi.DiffusionAnalyzer.from_ase(
        trajectory=compact_frames,
        specie=None,
        specie_indices=sc.array(
            dims=["particle"], values=local_mobile, unit="dimensionless"
        ),
        **common,
    )

    np.testing.assert_array_equal(reference, [4, 5])
    np.testing.assert_allclose(compact.dt.values, official.dt.values)
    np.testing.assert_allclose(
        compact.msd.to(unit="angstrom^2").values,
        official.msd.to(unit="angstrom^2").values,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_kinisi_adapter_custom_dt_smoke() -> None:
    pytest.importorskip("scipp")
    pytest.importorskip("kinisi.analyze")
    rng = np.random.default_rng(7)
    n_frames = 80
    n_particles = 4
    walk = np.concatenate(
        (
            np.zeros((1, n_particles, 3)),
            np.cumsum(
                rng.normal(scale=0.1, size=(n_frames - 1, n_particles, 3)),
                axis=0,
            ),
        ),
        axis=0,
    )
    frames = [
        Atoms(
            f"Li{n_particles}",
            positions=positions,
            cell=[100, 100, 100],
            pbc=True,
        )
        for positions in walk
    ]
    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(n_frames, dtype=float) * 2.0,
        positions_convention="unwrapped",
        md_timestep_fs=0.5,
        frame_stride_steps=4,
    )
    adapter = kinisi_transport(
        dataset,
        mobile_species="Li",
        ionic_charge_e=1,
        fit_start_ps=0.03,
        lag_step_ps=0.02,
        lag_stop_ps=0.12,
        temperature_K=600,
        random_seed=7,
        n_samples=10,
        n_walkers=16,
        n_burn=10,
        n_thin=1,
    )
    np.testing.assert_allclose(
        adapter["lag_time_ps"], np.asarray([0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.12])
    )
    assert len(adapter["kinisi_msd_A2"]) == 7
    assert adapter["tracer_diffusion"]["fit_stop_ps"] == pytest.approx(0.12)
    assert adapter["tracer_diffusion"]["D_posterior_m2_s"]["posterior_samples"] > 0


def _unwrapped_dataset(
    positions: np.ndarray, *, cell: float = 10.0
) -> TrajectoryDataset:
    n_frames = positions.shape[0]
    cells = np.broadcast_to(np.diag([cell, cell, cell]), (n_frames, 3, 3)).copy()
    return TrajectoryDataset(
        run_dir=Path("."),
        source_path=Path(".") / "trajectory.traj",
        positions=positions,
        cells=cells,
        pbc=np.ones(3, dtype=bool),
        symbols=("Li",),
        masses=np.asarray([7.0]),
        times_fs=np.arange(n_frames, dtype=float) * 10.0,
        steps=None,
        positions_convention="unwrapped",
        frame_interval_fs=10.0,
    )


def test_kinisi_position_semantics_safe_unwrapped_is_equivalent() -> None:
    """4.1: a dense, slow unwrapped trajectory reconstructs exactly."""

    positions = np.zeros((6, 1, 3), dtype=float)
    positions[:, 0, 0] = np.arange(6, dtype=float) * 0.1
    dataset = _unwrapped_dataset(positions, cell=10.0)
    semantics = _validate_kinisi_periodic_reconstruction(
        dataset, {"unwrap_safety_level": "not_applicable_exact_unwrapped_source"}
    )
    assert semantics["displacement_input_class"] == "exact_unwrapped"
    assert semantics["source_positions_convention"] == "unwrapped"
    assert semantics["exact_unwrapped_preserved_directly"] is True
    assert semantics["exact_unwrapped_reconstruction_equivalent"] is True
    assert semantics["exact_unwrapped_intervals_beyond_mic"] == 0
    assert semantics["checked_saved_intervals"] == 5
    assert semantics["maximum_exact_vs_mic_difference_A"] < 1.0e-6


def test_kinisi_position_semantics_beyond_mic_stays_exact() -> None:
    """PR-E 7.3: an exact unwrapped step > half a cell is physical, not an error."""

    # frame 0 x=1 A, frame 1 x=7 A in a 10 A cell: exact +6 A, MIC -4 A.
    positions = np.zeros((6, 1, 3), dtype=float)
    positions[0, 0, 0] = 1.0
    positions[1:, 0, 0] = 1.0 + np.arange(1, 6) * 6.0
    dataset = _unwrapped_dataset(positions, cell=10.0)
    semantics = _validate_kinisi_periodic_reconstruction(
        dataset, {"unwrap_safety_level": "not_applicable_exact_unwrapped_source"}
    )
    assert semantics["displacement_input_class"] == "exact_unwrapped"
    assert semantics["exact_unwrapped_preserved_directly"] is True
    assert semantics["exact_unwrapped_intervals_beyond_mic"] == 5
    assert semantics["exact_unwrapped_reconstruction_equivalent"] is False
    assert semantics["maximum_exact_vs_mic_difference_A"] > 4.0


def test_kinisi_transport_consumes_exact_crossing_without_mic_gate() -> None:
    """PR-E 7.3: the exact adapter consumes a >half-cell step directly."""

    pytest.importorskip("kinisi")
    positions = np.zeros((6, 1, 3), dtype=float)
    positions[0, 0, 0] = 1.0
    positions[1:, 0, 0] = 1.0 + np.arange(1, 6) * 6.0
    dataset = _unwrapped_dataset(positions, cell=10.0)
    result = kinisi_transport(
        dataset,
        mobile_species="Li",
        ionic_charge_e=1,
        fit_start_ps=0.0,
        lag_step_ps=0.01,
        lag_stop_ps=0.04,
        temperature_K=600.0,
    )
    assert result["displacement_input_class"] == "exact_unwrapped"
    semantics = result["kinisi_position_semantics"]
    assert semantics["publication_grade"] is True
    assert semantics["backend_displacement_verified"] is True
    assert semantics["exact_unwrapped_preserved_directly"] is True
    assert semantics["exact_unwrapped_intervals_beyond_mic"] == 5
    assert semantics["backend_displacement_max_abs_difference_A"] <= 1.0e-6


def test_kinisi_position_semantics_wrapped_records_heuristic_safety() -> None:
    """4.3: wrapped sources keep the heuristic safety level and null equivalence."""

    positions = np.zeros((6, 1, 3), dtype=float)
    positions[:, 0, 0] = np.arange(6, dtype=float) * 0.1
    n_frames = positions.shape[0]
    cells = np.broadcast_to(np.diag([10.0, 10.0, 10.0]), (n_frames, 3, 3)).copy()
    dataset = TrajectoryDataset(
        run_dir=Path("."),
        source_path=Path(".") / "trajectory.traj",
        positions=positions,
        cells=cells,
        pbc=np.ones(3, dtype=bool),
        symbols=("Li",),
        masses=np.asarray([7.0]),
        times_fs=np.arange(n_frames, dtype=float) * 10.0,
        steps=None,
        positions_convention="wrapped",
        frame_interval_fs=10.0,
    )
    semantics = _validate_kinisi_periodic_reconstruction(
        dataset, {"unwrap_safety_level": "comfortably_safe"}
    )
    assert semantics["source_positions_convention"] == "wrapped"
    assert semantics["exact_unwrapped_reconstruction_equivalent"] is None
    assert semantics["wrapped_source_safety"] == "comfortably_safe"


def test_kinisi_transport_safe_unwrapped_records_backend_semantics(monkeypatch) -> None:
    """4.4: a safe unwrapped canonical trajectory records provenance semantics."""

    sc = pytest.importorskip("scipp")
    dataset = _synthetic_transport_dataset()

    class FakeDiffusionAnalyzer:
        def __class_getitem__(cls, item):
            return cls

        @classmethod
        def from_ase(cls, **kwargs):
            analyzer = cls()
            analyzer.dt = kwargs.get("dt")
            if analyzer.dt is None:
                analyzer.dt = sc.array(
                    dims=["time interval"],
                    values=np.arange(1, 140, dtype=float) * 2.0,
                    unit="fs",
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
            self.D = sc.array(
                dims=["sample"], values=np.full(16, 1.0e-10), unit="m^2/s"
            )

    monkeypatch.setattr(
        transport_module,
        "_require_kinisi",
        lambda: (sc, FakeDiffusionAnalyzer, FakeDiffusionAnalyzer, "2.1.0"),
    )
    result = kinisi_transport(
        dataset,
        mobile_species="Li",
        ionic_charge_e=1,
        fit_start_ps=0.07,
        lag_step_ps=0.02,
        lag_stop_ps=0.2,
        temperature_K=600.0,
        allow_reconstructed_fallback=True,
    )
    semantics = result["kinisi_position_semantics"]
    assert semantics["displacement_input_class"] == "exact_unwrapped"
    assert semantics["source_positions_convention"] == "unwrapped"
    assert semantics["exact_unwrapped_reconstruction_equivalent"] is True
    assert semantics["exact_unwrapped_preserved_directly"] is True
    assert (
        semantics["backend_reconstruction"]
        == "mliport exact continuous displacement array"
    )
    # a non-kinisi analyzer cannot claim the verified exact path
    assert semantics["exact_displacement_adapter"] is False
    assert semantics["backend_displacement_verified"] is False
    assert result["publication_grade"] is False


def _write_transport_run(path) -> None:
    raw = path / "raw"
    raw.mkdir(parents=True)
    n_frames = 60
    with Trajectory(raw / "trajectory.traj", "w") as writer:
        for index in range(n_frames):
            x = index * 0.05  # slow unwrapped Li drift, well below half a cell
            atoms = Atoms(
                "LiS",
                positions=[[x, 1, 1], [5, 5, 5]],
                cell=[20, 20, 20],
                pbc=True,
            )
            atoms.info["mliport_step"] = index
            atoms.info["mliport_time_fs"] = float(index * 10)
            atoms.info["mliport_phase"] = "production"
            writer.write(atoms)
    with (raw / "md.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "time_fs",
                "phase",
                "temperature_K",
                "potential_energy_eV",
                "kinetic_energy_eV",
                "total_energy_eV",
                "volume_A3",
            ]
        )
        for index in range(n_frames):
            writer.writerow([index, index * 10, "production", 600, -2, 0.1, -1.9, 8000])
    (path / "artifacts.json").write_text(
        json.dumps(
            {
                "schema": "mliport.md-artifacts/2",
                "status": "completed",
                "trajectory": {
                    "md_timestep_fs": 1.0,
                    "frame_stride_steps": 10,
                    "frame_interval_fs": 10.0,
                    "positions_convention": "unwrapped",
                    "production_start_step": 0,
                },
            }
        )
    )
    (path / "resolved_config.json").write_text(
        json.dumps({"run_options": {"ensemble": "NVE", "temperature": 600}})
    )


def _fake_kinisi_for_artifacts(monkeypatch) -> None:
    sc = pytest.importorskip("scipp")

    class FakeDiffusionAnalyzer:
        @classmethod
        def from_ase(cls, **kwargs):
            analyzer = cls()
            analyzer.dt = kwargs.get("dt")
            if analyzer.dt is None:
                analyzer.dt = sc.array(
                    dims=["time interval"],
                    values=np.arange(1, 60, dtype=float) * 10.0,
                    unit="fs",
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
            self.D = sc.array(
                dims=["sample"], values=np.full(16, 1.0e-10), unit="m^2/s"
            )

    monkeypatch.setattr(
        transport_module,
        "_require_kinisi",
        lambda: (sc, FakeDiffusionAnalyzer, FakeDiffusionAnalyzer, "2.1.0"),
    )


def test_transport_runner_writes_summary_csv_plot_and_arrays(
    tmp_path, monkeypatch
) -> None:
    """22: transport success writes the full human-readable artifact set."""

    pytest.importorskip("scipp")
    pytest.importorskip("matplotlib")
    run = tmp_path / "transport-run"
    _write_transport_run(run)
    _fake_kinisi_for_artifacts(monkeypatch)
    outcome = run_analysis(
        AnalysisRequest(
            "transport",
            str(run),
            parameters={
                "mobile_species": "Li",
                "ionic_charge_e": 1.0,
                "fit_start_ps": 0.1,
                "lag_step_ps": 0.1,
                "lag_stop_ps": 0.5,
                "temperature_K": 600.0,
                "allow_reconstructed_fallback": True,
            },
        )
    )
    assert outcome["status"] == "success"
    output = run / "analysis" / "transport" / outcome["analysis_id"]
    for name in (
        "kinisi_arrays.npz",
        "transport_summary.csv",
        "transport_msd.png",
        "transport_msd.svg",
        "results.json",
        "provenance.json",
        "request.json",
    ):
        assert (output / name).is_file(), name
    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert {
        "kinisi_arrays.npz",
        "transport_summary.csv",
        "transport_msd.png",
        "transport_msd.svg",
    } <= set(payload["artifacts"])
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert (
        provenance["transport"]["kinisi_position_semantics"][
            "source_positions_convention"
        ]
        == "unwrapped"
    )
    with (output / "transport_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["mobile_species"] == "Li"
    assert row["lag_grid_mode"] == "custom"
    assert row["positions_convention"] == "unwrapped"
    assert float(row["D_mean_m2_s"]) == pytest.approx(1.0e-10)


def test_plot_transport_writes_png_and_svg(tmp_path) -> None:
    """24: plot_transport renders non-empty PNG and SVG from a synthetic result."""

    pytest.importorskip("matplotlib")
    from mliport.analysis.plots import plot_transport

    result = {
        "lag_time_ps": np.asarray([0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12]),
        "kinisi_msd_A2": np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        "kinisi_msd_variance_A4": np.asarray([0.0, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4]),
        "tracer_diffusion": {"fit_start_ps": 0.04, "fit_stop_ps": 0.12},
    }
    paths = plot_transport(result, tmp_path / "transport_msd")
    assert {path.name for path in paths} == {"transport_msd.png", "transport_msd.svg"}
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0


def test_cli_transport_summary_prints_posterior_and_fit_window(
    tmp_path, monkeypatch, capsys
) -> None:
    """23: the CLI prints a compact transport posterior summary."""

    run = tmp_path / "short-run"
    _write_short_run(run)

    def fake_dispatch(_request, _output_dir):
        return (
            {
                "mobile_species": "Li",
                "dimensions": "xyz",
                "temperature_mean_K": 700.0,
                "tracer_diffusion": {
                    "kinisi_version": "2.1.0",
                    "random_seed": 0,
                    "fit_start_ps": 40.0,
                    "fit_stop_ps": 200.0,
                    "lag_grid": {
                        "mode": "custom",
                        "nominal_step_ps": 2.0,
                        "requested_step_ps": 2.0,
                        "n_lag_points_total": 100,
                    },
                    "D_posterior_m2_s": {
                        "mean": 3.440443641e-9,
                        "std": 2.087898119e-10,
                        "credible_interval_95": [
                            3.027256045e-9,
                            3.845268093e-9,
                        ],
                    },
                },
                "nernst_einstein": {
                    "sigma_NE_tracer_mS_cm": 123.4,
                    "sigma_NE_tracer_posterior_mS_cm": {
                        "mean": 123.4,
                        "credible_interval_95": [110.0, 137.0],
                    },
                },
                "kinisi_position_semantics": {
                    "source_positions_convention": "unwrapped",
                    "backend_reconstruction": "kinisi periodic displacement reconstruction",
                },
            },
            ["kinisi_arrays.npz", "transport_summary.csv"],
        )

    monkeypatch.setattr(runner_module, "_dispatch", fake_dispatch)
    assert (
        main(
            [
                "analyze",
                str(run),
                "transport",
                "--mobile",
                "Li",
                "--charge",
                "1",
                "--fit-start-ps",
                "40",
                "--lag-step-ps",
                "2",
                "--lag-stop-ps",
                "200",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Tracer diffusion" in out
    assert "95% credible interval" in out
    assert "Fit window" in out
    assert "40 - 200 ps" in out
    assert "Kinisi lag grid" in out
    assert "Nernst-Einstein" in out


# ---------------------------------------------------------------------------
# R05: exact-displacement adapter and custom-lag resource guard
# ---------------------------------------------------------------------------


def _kinisi_msd_reference(displacements: np.ndarray, lag: int) -> float:
    """kinisi's raw MSD convention for one particle (edge samples included)."""
    disp = np.asarray(displacements, dtype=float)
    samples = np.concatenate([disp[lag - 1 : lag], disp[lag:] - disp[:-lag]], axis=0)
    return float(np.mean(np.sum(samples**2, axis=-1)))


def test_exact_parser_replaces_kinisi_skew_cell_misfold() -> None:
    """Review R05 counterexample: cell [[4,0,0],[12,4,0],[0,0,4]].

    kinisi's ASEParser turns the exact step (0, 1.8, 0) A into (4, 1.8, 0) A
    because it folds wrapped fractional coordinates image-by-image; the exact
    adapter must inject the real displacement instead.
    """
    sc = pytest.importorskip("scipp")
    pytest.importorskip("kinisi.analyze")
    from kinisi.ase import ASEParser

    cell = [[4.0, 0.0, 0.0], [12.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    continuous = np.array([[[0.0, 0.0, 0.0]], [[0.0, 1.8, 0.0]], [[0.0, 3.6, 0.0]]])
    frames = [
        Atoms("Li", positions=position[0:1], cell=cell, pbc=True)
        for position in continuous
    ]
    indices = sc.array(dims=["particle"], values=[0], unit="dimensionless")
    plain = ASEParser(
        atoms=frames,
        specie=None,
        time_step=sc.scalar(1.0, unit="fs"),
        step_skip=sc.scalar(1, unit="dimensionless"),
        dimension="xyz",
        specie_indices=indices,
        progress=False,
    )
    misfolded = np.asarray(plain.displacements.to(unit="angstrom").values)[:, 0, :]
    # the reviewer's measured kinisi reconstruction
    np.testing.assert_allclose(misfolded[1], [4.0, 1.8, 0.0], atol=1e-12)

    exact_parser = _exact_displacement_parser(
        sc=sc,
        ASEParser=ASEParser,
        frames=frames,
        continuous_positions_A=continuous,
        local_mobile=np.array([0]),
        time_step_fs=1.0,
        step_skip=1,
        dt_values_fs=None,
        dimensions="xyz",
    )
    used = np.asarray(exact_parser.displacements.to(unit="angstrom").values)[:, 0, :]
    np.testing.assert_allclose(
        used, [[0.0, 0.0, 0.0], [0.0, 1.8, 0.0], [0.0, 3.6, 0.0]], atol=1e-12
    )
    audit = exact_parser._mliport_exact_displacement_audit
    assert audit["backend_displacement_verified"] is True
    assert audit["backend_displacement_max_abs_difference_A"] <= 1e-12

    # deterministic raw MSD: kinisi's own calculate_msd on the exact parser
    from kinisi.displacement import calculate_msd

    msd_group = calculate_msd(exact_parser, progress=False)
    msd_values = np.asarray(msd_group["da"].to(unit="angstrom^2").values)
    np.testing.assert_allclose(
        msd_values,
        [_kinisi_msd_reference(continuous[:, 0, :], lag) for lag in (1, 2)],
        rtol=1e-12,
    )


def _skew_unwrapped_dataset() -> TrajectoryDataset:
    """Single Li walking +1.8 A/frame along y in a strongly skewed cell."""
    n_frames = 30
    cell = np.array([[6.0, 0.0, 0.0], [6.0, 6.0, 0.0], [0.0, 0.0, 6.0]])
    positions = np.zeros((n_frames, 2, 3), dtype=float)
    positions[:, 1, 0] = 0.0  # S framework atom stays at the origin
    positions[:, 0, 1] = 1.8 * np.arange(n_frames, dtype=float)
    frames = [Atoms("LiS", positions=frame, cell=cell, pbc=True) for frame in positions]
    return TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(n_frames, dtype=float) * 10.0,
        positions_convention="unwrapped",
    )


def test_kinisi_transport_uses_exact_displacements_on_skew_cell() -> None:
    """The fitted MSD must follow the exact steps, not kinisi's misfold."""
    pytest.importorskip("scipp")
    pytest.importorskip("kinisi.analyze")
    dataset = _skew_unwrapped_dataset()
    result = kinisi_transport(
        dataset,
        mobile_species="Li",
        ionic_charge_e=1,
        fit_start_ps=0.0,
        lag_step_ps=0.05,
        lag_stop_ps=0.2,
        temperature_K=600.0,
        n_samples=10,
        n_walkers=16,
        n_burn=10,
        n_thin=1,
    )
    semantics = result["kinisi_position_semantics"]
    assert semantics["exact_displacement_adapter"] is True
    assert semantics["backend_displacement_verified"] is True

    lag_frames = np.rint(np.asarray(result["lag_time_ps"]) * 100.0).astype(int)
    walk = np.concatenate(
        [np.zeros((1, 3)), np.cumsum(np.tile([0.0, 1.8, 0.0], (29, 1)), axis=0)]
    )
    exact_msd = np.array([_kinisi_msd_reference(walk, int(lag)) for lag in lag_frames])
    reported = np.asarray(result["kinisi_msd_A2"], dtype=float)
    np.testing.assert_allclose(reported, exact_msd, rtol=1e-9)
    # the image misfold would shift the walk by 6 A per lag; that MSD is
    # an order of magnitude larger than anything the exact walk can give
    misfolded_msd = exact_msd + (6.0 * lag_frames) ** 2
    assert np.all(reported < 0.5 * misfolded_msd)


def test_custom_lag_grid_guard_refuses_20000_points() -> None:
    """Review: 20000 custom lags imply a 3.2 GiB covariance matrix alone."""
    with pytest.raises(ValueError, match="covariance"):
        _resolve_kinisi_lag_grid(
            frame_interval_fs=1.0,
            total_duration_ps=20.0,
            fit_start_ps=0.0,
            lag_step_ps=0.001,
            lag_stop_ps=20.0,
            covariance_memory_limit_bytes=4 * 1024**3,
        )
    # a coarse grid with the same limit is accepted
    accepted = _resolve_kinisi_lag_grid(
        frame_interval_fs=1.0,
        total_duration_ps=20.0,
        fit_start_ps=0.0,
        lag_step_ps=1.0,
        lag_stop_ps=20.0,
        covariance_memory_limit_bytes=4 * 1024**3,
    )
    assert accepted["n_lag_points"] == 20
    assert accepted["estimated_covariance_peak_bytes"] < 4 * 1024**3


def test_custom_lag_guard_runs_before_kinisi_import(monkeypatch) -> None:
    dataset = _synthetic_transport_dataset()
    monkeypatch.setattr(
        transport_module,
        "_require_kinisi",
        lambda: pytest.fail("kinisi must not be imported for a guarded custom grid"),
    )
    with pytest.raises(ValueError, match="covariance"):
        kinisi_transport(
            dataset,
            mobile_species="Li",
            ionic_charge_e=1,
            fit_start_ps=0.05,
            lag_step_ps=0.002,
            lag_stop_ps=0.2,
            temperature_K=600.0,
            parser_memory_limit_gib=4.7e-5,
        )


# ---------------------------------------------------------------------------
# R06: content-based cache identity and transactional attempts
# ---------------------------------------------------------------------------


def test_analysis_fingerprint_hashes_sidecar_content_not_only_stat(tmp_path) -> None:
    run = tmp_path / "run"
    _write_short_run(run)
    first = source_fingerprint(run)
    artifacts = run / "artifacts.json"
    text = artifacts.read_text(encoding="utf-8")
    modified = text.replace('"status": "completed"', '"status": "cancelled"')
    assert modified != text and len(modified) == len(text), "same-size edit"
    artifacts.write_text(modified, encoding="utf-8")
    second = source_fingerprint(run)
    assert (
        first["run_metadata"]["artifacts.json"]["sha256"]
        != second["run_metadata"]["artifacts.json"]["sha256"]
    )
    # the trajectory itself is content-hashed too
    assert first["sha256"] == second["sha256"]
    assert first["sha256"] and len(first["sha256"]) == 64


def test_sidecar_change_invalidates_cached_analysis(tmp_path) -> None:
    run = tmp_path / "run"
    _write_short_run(run)
    first = run_analysis(AnalysisRequest("validate", str(run)))
    assert first["reused"] is False
    cached = run_analysis(AnalysisRequest("validate", str(run)))
    assert cached["reused"] is True

    md_csv = run / "raw" / "md.csv"
    text = md_csv.read_text(encoding="utf-8")
    md_csv.write_text(text.replace(",600,", ",601,"), encoding="utf-8")
    second = run_analysis(AnalysisRequest("validate", str(run)))
    assert second["analysis_id"] != first["analysis_id"]
    assert second["reused"] is False


def test_forced_failure_cannot_be_masked_by_old_success(tmp_path, monkeypatch) -> None:
    """Review R06 red case: fail(force) then recall must never return success."""
    run = tmp_path / "run"
    _write_short_run(run)
    first = run_analysis(AnalysisRequest("validate", str(run)))
    output = Path(first["output_dir"])
    assert json.loads((output / "results.json").read_text())["status"] == "success"

    def boom(*_args, **_kwargs):
        raise RuntimeError("backend failed")

    monkeypatch.setattr(runner_module, "_dispatch", boom)
    with pytest.raises(RuntimeError, match="backend failed"):
        run_analysis(AnalysisRequest("validate", str(run), force=True))

    published = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert published["status"] == "failed"
    pointer = json.loads((output / "attempts" / "latest.json").read_text())
    assert pointer["status"] == "failed"
    # the old success is preserved as an immutable attempt artefact
    preserved = sorted((output / "attempts").glob("previous-results-*.json"))
    assert preserved, "previous published result must be preserved"
    assert json.loads(preserved[-1].read_text())["status"] == "success"

    # Still failing: a non-force recall must not resurrect the old success.
    with pytest.raises(RuntimeError, match="backend failed"):
        run_analysis(AnalysisRequest("validate", str(run)))

    # Once the backend works again the run recomputes and publishes fresh.
    monkeypatch.undo()
    recovered = run_analysis(AnalysisRequest("validate", str(run)))
    assert recovered["status"] == "success"
    assert recovered["reused"] is False


def test_keyboard_interrupt_leaves_cancelled_attempt_not_success(
    tmp_path, monkeypatch
) -> None:
    run = tmp_path / "run"
    _write_short_run(run)

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner_module, "_dispatch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_analysis(AnalysisRequest("validate", str(run)))
    output_dirs = list((run / "analysis" / "validate").glob("*/results.json"))
    assert output_dirs, "a cancelled attempt must still publish an outcome"
    published = json.loads(output_dirs[0].read_text(encoding="utf-8"))
    assert published["status"] == "cancelled"
    pointer = json.loads(
        (output_dirs[0].parent / "attempts" / "latest.json").read_text()
    )
    assert pointer["status"] == "cancelled"


def test_tampered_published_result_is_recomputed_not_reused(tmp_path) -> None:
    run = tmp_path / "run"
    _write_short_run(run)
    first = run_analysis(AnalysisRequest("validate", str(run)))
    output = Path(first["output_dir"])
    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    payload["source_fingerprint"] = {"path": "tampered", "exists": True}
    (output / "results.json").write_text(json.dumps(payload), encoding="utf-8")

    second = run_analysis(AnalysisRequest("validate", str(run)))
    assert second["reused"] is False
    assert second["status"] == "success"


def test_concurrent_identical_analyses_publish_complete_documents(tmp_path) -> None:
    run = tmp_path / "run"
    _write_short_run(run)

    def _one(_index: int) -> str:
        try:
            return run_analysis(AnalysisRequest("validate", str(run)))["status"]
        except Exception as exc:  # noqa: BLE001 - reported to the assertion
            return type(exc).__name__

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(_one, range(2)))
    assert outcomes == ["success", "success"], outcomes

    outputs = list((run / "analysis" / "validate").glob("*/results.json"))
    assert len(outputs) == 1
    published = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert published["status"] == "success"
    attempts = sorted((outputs[0].parent / "attempts").glob("*/attempt.json"))
    assert attempts, "attempt records must exist"
    for attempt in attempts:
        assert json.loads(attempt.read_text())["status"] == "success"
    pointer = json.loads((outputs[0].parent / "attempts" / "latest.json").read_text())
    assert pointer["status"] == "success"


# ---------------------------------------------------------------------------
# R06/array export: stored-separately arrays must round-trip
# ---------------------------------------------------------------------------


def test_oversized_json_arrays_round_trip_through_npz(tmp_path) -> None:
    from mliport.analysis.runner import _write_json, load_stored_array

    big = np.linspace(0.0, 1.0, 5000).reshape(2500, 2)
    document = {"results": {"big": big, "small": np.arange(4)}}
    path = tmp_path / "results.json"
    _write_json(path, document)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    reference = loaded["results"]["big"]
    assert reference["stored_separately"] is True
    assert reference["format"] == "npz"
    assert reference["shape"] == [2500, 2]
    assert reference["dtype"] == "float64"
    assert len(reference["sha256"]) == 64
    npz_path = tmp_path / reference["path"]
    assert npz_path.is_file()
    # small arrays stay inline
    assert loaded["results"]["small"] == [0, 1, 2, 3]

    roundtrip = load_stored_array(reference, tmp_path)
    np.testing.assert_array_equal(roundtrip, big)

    # tampering with the stored content is detected
    with np.load(npz_path, allow_pickle=False) as data:
        stored = np.asarray(data["array"])
    np.savez(npz_path, array=stored + 1.0)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_stored_array(reference, tmp_path)

    # path traversal is refused
    with pytest.raises(ValueError, match="escapes"):
        load_stored_array({"path": "../escape.npz"}, tmp_path)
    with pytest.raises(ValueError, match="no path"):
        load_stored_array({}, tmp_path)


def test_write_columns_stores_2d_arrays_in_npz(tmp_path) -> None:
    from mliport.analysis.runner import _write_columns, load_stored_array

    coordinates = np.arange(30.0).reshape(10, 3)
    csv_path = tmp_path / "coordinates.csv"
    _write_columns(
        csv_path,
        {"time_fs": np.arange(10.0), "coordinates": coordinates},
    )
    csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines[0].startswith("time_fs")
    assert "coordinates" not in csv_lines[0], "2-D data must not be flattened"
    manifest = json.loads(csv_path.with_name("coordinates.csv.arrays.json").read_text())
    reference = manifest["arrays"]["coordinates"]
    assert reference["shape"] == [10, 3]
    np.testing.assert_array_equal(load_stored_array(reference, tmp_path), coordinates)


def test_long_msd_analysis_exports_arrays_and_round_trips(tmp_path) -> None:
    run = tmp_path / "long-run"
    raw = run / "raw"
    raw.mkdir(parents=True)
    n_frames = 2100
    with Trajectory(raw / "trajectory.traj", "w") as writer:
        for index in range(n_frames):
            atoms = Atoms(
                "LiS",
                positions=[[0.1 * index, 0.0, 0.0], [5.0, 5.0, 5.0]],
                cell=[20.0, 20.0, 20.0],
                pbc=True,
            )
            atoms.info["mliport_step"] = index
            atoms.info["mliport_time_fs"] = float(index * 10.0)
            atoms.info["mliport_phase"] = "production"
            writer.write(atoms)
    (run / "artifacts.json").write_text(
        json.dumps(
            {
                "schema": "mliport.md-artifacts/2",
                "status": "completed",
                "trajectory": {
                    "md_timestep_fs": 10.0,
                    "frame_stride_steps": 1,
                    "frame_interval_fs": 10.0,
                    "positions_convention": "unwrapped",
                },
            }
        )
    )
    (run / "resolved_config.json").write_text(
        json.dumps({"run_options": {"ensemble": "NVE", "temperature": 600}})
    )
    from mliport.analysis.runner import load_stored_array

    result = run_analysis(
        AnalysisRequest(
            "msd",
            str(run),
            parameters={"mobile_species": "Li"},
        )
    )
    output = Path(result["output_dir"])
    published = json.loads((output / "results.json").read_text(encoding="utf-8"))

    references: list[dict[str, Any]] = []

    def collect(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("stored_separately") is True:
                references.append(node)
            else:
                for item in node.values():
                    collect(item)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(published["results"])
    assert references, "the >2000-element MSD arrays must be stored separately"
    for reference in references[:3]:
        array = load_stored_array(reference, output)
        assert array.shape == tuple(reference["shape"])
        assert str(array.dtype) == reference["dtype"]


def test_cli_alpha_parameters_enter_request_and_result(tmp_path, capsys) -> None:
    """CLI and TUI share one implementation: the flags land in the request."""
    run = tmp_path / "short-run"
    _write_short_run(run)
    assert (
        main(
            [
                "analyze",
                str(run),
                "msd",
                "--mobile",
                "Li",
                "--alpha-window-decades",
                "0.3",
                "--alpha-min-origins",
                "3",
            ]
        )
        == 0
    )
    capsys.readouterr()
    requests = list((run / "analysis" / "msd").glob("*/request.json"))
    assert len(requests) == 1
    request = json.loads(requests[0].read_text(encoding="utf-8"))
    assert request["parameters"]["alpha_window_decades"] == pytest.approx(0.3)
    assert request["parameters"]["alpha_min_origins"] == 3
    result = json.loads(
        (requests[0].parent / "results.json").read_text(encoding="utf-8")
    )
    estimate = result["results"]["alpha_estimates_by_axes"]["xyz"]
    assert estimate["log_window_width_decades"] == pytest.approx(0.3)
    assert estimate["parameters"]["min_origins"] == 3
    # a different alpha window is a different request/cache identity
    assert (
        main(
            [
                "analyze",
                str(run),
                "msd",
                "--mobile",
                "Li",
                "--alpha-window-decades",
                "0.4",
            ]
        )
        == 0
    )
    assert len(list((run / "analysis" / "msd").glob("*/request.json"))) == 2
