"""Fit-independent MSD core cache, reuse of preparation and phase timings (#133)."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from ase import Atoms
from ase.io.trajectory import Trajectory

import mliport.analysis.drift as drift_module
import mliport.analysis.msd as msd_module
import mliport.analysis.runner as runner_module
import mliport.analysis.transport as transport_module
from mliport.analysis import TrajectoryDataset
from mliport.analysis.msd import unwrap_positions
from mliport.analysis.msd_cache import MsdCoreCache, msd_core_key
from mliport.analysis.runner import run_analysis
from mliport.analysis.schema import AnalysisRequest
from mliport.analysis.timing import PhaseTimings, phase, recording

N_FRAMES = 40


def _write_run(path, *, scale: float = 1.0) -> None:
    raw = path / "raw"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(5)
    li = np.array([[1.0, 1, 1], [4, 4, 4], [7, 2, 6]])
    sulfur = np.array([[5.0, 5, 5], [2, 8, 3]])
    walk = np.cumsum(rng.normal(size=(N_FRAMES, 3, 3)) * 0.3 * scale, axis=0)
    drift = np.cumsum(rng.normal(size=(N_FRAMES, 1, 3)) * 0.05, axis=0)
    with Trajectory(raw / "trajectory.traj", "w") as writer:
        for index in range(N_FRAMES):
            positions = np.vstack(
                [li + walk[index] + drift[index], sulfur + drift[index]]
            )
            atoms = Atoms("Li3S2", positions=positions, cell=[30] * 3, pbc=True)
            atoms.info["mliport_step"] = index * 2
            atoms.info["mliport_time_fs"] = float(index * 2)
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
        for index in range(N_FRAMES):
            writer.writerow(
                [index * 2, index * 2, "production", 600, -2, 0.1, -1.9, 27000]
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
                    "positions_convention": "unwrapped",
                    "production_start_step": 0,
                },
            }
        )
    )
    (path / "resolved_config.json").write_text(
        json.dumps({"run_options": {"ensemble": "NVE", "temperature": 600}})
    )


def _msd(run, **parameters):
    base = {
        "mobile_species": "Li",
        "drift_reference": "nonmobile",
        "drift_mode": "mass_weighted_com",
        "fit_start_ps": 0.01,
        "fit_stop_ps": 0.05,
    }
    base.update(parameters)
    return run_analysis(AnalysisRequest("msd", str(run), parameters=base))


@pytest.fixture()
def load_counter(monkeypatch):
    calls = {"n": 0}
    original = TrajectoryDataset.load.__func__

    def counting(cls, *args, **kwargs):
        calls["n"] += 1
        return original(cls, *args, **kwargs)

    monkeypatch.setattr(TrajectoryDataset, "load", classmethod(counting))
    return calls


def _read_results(outcome, run):
    path = run / "analysis" / "msd" / outcome["analysis_id"] / "results.json"
    return json.loads(path.read_text(encoding="utf-8"))["results"]


def _strip(result):
    return {k: v for k, v in result.items() if k not in {"timings", "core_cache"}}


def test_refit_only_reuses_the_cached_core_and_matches_a_fresh_run(
    tmp_path, load_counter
) -> None:
    pytest.importorskip("matplotlib")
    run = tmp_path / "run"
    _write_run(run)

    first = _msd(run)
    assert load_counter["n"] == 1
    assert _read_results(first, run)["core_cache"]["hit"] is False

    refit = _msd(run, fit_start_ps=0.02, fit_stop_ps=0.06)
    assert refit["analysis_id"] != first["analysis_id"]
    assert load_counter["n"] == 1, "trajectory must not be reloaded for a refit"
    refit_result = _read_results(refit, run)
    assert refit_result["core_cache"]["hit"] is True

    # The reused core gives exactly the numbers of a from-scratch analysis.
    other_run = tmp_path / "fresh"
    _write_run(other_run)
    scratch = _msd(other_run, fit_start_ps=0.02, fit_stop_ps=0.06)
    scratch_result = _read_results(scratch, other_run)
    assert scratch_result["core_cache"]["hit"] is False
    assert _strip(refit_result) == _strip(scratch_result)


@pytest.mark.parametrize(
    "change",
    [
        {"drift_reference": "none"},
        {"drift_mode": "arithmetic_mean"},
        {"alpha_window_decades": 0.4},
        {"alpha_min_points": 6},
        {"axes": "x"},
        {"mobile_species": "S"},
        {"method": "direct"},
        {"include_equilibration": True},
        {"frame_interval_fs": 2.0},
        {"positions_convention": "unwrapped"},
    ],
)
def test_changing_a_scientific_parameter_invalidates_the_core(
    tmp_path, load_counter, change
) -> None:
    pytest.importorskip("matplotlib")
    run = tmp_path / "run"
    _write_run(run)
    _msd(run)
    before = load_counter["n"]
    parameters = dict(change)
    if parameters.get("mobile_species") == "S":
        parameters["drift_reference"] = "none"
    outcome = _msd(run, **parameters)
    assert load_counter["n"] == before + 1
    assert _read_results(outcome, run)["core_cache"]["hit"] is False


def test_modified_trajectory_or_sidecar_invalidates_the_core(
    tmp_path, load_counter
) -> None:
    pytest.importorskip("matplotlib")
    run = tmp_path / "run"
    _write_run(run)
    _msd(run)
    assert _msd(run, fit_start_ps=0.02, fit_stop_ps=0.06)
    assert load_counter["n"] == 1

    # Metadata-only edit of a sidecar the loader consumes.
    config = run / "resolved_config.json"
    config.write_text(
        json.dumps({"run_options": {"ensemble": "NVE", "temperature": 601}})
    )
    outcome = _msd(run, fit_start_ps=0.02, fit_stop_ps=0.06)
    assert load_counter["n"] == 2
    assert _read_results(outcome, run)["core_cache"]["hit"] is False

    # Rewritten trajectory content.
    import shutil

    shutil.rmtree(run / "raw")
    _write_run_raw_only(run, scale=2.0)
    outcome = _msd(run, fit_start_ps=0.02, fit_stop_ps=0.06)
    assert load_counter["n"] == 3
    assert _read_results(outcome, run)["core_cache"]["hit"] is False


def _write_run_raw_only(run, *, scale: float) -> None:
    other = run.parent / "tmp_other"
    _write_run(other, scale=scale)
    (run / "raw").mkdir(exist_ok=True)
    for item in (other / "raw").iterdir():
        item.rename(run / "raw" / item.name)


def test_force_recomputes_and_corrupt_entry_is_a_miss(tmp_path, load_counter) -> None:
    pytest.importorskip("matplotlib")
    run = tmp_path / "run"
    _write_run(run)
    _msd(run)
    assert load_counter["n"] == 1
    forced = run_analysis(
        AnalysisRequest(
            "msd",
            str(run),
            parameters={
                "mobile_species": "Li",
                "drift_reference": "nonmobile",
                "drift_mode": "mass_weighted_com",
                "fit_start_ps": 0.02,
                "fit_stop_ps": 0.06,
            },
            force=True,
        )
    )
    assert load_counter["n"] == 2
    assert _read_results(forced, run)["core_cache"]["hit"] is False

    for npz in (run / "analysis" / "_msd_core").glob("*/core.npz"):
        npz.write_bytes(npz.read_bytes()[:-8] + b"corrupt!")
    outcome = _msd(run, fit_start_ps=0.03, fit_stop_ps=0.07)
    assert load_counter["n"] == 3
    assert _read_results(outcome, run)["core_cache"]["hit"] is False


def test_core_cache_roundtrip_preserves_values_and_types(tmp_path) -> None:
    rng = np.random.default_rng(1)
    frames = [
        Atoms(
            "Li2S", positions=rng.random((3, 3)) * 5 + i * 0.1, cell=[20] * 3, pbc=True
        )
        for i in range(30)
    ]
    dataset = TrajectoryDataset.from_frames(
        frames, times_fs=np.arange(30) * 2.0, positions_convention="unwrapped"
    )
    core = msd_module.compute_msd_core(
        dataset, mobile_species="Li", drift_reference="nonmobile"
    )
    cache = MsdCoreCache(tmp_path, msd_core_key({"x": 1}))
    assert cache.load() is None
    assert cache.store(core)
    loaded = cache.load()

    def same(a, b):
        if isinstance(a, dict):
            assert isinstance(b, dict)
            assert {str(k) for k in a} == {str(k) for k in b}
            for key in a:
                same(a[key], b[key])
        elif isinstance(a, np.ndarray):
            assert isinstance(b, np.ndarray)
            assert a.dtype == b.dtype
            assert np.array_equal(a, b, equal_nan=True)
        elif isinstance(a, (list, tuple)):
            assert type(a) is type(b)
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                same(x, y)
        elif isinstance(a, float) and np.isnan(a):
            assert np.isnan(b)
        else:
            assert a == b

    same(core, loaded)
    # Different identities never collide.
    assert msd_core_key({"x": 1}) != msd_core_key({"x": 2})


def test_blocked_reductions_match_the_unblocked_formulas(monkeypatch) -> None:
    rng = np.random.default_rng(3)
    positions = np.cumsum(rng.normal(size=(37, 9, 3)) * 0.4, axis=0)
    cell = np.diag([6.0, 7.0, 8.0])
    frames = [
        Atoms("Li5S4", positions=p % [6, 7, 8], cell=cell, pbc=True) for p in positions
    ]
    for convention, source in (("unwrapped", positions), ("wrapped", None)):
        if convention == "unwrapped":
            frames = [Atoms("Li5S4", positions=p, cell=cell, pbc=True) for p in source]
        else:
            frames = [
                Atoms("Li5S4", positions=p % [6, 7, 8], cell=cell, pbc=True)
                for p in positions
            ]
        dataset = TrajectoryDataset.from_frames(
            frames, times_fs=np.arange(37) * 1.0, positions_convention=convention
        )
        big = unwrap_positions(dataset)
        monkeypatch.setattr(msd_module, "_BLOCK_TARGET_BYTES", 1)
        small = unwrap_positions(dataset)
        monkeypatch.undo()
        assert np.array_equal(big[0], small[0])
        for key in ("max_fractional_step_mic", "max_cartesian_step_mic_A"):
            assert big[1][key] == pytest.approx(small[1][key], rel=1e-12)

    reference = np.arange(5, 9)
    masses = rng.uniform(1, 30, size=9)
    for mode in ("arithmetic_mean", "mass_weighted_com", "none"):
        one = drift_module.drift_displacement(
            positions, reference=reference, masses=masses, drift_mode=mode
        )
        monkeypatch.setattr(drift_module, "_BLOCK_TARGET_BYTES", 1)
        many = drift_module.drift_displacement(
            positions, reference=reference, masses=masses, drift_mode=mode
        )
        monkeypatch.undo()
        np.testing.assert_allclose(many, one, rtol=1e-13, atol=1e-13)


def test_unwrapped_continuous_positions_are_a_read_only_view() -> None:
    frames = [
        Atoms("LiS", positions=[[i * 0.1, 1, 1], [3, 3, 3]], cell=[9] * 3, pbc=True)
        for i in range(5)
    ]
    dataset = TrajectoryDataset.from_frames(
        frames, times_fs=np.arange(5) * 1.0, positions_convention="unwrapped"
    )
    continuous, _ = unwrap_positions(dataset)
    assert np.shares_memory(continuous, dataset.positions)
    with pytest.raises(ValueError):
        continuous[0, 0, 0] = 1.0


def test_transport_unwraps_once_and_blocked_validation_matches(monkeypatch) -> None:
    pytest.importorskip("kinisi")
    rng = np.random.default_rng(8)
    n_frames = 60
    li = np.array([[1.0, 1, 1], [4, 4, 4], [7, 2, 6], [2, 6, 2]])
    sulfur = np.array([[5.0, 5, 5], [2, 8, 3], [8, 8, 8]])
    walk = np.cumsum(rng.normal(size=(n_frames, 4, 3)) * 0.4, axis=0)
    frames = [
        Atoms(
            "Li4S3",
            positions=np.vstack([li + walk[i], sulfur]),
            cell=[10] * 3,
            pbc=True,
        )
        for i in range(n_frames)
    ]
    dataset = TrajectoryDataset.from_frames(
        frames, times_fs=np.arange(n_frames) * 100.0, positions_convention="unwrapped"
    )
    kwargs = {
        "mobile_species": "Li",
        "ionic_charge_e": 1.0,
        "fit_start_ps": 1.0,
        "drift_reference": "nonmobile",
        "temperature_K": 700.0,
        "n_samples": 20,
        "n_walkers": 8,
        "n_burn": 5,
        "n_thin": 1,
        "random_seed": 1,
        "drift_mode": "mass_weighted_com",
    }
    calls = {"n": 0}
    original = transport_module.unwrap_positions

    def counting(dataset_):
        calls["n"] += 1
        return original(dataset_)

    monkeypatch.setattr(transport_module, "unwrap_positions", counting)
    timings = PhaseTimings()
    with recording(timings):
        reference = transport_module.kinisi_transport(dataset, **kwargs)
    assert calls["n"] == 1, "the trajectory must be unwrapped once per transport run"
    names = [entry["name"] for entry in timings.to_dict()["phases"]]
    for expected in (
        "unwrap_and_step_diagnostics",
        "periodic_reconstruction_validation",
        "mobile_input_preparation",
        "kinisi_exact_displacement_parse",
        "kinisi_lag_statistics",
        "kinisi_analyzer_construction",
        "kinisi_posterior_sampling",
    ):
        assert expected in names
    construction = next(
        entry
        for entry in timings.to_dict()["phases"]
        if entry["name"] == "kinisi_analyzer_construction"
    )
    assert {"kinisi_exact_displacement_parse", "kinisi_lag_statistics"} <= set(
        construction["includes"]
    )

    # Frame-blocked validation and drift give the same published numbers.
    monkeypatch.setattr(transport_module, "_VALIDATION_BLOCK_BYTES", 1)
    monkeypatch.setattr(drift_module, "_BLOCK_TARGET_BYTES", 1)
    blocked = transport_module.kinisi_transport(dataset, **kwargs)
    assert blocked["tracer_diffusion"]["D_posterior_m2_s"]["mean"] == pytest.approx(
        reference["tracer_diffusion"]["D_posterior_m2_s"]["mean"], rel=1e-10
    )
    assert blocked["kinisi_position_semantics"][
        "maximum_exact_vs_mic_difference_A"
    ] == pytest.approx(
        reference["kinisi_position_semantics"]["maximum_exact_vs_mic_difference_A"],
        abs=1e-12,
    )


def test_phase_timings_nesting_and_noop_without_recorder(tmp_path) -> None:
    with phase("ignored"):  # no active recorder: must be a no-op
        pass
    timings = PhaseTimings()
    with recording(timings):
        with phase("outer"):
            with phase("inner_a"):
                pass
            with phase("inner_b"):
                pass
        with phase("later"):
            pass
    phases = {entry["name"]: entry for entry in timings.to_dict()["phases"]}
    assert phases["outer"]["includes"] == ["inner_a", "inner_b"]
    assert phases["later"]["includes"] == []
    assert phases["outer"]["seconds"] >= phases["inner_a"]["seconds"]

    pytest.importorskip("matplotlib")
    run = tmp_path / "run"
    _write_run(run)
    result = _read_results(_msd(run), run)
    names = [entry["name"] for entry in result["timings"]["phases"]]
    for expected in (
        "source_fingerprint",
        "dataset_load",
        "displacement_preparation",
        "msd_estimator",
        "alpha_local_exponent",
        "diagnostic_fit",
        "analysis_task",
    ):
        assert expected in names
    assert result["timings"]["schema"] == "mliport.phase-timings/1"
    # The cache hit skips the load and the heavy phases.
    refit = _read_results(_msd(run, fit_start_ps=0.02, fit_stop_ps=0.06), run)
    refit_names = [entry["name"] for entry in refit["timings"]["phases"]]
    assert "dataset_load" not in refit_names
    assert "msd_estimator" not in refit_names
    assert "msd_core_cache_lookup" in refit_names
    assert runner_module  # keep the import used
