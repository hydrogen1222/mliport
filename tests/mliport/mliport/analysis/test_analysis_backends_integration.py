"""Real optional-backend integration and synthetic collective known answers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mliport.analysis.dataset import TrajectoryDataset
from mliport.analysis.electrolyte import gemdat_electrolyte
from mliport.analysis.transport import kinisi_transport

gemdat = pytest.importorskip("gemdat")
kinisi = pytest.importorskip("kinisi")
pytest.importorskip("scipp")


def _brownian_dataset(
    mode: str,
    *,
    seed: int,
    n_frames: int,
    n_particles: int = 32,
    frame_interval_fs: float = 10.0,
) -> TrajectoryDataset:
    """Build paired Brownian walkers with a known collective ordering."""

    rng = np.random.default_rng(seed)
    n_steps = n_frames - 1
    if mode == "independent":
        increments = rng.normal(scale=0.03, size=(n_steps, n_particles, 3))
    else:
        if n_particles % 2:
            raise ValueError("Correlated Brownian fixtures require particle pairs")
        n_pairs = n_particles // 2
        common = rng.normal(scale=0.03, size=(n_steps, n_pairs, 3))
        noise = rng.normal(scale=0.006, size=(n_steps, n_pairs, 2, 3))
        if mode == "positive":
            paired = np.stack(
                (common + noise[:, :, 0], common + noise[:, :, 1]), axis=2
            )
        elif mode == "anti":
            paired = np.stack(
                (common + noise[:, :, 0], -common + noise[:, :, 1]), axis=2
            )
        else:
            raise ValueError(f"Unknown Brownian mode: {mode}")
        increments = paired.reshape(n_steps, n_particles, 3)
    positions = np.concatenate(
        (np.zeros((1, n_particles, 3)), np.cumsum(increments, axis=0)), axis=0
    )
    positions += 100.0
    frames = [
        Atoms(
            f"Li{n_particles}",
            positions=frame,
            cell=[200.0, 200.0, 200.0],
            pbc=True,
        )
        for frame in positions
    ]
    return TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(n_frames, dtype=float) * frame_interval_fs,
        positions_convention="unwrapped",
    )


def _run_collective(
    dataset: TrajectoryDataset, *, jump_diffusion: bool = False
) -> dict:
    return kinisi_transport(
        dataset,
        mobile_species="Li",
        ionic_charge_e=1,
        fit_start_ps=2.0,
        lag_step_ps=0.5,
        lag_stop_ps=10.0,
        temperature_K=600.0,
        collective_conductivity=True,
        jump_diffusion=jump_diffusion,
        random_seed=4,
        n_samples=20,
        n_walkers=16,
        n_burn=10,
        n_thin=1,
    )


@pytest.fixture(scope="module")
def independent_transport() -> dict:
    dataset = _brownian_dataset("independent", seed=123, n_frames=5000, n_particles=32)
    return _run_collective(dataset, jump_diffusion=True)


def test_real_kinisi_conductivity_and_jump_analyzers_smoke(
    independent_transport,
) -> None:
    from kinisi.analyze import ConductivityAnalyzer, JumpDiffusionAnalyzer

    assert ConductivityAnalyzer.__module__.startswith("kinisi.")
    assert JumpDiffusionAnalyzer.__module__.startswith("kinisi.")
    assert independent_transport["collective_conductivity"]["backend"] == "kinisi"
    jump_mean = independent_transport["jump_diffusion"]["D_J_posterior_m2_s"]["mean"]
    assert np.isfinite(jump_mean) and jump_mean > 0


def test_independent_brownian_haven_is_approximately_one(
    independent_transport,
) -> None:
    haven = independent_transport["haven_ratio"]["point_estimate"]
    assert haven == pytest.approx(1.0, rel=0.3)


@pytest.mark.parametrize(
    ("mode", "ordering"),
    [("positive", "greater"), ("anti", "less")],
)
def test_correlated_brownian_collective_conductivity_ordering(mode, ordering) -> None:
    result = _run_collective(
        _brownian_dataset(mode, seed=2, n_frames=2000, n_particles=32)
    )
    sigma_collective = result["collective_conductivity"][
        "sigma_collective_posterior_S_m"
    ]["mean"]
    sigma_ne = result["nernst_einstein"]["sigma_NE_tracer_posterior_S_m"]["mean"]
    if ordering == "greater":
        assert sigma_collective > 1.4 * sigma_ne
    else:
        assert sigma_collective < 0.2 * sigma_ne


def test_safe_point_01_ps_downsampling_preserves_long_time_transport() -> None:
    dense = _brownian_dataset("independent", seed=321, n_frames=5000, n_particles=32)
    sparse_positions = dense.positions[::10]
    sparse_frames = [
        Atoms(
            "Li32",
            positions=frame,
            cell=[200.0, 200.0, 200.0],
            pbc=True,
        )
        for frame in sparse_positions
    ]
    sparse = TrajectoryDataset.from_frames(
        sparse_frames,
        times_fs=np.arange(len(sparse_frames), dtype=float) * 100.0,
        positions_convention="unwrapped",
    )
    dense_result = _run_collective(dense)
    sparse_result = _run_collective(sparse)

    dense_d = dense_result["tracer_diffusion"]["D_posterior_m2_s"]["mean"]
    sparse_d = sparse_result["tracer_diffusion"]["D_posterior_m2_s"]["mean"]
    dense_sigma = dense_result["collective_conductivity"][
        "sigma_collective_posterior_S_m"
    ]["mean"]
    sparse_sigma = sparse_result["collective_conductivity"][
        "sigma_collective_posterior_S_m"
    ]["mean"]
    assert sparse_d == pytest.approx(dense_d, rel=0.1)
    assert sparse_sigma == pytest.approx(dense_sigma, rel=0.1)


def test_real_gemdat_mechanism_integration_smoke(tmp_path: Path) -> None:
    from pymatgen.core import Lattice, Structure

    pattern = np.r_[
        np.full(5, 2.0),
        [1.0, 0.0, -1.0],
        np.full(5, -2.0),
        [-1.0, 0.0, 1.0],
        np.full(5, 2.0),
    ]
    x_positions = np.tile(pattern, 30)[:500]
    frames = [
        Atoms(
            "LiS",
            positions=[[x, 5.0, 5.0], [5.0, 1.0, 1.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        for x in x_positions
    ]
    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(len(frames), dtype=float) * 100.0,
        positions_convention="unwrapped",
    )
    sites_path = tmp_path / "li_sites.cif"
    Structure(
        Lattice.cubic(10.0),
        ["Li", "Li"],
        [[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]],
    ).to(filename=str(sites_path))

    result = gemdat_electrolyte(
        dataset,
        mobile_species="Li",
        sites_path=sites_path,
        temperature_K=600.0,
        resolution_A=1.0,
        site_radius_A=0.8,
        minimal_residence=2,
        percolation_axes="x",
    )

    assert result.summary["backend"] == "GEMDAT"
    assert result.summary["gemdat_version"].startswith("1.")
    assert result.summary["number_of_jumps"] > 0
    assert "residence_time_ps" in result.tables["residence_times"].columns
    assert "jump_distance_A" in result.tables["jumps"].columns
    assert "jump_rate_s^-1" in result.tables["jump_rates"].columns
    np.testing.assert_allclose(result.tables["jumps"]["jump_distance_A"], 4.0)
    assert all(
        "path_integrated_free_energy_eV" not in path for path in result.paths.values()
    )
