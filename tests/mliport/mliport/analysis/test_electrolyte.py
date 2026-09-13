from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mliport.analysis.dataset import TrajectoryDataset
from mliport.analysis.electrolyte import _gemdat_trajectory, _path_arrays, jump_summary


class _FakeJumps:
    n_jumps = 12

    def jump_diffusivity(self, dimensions: int) -> float:
        return 6.0 / dimensions


def test_percolation_axes_do_not_change_jump_dimensions() -> None:
    jumps = _FakeJumps()
    along_x = jump_summary(jumps, jump_dimensions=3, percolation_axes="x")
    all_axes = jump_summary(jumps, jump_dimensions=3, percolation_axes="xyz")
    assert along_x["jump_diffusivity_m2_s"] == all_axes["jump_diffusivity_m2_s"]
    assert along_x["jump_dimensions"] == all_axes["jump_dimensions"] == 3


def test_gemdat_trajectory_adapter_uses_saved_frame_interval() -> None:
    pytest.importorskip("gemdat")
    frames = [
        Atoms(
            "LiS",
            positions=[[1 + 0.1 * frame, 1, 1], [5, 5, 5]],
            cell=[10, 10, 10],
            pbc=True,
        )
        for frame in range(6)
    ]
    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(6) * 2,
        positions_convention="unwrapped",
    )
    trajectory = _gemdat_trajectory(dataset, temperature_K=600)
    assert len(trajectory) == 6
    assert trajectory.time_step == pytest.approx(2e-15)
    assert trajectory.metadata["temperature"] == 600
    assert len(trajectory.filter("Li")) == 6


def test_path_arrays_do_not_claim_an_unphysical_path_integral() -> None:
    class Path:
        sites = np.asarray([[0, 0, 0], [1, 0, 0]])
        dims = np.asarray([2, 2, 2])
        energy = np.asarray([0.1, 0.4])

    class Lattice:
        @staticmethod
        def get_cartesian_coords(fractional):
            return np.asarray(fractional) * 10.0

    arrays = _path_arrays(Path(), Lattice())
    assert "path_integrated_free_energy_eV" not in arrays
    np.testing.assert_allclose(arrays["free_energy_eV"], [0.1, 0.4])
