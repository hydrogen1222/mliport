"""Raw-force storage modes in raw/trajectory.traj (#132)."""

from __future__ import annotations

import os

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

from mliport.runners.md_output import AsyncMDOutputWriter, MDFrameSnapshot
from mliport.trajectory_forces import (
    RAW_FORCES_INFO_KEY,
    RAW_FORCES_ULM_FIELD,
    RawForcesStorageError,
    detect_raw_forces_storage,
    open_trajectory_writer,
    read_frames_with_raw_forces,
    read_raw_forces,
)

N = 24


def _frame(rng: np.random.Generator):
    atoms = Atoms(
        ["Li"] * N, positions=rng.random((N, 3)) * 10, cell=[10] * 3, pbc=True
    )
    atoms.set_momenta(rng.normal(size=(N, 3)))
    applied = rng.normal(size=(N, 3))
    # Awkward values: raw != applied + (raw - applied) in float64.
    raw = applied + rng.normal(size=(N, 3)) * 1e-3 + 1e-17
    atoms.calc = SinglePointCalculator(atoms, energy=-1.5, forces=applied)
    return atoms, raw, applied


def _write(path, storage, n, rng, mode="w"):
    writer = open_trajectory_writer(path, mode=mode, raw_forces_storage=storage)
    saved = []
    for _ in range(n):
        atoms, raw, applied = _frame(rng)
        if storage == "binary":
            writer.write(atoms, raw_forces=raw)
        else:
            atoms.info[RAW_FORCES_INFO_KEY] = raw.tolist()
            writer.write(atoms)
        saved.append((atoms, raw, applied))
    writer.close()
    return saved


@pytest.mark.parametrize("storage", ["json", "binary"])
def test_roundtrip_bit_exact_random_access_and_plain_ase(tmp_path, storage) -> None:
    path = tmp_path / "t.traj"
    saved = _write(path, storage, 6, np.random.default_rng(3))
    assert detect_raw_forces_storage(path) == storage
    for i in (3, 0, 5, -1, 2):
        expected = saved[i][1]
        assert np.array_equal(read_raw_forces(path, i), expected)
    frames = read_frames_with_raw_forces(path)
    assert len(frames) == 6
    for frame, (_, raw, applied) in zip(frames, saved, strict=True):
        assert np.array_equal(np.asarray(frame.info[RAW_FORCES_INFO_KEY]), raw)
        # The constraint-applied forces stay the frame calculator forces.
        assert np.array_equal(frame.get_forces(), applied)
    # Generic ASE still reads structure, momenta and applied forces.
    plain = read(path, ":")
    assert np.array_equal(plain[2].get_forces(), saved[2][2])
    assert np.array_equal(plain[2].get_momenta(), saved[2][0].get_momenta())
    if storage == "binary":
        assert RAW_FORCES_INFO_KEY not in plain[2].info


def test_binary_is_smaller_and_numbers_written_once(tmp_path) -> None:
    sizes = {}
    for storage in ("json", "binary"):
        path = tmp_path / f"{storage}.traj"
        _write(path, storage, 8, np.random.default_rng(5))
        sizes[storage] = os.path.getsize(path)
    assert sizes["binary"] < 0.8 * sizes["json"]


@pytest.mark.parametrize("storage", ["json", "binary"])
def test_append_after_close_keeps_format_and_frames(tmp_path, storage) -> None:
    path = tmp_path / "t.traj"
    rng = np.random.default_rng(11)
    first = _write(path, storage, 3, rng)
    second = _write(path, storage, 2, rng, mode="a")
    saved = first + second
    assert detect_raw_forces_storage(path) == storage
    for i, (_, raw, _) in enumerate(saved):
        assert np.array_equal(read_raw_forces(path, i), raw)
    assert len(read(path, ":")) == 5


def test_append_with_a_different_storage_mode_fails_closed(tmp_path) -> None:
    path = tmp_path / "t.traj"
    _write(path, "binary", 2, np.random.default_rng(1))
    with pytest.raises(RawForcesStorageError):
        open_trajectory_writer(path, mode="a", raw_forces_storage="json")
    _write(path, "json", 1, np.random.default_rng(1), mode="w")
    with pytest.raises(RawForcesStorageError):
        open_trajectory_writer(path, mode="a", raw_forces_storage="binary")


def test_invalid_mode_and_missing_or_misshaped_raw_forces(tmp_path) -> None:
    with pytest.raises(RawForcesStorageError):
        open_trajectory_writer(tmp_path / "x.traj", raw_forces_storage="float32")
    writer = open_trajectory_writer(tmp_path / "y.traj", raw_forces_storage="binary")
    atoms, raw, _ = _frame(np.random.default_rng(0))
    with pytest.raises(RawForcesStorageError):
        writer.write(atoms)
    with pytest.raises(RawForcesStorageError):
        writer.write(atoms, raw_forces=raw[:-1])
    writer.close()


def test_legacy_json_file_without_raw_forces_reads_none(tmp_path) -> None:
    path = tmp_path / "old.traj"
    writer = open_trajectory_writer(path)
    atoms, _, _ = _frame(np.random.default_rng(2))
    writer.write(atoms)
    writer.close()
    assert detect_raw_forces_storage(path) is None
    assert read_raw_forces(path, 0) is None
    assert RAW_FORCES_INFO_KEY not in read_frames_with_raw_forces(path)[0].info
    assert RAW_FORCES_ULM_FIELD not in str(read(path).info)


@pytest.mark.parametrize("storage", ["json", "binary"])
def test_async_md_output_writes_selected_storage(tmp_path, storage) -> None:
    rng = np.random.default_rng(9)
    stream = AsyncMDOutputWriter(
        trajectory_path=tmp_path / "raw" / "trajectory.traj",
        csv_path=tmp_path / "raw" / "md.csv",
        write_trajectory=True,
        xdatcar_writer=None,
        xdatcar_path=tmp_path / "XDATCAR",
        outcar_writer=None,
        raw_forces_storage=storage,
    )
    raws = []
    for step in range(4):
        atoms, raw, _ = _frame(rng)
        if storage == "json":
            atoms.info[RAW_FORCES_INFO_KEY] = raw.tolist()
        raws.append(raw)
        summary = {
            "step": step,
            "time_fs": float(step),
            "phase": "production",
            "energy": 0.0,
            "kinetic_energy": 0.0,
            "total_energy": 0.0,
            "temperature": 0.0,
            "volume": 1000.0,
            "max_force_raw_eV_A": 0.0,
            "max_force_applied_eV_A": 0.0,
            "configurational_stress": None,
            "total_stress": None,
            "configurational_pressure_gpa": None,
            "total_pressure_gpa": None,
        }
        stream.submit(
            MDFrameSnapshot(
                atoms=atoms,
                summary=summary,
                outcar_forces=None,
                raw_forces=raw if storage == "binary" else None,
            )
        )
    stream.close()
    path = tmp_path / "raw" / "trajectory.traj"
    assert detect_raw_forces_storage(path) == storage
    for i, raw in enumerate(raws):
        assert np.array_equal(read_raw_forces(path, i), raw)
