"""Regression tests for scientific/output correctness found in the final audit."""

from __future__ import annotations

import json
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixSymmetry
from ase.io import write

from mliport.runners.batch import BatchRunner
from mliport.runners.optimization import OptimizationRunner
from mliport.runners.singlepoint import SinglePointRunner
from mliport.writers.json_writer import JsonWriter
from mliport.writers.xdatcar import XdatcarWriter


class _Wrapper:
    task = "molecule"
    has_stress = True

    def get_calculator(self):
        return _ZeroCalculator()

    def info(self):
        return {}


class _ZeroCalculator(Calculator):
    implemented_properties: ClassVar = ["energy", "forces", "stress"]

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(atoms), 3)),
            "stress": np.zeros(6),
        }


def test_cell_optimization_rejects_isolated_molecule(tmp_path):
    with pytest.raises(ValueError, match="isolated molecule"):
        OptimizationRunner(
            _Wrapper(), cell_opt=True, output_dir=tmp_path, verbose=False
        )


def test_cell_optimization_rejects_partial_pbc(tmp_path):
    class _BulkWrapper(_Wrapper):
        task = "bulk"

    atoms = Atoms(
        "Si2",
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=np.eye(3) * 5,
        pbc=[True, True, False],
    )
    runner = OptimizationRunner(
        _BulkWrapper(), cell_opt=True, output_dir=tmp_path, verbose=False
    )
    with pytest.raises(ValueError, match="full 3D periodicity"):
        runner.run(atoms)


def test_fix_symmetry_does_not_discard_existing_constraints(tmp_path):
    atoms = Atoms(
        "Si2",
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=np.eye(3) * 5,
        pbc=True,
    )
    atoms.set_constraint(FixAtoms(indices=[0]))

    class _BulkWrapper(_Wrapper):
        task = "bulk"

    class _InspectableRunner(OptimizationRunner):
        prepared = None

        def _prepare_atoms(self, source):
            self.prepared = super()._prepare_atoms(source)
            return self.prepared

    runner = _InspectableRunner(
        _BulkWrapper(),
        fix_symmetry=True,
        max_steps=0,
        output_dir=tmp_path,
        verbose=False,
    )
    runner.run(atoms)
    assert any(isinstance(c, FixAtoms) for c in runner.prepared.constraints)
    assert any(isinstance(c, FixSymmetry) for c in runner.prepared.constraints)


def test_json_keeps_zero_energy_per_atom(tmp_path):
    atoms = Atoms("He", positions=[[0, 0, 0]], cell=[5, 5, 5])
    path = tmp_path / "zero.json"
    JsonWriter().write(
        atoms,
        {"energy": 0.0, "forces": np.zeros((1, 3)), "time": 0.0},
        path,
    )
    data = json.loads(path.read_text())
    assert data["calculation"]["results"]["energy_per_atom"] == 0.0


def test_output_control_flags_affect_written_files(tmp_path):
    atoms = Atoms("He", positions=[[0, 0, 0]], cell=[5, 5, 5])
    runner = SinglePointRunner(
        _Wrapper(),
        output_dir=tmp_path,
        write_forces=False,
        write_stress=False,
        write_json=False,
        verbose=False,
    )
    runner.run(atoms)
    outcar = (tmp_path / "OUTCAR").read_text()
    assert " FORCES (eV/Å)" not in outcar
    assert " STRESS TENSOR" not in outcar
    assert not (tmp_path / "mliport_results.json").exists()


def test_explicit_charge_and_spin_override_structure_metadata(tmp_path):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]])
    atoms.info.update({"charge": 0, "spin": 1})
    runner = SinglePointRunner(
        _Wrapper(),
        output_dir=tmp_path,
        charge=-1,
        spin=2,
        verbose=False,
    )

    prepared = runner._prepare_atoms(atoms)

    assert prepared.info["charge"] == -1
    assert prepared.info["spin"] == 2
    # Preparation works on a copy and must not rewrite the caller's object.
    assert atoms.info == {"charge": 0, "spin": 1}


@pytest.mark.parametrize("pbc", [False, [True, True, False]])
def test_bulk_task_never_silently_enables_pbc(tmp_path, pbc):
    class _BulkWrapper(_Wrapper):
        task = "bulk"

    atoms = Atoms("Li", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=pbc)
    runner = SinglePointRunner(_BulkWrapper(), output_dir=tmp_path, verbose=False)
    with pytest.raises(ValueError, match="requires full 3D periodicity"):
        runner._prepare_atoms(atoms)
    assert np.array_equal(atoms.pbc, np.broadcast_to(np.asarray(pbc), (3,)))


@pytest.mark.parametrize("pbc", [True, [True, False, False]])
def test_molecule_task_never_silently_disables_pbc(tmp_path, pbc):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]], cell=[8, 8, 8], pbc=pbc)
    runner = SinglePointRunner(_Wrapper(), output_dir=tmp_path, verbose=False)
    with pytest.raises(ValueError, match="requires a nonperiodic structure"):
        runner._prepare_atoms(atoms)
    assert np.array_equal(atoms.pbc, np.broadcast_to(np.asarray(pbc), (3,)))


def test_unknown_task_fails_closed(tmp_path):
    class _UnknownWrapper(_Wrapper):
        task = "bulkk"

    atoms = Atoms("Li", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    runner = SinglePointRunner(_UnknownWrapper(), output_dir=tmp_path, verbose=False)
    with pytest.raises(ValueError, match="Unknown task"):
        runner._prepare_atoms(atoms)


def test_xdatcar_preserves_interleaved_symbol_blocks(tmp_path):
    atoms = Atoms(
        ["Li", "S", "Li"],
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 5,
        pbc=True,
    )
    path = tmp_path / "XDATCAR"
    writer = XdatcarWriter()
    writer.write_header(atoms, path)
    writer.append_frame(path, atoms)
    lines = path.read_text().splitlines()
    assert lines[5].split() == ["Li", "S", "Li"]
    assert lines[6].split() == ["1", "1", "1"]
    assert lines[7].startswith("Direct configuration=")


def test_xdatcar_requires_header(tmp_path):
    atoms = Atoms("Li", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    with pytest.raises(RuntimeError, match="write_header"):
        XdatcarWriter().append_frame(tmp_path / "XDATCAR", atoms)


def test_batch_default_discovery_and_explicit_pattern(tmp_path, monkeypatch):
    structures = tmp_path / "structures"
    structures.mkdir()
    atoms = Atoms("He", positions=[[0, 0, 0]])
    write(structures / "one.xyz", atoms)
    write(
        structures / "two.vasp",
        Atoms("He", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
        format="vasp",
    )
    runner = BatchRunner(_Wrapper(), output_dir=tmp_path / "out", verbose=False)
    seen = []

    def fake_run(files):
        seen.append({path.name for path in files})
        return {"total": len(files), "success": len(files), "failed": 0}

    monkeypatch.setattr(runner, "run_from_files", fake_run)
    runner.run_from_directory(structures)
    runner.run_from_directory(structures, pattern="*.xyz")
    assert seen == [{"one.xyz", "two.vasp"}, {"one.xyz"}]


def test_batch_same_stem_different_formats_do_not_overwrite(tmp_path):
    structures = tmp_path / "structures"
    structures.mkdir()
    atoms = Atoms("He", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    write(structures / "sample.xyz", atoms)
    write(structures / "sample.cif", atoms)
    out = tmp_path / "out"

    class _BulkWrapper(_Wrapper):
        task = "bulk"

    summary = BatchRunner(_BulkWrapper(), output_dir=out, verbose=False).run_from_files(
        [structures / "sample.xyz", structures / "sample.cif"]
    )
    assert summary["success"] == 2
    assert (out / "sample_xyz" / "mliport_results.json").exists()
    assert (out / "sample_cif" / "mliport_results.json").exists()


def test_batch_optimization_path_smoke(tmp_path):
    structure = tmp_path / "one.xyz"
    write(structure, Atoms("He", positions=[[0, 0, 0]], cell=[5, 5, 5]))
    summary = BatchRunner(
        _Wrapper(),
        calc_type="opt",
        output_dir=tmp_path / "batch-opt",
        verbose=False,
        max_steps=1,
    ).run_from_files([structure])
    assert summary["success"] == 1
    assert (tmp_path / "batch-opt" / "one" / "CONTCAR").exists()


def test_batch_rejects_unsafe_shared_calculator_threads(tmp_path):
    with pytest.raises(NotImplementedError, match="shared ASE calculator"):
        BatchRunner(_Wrapper(), output_dir=tmp_path, parallel=True, max_workers=2)
