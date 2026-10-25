"""CLI/API integration for confsearch (spec section 12).

The dry-run and placement paths need no backend, so they are testable in CI;
the model-loading paths are covered by the engine tests with a synthetic
calculator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from mliport.cli import main
from tests.mliport.mliport.confsearch.calculators import EggCrateCalculator

SLAB_ATOMS = 16


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    slab = Atoms(
        "S" * SLAB_ATOMS,
        positions=[
            [0.5 + 3.0 * i, 0.5 + 3.0 * j, 1.0] for i in range(4) for j in range(4)
        ],
        cell=[12.0, 12.0, 20.0],
        pbc=True,
    )
    angle = np.deg2rad(109.47)
    molecule = Atoms(
        "CSO",
        positions=[
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [1.5 - 1.5 * np.cos(angle), 1.5 * np.sin(angle), 0.0],
        ],
        pbc=False,
    )
    slab_path = tmp_path / "slab.vasp"
    molecule_path = tmp_path / "molecule.xyz"
    write(slab_path, slab, format="vasp")
    write(molecule_path, molecule, format="xyz")
    return slab_path, molecule_path


def _start_file(tmp_path: Path) -> Path:
    slab_path, molecule_path = _write_inputs(tmp_path)
    slab = read(slab_path)
    molecule = read(molecule_path)
    combined = Atoms(
        numbers=np.concatenate([slab.numbers, molecule.numbers]),
        positions=np.concatenate(
            [slab.get_positions(), molecule.get_positions() + np.array([6.0, 6.0, 2.0])]
        ),
        cell=slab.cell,
        pbc=slab.pbc,
    )
    path = tmp_path / "start.extxyz"
    write(path, combined, format="extxyz")
    return path


@pytest.mark.parametrize(
    "argv",
    [
        ["confsearch", "--help"],
        ["confsearch", "run", "--help"],
        ["confsearch", "place", "--help"],
        ["confsearch", "resume", "--help"],
    ],
)
def test_confsearch_help_exits_cleanly(argv):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 0


def test_confsearch_place_writes_starts_without_a_model(tmp_path, capsys):
    slab_path, molecule_path = _write_inputs(tmp_path)
    output = tmp_path / "placed"
    code = main(
        [
            "confsearch",
            "place",
            "--slab",
            str(slab_path),
            "--molecule",
            str(molecule_path),
            "--n-starts",
            "2",
            "--seed",
            "5",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    starts = sorted((output / "starts").glob("placed_*.extxyz"))
    assert len(starts) == 2
    placements = sorted((output / "starts").glob("placed_*.placement.json"))
    assert len(placements) == 2
    payload = json.loads(placements[0].read_text())
    assert payload["kind"] == "contact"
    assert payload["min_pair_q"] == pytest.approx(1.5, abs=1e-3)
    placed = read(starts[0])
    assert len(placed) == SLAB_ATOMS + 3
    assert "Placed 2 start(s)" in capsys.readouterr().out


def test_confsearch_run_dry_run_needs_no_model(tmp_path, capsys):
    start = _start_file(tmp_path)
    output = tmp_path / "dry"
    code = main(
        [
            "confsearch",
            "run",
            str(start),
            "--adsorbate-indices",
            "16-18",
            "--model-type",
            "mace",
            "--task",
            "bulk",
            "--model",
            str(tmp_path / "missing.pt"),
            "--output",
            str(output),
            "--cycles",
            "2",
            "--dry-run",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "dry run" in out
    assert "Starts: 1" in out


def test_api_dry_run_matches_the_cli_contract(tmp_path, capsys):
    from mliport.api import run_confsearch

    start = _start_file(tmp_path)
    result = run_confsearch(
        starts=[str(start)],
        adsorbate_indices="16-18",
        model_type="mace",
        task="bulk",
        model_path=str(tmp_path / "missing.pt"),
        output_dir=tmp_path / "api-dry",
        cycles=3,
        dry_run=True,
        verbose=False,
    )
    assert result["status"] == "dry_run"
    assert result["n_starts"] == 1
    assert result["budget"]["cycles"] == 3
    assert result["budget"]["md_steps_total"] > 0


def test_confsearch_run_requires_an_output_directory(tmp_path, capsys):
    start = _start_file(tmp_path)
    code = main(
        [
            "confsearch",
            "run",
            str(start),
            "--adsorbate-indices",
            "16-18",
            "--model-type",
            "mace",
            "--task",
            "bulk",
            "--model",
            str(tmp_path / "missing.pt"),
            "--dry-run",
        ]
    )
    assert code != 0
    assert "explicit --output" in capsys.readouterr().out


def test_workflow_rejects_direct_force_models_unless_allowed(tmp_path, monkeypatch):
    """Section 6.3 gate: direct_forces must fail closed by default."""
    from mliport.config import resolve_config
    from mliport.confsearch.errors import ConfsearchGradientConsistencyError
    from mliport.confsearch.workflow import run_confsearch_workflow

    start = _start_file(tmp_path)
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"identity model")
    resolved = resolve_config(
        calc_type="confsearch",
        cli={
            "model_type": "mace",
            "model_path": str(model_file),
            "task": "bulk",
            "cs_starts": str(start),
            "cs_adsorbate_indices": "16-18",
        },
    )

    class Wrapper:
        task = "bulk"

        def info(self) -> dict:
            return {
                "model_type": "mace",
                "task": "bulk",
                "direct_forces": True,
                "backend_version": "test",
                "framework_versions": {"torch": "test"},
                "model_path": str(model_file),
            }

        def get_calculator(self):
            return EggCrateCalculator()

    with pytest.raises(ConfsearchGradientConsistencyError, match="direct_forces"):
        run_confsearch_workflow(Wrapper(), resolved, output_dir=tmp_path / "direct")
