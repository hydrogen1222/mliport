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


def test_confsearch_export_reexports_a_finished_run(tmp_path, capsys):
    from tests.mliport.mliport.confsearch.test_engine import _run

    output, results = _run(tmp_path, [(0.5, 0.5)], name="export-source")
    assert results["exports"]
    destination = tmp_path / "reexport"
    code = main(
        [
            "confsearch",
            "export",
            str(output),
            "--top-k",
            "1",
            "--output",
            str(destination),
        ]
    )
    assert code == 0
    rank = destination / "rank_001"
    for name in ("POSCAR", "structure.extxyz", "minimum.json", "cp2k_cell.inc"):
        assert (rank / name).is_file(), name
    assert "Re-exported 1 candidate(s)" in capsys.readouterr().out


class _SyntheticWrapper:
    """Backend-shaped wrapper around a synthetic calculator (no MLIP needed)."""

    task = "bulk"

    def __init__(self, calculator, model_path: Path) -> None:
        self._calculator = calculator
        self._model_path = model_path

    def info(self) -> dict:
        return {
            "model_type": "mace",
            "task": "bulk",
            "backend_version": "test",
            "framework_versions": {"torch": "test"},
            "default_dtype": "float64",
            "model_path": str(self._model_path),
        }

    def get_calculator(self):
        return self._calculator


def _workflow_setup(tmp_path: Path, *, cycles: int):
    from mliport.config import resolve_config
    from tests.mliport.mliport.confsearch.test_engine import _start

    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"identity model")
    start_file = tmp_path / "start.extxyz"
    write(start_file, _start((0.5, 0.5)), format="extxyz")
    resolved = resolve_config(
        calc_type="confsearch",
        cli={
            "model_type": "mace",
            "model_path": str(model_file),
            "task": "bulk",
            "cs_starts": str(start_file),
            "cs_adsorbate_indices": "16-18",
            "cs_cycles": cycles,
            "cs_hold_ps": 0.05,
            "cs_cool_ps": 0.05,
            "cs_snapshot_interval_ps": 0.02,
            "timestep": 0.5,
            "cs_temperature_high": 800.0,
            "cs_temperature_low": 300.0,
            "cs_quench_fmax": 0.01,
            "cs_quench_steps": 2000,
            "cs_refine": False,
            "cs_refine_fmax": 0.005,
            "seed": 7,
        },
    )
    return resolved, _SyntheticWrapper(EggCrateCalculator(), model_file)


def test_workflow_runs_the_full_pipeline_with_a_synthetic_model(tmp_path):
    from mliport.confsearch.workflow import run_confsearch_workflow

    resolved, wrapper = _workflow_setup(tmp_path, cycles=1)
    output = tmp_path / "workflow-run"
    result = run_confsearch_workflow(wrapper, resolved, output_dir=output)
    assert result.status == "completed"
    for name in (
        "run_context.json",
        "search_results.json",
        "run.log",
        "artifacts.json",
    ):
        assert (output / name).is_file(), name
    context = json.loads((output / "run_context.json").read_text())
    assert context["gradient_preflight"]["status"] == "passed"
    assert context["model"]["model_type"] == "mace"
    assert context["regions"]["adsorbate"] == [16, 17, 18]
    assert (output / "run.log").read_text().strip()


def test_workflow_extension_validates_then_archives_and_reruns_the_budget(tmp_path):
    """Spec 11.3 revision 5: the fingerprint is checked before anything moves."""
    from dataclasses import replace

    from mliport.confsearch.workflow import run_confsearch_workflow

    resolved, wrapper = _workflow_setup(tmp_path, cycles=1)
    output = tmp_path / "extend-run"
    first = run_confsearch_workflow(wrapper, resolved, output_dir=output)
    assert first.status == "completed"
    first_quenches = first.results["diagnostics"]["n_quenches"]
    assert not (output / "history").exists()

    resolved_extended = replace(
        resolved, run_options=dict(resolved.run_options, cs_cycles=2)
    )
    second = run_confsearch_workflow(
        wrapper,
        resolved_extended,
        output_dir=output,
        resume=True,
        allow_cycle_change=True,
    )
    assert second.status == "completed"
    # The workflow archived the previous results and recorded the extension.
    history = output / "history" / "extension_1"
    assert (history / "search_results.json").is_file()
    context = json.loads((output / "run_context.json").read_text())
    assert context["extensions"][0]["added_cycles"] == 1
    assert context["options"]["cs_cycles"] == 2
    # The extension actually re-ran the budget: more quenches and a new cycle
    # directory in the raw layout.
    assert second.results["diagnostics"]["n_quenches"] > first_quenches
    cycles_after = sorted(
        path.name for path in (output / "raw").glob("replica_*/cycle_*")
    )
    assert any(name.endswith("cycle_002") for name in cycles_after), cycles_after


def test_extend_cli_refuses_before_archiving_when_another_option_changed(
    tmp_path, capsys
):
    """A refused extension must leave the run directory untouched."""
    from mliport.confsearch.workflow import run_confsearch_workflow

    resolved, wrapper = _workflow_setup(tmp_path, cycles=1)
    output = tmp_path / "cli-extend-refused"
    run_confsearch_workflow(wrapper, resolved, output_dir=output)
    top_before = sorted(path.name for path in (output / "top").glob("*"))

    # A *different* model file is a different model identity (a moved copy of an
    # identical file would be accepted, spec 11.2 revision 5).
    other_model = tmp_path / "other-model.pt"
    other_model.write_bytes(b"a different model")
    code = main(
        [
            "confsearch",
            "extend",
            str(output),
            "--cycles",
            "1",
            "--model",
            str(other_model),
        ]
    )
    captured = capsys.readouterr()
    assert code != 0
    # Nothing was archived and the previous results are still in place.
    assert not (output / "history").exists()
    assert sorted(path.name for path in (output / "top").glob("*")) == top_before
    context = json.loads((output / "run_context.json").read_text())
    assert context["extensions"] == []
    assert context["options"]["cs_cycles"] == 1
    assert "fingerprint" in captured.out + captured.err


def test_extend_cli_adds_cycles_archives_and_reruns(tmp_path, capsys, monkeypatch):
    """CLI-level extend: archive after validation, strictly more quenches."""
    from mliport.confsearch.workflow import run_confsearch_workflow

    resolved, wrapper = _workflow_setup(tmp_path, cycles=1)
    output = tmp_path / "cli-extend"
    first = run_confsearch_workflow(wrapper, resolved, output_dir=output)
    first_quenches = first.results["diagnostics"]["n_quenches"]

    # The CLI builds a real backend; the synthetic calculator is substituted at
    # the engine seam so the CLI path itself is exercised.
    monkeypatch.setattr(
        "mliport.engine.CalculationEngine._create_calculator", lambda self: wrapper
    )
    code = main(["confsearch", "extend", str(output), "--cycles", "1"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "confsearch status: completed" in out

    history = output / "history" / "extension_1"
    assert (history / "search_results.json").is_file()
    context = json.loads((output / "run_context.json").read_text())
    assert context["extensions"][0]["added_cycles"] == 1
    assert context["options"]["cs_cycles"] == 2
    results = json.loads((output / "search_results.json").read_text())
    assert results["diagnostics"]["n_quenches"] > first_quenches
    assert any(
        path.name.endswith("cycle_002")
        for path in (output / "raw").glob("replica_*/cycle_*")
    )


def test_confsearch_merge_combines_comparable_runs(tmp_path, capsys):
    from tests.mliport.mliport.confsearch.test_engine import _run

    first_dir, first = _run(tmp_path, [(0.5, 0.5)], name="merge-a")
    second_dir, second = _run(tmp_path, [(0.5, 0.5)], name="merge-b")
    assert first["ranking"]["n_rankable"] >= 1
    destination = tmp_path / "merged"
    code = main(
        [
            "confsearch",
            "merge",
            str(first_dir),
            str(second_dir),
            "--output",
            str(destination),
        ]
    )
    assert code == 0
    merged = json.loads((destination / "search_results.json").read_text())
    assert merged["status"] == "completed"
    assert merged["ranking"]["n_rankable"] >= 1
    assert merged["merge"]["sources"] == ["merge-a", "merge-b"]
    assert merged["merge"]["refinement_reused"] is False
    # The same basin from two sources must merge into one unique minimum with
    # the combined hit count.
    entries = merged["ranking"]["refined"] + merged["ranking"]["unrefined"]
    assert max(entry["hit_count"] for entry in entries) >= 2
    # Spec 11.4: the origin carries the source run id (not a renumbered replica).
    assert all(
        entry["origin"]["source"] in {"merge:merge-a", "merge:merge-b"}
        for entry in entries
    )
    assert "Merged 2 run(s)" in capsys.readouterr().out


def test_confsearch_merge_refuses_incomparable_runs(tmp_path, capsys):
    from tests.mliport.mliport.confsearch.test_engine import _run

    first_dir, _first = _run(tmp_path, [(0.5, 0.5)], name="cmp-a")
    second_dir, _second = _run(
        tmp_path, [(0.5, 0.5)], options=None, name="cmp-b", seed=99
    )
    context = json.loads((second_dir / "run_context.json").read_text())
    context["options"]["cs_dedup_energy_tol"] = 0.5
    (second_dir / "run_context.json").write_text(json.dumps(context))
    code = main(
        [
            "confsearch",
            "merge",
            str(first_dir),
            str(second_dir),
            "--output",
            str(tmp_path / "merged-bad"),
        ]
    )
    assert code != 0
    assert "comparability key differs" in capsys.readouterr().out


def test_confsearch_resume_cli_accepts_a_recorded_run(tmp_path, capsys, monkeypatch):
    """A resume must rebuild the config layer from run_context.json.

    Regression: the recorded ``cs_starts`` is a list, and handing that list
    straight back to the config layer made the resolver validate the
    stringified list as a path ("cs_starts: path not found").
    """
    from mliport.confsearch.workflow import run_confsearch_workflow
    from tests.mliport.mliport.confsearch.test_resume_identity import _workflow_setup

    resolved, wrapper = _workflow_setup(tmp_path, cycles=1)
    output = tmp_path / "resume-run"
    run_confsearch_workflow(wrapper, resolved, output_dir=output)

    # The engine builds a real backend; substitute the synthetic wrapper at the
    # engine seam so the CLI path itself is exercised.
    monkeypatch.setattr(
        "mliport.engine.CalculationEngine._create_calculator", lambda self: wrapper
    )
    code = main(["confsearch", "resume", str(output)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "confsearch status: completed" in out
    context = json.loads((output / "run_context.json").read_text())
    assert len(context["resumes"]) == 1


def test_confsearch_merge_refuses_a_different_cell_with_the_same_index_sets(
    tmp_path, capsys
):
    """Spec 11.4 revision 5: index sets alone are not enough."""
    from tests.mliport.mliport.confsearch.test_engine import _run

    first_dir, _first = _run(tmp_path, [(0.5, 0.5)], name="cell-a")
    second_dir, _second = _run(tmp_path, [(0.5, 0.5)], name="cell-b")
    context = json.loads((second_dir / "run_context.json").read_text())
    cell = context["system"]["cell_A"]
    # Same index sets, same regions, same everything -- except the cell.
    cell[2][2] = float(cell[2][2]) + 1.0
    (second_dir / "run_context.json").write_text(json.dumps(context))
    code = main(
        [
            "confsearch",
            "merge",
            str(first_dir),
            str(second_dir),
            "--output",
            str(tmp_path / "merged-cell"),
        ]
    )
    assert code != 0
    out = capsys.readouterr().out
    assert "comparability key differs" in out
    assert "system" in out


def test_confsearch_merge_refuses_a_context_without_the_system_block(tmp_path, capsys):
    """A run recorded before revision 5 is refused, not silently compared."""
    from tests.mliport.mliport.confsearch.test_engine import _run

    first_dir, _first = _run(tmp_path, [(0.5, 0.5)], name="oldsys-a")
    second_dir, _second = _run(tmp_path, [(0.5, 0.5)], name="oldsys-b")
    context = json.loads((second_dir / "run_context.json").read_text())
    context.pop("system", None)
    (second_dir / "run_context.json").write_text(json.dumps(context))
    code = main(
        [
            "confsearch",
            "merge",
            str(first_dir),
            str(second_dir),
            "--output",
            str(tmp_path / "merged-oldsys"),
        ]
    )
    assert code != 0
    assert "does not record the atomic numbers" in capsys.readouterr().out
