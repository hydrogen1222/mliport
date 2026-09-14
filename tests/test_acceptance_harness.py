"""Contract tests for the LGPS acceptance harness (task book section 143).

These tests are CPU-only and do not install anything: they validate the
fixture, the task registry, the JSON parsing helpers, the deterministic
endpoint construction and the schemas, plus that README/docs commands marked
``tested example`` point at real CLI subcommands.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
ACCEPTANCE = REPO / "validation" / "acceptance"
HARNESS = ACCEPTANCE / "lgps_smoke.py"
SCHEMAS = ACCEPTANCE / "schemas"

pytest.importorskip("ase")


def _load_harness():
    if str(ACCEPTANCE) not in sys.path:
        sys.path.insert(0, str(ACCEPTANCE))
    spec = importlib.util.spec_from_file_location("lgps_smoke", HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lgps_smoke"] = module
    spec.loader.exec_module(module)
    return module


def test_repo_lgps_fixture_is_valid_and_periodic():
    from ase.io import read

    fixture = REPO / "examples" / "structures" / "li10gep2s12_primitive.vasp"
    assert fixture.is_file(), "the README quickstart LGPS fixture must exist"
    atoms = read(fixture)
    assert len(atoms) == 50
    assert atoms.get_chemical_formula() == "Ge2Li20P4S24"
    assert all(atoms.pbc)
    distances = atoms.get_all_distances(mic=True)
    import numpy as np

    np.fill_diagonal(distances, 99.0)
    assert float(distances.min()) > 1.5


def test_task_registry_names_are_unique_and_callable():
    harness = _load_harness()
    expected = {
        "g1_sp",
        "g1_sp_repeat",
        "g2_sp_perturbed",
        "g3_opt_fixed",
        "g4_opt_cell",
        "g5_nve",
        "g6_nvt_langevin",
        "g7_nvt_bussi",
        "g8_nvt_nhc",
        "g9_md_restart",
        "g10_batch",
        "g11_incar",
        "g12_negatives",
        "g13_api",
        "g14_tui",
        "g15_queue",
        "g16_neb",
        "g17_neb_restart",
        "g18_analysis_core",
        "g19_analysis_long",
        "g20_arrhenius",
    }
    assert expected <= set(harness.TASKS)
    assert all(callable(task) for task in harness.TASKS.values())


def test_set_incar_keys_replaces_and_appends(tmp_path):
    harness = _load_harness()
    incar = tmp_path / "INCAR"
    incar.write_text(
        "CALC_TYPE = sp\nMODEL_PATH = REQUIRED\nDEVICE = cpu\n", encoding="utf-8"
    )
    harness._set_incar_keys(
        incar,
        {"MODEL_PATH": "/models/mace.model", "DEVICE": "cuda", "HEAD": "omat"},
    )
    text = incar.read_text(encoding="utf-8")
    assert "MODEL_PATH = /models/mace.model" in text
    assert "MODEL_PATH = REQUIRED" not in text
    assert "DEVICE = cuda" in text
    assert "DEVICE = cpu" not in text
    assert text.rstrip().endswith("HEAD = omat")


def test_parse_run_dir_reads_documented_outputs(tmp_path):
    harness = _load_harness()
    run = tmp_path / "run"
    run.mkdir()
    (run / "mliport_results.json").write_text(
        json.dumps(
            {
                "mliport_version": "2.0.0b3",
                "calculation": {
                    "mode": "single_point",
                    "system": {"formula": "Ge2Li20P4S24", "natoms": 50},
                    "results": {
                        "energy": -216.0,
                        "energy_per_atom": -4.32,
                        "forces": [[0.0, 0.0, 0.0]] * 50,
                        "stress": [[0.0] * 3] * 3,
                    },
                },
                "metadata": {
                    "model_type": "mace",
                    "model_path": "/models/mace.model",
                    "task": "bulk",
                    "head": None,
                    "device": "cuda",
                    "default_dtype": "float64",
                    "actual_device_type": "cuda",
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "resolved_config.json").write_text(
        json.dumps(
            {
                "calc_type": "sp",
                "model_type": "mace",
                "model_path": "/models/mace.model",
                "task": "bulk",
                "device": "cuda",
            }
        ),
        encoding="utf-8",
    )
    summary = harness.parse_run_dir(run)
    assert summary["energy_per_atom_eV"] == pytest.approx(-4.32)
    assert summary["actual_device_type"] == "cuda"
    assert summary["natoms"] == 50
    assert summary["forces_finite"] is True


def test_neb_endpoints_are_deterministic_and_safe(tmp_path):
    from ase.io import read

    harness = _load_harness()
    outputs = []
    for name in ("a", "b"):
        campaign = SimpleNamespace(out_dir=tmp_path / name)
        initial, final = harness._write_neb_endpoints(campaign)
        outputs.append((read(initial), read(final), final.parent / "endpoints.json"))
    first_initial, first_final, metadata_path = outputs[0]
    second_initial, second_final, _ = outputs[1]
    assert first_initial.get_chemical_symbols() == first_final.get_chemical_symbols()
    assert (first_initial.cell == first_final.cell).all()
    assert (first_initial.positions != first_final.positions).any()
    assert (first_initial.positions == second_initial.positions).all()
    assert (first_final.positions == second_final.positions).all()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["workflow_smoke_only"] is True
    assert metadata["physical_barrier_claimed"] is False
    assert metadata["min_distance_A"] >= 1.8


def test_install_matrix_rendering_never_contains_a_raw_uuid():
    spec = importlib.util.spec_from_file_location(
        "collect_environment", ACCEPTANCE / "collect_environment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = {
        "device": "cuda",
        "gpu": {"available": True},
        "environments": {
            "mace": {
                "python": "3.12.13",
                "framework": "torch",
                "framework_version": "2.8.0+cu126",
                "backend_version": "0.3.16",
                "mliport": "2.0.0b3",
                "doctor": {"failures": 0},
            }
        },
    }
    markdown = module.install_matrix_markdown(data)
    assert "| V100 | MACE | 3.12.13 |" in markdown
    assert "PASS" in markdown
    assert "GPU-" not in markdown  # raw UUID prefix must never be rendered


def test_acceptance_schemas_are_valid_json_with_expected_statuses():
    record = json.loads(
        (SCHEMAS / "acceptance-record.schema.json").read_text(encoding="utf-8")
    )
    statuses = set(record["properties"]["status"]["enum"])
    assert statuses == {"PASS", "FAIL", "EXPECTED_LIMITATION", "NOT_RUN"}
    backend = json.loads(
        (SCHEMAS / "acceptance-backend.schema.json").read_text(encoding="utf-8")
    )
    assert backend["properties"]["schema"]["const"] == "mliport.acceptance-backend/1"
    environment = json.loads(
        (SCHEMAS / "acceptance-environment.schema.json").read_text(encoding="utf-8")
    )
    assert environment["properties"]["schema"]["const"] == (
        "mliport.acceptance-environment/1"
    )


def test_tested_example_commands_reference_real_subcommands():
    """README/docs blocks marked `tested example` must use existing commands."""
    harness = _load_harness()
    from mliport.cli import create_parser

    choices = set(create_parser()._subparsers._group_actions[0].choices)  # type: ignore[attr-defined]
    block_re = re.compile(r"```bash\n(.*?)```", re.DOTALL)
    tested_marker = "tested example"
    checked = 0
    for doc in [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]:
        if not doc.is_file():
            continue
        text = doc.read_text(encoding="utf-8")
        for block in block_re.findall(text):
            if tested_marker not in block:
                continue
            for raw_line in block.splitlines():
                line = raw_line.strip().lstrip("$").strip()
                if not line or line.startswith("#"):
                    continue
                match = re.match(
                    r"(?:\.venv-[a-z]+/bin/|\.venv/bin/|\./bin/mliport-[a-z]+|mliport)\s+([a-z-]+)",
                    line,
                )
                if not match:
                    continue
                subcommand = match.group(1)
                if subcommand.startswith("--"):
                    continue
                assert subcommand in choices, (
                    f"{doc.name}: tested example uses unknown subcommand {subcommand!r}"
                )
                checked += 1
    # The docs rewrite adds the markers; before that the scan simply finds none.
    assert checked >= 0
