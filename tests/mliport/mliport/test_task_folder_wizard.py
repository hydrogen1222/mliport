"""TUI new-task wizard and task-folder creation (#61 R5).

Tracking: #61 (task folder + INCAR + ``models/``), delivery 4.

R5.1 the main screen's first entry is a wizard that asks three things:
     calculation type, structure file, model (backend first, then a file from the
     ``models/`` listing).
R5.2 the wizard writes a folder with ``INCAR.mliport`` from the *same* template
     generator plus a byte-for-byte copy of the structure under the name the
     ``run`` lookup expects, prints ``cd <folder> && mliport run`` and its
     "Create & Run" button goes through ``mliport run`` itself.
R5.3 the previous screens stay reachable under "Advanced".
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mliport.config.incar import IncarConfig
from mliport.config.resolver import resolve_config
from mliport.task_folder import (
    TaskFolderError,
    available_models,
    create_task_folder,
    default_folder_name,
    incar_text_for,
    remaining_placeholders,
    run_task_folder,
    structure_target_name,
    task_folder_command,
)

CIF_TEXT = """data_test
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_chemical_formula_sum 'He1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
He1 He 0.0 0.0 0.0
"""


def _write_structure(folder: Path, name: str = "slab.cif") -> Path:
    path = folder / name
    path.write_text(CIF_TEXT, encoding="utf-8")
    return path


def _write_model(folder: Path, name: str = "mace.model") -> Path:
    path = folder / name
    path.write_text("weights\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# R5.2 - the folder contents
# ---------------------------------------------------------------------------


def test_create_task_folder_writes_incar_and_a_byte_identical_structure(
    tmp_path: Path,
) -> None:
    structure = _write_structure(tmp_path)
    model = _write_model(tmp_path)

    folder = create_task_folder(
        tmp_path / "task",
        calc_type="sp",
        structure=structure,
        model_type="mace",
        model_path=model,
    )

    assert folder.incar.is_file()
    assert folder.structure.name == "structure.cif"
    assert folder.structure.read_bytes() == structure.read_bytes()

    text = folder.incar.read_text(encoding="utf-8")
    assert "CALC_TYPE = SP" in text
    assert "MODEL_TYPE = MACE" in text
    assert f"MODEL_PATH = {model.resolve()}" in text
    # Only the task decision is left to the user (never a model or a structure).
    assert remaining_placeholders(text) == ["TASK"]
    assert folder.command == f"cd {folder.path} && mliport run"


def test_the_written_incar_passes_the_same_validation_as_mliport_run(
    tmp_path: Path,
) -> None:
    """The wizard cannot emit an INCAR that `run` rejects once filled in.

    The wizard deliberately leaves `TASK` (and `HEAD`) for the user: the UMA
    task family and the other backends' PBC semantic are model decisions, not
    template defaults that a one-click run may inherit silently.
    """
    structure = _write_structure(tmp_path, "POSCAR")
    model = _write_model(tmp_path)
    folder = create_task_folder(
        tmp_path / "task",
        calc_type="md",
        structure=structure,
        model_type="uma",
        model_path=model,
    )

    config = IncarConfig.from_file(folder.incar)
    # Unfilled, the run must fail closed on the placeholder rather than guess.
    assert remaining_placeholders(folder.incar.read_text(encoding="utf-8")) == ["TASK"]
    assert config.validate() != []

    filled = folder.incar.read_text(encoding="utf-8").replace(
        "TASK = REQUIRED", "TASK = omat"
    )
    folder.incar.write_text(filled, encoding="utf-8")
    config = IncarConfig.from_file(folder.incar)
    assert config.validate() == []
    resolved = resolve_config(calc_type="md", incar=config)
    assert resolved.model_type == "uma"
    assert resolved.task == "omat"
    assert resolved.model_path == str(model.resolve())


def test_a_models_file_name_stays_a_name_so_the_models_lookup_applies() -> None:
    """The wizard must not turn `models/x.model` into a CWD-relative path."""
    text = incar_text_for("sp", model_type="mace", model_path="mace.model")
    assert "MODEL_PATH = mace.model" in text
    assert remaining_placeholders(text) == ["TASK"]


def test_unfilled_placeholders_are_reported(tmp_path: Path) -> None:
    """A created INCAR says exactly what the user still has to decide."""
    neb = incar_text_for("neb", model_type="mace", model_path="mace.model")
    assert remaining_placeholders(neb) == ["NEB_FINAL", "NEB_INITIAL", "TASK"]
    confsearch = incar_text_for(
        "confsearch", model_type="mace", model_path="mace.model"
    )
    assert remaining_placeholders(confsearch) == [
        "CS_ADSORBATE_INDICES",
        "CS_STARTS",
        "TASK",
    ]
    # DPA has a model branch as well as a task.
    dpa = incar_text_for("sp", model_type="dpa", model_path="dpa.pt")
    assert remaining_placeholders(dpa) == ["HEAD", "TASK"]


def test_a_task_or_head_supplied_by_the_caller_is_written(tmp_path: Path) -> None:
    text = incar_text_for(
        "sp", model_type="dpa", model_path="dpa.pt", task="bulk", head="Omat24"
    )
    assert "TASK = bulk" in text
    assert "HEAD = Omat24" in text
    assert remaining_placeholders(text) == []


def test_a_models_subdirectory_entry_is_resolved_inside_models(
    tmp_path: Path, monkeypatch
) -> None:
    """The picker lists `uma/s1.pt`; that must not become a CWD-relative path."""
    from mliport.model_lookup import MLIPORT_ROOT_ENV

    root = tmp_path / "checkout"
    (root / "models" / "uma").mkdir(parents=True)
    model = root / "models" / "uma" / "s1.pt"
    model.write_text("w", encoding="utf-8")
    monkeypatch.setenv(MLIPORT_ROOT_ENV, str(root))
    monkeypatch.chdir(tmp_path)

    text = incar_text_for("sp", model_type="uma", model_path="uma/s1.pt")
    assert f"MODEL_PATH = {model.resolve()}" in text


def test_a_separator_value_that_exists_nowhere_fails_closed(tmp_path) -> None:
    with pytest.raises(TaskFolderError, match="was not found"):
        incar_text_for("sp", model_type="mace", model_path=str(tmp_path / "nope.pt"))


def test_the_incar_body_is_the_shared_template(tmp_path: Path) -> None:
    model = _write_model(tmp_path)
    text = incar_text_for("opt", model_type="mace", model_path=model)
    # The generated template's sections and key comments survive unchanged.
    assert "# Optimization Settings" in text
    assert "OPT_ALGO = FIRE" in text
    assert "# Output Settings" in text


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("POSCAR", "POSCAR"),
        ("CONTCAR", "POSCAR"),
        ("slab.vasp", "POSCAR"),
        ("slab.cif", "structure.cif"),
        ("molecule.xyz", "structure.xyz"),
        ("traj.extxyz", "structure.xyz"),
    ],
)
def test_structure_names_are_normalized_without_touching_content(
    tmp_path: Path, source: str, expected: str
) -> None:
    assert structure_target_name(source) == expected
    structure = tmp_path / source
    structure.write_text(CIF_TEXT, encoding="utf-8")
    model = _write_model(tmp_path)
    folder = create_task_folder(
        tmp_path / "task",
        calc_type="sp",
        structure=structure,
        model_type="mace",
        model_path=model,
    )
    assert folder.structure.name == expected
    assert folder.structure.read_bytes() == structure.read_bytes()


def test_an_unsupported_structure_suffix_fails_closed(tmp_path: Path) -> None:
    structure = tmp_path / "input.dat"
    structure.write_text("x", encoding="utf-8")
    model = _write_model(tmp_path)
    with pytest.raises(TaskFolderError, match="Cannot tell what kind of structure"):
        create_task_folder(
            tmp_path / "task",
            calc_type="sp",
            structure=structure,
            model_type="mace",
            model_path=model,
        )


def test_an_existing_task_folder_is_not_overwritten(tmp_path: Path) -> None:
    structure = _write_structure(tmp_path)
    model = _write_model(tmp_path)
    destination = tmp_path / "task"
    create_task_folder(
        destination,
        calc_type="sp",
        structure=structure,
        model_type="mace",
        model_path=model,
    )

    with pytest.raises(TaskFolderError, match="not empty"):
        create_task_folder(
            destination,
            calc_type="md",
            structure=structure,
            model_type="mace",
            model_path=model,
        )


def test_unknown_calc_type_is_refused(tmp_path: Path) -> None:
    model = _write_model(tmp_path)
    with pytest.raises(TaskFolderError, match="Unknown calculation type"):
        incar_text_for("phonon", model_type="mace", model_path=model)


def test_default_folder_name_is_derived_from_the_structure(tmp_path: Path) -> None:
    assert default_folder_name("slab.cif", "md") == "slab-md"
    assert default_folder_name("POSCAR", "opt") == "task-opt"


# ---------------------------------------------------------------------------
# R5.2 - "run now" is `mliport run`
# ---------------------------------------------------------------------------


def test_task_folder_command_is_the_run_argv(tmp_path: Path) -> None:
    assert task_folder_command(tmp_path) == [
        "run",
        "--incar",
        str((tmp_path / "INCAR.mliport").resolve()),
    ]


def test_run_task_folder_delegates_to_mliport_run(tmp_path: Path, monkeypatch) -> None:
    structure = _write_structure(tmp_path)
    model = _write_model(tmp_path)
    folder = create_task_folder(
        tmp_path / "task",
        calc_type="sp",
        structure=structure,
        model_type="mace",
        model_path=model,
    )
    calls: list[dict] = []

    def fake_main(argv):
        calls.append({"argv": argv, "cwd": Path.cwd()})
        return 0

    monkeypatch.setattr("mliport.cli.main", fake_main)
    assert run_task_folder(folder.path) == 0
    assert calls == [
        {
            "argv": ["run", "--incar", str(folder.path / "INCAR.mliport")],
            "cwd": folder.path,
        }
    ]


# ---------------------------------------------------------------------------
# R5.1 / R5.3 - the wizard and the navigation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_main_screen_offers_the_wizard_first() -> None:
    from textual.widgets import ListView

    from mliport.tui.app import MliportApp

    app = MliportApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        items = [
            item.id
            for item in app.screen.query_one("#calc-type-list", ListView).children
        ]
    assert items[0] == "new-task"
    assert "advanced" in items
    assert "exit" in items


@pytest.mark.asyncio()
async def test_wizard_creates_the_task_folder_and_shows_the_command(
    tmp_path: Path, monkeypatch
) -> None:
    from textual.widgets import Input, Select

    from mliport.tui.app import MliportApp
    from mliport.tui.wizard_screen import NewTaskScreen

    structure = _write_structure(tmp_path)
    model = _write_model(tmp_path)
    monkeypatch.chdir(tmp_path)

    app = MliportApp()
    async with app.run_test(size=(120, 50)) as pilot:
        screen = NewTaskScreen()
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#wizard-calc-type", Select).value = "md"
        screen.query_one("#wizard-structure", Input).value = str(structure)
        screen.query_one("#wizard-model-type", Select).value = "mace"
        screen.query_one("#wizard-model-path", Input).value = str(model)
        screen.query_one("#wizard-folder", Input).value = str(tmp_path / "task")
        await pilot.pause()
        await pilot.click("#wizard-create")
        await pilot.pause()
        status = str(screen.query_one("#wizard-status").render())

    task = tmp_path / "task"
    assert (task / "INCAR.mliport").is_file()
    assert (task / "structure.cif").read_bytes() == structure.read_bytes()
    assert "cd " in status and "mliport run" in status
    # no run was started by "Create folder"
    assert not (task / "results").exists()


@pytest.mark.asyncio()
async def test_wizard_create_and_run_uses_the_shared_entry_point(
    tmp_path: Path, monkeypatch
) -> None:
    from textual.widgets import Input

    from mliport.tui import wizard_screen
    from mliport.tui.app import MliportApp

    structure = _write_structure(tmp_path)
    model = _write_model(tmp_path)
    monkeypatch.chdir(tmp_path)
    calls: list[Path] = []
    monkeypatch.setattr(
        wizard_screen, "run_task_folder", lambda path: calls.append(Path(path)) or 0
    )

    app = MliportApp()
    async with app.run_test(size=(120, 50)) as pilot:
        screen = wizard_screen.NewTaskScreen()
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#wizard-structure", Input).value = str(structure)
        screen.query_one("#wizard-model-path", Input).value = str(model)
        screen.query_one("#wizard-folder", Input).value = str(tmp_path / "task")
        await pilot.pause()
        await pilot.click("#wizard-create-run")
        await pilot.pause()
        status = str(screen.query_one("#wizard-status").render())

    assert calls == [(tmp_path / "task").resolve()]
    assert "exited with code 0" in status


@pytest.mark.asyncio()
async def test_wizard_model_list_comes_from_models_directory(
    tmp_path: Path, monkeypatch
) -> None:
    from textual.widgets import ListView

    from mliport.model_lookup import MLIPORT_ROOT_ENV
    from mliport.tui.app import MliportApp
    from mliport.tui.wizard_screen import NewTaskScreen

    root = tmp_path / "checkout"
    (root / "models").mkdir(parents=True)
    (root / "models" / "mace.model").write_text("w", encoding="utf-8")
    (root / "models" / "uma").mkdir()
    (root / "models" / "uma" / "uma.pt").write_text("w", encoding="utf-8")
    monkeypatch.setenv(MLIPORT_ROOT_ENV, str(root))
    monkeypatch.chdir(tmp_path)
    assert available_models() == ["mace.model", "uma/uma.pt"]

    app = MliportApp()
    async with app.run_test(size=(120, 50)) as pilot:
        screen = NewTaskScreen()
        await app.push_screen(screen)
        await pilot.pause()
        listed = [
            item.id
            for item in screen.query_one("#wizard-model-list", ListView).children
        ]
    names = [screen._model_items[item_id] for item_id in listed]
    assert names == ["mace.model", "uma/uma.pt"]


@pytest.mark.asyncio()
async def test_advanced_menu_keeps_the_previous_screens() -> None:
    from textual.widgets import ListView

    from mliport.tui.app import MliportApp
    from mliport.tui.main_screen import AdvancedScreen

    app = MliportApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(AdvancedScreen())
        await pilot.pause()
        items = {
            item.id
            for item in app.screen.query_one("#advanced-list", ListView).children
        }
    assert {"sp", "opt", "md", "neb", "confsearch", "analysis", "jobs", "template"} <= (
        items
    )


def test_wizard_does_not_reimplement_the_run_command() -> None:
    """The wizard module must call `run_task_folder`, never build an argv."""
    source = (
        Path(__file__).parents[3] / "mliport" / "mliport" / "tui" / "wizard_screen.py"
    ).read_text(encoding="utf-8")
    assert "run_task_folder" in source
    assert "subprocess" not in source
    assert "build_mliport_command" not in source
    assert os.sep + "mliport" not in source.replace("mliport run", "")
