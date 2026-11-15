"""New-task wizard: the TUI's first entry (spec R5 of the #61 usage round).

Three questions, then a confirmation:

1. calculation type (``sp``/``opt``/``md``/``neb``/``confsearch``),
2. structure file,
3. model: the backend (``MODEL_TYPE``) first, then a file from the repository's
   ``models/`` listing (or a path typed in).

The confirmation shows the derived folder and the exact command
``cd <folder> && mliport run``. Both buttons write the same folder; "Create &
Run" then calls :func:`mliport.task_folder.run_task_folder`, which is
``mliport.cli.main(["run", ...])` - there is no second run implementation.
"""

from __future__ import annotations

from pathlib import Path

from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Input, ListItem, ListView, Select, Static

from mliport.task_folder import (
    TaskFolderError,
    available_models,
    create_task_folder,
    default_folder_name,
    remaining_placeholders,
    run_task_folder,
)

_CALC_LABELS = (
    ("sp", "Single point (SP)"),
    ("opt", "Structure optimization (OPT)"),
    ("md", "Molecular dynamics (MD)"),
    ("neb", "NEB (CALCULATION = NEB)"),
    ("confsearch", "Configuration search (CONFSEARCH)"),
)


class NewTaskScreen(Screen):
    """Create a task folder, then run it the same way ``mliport run`` does."""

    def compose(self):
        """Compose the wizard."""
        yield Container(
            Static("New task", id="title"),
            Static(
                "1. Calculation type, 2. structure file, 3. model. "
                "The folder gets an INCAR.mliport and a copy of the structure.",
                id="subtitle",
            ),
            Select(
                [(label, value) for value, label in _CALC_LABELS],
                prompt="Calculation type",
                value="sp",
                allow_blank=False,
                id="wizard-calc-type",
            ),
            Input(
                placeholder="Structure file (POSCAR, .cif, .vasp, .xyz)",
                id="wizard-structure",
            ),
            Select(
                [
                    ("UMA (fairchem)", "uma"),
                    ("MACE", "mace"),
                    ("DPA", "dpa"),
                    ("GRACE", "grace"),
                    ("NEP", "nep"),
                ],
                prompt="Model backend (MODEL_TYPE)",
                value="mace",
                allow_blank=False,
                id="wizard-model-type",
            ),
            Input(
                placeholder=(
                    "Model file: a name from models/ or a path to a checkpoint"
                ),
                id="wizard-model-path",
            ),
            Static("", id="wizard-model-list-title"),
            ListView(id="wizard-model-list"),
            Static("", id="wizard-folder-hint"),
            Input(placeholder="Folder to create", id="wizard-folder"),
            Static("", id="wizard-status"),
            Horizontal(
                Button("◀ Back", id="wizard-back"),
                Button("Create folder", id="wizard-create"),
                Button("Create && Run", variant="success", id="wizard-create-run"),
                id="wizard-buttons",
            ),
            id="main-container",
        )

    def on_mount(self) -> None:
        """Populate the model list and the default structure/folder."""
        self._refresh_model_list()
        self._suggest_structure()
        self._refresh_folder_default()

    # -- model list ------------------------------------------------------
    def _refresh_model_list(self) -> None:
        listing = self.query_one("#wizard-model-list", ListView)
        listing.clear()
        # Widget ids allow only letters, digits, "_" and "-", so the model file
        # name is kept in a mapping instead of inside the id.
        self._model_items: dict[str, str] = {}
        try:
            models = available_models()
        except ValueError as exc:  # unknown MLIPORT_ROOT
            self._set_status(str(exc))
            models = []
        if not models:
            self.query_one("#wizard-model-list-title", Static).update(
                "No models/ listing available: type the model path above, or set "
                "MLIPORT_ROOT to the mliport checkout."
            )
            return
        self.query_one("#wizard-model-list-title", Static).update("Models in models/:")
        for index, name in enumerate(models):
            item_id = f"model-{index}"
            self._model_items[item_id] = name
            listing.append(ListItem(Static(name), id=item_id))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """A click on a listed model fills the model path."""
        item_id = event.item.id or ""
        name = getattr(self, "_model_items", {}).get(item_id)
        if name is None:
            return
        self.query_one("#wizard-model-path", Input).value = name
        self._refresh_folder_default()

    # -- defaults --------------------------------------------------------
    def _suggest_structure(self) -> None:
        field = self.query_one("#wizard-structure", Input)
        if field.value:
            return
        for candidate in ("POSCAR", "CONTCAR", "structure.cif", "structure.xyz"):
            if Path(candidate).is_file():
                field.value = candidate
                return
        for pattern in ("*.cif", "*.xyz", "*.vasp", "*.extxyz"):
            matches = sorted(Path.cwd().glob(pattern))
            if matches:
                field.value = str(matches[0])
                return

    def _refresh_folder_default(self) -> None:
        structure = self.query_one("#wizard-structure", Input).value.strip()
        calc_type = str(self.query_one("#wizard-calc-type", Select).value or "sp")
        field = self.query_one("#wizard-folder", Input)
        hint = self.query_one("#wizard-folder-hint", Static)
        if field.value.strip():
            hint.update(f"Create: {field.value.strip()}/INCAR.mliport")
            return
        if structure:
            suggested = default_folder_name(structure, calc_type)
            field.placeholder = f"Folder to create (default: {suggested})"
            hint.update(f"Create: {suggested}/INCAR.mliport")
        else:
            hint.update("Create: <structure>-<calc type>/INCAR.mliport")

    def on_input_changed(self, event: Input.Changed) -> None:  # noqa: N802
        """Keep the folder suggestion in step with the structure path."""
        if event.input.id in {"wizard-structure", "wizard-folder"}:
            self._refresh_folder_default()

    def on_select_changed(self, event: Select.Changed) -> None:  # noqa: N802
        """Keep the folder suggestion in step with the calculation type."""
        if event.select.id == "wizard-calc-type":
            self._refresh_folder_default()

    # -- actions ---------------------------------------------------------
    def _set_status(self, message: str) -> None:
        self.query_one("#wizard-status", Static).update(message)

    def _folder_value(self) -> str:
        field = self.query_one("#wizard-folder", Input)
        explicit = field.value.strip()
        if explicit:
            return explicit
        structure = self.query_one("#wizard-structure", Input).value.strip()
        calc_type = str(self.query_one("#wizard-calc-type", Select).value or "sp")
        return default_folder_name(structure or "task", calc_type)

    def _create(self):
        """Create the task folder, reporting every failure in the status line."""
        structure = self.query_one("#wizard-structure", Input).value.strip()
        model_path = self.query_one("#wizard-model-path", Input).value.strip()
        model_type = str(self.query_one("#wizard-model-type", Select).value or "")
        calc_type = str(self.query_one("#wizard-calc-type", Select).value or "sp")
        try:
            if not structure:
                raise TaskFolderError("Choose a structure file first.")
            if not model_path:
                raise TaskFolderError(
                    "Choose a model from models/ or type its path first."
                )
            folder = create_task_folder(
                self._folder_value(),
                calc_type=calc_type,
                structure=structure,
                model_type=model_type,
                model_path=model_path,
            )
        except TaskFolderError as exc:
            self._set_status(f"Error: {exc}")
            return None
        except OSError as exc:
            self._set_status(f"Error: {exc}")
            return None
        message = f"Created {folder.path}\nRun it with: {folder.command}"
        # Some calculation types need more than a structure and a model (NEB
        # endpoints, confsearch starts). Say so instead of letting the first
        # `mliport run` discover it.
        pending = remaining_placeholders(folder.incar.read_text(encoding="utf-8"))
        if pending:
            message += (
                "\nStill to fill in: " + ", ".join(pending) + " in " + folder.incar.name
            )
        self._set_status(message)
        self.app.notify(f"Task folder created: {folder.path}", title="Success")
        return folder

    def on_button_pressed(self, event: Button.Pressed) -> None:  # noqa: N802
        """Handle the three buttons."""
        button_id = event.button.id
        if button_id == "wizard-back":
            self.app.pop_screen()
            return
        folder = self._create()
        if folder is None:
            return
        if button_id == "wizard-create-run":
            code = run_task_folder(folder.path)
            self._set_status(f"`mliport run` exited with code {code}")


__all__ = ["NewTaskScreen"]
