"""New-task wizard: the TUI's first entry (spec R5 of the #61 usage round).

Three questions, then a confirmation:

1. calculation type (``sp``/``opt``/``md``/``neb``/``confsearch``),
2. structure file,
3. model: the backend (``MODEL_TYPE``) first, a file from the repository's
   ``models/`` listing (or a path typed in), then the model task (and head).

The confirmation shows the derived folder and the exact command
``cd <folder> && mliport run``. Both buttons write the same folder; "Create &
Run" only starts a job when the folder has no unresolved ``REQUIRED`` value,
and then enqueues it through the shared scheduler-controlled JobManager
(worker/status/log path) instead of running the CLI on the Textual event loop.
"""

from __future__ import annotations

from pathlib import Path

from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Input, ListItem, ListView, Select, Static

from mliport.backend_selection import task_options_for
from mliport.task_folder import (
    TaskFolderError,
    available_models,
    create_task_folder,
    default_folder_name,
    submit_task_folder,
    task_folder_readiness,
)
from mliport.tui.run_screen import RunScreen

_CALC_LABELS = (
    ("sp", "Single point (SP)"),
    ("opt", "Structure optimization (OPT)"),
    ("md", "Molecular dynamics (MD)"),
    ("neb", "NEB (CALCULATION = NEB)"),
    ("confsearch", "Configuration search (CONFSEARCH)"),
)

_HEAD_BACKENDS = {"mace", "dpa"}


class NewTaskScreen(Screen):
    """Create a task folder, then run it the same way ``mliport run`` does."""

    def compose(self):
        """Compose the wizard."""
        yield Container(
            Static("New task", id="title"),
            Static(
                "1. Calculation type, 2. structure file, 3. model and task. "
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
            Select(
                [],
                prompt="Model task / system type (required)",
                value=Select.NULL,
                allow_blank=True,
                id="wizard-task",
            ),
            Input(
                placeholder="MACE head / DPA branch (required when the model has one)",
                id="wizard-head",
            ),
            Select(
                [
                    ("current (CLI default)", "current"),
                    ("legacy (DPA-2 pre-release runtime)", "legacy"),
                ],
                prompt="DPA runtime (blank = CLI default 'current')",
                value=Select.NULL,
                allow_blank=True,
                id="wizard-dpa-runtime",
            ),
            Static("", id="wizard-run-note"),
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
        """Populate the model list, task choices and the default paths."""
        self._submitted_folder: Path | None = None
        self._refresh_model_list()
        self._refresh_task_options()
        self._refresh_head_state()
        self._refresh_run_note()
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

    # -- task / head -----------------------------------------------------
    def _model_type(self) -> str:
        return str(self.query_one("#wizard-model-type", Select).value or "")

    def _refresh_task_options(self) -> None:
        """Offer the task families of the selected backend, with no guess."""
        selector = self.query_one("#wizard-task", Select)
        options = task_options_for(self._model_type())
        selector.set_options(options)
        # Always blank: the wizard must not silently inherit a template/default
        # task (AGENTS.md rule 3).  The user makes this decision explicitly.
        selector.value = Select.NULL

    def _refresh_head_state(self) -> None:
        """Show the head control only for backends that read it."""
        model_type = self._model_type()
        has_head = model_type in _HEAD_BACKENDS
        is_dpa = model_type == "dpa"
        for selector in ("#wizard-head",):
            self.query_one(selector).display = has_head
        self.query_one("#wizard-head", Input).disabled = not has_head
        self.query_one("#wizard-dpa-runtime", Select).display = is_dpa
        self.query_one("#wizard-dpa-runtime", Select).disabled = not is_dpa
        if model_type == "dpa":
            self.query_one("#wizard-head", Input).placeholder = (
                "DPA branch (required for multi-task checkpoints; "
                "blank leaves HEAD = REQUIRED)"
            )
        else:
            self.query_one(
                "#wizard-head", Input
            ).placeholder = (
                "MACE head (optional; blank leaves the model/in-template default)"
            )

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

    def _refresh_run_note(self) -> None:
        """State honestly whether Create & Run can execute this calc type."""
        calc_type = str(self.query_one("#wizard-calc-type", Select).value or "sp")
        note = self.query_one("#wizard-run-note", Static)
        if calc_type == "neb":
            note.update(
                "NEB needs NEB_INITIAL and NEB_FINAL (and the task) in the INCAR; "
                "until they are filled in, Create & Run creates the folder but "
                "does not start a run."
            )
        elif calc_type == "confsearch":
            note.update(
                "confsearch needs CS_STARTS + CS_ADSORBATE_INDICES (mode A) or "
                "slab/molecule inputs (mode B); until they are filled in, Create "
                "& Run creates the folder but does not start a run. Resume/extend/"
                "merge/export are CLI-only."
            )
        else:
            note.update(
                "Create & Run needs an explicit task; DPA multi-task checkpoints "
                "also need a head. Unresolved REQUIRED values are never run."
            )

    def on_input_changed(self, event: Input.Changed) -> None:  # noqa: N802
        """Keep the folder suggestion in step with the structure path."""
        if event.input.id in {"wizard-structure", "wizard-folder"}:
            self._refresh_folder_default()

    def on_select_changed(self, event: Select.Changed) -> None:  # noqa: N802
        """Keep backend-dependent choices and the folder suggestion current."""
        if event.select.id == "wizard-model-type":
            self._refresh_task_options()
            self._refresh_head_state()
            # Backend-specific leftovers must not follow the new model.
            self.query_one("#wizard-head", Input).value = ""
            self.query_one("#wizard-dpa-runtime", Select).value = Select.NULL
        elif event.select.id == "wizard-calc-type":
            self._refresh_run_note()
        if event.select.id in {"wizard-calc-type", "wizard-model-type"}:
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
        model_type = self._model_type()
        calc_type = str(self.query_one("#wizard-calc-type", Select).value or "sp")
        task_selector = self.query_one("#wizard-task", Select)
        task = None if task_selector.value is Select.NULL else str(task_selector.value)
        head = None
        if model_type in _HEAD_BACKENDS:
            head = self.query_one("#wizard-head", Input).value.strip() or None
        dpa_runtime = None
        if model_type == "dpa":
            runtime_selector = self.query_one("#wizard-dpa-runtime", Select)
            if runtime_selector.value is not Select.NULL:
                dpa_runtime = str(runtime_selector.value)
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
                task=task,
                head=head,
                dpa_runtime=dpa_runtime,
            )
        except TaskFolderError as exc:
            self._set_status(f"Error: {exc}")
            return None
        except OSError as exc:
            self._set_status(f"Error: {exc}")
            return None
        readiness = task_folder_readiness(folder.path)
        if readiness.runnable:
            message = (
                f"Created {folder.path} — ready to run.\n"
                f"Run it with: {folder.command}"
            )
        else:
            message = (
                f"Created {folder.path} — NOT runnable yet.\n"
                f"{readiness.explanation()}\n"
                f"After filling it in: {folder.command}"
            )
        self._set_status(message)
        self.app.notify(f"Task folder created: {folder.path}", title="Created")
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
        if button_id != "wizard-create-run":
            return
        readiness = task_folder_readiness(folder.path)
        if not readiness.runnable:
            # A created folder is never deleted just because a run was asked
            # for: it is the user's file, and it can be completed and run later.
            self._set_status(
                f"Created {folder.path} but did NOT start it.\n"
                f"{readiness.explanation()}\n"
                f"When complete: {folder.command}"
            )
            self.app.notify(
                "Task folder created but not started: " + readiness.explanation(),
                title="Not runnable yet",
                severity="warning",
            )
            return
        if self._submitted_folder == folder.path:
            # Double click / repeated Enter must not create a second job.
            self._set_status(f"Already submitted: {folder.path}")
            return
        try:
            job_id = submit_task_folder(folder.path)
        except (TaskFolderError, ValueError, OSError) as exc:
            self._set_status(f"Created {folder.path} but could not start it: {exc}")
            self.app.notify(f"Could not start the job: {exc}", severity="error")
            return
        self._submitted_folder = folder.path
        for button_id in ("wizard-create", "wizard-create-run"):
            self.query_one(f"#{button_id}", Button).disabled = True
        self.app.notify(
            f"Queued background job: {job_id}. Start the scheduler from Jobs "
            "or with `mliport queue start` if it is not running.",
            title="Queued",
        )
        self.app.push_screen(RunScreen(monitor_job_id=job_id))


__all__ = ["NewTaskScreen"]
