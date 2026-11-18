"""Create and run a task folder (spec R5 of the #61 usage round).

A *task folder* is the unit of daily mliport use: an ``INCAR.mliport`` next to
the structure file, so that ``mliport run`` needs no arguments at all. This
module owns the task-folder operations behind the TUI wizard, and nothing else:

* :func:`create_task_folder` writes ``INCAR.mliport`` from the same template
  generator the CLI uses (with ``MODEL_TYPE`` / ``MODEL_PATH`` and the explicit
  task/head filled in) and copies the chosen structure *byte for byte* under the
  name the ``run`` structure lookup expects (``POSCAR`` / ``structure.cif`` /
  ``structure.xyz``);
* :func:`task_folder_readiness` keeps creation and executability separate: an
  unresolved ``REQUIRED`` value is reported, never replaced by a default;
* :func:`submit_task_folder` starts a ready folder as a detached background job
  whose command is ``mliport run`` itself, through the shared
  :class:`mliport.jobs.JobManager` worker/status/log path. The TUI never builds
  its own run command and never blocks its event loop on the calculation;
* :func:`run_task_folder` remains the blocking ``mliport.cli.main`` entry point
  for non-interactive callers.

All functions fail closed: an unknown calculation type, an unsupported structure
suffix, an existing non-empty destination, a template without the two model
lines, or a folder that is not ready is an error, never a guess.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

TASK_CALC_TYPES = ("sp", "opt", "md", "neb", "confsearch")

#: Source suffix -> the file name ``mliport run`` looks for (spec R5.2). The
#: content is copied unchanged; only the name is normalized.
STRUCTURE_TARGETS: dict[str, str] = {
    "": "POSCAR",  # POSCAR / CONTCAR style files have no suffix
    ".vasp": "POSCAR",
    ".poscar": "POSCAR",
    ".contcar": "POSCAR",
    ".cif": "structure.cif",
    ".mcif": "structure.cif",
    ".xyz": "structure.xyz",
    ".extxyz": "structure.xyz",
}

INCAR_NAME = "INCAR.mliport"


class TaskFolderError(ValueError):
    """A task folder could not be created without guessing."""


@dataclass(frozen=True)
class TaskFolderReadiness:
    """Whether a created task folder is ready to be executed as written.

    ``create_task_folder`` deliberately succeeds with ``REQUIRED`` placeholders
    for decisions the wizard must not guess (task/head and the extra inputs of
    NEB/confsearch).  Creation and executability are therefore separate facts:
    callers that want to start a run must consult :func:`task_folder_readiness`
    (or :func:`submit_task_folder`, which refuses to queue an unready folder).
    """

    calc_type: str
    placeholders: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def runnable(self) -> bool:
        return not self.placeholders and not self.errors

    def explanation(self) -> str:
        """A user-facing reason why the folder cannot run yet."""
        if self.placeholders:
            return (
                "This task folder still has unfilled REQUIRED values: "
                + ", ".join(self.placeholders)
                + ". Fill them in "
                + INCAR_NAME
                + " first."
            )
        if self.errors:
            return "This task folder is not valid yet: " + "; ".join(self.errors)
        return ""


@dataclass(frozen=True)
class TaskFolder:
    """A created task folder."""

    path: Path
    incar: Path
    structure: Path
    calc_type: str
    model_type: str
    model_path: str

    @property
    def command(self) -> str:
        """The command a user can paste into a shell."""
        return f"cd {self.path} && mliport run"


def structure_target_name(source: str | Path) -> str:
    """The task-folder file name for a structure file (content unchanged)."""
    name = Path(source).name
    if name.lower() in {"poscar", "contcar"}:
        return "POSCAR"
    suffix = Path(source).suffix.lower()
    target = STRUCTURE_TARGETS.get(suffix)
    if target is None:
        raise TaskFolderError(
            f"Cannot tell what kind of structure {name!r} is. Use a POSCAR-like "
            "file, a .cif, a .vasp or an .xyz/.extxyz file."
        )
    return target


def _model_reference_for_task_folder(model_path: str | Path) -> str:
    """The ``MODEL_PATH`` value to write for the model the user picked.

    A bare name stays a name, so the ``models/`` convention (#61 R2) resolves
    it at run time. Anything else must exist now, and is recorded absolutely so
    the task folder does not depend on the directory the wizard ran in. A
    ``models``-relative entry such as ``uma/s1.pt`` (what the picker lists) is
    resolved inside ``models/`` instead of against the working directory.
    """
    from mliport.model_lookup import (  # noqa: PLC0415
        is_file_name,
        mliport_root,
        models_dir,
    )

    raw = str(model_path).strip()
    if not raw:
        raise TaskFolderError("A model file (MODEL_PATH) is required.")
    if is_file_name(raw):
        # A bare name is either a models/ entry (resolved by `run` at run time)
        # or a file in the wizard's directory.  Record the latter absolutely so
        # the background worker does not depend on the wizard's cwd.
        local = Path(raw).expanduser()
        if local.is_file():
            return str(local.resolve())
        return raw
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    root = mliport_root()
    if root is not None:
        in_models = models_dir(root) / raw
        if in_models.exists():
            return str(in_models.resolve())
        raise TaskFolderError(
            f"Model {raw!r} was not found: neither {candidate.resolve()} nor "
            f"{in_models} exists."
        )
    raise TaskFolderError(
        f"Model {raw!r} was not found at {candidate.resolve()}, and no mliport "
        "root is known, so models/ cannot be searched. Pass the model path, or "
        "set MLIPORT_ROOT."
    )


def incar_text_for(
    calc_type: str,
    *,
    model_type: str,
    model_path: str | Path,
    task: str | None = None,
    head: str | None = None,
    dpa_runtime: str | None = None,
) -> str:
    """The template for ``calc_type`` with the chosen model filled in.

    The rest of the file is the generated template, byte for byte: the wizard
    must not maintain a second description of the options (spec R5.2).

    ``TASK`` (and ``HEAD`` where the backend has an active one) is written as a
    ``REQUIRED`` placeholder unless the caller supplies it: the task family of
    UMA and the PBC semantic of the other backends are model decisions that the
    wizard must not silently inherit from a template default (AGENTS.md rule 3,
    "never silently change model task/head").
    """
    from mliport.config.defaults import build_incar_default  # noqa: PLC0415

    calc = str(calc_type).strip().lower()
    if calc not in TASK_CALC_TYPES:
        raise TaskFolderError(
            f"Unknown calculation type {calc_type!r}; choose from "
            f"{', '.join(TASK_CALC_TYPES)}."
        )
    engine = str(model_type).strip().lower()
    if not engine:
        raise TaskFolderError("A model backend (MODEL_TYPE) is required.")
    resolved_model = _model_reference_for_task_folder(model_path)

    text = build_incar_default(calc, engine=engine)
    lines: list[str] = []
    seen_model_type = False
    seen_model_path = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("MODEL_TYPE"):
            lines.append(f"MODEL_TYPE = {engine.upper()}")
            seen_model_type = True
            continue
        if stripped.startswith("MODEL_PATH"):
            lines.append(f"MODEL_PATH = {resolved_model}")
            seen_model_path = True
            if engine == "dpa" and dpa_runtime:
                lines.append(f"DPA_RUNTIME = {dpa_runtime}")
            continue
        if stripped.startswith("TASK") and "=" in stripped:
            lines.append(f"TASK = {task.strip() if task else 'REQUIRED'}")
            continue
        if stripped.startswith("HEAD") and "=" in stripped:
            lines.append(f"HEAD = {head.strip() if head else 'REQUIRED'}")
            continue
        lines.append(line)
    if not (seen_model_type and seen_model_path):
        raise TaskFolderError(
            "The generated template does not declare MODEL_TYPE and MODEL_PATH; "
            "refusing to write an INCAR the wizard cannot fill in."
        )
    return "\n".join(lines) + "\n"


def create_task_folder(
    destination: str | Path,
    *,
    calc_type: str,
    structure: str | Path,
    model_type: str,
    model_path: str | Path,
    task: str | None = None,
    head: str | None = None,
    dpa_runtime: str | None = None,
    overwrite: bool = False,
) -> TaskFolder:
    """Create a task folder and return what was written.

    Args:
        destination: Directory to create (must be absent or empty).
        calc_type: ``sp``/``opt``/``md``/``neb``/``confsearch``.
        structure: Existing structure file; copied byte for byte.
        model_type: Backend chosen for ``MODEL_TYPE``.
        model_path: Model file chosen; recorded as an absolute path.
        task: Explicit model task/system semantic.  ``None`` leaves ``TASK``
            as a ``REQUIRED`` placeholder (the wizard then refuses to run it).
        head: Explicit MACE head/DPA branch.  ``None`` leaves ``HEAD`` as a
            ``REQUIRED`` placeholder when the template has that line.
        dpa_runtime: Explicit DPA runtime (``current``/``legacy``); ``None``
            keeps the CLI default strategy (``current``) without writing it.
        overwrite: Allow writing into an existing directory (only when it holds
            neither an INCAR nor a structure of ours).

    Raises:
        TaskFolderError: unknown calc type/suffix, missing inputs, or a
            destination that already holds a task.
    """
    source = Path(structure).expanduser()
    if not source.is_file():
        raise TaskFolderError(f"Structure file not found: {source}")
    target_name = structure_target_name(source)

    folder = Path(destination).expanduser()
    if folder.exists() and not folder.is_dir():
        raise TaskFolderError(f"{folder} exists and is not a directory")
    if folder.is_dir() and any(folder.iterdir()):
        if not overwrite:
            raise TaskFolderError(
                f"{folder} is not empty; choose a new folder name or pass "
                "overwrite to add the task files to the existing contents"
            )
        for name in (INCAR_NAME, target_name):
            if (folder / name).exists():
                raise TaskFolderError(
                    f"{folder / name} already exists; refusing to overwrite a "
                    "previous task's file"
                )
    folder.mkdir(parents=True, exist_ok=True)

    incar_text = incar_text_for(
        calc_type,
        model_type=model_type,
        model_path=model_path,
        task=task,
        head=head,
        dpa_runtime=dpa_runtime,
    )
    structure_copy = folder / target_name
    # Byte-for-byte copy: the wizard must never rewrite a structure.
    shutil.copyfile(source, structure_copy)
    incar = folder / INCAR_NAME
    incar.write_text(incar_text, encoding="utf-8")
    return TaskFolder(
        path=folder.resolve(),
        incar=incar.resolve(),
        structure=structure_copy.resolve(),
        calc_type=str(calc_type).strip().lower(),
        model_type=str(model_type).strip().lower(),
        model_path=incar_model_path(incar_text),
    )


def incar_model_path(incar_text: str) -> str:
    """The ``MODEL_PATH`` value a generated INCAR carries."""
    for line in incar_text.splitlines():
        if line.strip().startswith("MODEL_PATH"):
            return line.split("=", 1)[1].split("#", 1)[0].strip()
    return ""


def remaining_placeholders(incar_text: str) -> list[str]:
    """Keys a created INCAR still leaves for the user to fill in (``REQUIRED``)."""
    from mliport.config.incar import IncarConfig  # noqa: PLC0415

    config = IncarConfig.from_string(incar_text)
    return sorted(
        key for key, value in config.items() if str(value).strip() == "REQUIRED"
    )


def task_folder_command(
    folder: str | Path, *, structure: str | Path | None = None
) -> list[str]:
    """The argv of ``mliport run`` for a created task folder.

    This is the single entry point the "run now" path uses, so the wizard and
    the command line execute the same code.  Passing the structure explicitly
    means the command does not depend on the caller's working directory (and
    therefore never needs a process-wide ``chdir``).
    """
    path = Path(folder).expanduser().resolve()
    argv = ["run", "--incar", str(path / INCAR_NAME)]
    if structure is not None:
        argv += ["--structure", str(Path(structure).expanduser().resolve())]
    return argv


def run_task_folder(folder: str | Path) -> int:
    """Run a "Create & Run" request through ``mliport run`` itself (blocking).

    Kept for non-interactive callers and tests.  The TUI never calls this on its
    event loop: it queues :func:`submit_task_folder` instead.  The task folder's
    structure is passed explicitly so no global working-directory change is
    needed.
    """
    from mliport.cli import main  # noqa: PLC0415

    path = Path(folder).expanduser().resolve()
    structure = _task_folder_structure(path)
    return main(task_folder_command(path, structure=structure))


def _task_folder_structure(folder: Path) -> Path | None:
    """The structure file ``run`` discovers in ``folder`` (POSCAR first)."""
    for name in ("POSCAR", "CONTCAR", "structure.cif", "structure.xyz"):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def task_folder_readiness(folder: str | Path) -> TaskFolderReadiness:
    """Report whether a created task folder can run as written.

    This is intentionally a read-only check of the same INCAR ``mliport run``
    reads.  It never repairs placeholders and never falls back to template
    defaults for task/head.
    """
    from mliport.config.incar import IncarConfig  # noqa: PLC0415

    path = Path(folder).expanduser().resolve()
    incar = path / INCAR_NAME
    if not incar.is_file():
        return TaskFolderReadiness(
            calc_type="", errors=(f"{INCAR_NAME} was not found in {path}",)
        )
    config = IncarConfig.from_file(incar)
    calc_type = str(
        config.get_str("CALC_TYPE", config.get_str("CALCULATION", "")) or ""
    ).lower()
    placeholders = tuple(remaining_placeholders(incar.read_text(encoding="utf-8")))
    errors: tuple[str, ...] = ()
    if not placeholders:
        try:
            errors = tuple(config.validate())
        except ValueError as exc:
            errors = (str(exc),)
    return TaskFolderReadiness(
        calc_type=calc_type, placeholders=placeholders, errors=errors
    )


def submit_task_folder(
    folder: str | Path,
    *,
    manager=None,
    python: str | None = None,
    display_name: str | None = None,
):
    """Start a ready task folder as a persistent background job.

    The job goes through the same :class:`mliport.jobs.JobManager` worker,
    lease, status and log machinery as every other queued TUI job; it is not a
    second execution engine.  The worker is detached, so leaving the TUI does
    not cancel it.

    Raises:
        TaskFolderError: the folder has unfilled placeholders, invalid config,
            a missing structure, or the job could not be submitted.
    """
    from ase.io import read  # noqa: PLC0415

    from mliport.config.resolver import resolve_config  # noqa: PLC0415
    from mliport.jobs import JobManager  # noqa: PLC0415

    path = Path(folder).expanduser().resolve()
    readiness = task_folder_readiness(path)
    if not readiness.runnable:
        raise TaskFolderError(readiness.explanation())
    incar = path / INCAR_NAME
    structure = _task_folder_structure(path)
    if structure is None:
        raise TaskFolderError(
            f"No structure file ({', '.join(('POSCAR', 'CONTCAR', 'structure.cif', 'structure.xyz'))}) was found in {path}."
        )

    from mliport.config.incar import IncarConfig  # noqa: PLC0415

    config = IncarConfig.from_file(incar)
    calc_type = readiness.calc_type
    if calc_type not in TASK_CALC_TYPES:
        raise TaskFolderError(
            f"Unknown calculation type {calc_type!r} in {incar}; cannot submit it."
        )
    resolved = resolve_config(calc_type=calc_type, incar=config)
    atoms = read(structure)
    command = task_folder_command(path, structure=structure)
    argv = [python or sys.executable, "-m", "mliport.cli", *command]
    job_manager = manager if manager is not None else JobManager()
    job_id = job_manager.new_job_id()
    try:
        job_manager.submit(
            job_id,
            calc_type,
            str(structure),
            atoms.get_chemical_formula(),
            len(atoms),
            str(resolved.device),
            argv,
            python=python or sys.executable,
            display_name=display_name or path.name,
        )
    except Exception as exc:
        # Never let a failed claim/spawn escape into the Textual event loop as
        # an unhandled exception; the caller shows this as a creation result.
        raise TaskFolderError(f"Could not start the background job: {exc}") from exc
    return job_id


def available_models(root: str | Path | None = None) -> list[str]:
    """Model file names the wizard can offer, from the ``models/`` directory."""
    from mliport.model_lookup import (  # noqa: PLC0415
        available_model_files,
        mliport_root,
    )

    resolved = mliport_root() if root is None else root
    return available_model_files(resolved)


def default_folder_name(structure: str | Path, calc_type: str) -> str:
    """A readable default folder name for the wizard's confirmation step."""
    stem = Path(structure).stem
    if stem.lower() in {"poscar", "contcar"}:
        stem = "task"
    return f"{stem}-{str(calc_type).strip().lower()}"


__all__ = [
    "INCAR_NAME",
    "STRUCTURE_TARGETS",
    "TASK_CALC_TYPES",
    "TaskFolder",
    "TaskFolderError",
    "TaskFolderReadiness",
    "available_models",
    "create_task_folder",
    "default_folder_name",
    "incar_model_path",
    "incar_text_for",
    "remaining_placeholders",
    "run_task_folder",
    "structure_target_name",
    "submit_task_folder",
    "task_folder_command",
    "task_folder_readiness",
]
