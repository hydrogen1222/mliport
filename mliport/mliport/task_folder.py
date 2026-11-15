"""Create and run a task folder (spec R5 of the #61 usage round).

A *task folder* is the unit of daily mliport use: an ``INCAR.mliport`` next to
the structure file, so that ``mliport run`` needs no arguments at all. This
module owns the two operations behind the TUI wizard, and nothing else:

* :func:`create_task_folder` writes ``INCAR.mliport`` from the same template
  generator the CLI uses (with ``MODEL_TYPE`` / ``MODEL_PATH`` filled in) and
  copies the chosen structure *byte for byte* under the name the ``run``
  structure lookup expects (``POSCAR`` / ``structure.cif`` / ``structure.xyz``);
* :func:`run_task_folder` runs it through :func:`mliport.cli.main`, i.e. the
  exact code path of ``mliport run``. The TUI must never build its own run
  command for this.

Both functions fail closed: an unknown calculation type, an unsupported
structure suffix, an existing non-empty destination or a template without the
two model lines is an error, never a guess.
"""

from __future__ import annotations

import shutil
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


def incar_text_for(
    calc_type: str,
    *,
    model_type: str,
    model_path: str | Path,
) -> str:
    """The template for ``calc_type`` with the chosen model filled in.

    The rest of the file is the generated template, byte for byte: the wizard
    must not maintain a second description of the options (spec R5.2).
    """
    from mliport.config.defaults import build_incar_default  # noqa: PLC0415
    from mliport.model_lookup import is_file_name  # noqa: PLC0415

    calc = str(calc_type).strip().lower()
    if calc not in TASK_CALC_TYPES:
        raise TaskFolderError(
            f"Unknown calculation type {calc_type!r}; choose from "
            f"{', '.join(TASK_CALC_TYPES)}."
        )
    engine = str(model_type).strip().lower()
    if not engine:
        raise TaskFolderError("A model backend (MODEL_TYPE) is required.")
    raw_model = str(model_path).strip()
    if not raw_model:
        raise TaskFolderError("A model file (MODEL_PATH) is required.")
    # A bare file name keeps the models/ convention working (#61 R2); anything
    # else is recorded as an absolute path so the folder does not depend on the
    # directory the wizard was started from.
    resolved_model = (
        raw_model
        if is_file_name(raw_model)
        else str(Path(raw_model).expanduser().resolve())
    )

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
    overwrite: bool = False,
) -> TaskFolder:
    """Create a task folder and return what was written.

    Args:
        destination: Directory to create (must be absent or empty).
        calc_type: ``sp``/``opt``/``md``/``neb``/``confsearch``.
        structure: Existing structure file; copied byte for byte.
        model_type: Backend chosen for ``MODEL_TYPE``.
        model_path: Model file chosen; recorded as an absolute path.
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

    incar_text = incar_text_for(calc_type, model_type=model_type, model_path=model_path)
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


def task_folder_command(folder: str | Path) -> list[str]:
    """The argv of ``mliport run`` for a created task folder.

    This is the single entry point the "run now" path uses, so the wizard and
    the command line execute the same code.
    """
    path = Path(folder).expanduser().resolve()
    return ["run", "--incar", str(path / INCAR_NAME)]


def run_task_folder(folder: str | Path) -> int:
    """Run a "Create & Run" request through ``mliport run`` itself."""
    from mliport.cli import main  # noqa: PLC0415

    cwd = Path.cwd()
    path = Path(folder).expanduser().resolve()
    try:
        # `run` discovers the structure in the working directory, so the task
        # folder is the working directory for this call - exactly what the
        # printed `cd <folder> && mliport run` does.
        import os  # noqa: PLC0415

        os.chdir(path)
        return main(task_folder_command("."))
    finally:
        os.chdir(cwd)


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
    "available_models",
    "create_task_folder",
    "default_folder_name",
    "incar_model_path",
    "incar_text_for",
    "remaining_placeholders",
    "run_task_folder",
    "structure_target_name",
    "task_folder_command",
]
