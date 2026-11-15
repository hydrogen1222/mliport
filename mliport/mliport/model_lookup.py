"""The repository ``models/`` convention for ``MODEL_PATH`` (spec R2, #61).

Daily mliport use keeps model weights in one place: ``models/`` at the root of
the mliport checkout. A ``MODEL_PATH`` that is a *file name* - a value without
any path separator - names a file there; anything that looks like a path keeps
its existing meaning and is never second-guessed.

Lookup order for a file name (fail closed at every step):

1. the same file name next to the file that declared it (an INCAR folder, the
   settings.ini directory, the current directory for a CLI value) - the
   historical behaviour,
2. ``<MLIPORT_ROOT>/models/<name>``,
3. ``<MLIPORT_ROOT>/models/*/<name>`` - exactly one level of subdirectories.

Two matches in step 3 are an error: choosing one of two different checkpoints
would silently change the model identity a result is recorded against.

``MLIPORT_ROOT`` is taken from the environment when set, otherwise from the
parent of ``sys.prefix`` when that is an installer-created ``.venv-<engine>``
directory, otherwise it is unknown and a file name is refused with instructions
rather than resolved against a guess.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

MLIPORT_ROOT_ENV = "MLIPORT_ROOT"
MODELS_DIRNAME = "models"


class ModelLookupError(ValueError):
    """A model reference could not be resolved without guessing."""


def _installer_venv_names() -> frozenset[str]:
    """Filesystem names of the environments the installer creates."""
    from mliport.install.compatibility import ALL_BACKENDS  # noqa: PLC0415

    return frozenset(backend.venv_name for backend in ALL_BACKENDS.values())


def mliport_root(
    *,
    env: Mapping[str, str] | None = None,
    prefix: str | Path | None = None,
) -> Path | None:
    """Return the mliport checkout root, or ``None`` when it is unknown.

    Args:
        env: Environment mapping (defaults to ``os.environ``).
        prefix: Python prefix to derive the root from (defaults to
            ``sys.prefix``). Only an installer-created ``.venv-<engine>``
            directory is accepted; a hand-made environment says nothing about
            where the repository is.
    """
    environ = os.environ if env is None else env
    configured = str(environ.get(MLIPORT_ROOT_ENV, "")).strip()
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            raise ModelLookupError(
                f"{MLIPORT_ROOT_ENV}={configured!r} is not an absolute path. "
                "Set it to the absolute path of the mliport checkout."
            )
        return root.resolve()
    prefix_path = Path(sys.prefix if prefix is None else prefix).expanduser()
    if prefix_path.name in _installer_venv_names():
        candidate = prefix_path.parent.resolve()
        # A matching environment *name* is not proof that its parent is the
        # mliport checkout: ``.venv`` is the name of UMA's environment and also
        # of any project-local virtualenv. Without a checkout marker the root
        # stays unknown, so a file name fails closed with instructions instead
        # of resolving a model out of an unrelated directory.
        if looks_like_checkout(candidate):
            return candidate
    return None


def looks_like_checkout(root: str | Path) -> bool:
    """True when ``root`` carries the markers of an mliport checkout."""
    directory = Path(root)
    if (directory / MODELS_DIRNAME).is_dir():
        return True
    launchers = directory / "bin"
    if launchers.is_dir() and any(launchers.glob("mliport-*")):
        return True
    return (directory / "mliport" / "pyproject.toml").is_file()


def models_dir(root: str | Path) -> Path:
    """The ``models/`` directory of a checkout root."""
    return Path(root) / MODELS_DIRNAME


def is_file_name(value: str | os.PathLike[str]) -> bool:
    """True when ``value`` names a file without any path separator."""
    text = str(value)
    separators = {os.sep, "/", os.altsep} - {None, ""}
    return not any(separator in text for separator in separators)


def available_model_files(root: str | Path | None) -> list[str]:
    """Model file names in ``models/`` (and its one level of subdirectories).

    Returned as sorted ``models``-relative paths, so an error message or
    ``mliport doctor`` can show the user what is actually there.
    """
    if root is None:
        return []
    directory = models_dir(root)
    if not directory.is_dir():
        return []
    found: list[str] = []
    with os.scandir(directory) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            if entry.is_file():
                found.append(entry.name)
            elif entry.is_dir():
                nested = [
                    child.name
                    for child in sorted(
                        os.scandir(entry.path), key=lambda item: item.name
                    )
                    if child.is_file() or child.is_dir()
                ]
                if nested:
                    found.extend(f"{entry.name}/{name}" for name in nested)
                else:
                    # An empty directory in models/ is a model directory in its
                    # own right (a GRACE SavedModel, for example).
                    found.append(f"{entry.name}/")
    return found


def _searched_locations(name: str, root: Path | None, declared_near: Path) -> list[str]:
    locations = [f"{declared_near} (next to the file that declares it)"]
    if root is not None:
        locations.append(str(models_dir(root) / name))
        locations.append(f"{models_dir(root)}/*/{name} (one level of subdirectories)")
    return locations


def _not_found_message(name: str, root: Path | None, declared_near: Path) -> str:
    lines = [f"Model {name!r} was not found. Searched:"]
    lines.extend(
        f"  - {location}" for location in _searched_locations(name, root, declared_near)
    )
    if root is None:
        lines.append(
            f"No mliport root directory was found, so {MODELS_DIRNAME}/ could not be "
            f"searched. Set {MLIPORT_ROOT_ENV} to the absolute path of the mliport "
            f"checkout, or pass an explicit path in MODEL_PATH."
        )
        return "\n".join(lines)
    available = available_model_files(root)
    if available:
        lines.append(f"Files in {models_dir(root)}:")
        lines.extend(f"  - {item}" for item in available)
    else:
        lines.append(f"{models_dir(root)} holds no models.")
    return "\n".join(lines)


def _missing_root_message(name: str, declared_near: Path) -> str:
    return "\n".join(
        [
            f"Model {name!r} is a bare file name and was not found at "
            f"{declared_near}.",
            f"Looking it up in {MODELS_DIRNAME}/ needs a known mliport root, and none "
            "could be determined:",
            f"  - {MLIPORT_ROOT_ENV} is not set, and",
            "  - the running environment is not an installer-created "
            ".venv-<engine> directory.",
            f"Set {MLIPORT_ROOT_ENV} to the absolute path of the mliport checkout "
            f"(the directory that holds {MODELS_DIRNAME}/), or pass an explicit "
            "path in MODEL_PATH.",
        ]
    )


def lookup_model_file_name(
    name: str,
    *,
    declared_near: Path,
    root: str | Path | None = None,
) -> Path:
    """Resolve a bare model file name against ``models/``.

    Args:
        name: The bare file name from ``MODEL_PATH``.
        declared_near: The path the historical behaviour would have used (the
            declaring file's directory joined with ``name``), for the error.
        root: Checkout root; resolved from the environment/``sys.prefix`` when
            omitted.

    Raises:
        ModelLookupError: no match, an unknown root, or two matches.
    """
    resolved_root = mliport_root() if root is None else Path(root).resolve()
    if resolved_root is None:
        raise ModelLookupError(_missing_root_message(name, declared_near))
    directory = models_dir(resolved_root)
    matches: list[Path] = []
    direct = directory / name
    if direct.exists():
        matches.append(direct)
    else:
        if directory.is_dir():
            with os.scandir(directory) as entries:
                for entry in sorted(entries, key=lambda item: item.name):
                    if not entry.is_dir():
                        continue
                    candidate = Path(entry.path) / name
                    if candidate.exists():
                        matches.append(candidate)
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        listed = "\n".join(f"  - {match}" for match in matches)
        raise ModelLookupError(
            f"Model name {name!r} matches more than one entry in "
            f"{directory}:\n{listed}\n"
            "Name the model by path instead, or keep one copy per name."
        )
    raise ModelLookupError(_not_found_message(name, resolved_root, declared_near))


def resolve_model_reference(
    declared: str | os.PathLike[str],
    *,
    base_dir: str | Path | None,
    root: str | Path | None = None,
) -> Path:
    """Resolve one ``MODEL_PATH`` value to an absolute path.

    A value with a path separator keeps the historical behaviour exactly: it is
    resolved against the declaring file's directory and is never checked against
    ``models/``. A bare file name prefers a same-named file next to the declaring
    file and only then searches ``models/``.
    """
    text = str(declared).strip()
    if not text:
        raise ModelLookupError("MODEL_PATH must not be empty")
    path = Path(text).expanduser()
    if not path.is_absolute():
        if base_dir is None:
            raise ModelLookupError(
                f"Path option 'model_path' has no source base directory to "
                f"resolve {text!r} against"
            )
        path = Path(base_dir) / path
    if not is_file_name(text):
        return path.resolve()
    if path.exists():
        # A file, or a directory when the backend's model is a directory
        # (GRACE SavedModel). Either way the user's own path wins over models/.
        return path.resolve()
    return lookup_model_file_name(text, declared_near=path, root=root)


def describe_models_directory(root: str | Path | None) -> list[str]:
    """Human-readable lines describing ``models/`` for ``mliport doctor``."""
    if root is None:
        return [
            f"No mliport root directory found: {MLIPORT_ROOT_ENV} is not set and the "
            "environment is not an installer-created .venv-<engine> directory.",
        ]
    directory = models_dir(root)
    lines = [f"{directory}"]
    files = available_model_files(root)
    if files:
        lines.extend(f"  {item}" for item in files)
    else:
        lines.append("  (no model files)")
    return lines


def model_search_paths(root: str | Path | None) -> Iterable[str]:
    """The directories a bare ``MODEL_PATH`` name is searched in."""
    if root is None:
        return ()
    directory = models_dir(root)
    return (str(directory), f"{directory}/*")


__all__ = [
    "MLIPORT_ROOT_ENV",
    "MODELS_DIRNAME",
    "ModelLookupError",
    "available_model_files",
    "describe_models_directory",
    "is_file_name",
    "lookup_model_file_name",
    "looks_like_checkout",
    "mliport_root",
    "model_search_paths",
    "models_dir",
    "resolve_model_reference",
]
