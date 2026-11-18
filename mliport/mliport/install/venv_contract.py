# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Contract for a target environment that already exists.

Every mliport backend lives in its own virtual environment (``.venv-mace``,
``.venv-dpa`` …), and the install plan creates it with ``uv venv
--allow-existing`` so that a re-run does not have to delete a working
environment first.  ``--allow-existing`` on its own is **not** a contract: uv
happily initialises a venv inside a non-venv directory and silently replaces a
venv whose interpreter has another Python version, both of which destroy work
that mliport did not create.

This module turns that flag into an explicit, fail-closed decision
(``AGENTS.md`` rule 2) *before* any step runs:

``ABSENT``
    Nothing exists at the path: the plan creates the environment.

``REUSE``
    A virtual environment (``pyvenv.cfg`` + interpreter) of the requested
    Python version that carries no other backend's packages: the environment
    belongs to this backend's contract, so reusing it is safe.  The pip steps
    then re-resolve the pinned packages in place.

``INCOMPATIBLE``
    Anything else: a plain directory (or file) that is not a venv, an
    interpreter that cannot report its version, a venv of a different Python,
    or a venv that contains another backend's framework/packages.  mliport
    refuses, changes nothing, and tells the user to pass ``--clean``.

The check never mutates the filesystem and never imports anything from the
target environment: it only reads ``pyvenv.cfg``, lists ``site-packages`` and
asks the platform-specific venv interpreter for ``major.minor``.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from mliport.install.compatibility import ALL_BACKENDS

if TYPE_CHECKING:
    from mliport.install.compatibility import BackendSpec

#: Import/distribution names that identify one backend inside a venv.
#: A venv is only reusable for backend *X* when it carries none of the probes
#: of the other backends: mixing two frameworks in one environment breaks the
#: model/ dtype / task identity mliport guarantees.
_BACKEND_PROBES: dict[str, tuple[str, ...]] = {
    "uma": ("fairchem", "fairchem_core"),
    "mace": ("mace", "mace_torch"),
    "dpa": ("deepmd", "deepmd_kit"),
    "grace": ("tensorpotential",),
    "nep": ("calorine",),
}

#: Framework-level probes, so a foreign framework is caught even when the
#: backend package itself was never installed.
_FRAMEWORK_PROBES: dict[str, tuple[str, ...]] = {
    "torch": ("torch",),
    "tensorflow": ("tensorflow",),
    "numpy": (),
}

#: Display labels for the refusals, so the message names what it found.
_PROBE_LABELS: dict[str, str] = {
    "fairchem": "UMA (FAIRChem)",
    "fairchem_core": "UMA (FAIRChem)",
    "mace": "MACE",
    "mace_torch": "MACE",
    "deepmd": "DPA (DeepMD-kit)",
    "deepmd_kit": "DPA (DeepMD-kit)",
    "tensorpotential": "GRACE",
    "calorine": "NEP (Calorine / GPUMD)",
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
}

_INTERPRETER_TIMEOUT_S = 10

_VERSION_PROBE = (
    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
)


class Verdict(str, Enum):
    """Outcome of inspecting the target environment path."""

    ABSENT = "absent"
    REUSE = "reuse"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class VenvState:
    """Result of :func:`inspect_target_venv`.

    Attributes:
        path: The environment path that was inspected.
        verdict: One of :class:`Verdict`.
        reason: Machine-readable code for ``INCOMPATIBLE`` (``""`` otherwise).
        detail: Human-readable explanation for ``INCOMPATIBLE``.
        python_version: ``major.minor`` reported by the venv, ``""`` if unknown.
        foreign: Names of the foreign packages/frameworks that were found.
    """

    path: Path
    verdict: Verdict
    reason: str = ""
    detail: str = ""
    python_version: str = ""
    foreign: tuple[str, ...] = ()

    @property
    def reusable(self) -> bool:
        return self.verdict is Verdict.REUSE

    @property
    def refused(self) -> bool:
        return self.verdict is Verdict.INCOMPATIBLE


def _interpreter(venv: Path) -> Path | None:
    """The venv interpreter path (POSIX and Windows layouts)."""
    for candidate in (venv / "bin" / "python", venv / "Scripts" / "python.exe"):
        if candidate.is_file():
            return candidate
    return None


def _site_packages(venv: Path) -> list[Path]:
    """All ``site-packages`` directories of a venv, without reading metadata."""
    found = sorted(venv.glob("lib/python*/site-packages"))
    windows = venv / "Lib" / "site-packages"
    if windows.is_dir():
        found.append(windows)
    return [path for path in found if path.is_dir()]


def _present(site_packages: list[Path], probe: str) -> bool:
    """True if ``probe`` (module dir or ``.dist-info``) is installed."""
    for directory in site_packages:
        if (directory / probe).exists():
            return True
        if any(directory.glob(f"{probe}-*.dist-info")):
            return True
    return False


_BACKEND_FAMILIES = {
    # Current and legacy DPA runtimes are separate environments but install
    # the same import/distribution names; those packages are the owning
    # backend in either environment, not foreign packages.
    "dpa-legacy": "dpa",
}


def _backend_family(engine: str) -> str:
    return _BACKEND_FAMILIES.get(engine, engine)


def _foreign_probes(backend: BackendSpec) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Probes that must be absent for ``backend`` to reuse an environment.

    Returns ``(package_probes, framework_probes)``: the own backend's own
    distribution and its own framework are excluded, everything else is
    foreign.
    """
    packages: list[str] = []
    own_family = _backend_family(backend.engine)
    for engine, probes in _BACKEND_PROBES.items():
        if _backend_family(engine) != own_family:
            packages.extend(probes)
    frameworks: list[str] = []
    for framework, probes in _FRAMEWORK_PROBES.items():
        if framework != backend.framework:
            frameworks.extend(probes)
    return tuple(packages), tuple(frameworks)


def _reported_python(interpreter: Path) -> str | None:
    """``major.minor`` of the venv interpreter, ``None`` when unreadable."""
    try:
        result = subprocess.run(
            [str(interpreter), "-c", _VERSION_PROBE],
            capture_output=True,
            text=True,
            timeout=_INTERPRETER_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    reported = result.stdout.strip()
    parts = reported.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        return None
    return reported


def _major_minor(version: str) -> str:
    parts = str(version).strip().split(".")
    return ".".join(parts[:2])


def inspect_target_venv(
    venv: Path | str,
    *,
    backend: BackendSpec,
    python_version: str,
) -> VenvState:
    """Classify an existing environment path before the plan touches it.

    Args:
        venv: Target environment path (relative to the working directory).
        backend: The backend whose environment this is supposed to be.
        python_version: Requested ``major.minor`` (e.g. ``"3.12"``).

    Returns:
        A :class:`VenvState`.  The function is read-only: nothing is created,
        deleted or installed, and nothing is imported from the venv.
    """
    path = Path(venv)
    if not path.exists():
        return VenvState(path=path, verdict=Verdict.ABSENT)
    if not path.is_dir():
        return VenvState(
            path=path,
            verdict=Verdict.INCOMPATIBLE,
            reason="not_a_venv",
            detail=f"{path} exists but is not a directory",
        )
    if not (path / "pyvenv.cfg").is_file():
        return VenvState(
            path=path,
            verdict=Verdict.INCOMPATIBLE,
            reason="not_a_venv",
            detail=f"{path} exists but is not a virtual environment (no pyvenv.cfg)",
        )
    interpreter = _interpreter(path)
    if interpreter is None:
        return VenvState(
            path=path,
            verdict=Verdict.INCOMPATIBLE,
            reason="no_interpreter",
            detail=f"{path} has no Python interpreter",
        )
    reported = _reported_python(interpreter)
    if reported is None:
        return VenvState(
            path=path,
            verdict=Verdict.INCOMPATIBLE,
            reason="unreadable_interpreter",
            detail=f"the interpreter of {path} cannot report its Python version",
        )
    if reported != _major_minor(python_version):
        return VenvState(
            path=path,
            verdict=Verdict.INCOMPATIBLE,
            reason="python_mismatch",
            detail=(
                f"{path} uses Python {reported}, but {_major_minor(python_version)} "
                "was requested"
            ),
            python_version=reported,
        )
    site_packages = _site_packages(path)
    package_probes, framework_probes = _foreign_probes(backend)
    found = [
        probe
        for probe in (*package_probes, *framework_probes)
        if _present(site_packages, probe)
    ]
    if found:
        labels = sorted({_PROBE_LABELS.get(probe, probe) for probe in found})
        return VenvState(
            path=path,
            verdict=Verdict.INCOMPATIBLE,
            reason="foreign_packages",
            detail=(
                f"{path} already holds {', '.join(labels)} from another backend, "
                f"so it is not a {backend.label} environment"
            ),
            python_version=reported,
            foreign=tuple(found),
        )
    return VenvState(path=path, verdict=Verdict.REUSE, python_version=reported)


def incompatible_message(state: VenvState) -> str:
    """Actionable one-block message for a refused environment."""
    return (
        f"existing {state.path} is incompatible: {state.detail}. "
        "Nothing was changed. Re-run with --clean to delete and rebuild that "
        "environment."
    )


def reuse_message(state: VenvState, *, backend: BackendSpec) -> str:
    """Informational line for a reused environment."""
    return (
        f"Reusing existing {state.path} (Python {state.python_version}, "
        f"compatible with {backend.label}): the pinned packages are re-resolved "
        "in place."
    )


def backend_for_venv(venv_name: str) -> BackendSpec | None:
    """The backend owning ``venv_name`` (``None`` when unknown)."""
    for backend in ALL_BACKENDS.values():
        if backend.venv_name == venv_name:
            return backend
    return None
