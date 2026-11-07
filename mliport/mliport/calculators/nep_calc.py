# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for the mliport project: NEP (Calorine) engine wrapper.

"""
NEP engine wrapper.

Implements :class:`~mliport.base_calculator.BaseMLIPCalculator` for NEP
potentials through the ``calorine`` package.

NEP is the only backend with **two execution runtimes**, and the runtime is an
explicit user decision that is never inferred from the device, the task or
whatever happens to be installed (see ``mliport.backend_selection``):

* ``nep_runtime="ase"`` -- this module. ``calorine.calculators.CPUNEP`` on
  ``cpu`` (float64, in-process) or ``calorine.calculators.GPUNEP`` on
  ``cuda``/``cuda:N`` (float32). ``GPUNEP`` is an ASE ``FileIOCalculator``:
  every ASE evaluation writes GPUMD inputs and runs **one GPUMD subprocess**,
  so it is coordination-expensive and gives no native-GPUMD MD throughput.
* ``nep_runtime="gpumd"`` -- a native, continuously running GPUMD process. That
  is a separate runner rather than an ASE calculator, so this wrapper rejects
  it and points the caller at ``mliport.gpumd`` instead of pretending the two
  paths are interchangeable (they differ in precision, units of truth and
  per-step cost).

Everything here fails closed: invalid runtimes, devices, tasks, model paths and
GPUMD commands raise instead of being silently downgraded, because a silent
fallback would change the potential's energy identity behind the user's back.
"""

from __future__ import annotations

import os
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from mliport.base_calculator import BaseMLIPCalculator

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from ase.calculators.calculator import Calculator


#: Execution runtimes accepted by ``NEP_RUNTIME``. There is deliberately no
#: ``auto`` value: the runtime selects a different execution model, so it must
#: be a user decision (see ``mliport.backend_selection.NEP_RUNTIMES``).
VALID_NEP_RUNTIMES: tuple[str, ...] = ("ase", "gpumd")

#: Package that supplies the ASE-facing NEP calculators.
_BACKEND_DISTRIBUTION = "calorine"

#: CPUNEP is a float64 in-process implementation; GPUNEP evaluates in float32
#: inside GPUMD. The dtype is therefore a property of the runtime, not a knob.
_CPU_DTYPE = "float64"
_GPU_DTYPE = "float32"

#: Characters that turn a path into a shell program. ``calorine``'s GPUNEP
#: builds its own command line (it even appends its own stdout/stderr
#: redirection), so accepting a shell string here would either be an injection
#: vector or be silently reinterpreted.
_SHELL_METACHARACTERS = frozenset(";|&$`<>()")

_DEVICE_PATTERN = re.compile(r"cpu|cuda(?::(?P<index>\d+))?")
_NEP_VERSION_PATTERN = re.compile(r"(nep\d+)(?:_[a-z0-9]+)*", re.IGNORECASE)
_ELEMENT_SYMBOL_PATTERN = re.compile(r"[A-Z][a-z]?")

_CALORINE_IMPORT_MESSAGE = (
    "NEP support via the ASE runtime requires the 'calorine' package.\n"
    "Install it in the dedicated .venv-nep environment; see the repository "
    "root README.md or README_CN.md."
)


def _distribution_version(name: str) -> str | None:
    """Installed version of ``name``, or ``None`` when it is not installed."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _parse_cutoff_row(
    line: str,
) -> tuple[float, float, float | None, float | None]:
    """Parse one ``cutoff`` header row of a ``nep.txt`` file.

    A modern row is ``cutoff <radial> <angular> <max_neighbours_radial>
    <max_neighbours_angular>``; older/minimal rows stop after the two cutoffs.
    The radial and angular cutoffs are the first two numbers after the keyword,
    *not* the last two: the trailing pair are neighbour-list capacities, and
    reporting them as cutoffs would misdescribe the model's interaction range.

    Args:
        line: One non-empty header line, already stripped.

    Returns:
        ``(cutoff_radial, cutoff_angular, max_neighbours_radial,
        max_neighbours_angular)``; the neighbour entries are ``None`` when the
        row does not carry them.

    Raises:
        ValueError: If the row carries fewer than two numbers or a non-number.
    """
    fields = line.split()
    values: list[float] = []
    for token in fields[1:]:
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(
                f"Malformed NEP cutoff row {line!r}: {token!r} is not a number."
            ) from exc
    if len(values) < 2:
        raise ValueError(
            f"Malformed NEP cutoff row {line!r}: expected radial and angular "
            "cutoffs after the 'cutoff' keyword."
        )
    neighbours = (
        values[2] if len(values) > 2 else None,
        values[3] if len(values) > 3 else None,
    )
    if values[0] <= 0.0 or values[1] <= 0.0:
        raise ValueError(
            f"Malformed NEP cutoff row {line!r}: cutoffs must be positive."
        )
    return values[0], values[1], neighbours[0], neighbours[1]


def parse_nep_model_header(path: str | Path) -> dict[str, Any]:
    """Inspect either the legacy or GPUMD format-versioned NEP model header.

    Reject unsupported energy/temperature/charge/dipole model families before
    the Calorine calculator is constructed. The shared parser also serves the
    native GPUMD path, avoiding divergent model-identity interpretations.
    """
    from mliport.nep_model_format import read_nep_model_metadata  # noqa: PLC0415

    data = read_nep_model_metadata(path)
    version = data["legacy_model_version"]
    return {
        "element_types": list(data["element_types"]),
        "model_version": None if version is None else f"nep{version}",
        "model_token": data["model_token"],
        "format_version": data["format_version"],
        "model_type": data["model_type"],
        "declared_element_count": data["declared_element_count"],
        "cutoff_radial": data["cutoff_radial"],
        "cutoff_angular": data["cutoff_angular"],
        "cutoff_by_element_A": data["cutoff_by_element_A"],
        "max_neighbours_radial": data["max_neighbours_radial"],
        "max_neighbours_angular": data["max_neighbours_angular"],
        "zbl": data["zbl"],
        "energy_reference_eV": data["energy_reference_eV"],
    }

def _normalize_device(device: str | int | None) -> tuple[str, int | None]:
    """Validate a device string for NEP and return its canonical form.

    Only ``cpu``, ``cuda``, ``cuda:N`` and ``gpu`` are meaningful here:
    ``GPUNEP`` runs GPUMD as a subprocess and selects a GPU with
    ``CUDA_VISIBLE_DEVICES``, so there is no ``mps``/``xpu``/``cuda.abc``
    backend to fall back to. Rejecting them keeps the recorded device identity
    truthful.

    Args:
        device: Requested device string.

    Returns:
        ``(device_type, logical_index)`` where ``device_type`` is ``"cpu"`` or
        ``"cuda"`` and the index is ``None`` for CPU or ``0`` for a bare
        ``cuda``/``gpu``.

    Raises:
        ValueError: If the device is not one of the accepted spellings.
    """
    text = "" if device is None else str(device).strip().lower()
    # mliport's legacy "gpu" synonym; GPUMD/torch name the device "cuda".
    if text == "gpu":
        text = "cuda"
    match = _DEVICE_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(
            f"Invalid NEP device {device!r}. Use cpu, cuda, cuda:N or gpu "
            "(gpu is a synonym for cuda)."
        )
    if text == "cpu":
        return "cpu", None
    return "cuda", int(match.group("index") or 0)


def _validate_gpumd_command(command: str | Path) -> Path:
    """Validate the GPUMD executable that ``GPUNEP`` will launch.

    ``calorine``'s ``GPUNEP`` hands its ``command`` to a shell, so a shell
    string here would be an injection vector and, worse, could be silently
    rewritten (GPUNEP appends its own stdout/stderr redirection). A single
    executable path is accepted; a path containing spaces is fine.

    Args:
        command: Path to the ``gpumd`` binary.

    Returns:
        The validated, existing executable path.

    Raises:
        ValueError: If the value is empty, contains shell metacharacters, is
            not a regular file, or is not executable.
        FileNotFoundError: If the path does not exist.
    """
    text = str(command).strip()
    if not text:
        raise ValueError(
            "gpumd_command must be the path of the GPUMD executable, not an "
            "empty string."
        )
    metacharacters = sorted({char for char in text if char in _SHELL_METACHARACTERS})
    # A newline or tab means several tokens pasted together, which is the
    # shell-string case too. A plain space is allowed: paths can contain them.
    if metacharacters or any(char.isspace() and char != " " for char in text):
        detail = (
            f"contains {''.join(metacharacters)!r}"
            if metacharacters
            else "contains whitespace other than plain spaces"
        )
        raise ValueError(
            f"gpumd_command {str(command)!r} looks like a shell command string "
            f"({detail}). GPUNEP builds its own command line, so mliport "
            "accepts one executable path only -- no '&&', '|', ';', '>', '<', "
            "'$VAR', backticks or subshells."
        )
    path = Path(text)
    if not path.exists():
        raise FileNotFoundError(f"GPUMD executable not found: {path}")
    if not path.is_file():
        raise ValueError(f"gpumd_command {path} is not a regular file.")
    if not os.access(path, os.X_OK):
        raise ValueError(
            f"gpumd_command {path} is not executable; run 'chmod +x {path}' or "
            "point at the gpumd binary."
        )
    return path


class NEPCalculatorWrapper(BaseMLIPCalculator):
    """NEP potential wrapper over the Calorine ASE calculators.

    ``device="cpu"`` builds ``CPUNEP`` (float64, in-process); a ``cuda`` device
    builds ``GPUNEP`` (float32, one GPUMD subprocess per ASE evaluation) and
    therefore requires an explicit ``gpumd_command``. The ``gpumd`` runtime is
    not available here at all -- use ``mliport.gpumd`` for the native runner.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        task: str = "bulk",
        *,
        nep_runtime: str = "ase",
        gpumd_command: str | Path | None = None,
    ):
        """
        Initialize the NEP calculator wrapper.

        Args:
            model_path: Path to a single ``nep.txt`` model file. A directory is
                rejected: it has no single model identity to record in the run
                provenance.
            device: ``cpu``, ``cuda``, ``cuda:N`` or ``gpu`` (a synonym for
                ``cuda``).
            task: PBC hint, ``bulk`` (periodic) or ``molecule`` (non-periodic).
            nep_runtime: Execution runtime. Only ``ase`` can be served by an ASE
                calculator; ``gpumd`` raises and points at ``mliport.gpumd``.
            gpumd_command: Path to the GPUMD executable. Required for a
                ``cuda`` device (GPUNEP shells out to GPUMD); a single path,
                never a shell command string.

        Raises:
            ValueError: For an unsupported runtime, task or device, for a
                ``gpumd`` runtime, for a model path that is a directory, for a
                missing/invalid ``gpumd_command``, or for a ``cuda`` device
                without one (validated before any calculator is constructed).
            FileNotFoundError: If ``model_path`` or ``gpumd_command`` is
                missing.
        """
        runtime = str(nep_runtime).strip().lower()
        if runtime not in VALID_NEP_RUNTIMES:
            raise ValueError(
                f"Invalid nep_runtime {nep_runtime!r}. Choose one of: "
                + ", ".join(VALID_NEP_RUNTIMES)
            )
        if runtime == "gpumd":
            raise ValueError(
                "nep_runtime='gpumd' selects the native GPUMD process runner, "
                "which is not an ASE calculator, so NEPCalculatorWrapper cannot "
                "provide it. Use mliport.gpumd for the native runtime, or "
                "nep_runtime='ase' for the Calorine ASE calculator."
            )
        self._nep_runtime = runtime

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"NEP model file not found: {self.model_path}")
        if not self.model_path.is_file():
            raise ValueError(
                f"NEP model path {self.model_path} is a directory. A NEP model "
                "is a single nep.txt file, and a directory has no single model "
                "identity, so it cannot be recorded in the run provenance."
            )

        self._task = str(task).strip().lower()
        if self._task not in {"bulk", "molecule"}:
            raise ValueError(
                f"Invalid NEP task {task!r}. Must be 'bulk' (periodic) or "
                "'molecule' (non-periodic)."
            )

        self._device, self._device_logical_index = _normalize_device(device)
        self._default_dtype = _CPU_DTYPE if self._device == "cpu" else _GPU_DTYPE
        self._gpumd_command: Path | None = None
        if gpumd_command is not None:
            # Validated even on cpu: silently ignoring a bad GPUMD path would
            # hide the typo until a user switches to a GPU device.
            self._gpumd_command = _validate_gpumd_command(gpumd_command)
        if self._device != "cpu" and self._gpumd_command is None:
            raise ValueError(
                f"The NEP ASE runtime on {self._device!r} needs an explicit "
                "GPUMD executable: pass gpumd_command=<path to gpumd>. "
                "Calorine's GPUNEP evaluates by running a GPUMD subprocess per "
                "ASE call, so it has no Python-side CUDA backend. Use "
                "device='cpu' (CPUNEP) if you have no GPUMD binary."
            )

        self._calculator: Calculator | None = None
        # The header is parsed on first use (require_elements / info / first
        # calculation) rather than here: the constructor validates the model
        # *path*, while the file contents are the backend's and the element
        # guard's business, and info() must work without ever importing the
        # backend or touching the model.
        self._header: dict[str, Any] | None = None

    # -- model header ------------------------------------------------------

    def _model_header(self) -> dict[str, Any]:
        """Return the parsed ``nep.txt`` header, cached after the first read."""
        if self._header is None:
            self._header = parse_nep_model_header(self.model_path)
        return self._header

    def require_elements(self, symbols: Sequence[str]) -> None:
        """Fail before any GPU allocation if the structure has foreign elements.

        ``CPUNEP``/``GPUNEP`` reject unsupported species themselves, but only
        once the calculator is built -- and for ``GPUNEP`` that means after a
        GPUMD subprocess has been scheduled. A pre-flight check gives a clear
        error before any GPU is allocated, and the native GPUMD side reuses the
        same header check.

        Args:
            symbols: Chemical symbols of the structure to validate.

        Raises:
            ValueError: If any symbol is not listed in the model header, with
                the unsupported elements and the supported set in the message.
        """
        supported = list(self._model_header()["element_types"])
        present = {str(symbol).strip() for symbol in symbols}
        unsupported = sorted(present.difference(supported))
        if unsupported:
            raise ValueError(
                f"Structure contains element(s) {unsupported} that the NEP "
                f"model {self.model_path} does not support. Supported "
                f"elements: {supported}."
            )

    # -- calculator --------------------------------------------------------

    def get_calculator(self) -> Calculator:
        """Return the cached Calorine ASE calculator, importing it lazily.

        ``calorine`` is imported on first use only, so mliport works in an
        environment without it until a NEP run is actually requested. The
        calculator is cached because ``GPUNEP``'s construction is expensive and
        because identity reporting must describe the single calculator that is
        really being used.

        Returns:
            A ``CPUNEP`` (cpu) or ``GPUNEP`` (cuda) instance.

        Raises:
            ImportError: If ``calorine`` is not installed; the message names the
                package and the dedicated ``.venv-nep`` environment.
        """
        if self._calculator is None:
            try:
                from calorine.calculators import (  # noqa: PLC0415
                    CPUNEP,
                    GPUNEP,
                )
            except ImportError as exc:
                raise ImportError(_CALORINE_IMPORT_MESSAGE) from exc
            if self._device == "cpu":
                calculator: Calculator = CPUNEP(
                    model_filename=str(self.model_path),
                )
            else:
                calculator = GPUNEP(
                    model_filename=str(self.model_path),
                    command=str(self._gpumd_command),
                    gpu_identifier_index=self._device_logical_index,
                )
            self._calculator = calculator
        return self._calculator

    @property
    def task(self) -> str:
        """PBC hint (``bulk``/``molecule``); NEP itself is task-agnostic."""
        return self._task

    @property
    def has_stress(self) -> bool:
        """Whether the constructed calculator really implements stress.

        Derived from the live ``implemented_properties`` rather than hard-coded:
        NEP models differ (charge, dipole and polarizability models advertise a
        different property set), and a claim that does not match the backend
        would put a meaningless "stress" column in the output.
        """
        return "stress" in self.implemented_properties

    def info(self) -> dict:
        """Return model metadata for OUTCAR/JSON provenance.

        Everything that can be answered without importing ``calorine`` is
        always present: the model header is parsed with the stdlib-only helper,
        so element types, model version and cutoffs are available for the
        native GPUMD side too. Fields that only exist once the calculator has
        been built (``implemented_properties``, ``has_stress``) are reported as
        ``None`` instead of triggering a backend import, so a metadata dump can
        never explode or start a GPUMD subprocess by accident.

        Returns:
            Dictionary with at least ``model_type``, ``model_path``,
            ``device``, ``nep_runtime``, ``task``, ``implemented_properties``,
            ``has_stress``, ``backend_distribution``, ``backend_version``,
            ``framework_versions``, ``default_dtype``, ``actual_device_type``,
            ``actual_device_logical_index``, ``nep_element_types``,
            ``nep_model_version`` and ``per_call_subprocess``.
        """
        calculator = self._calculator
        implemented_properties = (
            None if calculator is None else list(calculator.implemented_properties)
        )
        header: dict[str, Any] | None = None
        header_error: str | None = None
        try:
            header = self._model_header()
        except ValueError as exc:
            # info() is a reporting call: report the header problem in-band
            # rather than raising, so an OUTCAR/JSON writer cannot be the thing
            # that discovers a broken model file.
            header_error = str(exc)

        info: dict[str, Any] = {
            "model_type": "nep",
            "model_path": str(self.model_path),
            "device": self._device,
            "nep_runtime": self._nep_runtime,
            "task": self._task,
            "gpumd_command": (
                None if self._gpumd_command is None else str(self._gpumd_command)
            ),
            "implemented_properties": implemented_properties,
            "has_stress": (
                None
                if implemented_properties is None
                else "stress" in implemented_properties
            ),
            "backend_distribution": _BACKEND_DISTRIBUTION,
            "backend_version": _distribution_version(_BACKEND_DISTRIBUTION),
            "framework_versions": {
                "calorine": _distribution_version("calorine"),
                "ase": _distribution_version("ase"),
            },
            "default_dtype": self._default_dtype,
            "actual_device_type": self._device,
            "actual_device_logical_index": self._device_logical_index,
            "nep_element_types": (
                None if header is None else list(header["element_types"])
            ),
            "nep_model_version": None if header is None else header["model_version"],
            "nep_cutoff_radial": None if header is None else header["cutoff_radial"],
            "nep_cutoff_angular": None if header is None else header["cutoff_angular"],
            "per_call_subprocess": self._device != "cpu",
        }
        if header_error is not None:
            info["nep_model_header_error"] = header_error
        return info


__all__ = [
    "VALID_NEP_RUNTIMES",
    "NEPCalculatorWrapper",
    "parse_nep_model_header",
]
