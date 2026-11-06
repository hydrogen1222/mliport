"""Fail-closed parsers for native GPUMD output.

GPUMD writes only two machine-readable products that mliport consumes:

* ``thermo.out`` -- 18 fixed columns with a ``#`` header, stress in GPa;
* extended-XYZ dumps -- a per-frame header carrying a *total* energy, the
  native virial (eV) and the native stress (eV/Angstrom**3), plus per-atom
  property columns.

The parsers here never guess.  A truncated frame, a mismatched atom count, a
missing required property column, an unexpected column count or a non-finite
value raises :class:`GPUMDParseError`.  Wrapped ``pos`` and
``unwrapped_position`` are kept as separate arrays; one is never substituted
for the other.

Sign convention
---------------
GPUMD writes ``stress`` with the *same* sign as ``virial / volume``.  ASE's
stress is the negative of that value, so ``ase_stress = -gpumd_stress``.  The
:class:`Frame` fields ``stress_eV_A3`` and ``stress_GPa`` are ASE-convention;
the raw GPUMD tensors are preserved under ``Frame.extra`` for provenance.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms

__all__ = [
    "Frame",
    "GPUMDParseError",
    "iter_extxyz_frames",
    "md_frames_summary",
    "optimization_result",
    "parse_extxyz_frames",
    "parse_thermo_out",
    "single_point_from_dump",
    "voigt_stress",
    "write_normalized_trajectory",
]

#: Conversion from eV/Angstrom**3 to GPa.
EV_A3_TO_GPA = 160.2176634

#: The only ``thermo.out`` column layout documented for GPUMD 5.9.1.
THERMO_COLUMNS: tuple[str, ...] = (
    "T",
    "KE",
    "PE",
    "sxx",
    "syy",
    "szz",
    "syz",
    "sxz",
    "sxy",
    "ax",
    "ay",
    "az",
    "bx",
    "by",
    "bz",
    "cx",
    "cy",
    "cz",
)

#: Voigt order used by GPUMD / ASE for stress tensors.
VOIGT_ORDER: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (0, 1),
)

_HEADER_PAIR = re.compile(
    r"([A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*" r"(?:\"([^\"]*)\"|'([^']*)'|(\S+))"
)
_MINIMIZE_STEP = re.compile(
    r"step\s+(\d+):\s+total_potential\s*=\s*([-+0-9.eE]+)\s*eV,"
    r"\s*f_max\s*=\s*([-+0-9.eE]+)\s*eV/A"
)
_MINIMIZE_COMMAND = re.compile(r"^\s*minimize\s+(?P<body>.+)$")


class GPUMDParseError(ValueError):
    """Raised when native GPUMD output is missing, truncated or inconsistent."""


@dataclass
class Frame:
    """One extended-XYZ frame from a GPUMD ``dump_xyz`` file.

    Attributes:
        atoms: ASE ``Atoms`` carrying wrapped positions, cell and PBC.
        step: Step index if the header records one (GPUMD does not; the
            normalised mliport trajectory does).
        time_fs: Simulation time in fs from ``Time=``.
        energy_eV: Total potential energy in eV.
        virial_eV: Native GPUMD virial tensor in eV (row-major, same sign as
            GPUMD stress times volume), or None when absent.
        stress_GPa: ASE-convention stress tensor in GPa, or None.
        stress_eV_A3: ASE-convention stress tensor in eV/Angstrom**3, or None.
        forces: Per-atom forces in eV/Angstrom, or None.
        velocities: Per-atom velocities in Angstrom/fs, or None.
        unwrapped_positions: Per-atom unwrapped positions, or None.  This is
            never filled from the wrapped ``pos`` columns.
        extra: Remaining per-atom arrays plus provenance entries such as
            ``raw_header``, ``native_stress_eV_A3`` and ``properties``.
    """

    atoms: Atoms
    step: int | None
    time_fs: float | None
    energy_eV: float
    virial_eV: np.ndarray | None
    stress_GPa: np.ndarray | None
    stress_eV_A3: np.ndarray | None
    forces: np.ndarray | None
    velocities: np.ndarray | None
    unwrapped_positions: np.ndarray | None
    extra: dict[str, Any] = field(default_factory=dict)


def _finite(value: Any, *, context: str) -> float:
    """Parse a finite float or raise :class:`GPUMDParseError`."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise GPUMDParseError(f"{context}: expected a number, got {value!r}") from exc
    if not math.isfinite(numeric):
        raise GPUMDParseError(f"{context}: non-finite value {value!r}")
    return numeric


def _parse_header(line: str) -> dict[str, str]:
    """Parse an extended-XYZ comment line into a lower-cased key mapping."""
    header: dict[str, str] = {}
    for match in _HEADER_PAIR.finditer(line):
        key = match.group(1).lower()
        value = match.group(2)
        if value is None:
            value = match.group(3)
        if value is None:
            value = match.group(4)
        header[key] = value
    return header


def _header_float(header: dict[str, str], *names: str) -> float | None:
    """Return the first present finite float header value."""
    for name in names:
        if name in header:
            return _finite(header[name], context=f"header {name!r}")
    return None


def _header_int(header: dict[str, str], *names: str) -> int | None:
    """Return the first present integer header value."""
    for name in names:
        if name not in header:
            continue
        text = header[name].strip()
        try:
            return int(text)
        except ValueError:
            numeric = _finite(text, context=f"header {name!r}")
            if float(numeric).is_integer():
                return int(numeric)
            raise GPUMDParseError(
                f"header {name!r}: expected an integer, got {text!r}"
            ) from None
    return None


def _parse_vector(value: str, count: int, *, name: str) -> np.ndarray:
    """Parse exactly ``count`` finite floats from a header field."""
    tokens = value.split()
    if len(tokens) != count:
        raise GPUMDParseError(
            f"header {name!r}: expected {count} values, got {len(tokens)}"
        )
    values = np.array(
        [_finite(token, context=f"header {name!r}") for token in tokens],
        dtype=float,
    )
    return values


def _parse_lattice(header: dict[str, str]) -> np.ndarray | None:
    """Parse the row-major ``Lattice`` header field."""
    if "lattice" not in header:
        return None
    values = _parse_vector(header["lattice"], 9, name="Lattice")
    return values.reshape(3, 3)


def _parse_pbc(header: dict[str, str]) -> tuple[bool, bool, bool]:
    """Parse a PBC header such as ``pbc="T T F"``."""
    if "pbc" not in header:
        return (False, False, False)
    tokens = header["pbc"].replace(",", " ").split()
    if len(tokens) != 3:
        raise GPUMDParseError(f"header 'pbc': expected 3 flags, got {header['pbc']!r}")
    flags: list[bool] = []
    for token in tokens:
        lowered = token.strip().lower()
        if lowered in {"t", "true", ".true.", "1"}:
            flags.append(True)
        elif lowered in {"f", "false", ".false.", "0"}:
            flags.append(False)
        else:
            raise GPUMDParseError(f"header 'pbc': invalid flag {token!r}")
    return (flags[0], flags[1], flags[2])


def _parse_properties(value: str) -> tuple[dict[str, dict[str, Any]], int]:
    """Parse an extended-XYZ ``Properties=`` specification."""
    tokens = value.split(":")
    if not tokens or len(tokens) % 3 != 0:
        raise GPUMDParseError(
            f"Properties specification is not name:type:count triples: {value!r}"
        )
    properties: dict[str, dict[str, Any]] = {}
    start = 0
    for index in range(0, len(tokens), 3):
        name = tokens[index].strip().lower()
        kind = tokens[index + 1].strip().upper()
        count_text = tokens[index + 2].strip()
        if not name:
            raise GPUMDParseError(
                f"Properties specification has an empty name: {value!r}"
            )
        try:
            count = int(count_text)
        except ValueError as exc:
            raise GPUMDParseError(
                f"Properties {name!r}: non-integer column count {count_text!r}"
            ) from exc
        if count <= 0:
            raise GPUMDParseError(
                f"Properties {name!r}: column count must be > 0, got {count}"
            )
        if name in properties:
            raise GPUMDParseError(f"Properties declares {name!r} twice")
        properties[name] = {"kind": kind, "count": count, "start": start}
        start += count
    if "species" not in properties or "pos" not in properties:
        raise GPUMDParseError("Properties must declare both 'species' and 'pos'")
    if properties["pos"]["count"] != 3:
        raise GPUMDParseError("Properties 'pos' must have 3 columns")
    return properties, start


def _property_array(
    numeric: dict[str, np.ndarray],
    *names: str,
) -> np.ndarray | None:
    """Return the first present per-atom property array."""
    for name in names:
        if name in numeric:
            return numeric[name]
    return None


def _build_frame(
    *,
    natoms: int,
    header_line: str,
    atom_lines: list[str],
    frame_index: int,
) -> Frame:
    """Build one :class:`Frame` from its raw lines."""
    header = _parse_header(header_line)
    if "properties" not in header:
        raise GPUMDParseError(
            f"frame {frame_index}: header has no Properties= specification"
        )
    properties, total_columns = _parse_properties(header["properties"])

    declared_natoms = _header_int(header, "natoms", "num_atoms", "number_of_atoms")
    if declared_natoms is not None and declared_natoms != natoms:
        raise GPUMDParseError(
            f"frame {frame_index}: header declares {declared_natoms} atoms but "
            f"the count line says {natoms}"
        )

    species_start = properties["species"]["start"]
    symbols: list[str] = []
    for line_index, line in enumerate(atom_lines):
        tokens = line.split()
        if len(tokens) != total_columns:
            raise GPUMDParseError(
                f"frame {frame_index} atom {line_index}: expected "
                f"{total_columns} columns, got {len(tokens)}"
            )
        symbol = tokens[species_start]
        if not symbol or not symbol[0].isalpha():
            raise GPUMDParseError(
                f"frame {frame_index} atom {line_index}: invalid species " f"{symbol!r}"
            )
        symbols.append(symbol)

    numeric: dict[str, np.ndarray] = {}
    for name, spec in properties.items():
        if name == "species":
            continue
        start = spec["start"]
        count = spec["count"]
        rows = []
        for line_index, line in enumerate(atom_lines):
            tokens = line.split()
            rows.append(
                [
                    _finite(
                        token,
                        context=(
                            f"frame {frame_index} atom {line_index} "
                            f"property {name!r}"
                        ),
                    )
                    for token in tokens[start : start + count]
                ]
            )
        numeric[name] = np.asarray(rows, dtype=float)

    if symbols == []:
        raise GPUMDParseError(f"frame {frame_index}: frame has no atoms")
    positions = numeric.get("pos")
    if positions is None or positions.shape != (natoms, 3):
        raise GPUMDParseError(f"frame {frame_index}: pos must have shape ({natoms}, 3)")

    cell = _parse_lattice(header)
    pbc = _parse_pbc(header)
    if any(pbc) and cell is None:
        raise GPUMDParseError(
            f"frame {frame_index}: periodic pbc={pbc} but no Lattice= header"
        )
    try:
        from ase import Atoms  # noqa: PLC0415 - ASE import kept lazy
    except ImportError as exc:  # pragma: no cover - ASE is a hard dependency
        raise GPUMDParseError("ASE is required to build frame Atoms objects") from exc
    try:
        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=cell,
            pbc=pbc,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GPUMDParseError(
            f"frame {frame_index}: cannot build ASE Atoms: {exc}"
        ) from exc

    energy = _header_float(header, "energy", "energy_ev", "potential_energy")
    if energy is None:
        raise GPUMDParseError(
            f"frame {frame_index}: header has no finite energy= field"
        )
    time_fs = _header_float(header, "time", "time_fs")
    step = _header_int(header, "step", "step_index")

    virial: np.ndarray | None = None
    if "virial" in header:
        virial = _parse_vector(header["virial"], 9, name="virial").reshape(3, 3)
    native_stress: np.ndarray | None = None
    if "stress" in header:
        native_stress = _parse_vector(header["stress"], 9, name="stress").reshape(3, 3)

    stress_eV_A3 = None if native_stress is None else -native_stress
    stress_GPa = None if stress_eV_A3 is None else stress_eV_A3 * EV_A3_TO_GPA

    forces = _property_array(numeric, "forces", "force")
    velocities = _property_array(numeric, "vel", "velocities")
    unwrapped = _property_array(numeric, "unwrapped_position", "unwrapped_positions")
    if unwrapped is not None and unwrapped.shape != (natoms, 3):
        raise GPUMDParseError(
            f"frame {frame_index}: unwrapped_position must have shape "
            f"({natoms}, 3), got {unwrapped.shape}"
        )
    if forces is not None and forces.shape != (natoms, 3):
        raise GPUMDParseError(
            f"frame {frame_index}: forces must have shape ({natoms}, 3)"
        )
    if velocities is not None and velocities.shape != (natoms, 3):
        raise GPUMDParseError(
            f"frame {frame_index}: velocities must have shape ({natoms}, 3)"
        )

    consumed = {"pos", "forces", "force", "vel", "velocities"}
    extra: dict[str, Any] = {
        "raw_header": header_line,
        "frame_index": frame_index,
        "properties": properties,
        "lattice": None if cell is None else cell.tolist(),
        "pbc": list(pbc),
        "native_virial_eV": None if virial is None else virial.tolist(),
        "native_stress_eV_A3": (
            None if native_stress is None else native_stress.tolist()
        ),
        "native_stress_GPa": (
            None if native_stress is None else (native_stress * EV_A3_TO_GPA).tolist()
        ),
        "ase_virial_eV": None if virial is None else (-virial).tolist(),
        "unwrapped_position_present": unwrapped is not None,
        "wrapped_positions": positions,
    }
    for name, array in numeric.items():
        if name in consumed:
            continue
        extra[name] = array

    return Frame(
        atoms=atoms,
        step=step,
        time_fs=time_fs,
        energy_eV=energy,
        virial_eV=virial,
        stress_GPa=stress_GPa,
        stress_eV_A3=stress_eV_A3,
        forces=forces,
        velocities=velocities,
        unwrapped_positions=unwrapped,
        extra=extra,
    )


def iter_extxyz_frames(path: str | Path) -> Iterator[Frame]:
    """Yield :class:`Frame` objects from an extended-XYZ file.

    Args:
        path: Extended-XYZ file produced by ``dump_xyz`` (or a normalised
            mliport trajectory).

    Yields:
        One :class:`Frame` per record, in file order.

    Raises:
        GPUMDParseError: If the file is missing, empty or truncated, or if any
            frame is internally inconsistent.
    """
    source = Path(path)
    if not source.is_file():
        raise GPUMDParseError(f"Extended-XYZ file not found: {source}")

    with source.open("r", encoding="utf-8", errors="strict") as handle:
        frame_index = 0
        while True:
            count_line = handle.readline()
            if count_line == "":
                if frame_index == 0:
                    raise GPUMDParseError(f"Extended-XYZ file is empty: {source}")
                return
            if not count_line.strip():
                continue
            try:
                natoms = int(count_line.strip())
            except ValueError as exc:
                raise GPUMDParseError(
                    f"{source}: expected an atom count, got {count_line.strip()!r}"
                ) from exc
            if natoms <= 0:
                raise GPUMDParseError(f"{source}: atom count must be > 0, got {natoms}")
            header_line = handle.readline()
            if header_line == "":
                raise GPUMDParseError(
                    f"{source}: frame {frame_index} has a count line but no header"
                )
            header_line = header_line.rstrip("\r\n")
            atom_lines: list[str] = []
            for atom_index in range(natoms):
                line = handle.readline()
                if line == "":
                    raise GPUMDParseError(
                        f"{source}: frame {frame_index} is truncated after "
                        f"{atom_index} of {natoms} atoms"
                    )
                if not line.strip():
                    raise GPUMDParseError(
                        f"{source}: frame {frame_index} atom {atom_index} is blank"
                    )
                atom_lines.append(line.rstrip("\r\n"))
            yield _build_frame(
                natoms=natoms,
                header_line=header_line,
                atom_lines=atom_lines,
                frame_index=frame_index,
            )
            frame_index += 1


def parse_extxyz_frames(path: str | Path) -> list[Frame]:
    """Parse every :class:`Frame` in an extended-XYZ file."""
    return list(iter_extxyz_frames(path))


def parse_thermo_out(path: str | Path) -> dict[str, np.ndarray]:
    """Parse a GPUMD ``thermo.out`` file into named arrays.

    GPUMD does not write a step index; the returned ``step`` array is therefore
    the 1-based row number (documented explicitly so no caller confuses it with
    an MD integration step).

    Args:
        path: ``thermo.out`` file.

    Returns:
        Mapping from each documented column name to a float64 array, plus
        ``step``.

    Raises:
        GPUMDParseError: If the header is missing, its format version is not
            supported, a numeric row has the wrong field count, the file is
            empty, or any value is non-finite.
    """
    source = Path(path)
    if not source.is_file():
        raise GPUMDParseError(f"thermo.out not found: {source}")
    lines = source.read_text(encoding="utf-8", errors="strict").splitlines()
    if not lines:
        raise GPUMDParseError(f"thermo.out is empty: {source}")

    columns: tuple[str, ...] | None = None
    format_version: str | None = None
    declared_atoms: int | None = None
    rows: list[list[float]] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            payload = stripped.lstrip("#").strip()
            tokens = payload.split()
            if not tokens:
                continue
            keyword = tokens[0].lower()
            if keyword == "columns":
                columns = tuple(token.strip() for token in tokens[1:])
            elif keyword == "format_version":
                if len(tokens) != 2:
                    raise GPUMDParseError(
                        f"{source}:{line_number}: malformed format_version line"
                    )
                format_version = tokens[1]
            elif keyword == "num_atoms":
                try:
                    declared_atoms = int(tokens[1])
                except (IndexError, ValueError) as exc:
                    raise GPUMDParseError(
                        f"{source}:{line_number}: malformed num_atoms line"
                    ) from exc
            continue

        tokens = stripped.split()
        if columns is None:
            raise GPUMDParseError(
                f"{source}:{line_number}: numeric row before the # columns header"
            )
        if len(tokens) != len(columns):
            raise GPUMDParseError(
                f"{source}:{line_number}: expected {len(columns)} fields, "
                f"got {len(tokens)}"
            )
        rows.append(
            [
                _finite(token, context=f"{source}:{line_number} field {index}")
                for index, token in enumerate(tokens)
            ]
        )

    if columns is None:
        raise GPUMDParseError(f"{source}: no '# columns ...' header found")
    if tuple(columns) != THERMO_COLUMNS:
        raise GPUMDParseError(
            f"{source}: unsupported thermo columns {columns!r}; expected "
            f"{THERMO_COLUMNS!r}"
        )
    if format_version is not None and format_version != "1":
        raise GPUMDParseError(
            f"{source}: unsupported thermo format_version {format_version!r}"
        )
    if declared_atoms is not None and declared_atoms <= 0:
        raise GPUMDParseError(f"{source}: num_atoms must be > 0, got {declared_atoms}")
    if not rows:
        raise GPUMDParseError(f"{source}: no numeric thermo rows")

    matrix = np.asarray(rows, dtype=float)
    result = {name: matrix[:, index] for index, name in enumerate(THERMO_COLUMNS)}
    result["step"] = np.arange(1, matrix.shape[0] + 1, dtype=int)
    return result


def voigt_stress(stress: Any) -> np.ndarray:
    """Return the 6-component Voigt form of a 3x3 stress tensor.

    Order is GPUMD/ASE ``xx yy zz yz xz xy``.

    Args:
        stress: 3x3 matrix in any single stress convention.

    Returns:
        Float array of length 6.
    """
    matrix = np.asarray(stress, dtype=float)
    if matrix.shape != (3, 3):
        raise GPUMDParseError(f"stress must be 3x3, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise GPUMDParseError("stress contains NaN or Inf")
    return np.array([matrix[i, j] for i, j in VOIGT_ORDER], dtype=float)


def _require_forces(frame: Frame, *, context: str) -> np.ndarray:
    """Return finite forces or raise."""
    if frame.forces is None:
        raise GPUMDParseError(f"{context}: frame has no forces column")
    forces = np.asarray(frame.forces, dtype=float)
    if forces.shape != (len(frame.atoms), 3):
        raise GPUMDParseError(
            f"{context}: forces shape {forces.shape} disagrees with "
            f"{len(frame.atoms)} atoms"
        )
    if not np.all(np.isfinite(forces)):
        raise GPUMDParseError(f"{context}: forces contain NaN or Inf")
    return forces


def single_point_from_dump(path: str | Path) -> dict[str, Any]:
    """Extract a single-point result from one ``sp.xyz`` dump.

    Args:
        path: Extended-XYZ file written by the SP ``run.in`` (exactly one
            frame is required).

    Returns:
        Mapping with ``energy`` (total eV), ``energy_per_atom``, ``forces``
        (Nx3 eV/Angstrom), ``stress`` (ASE Voigt, eV/Angstrom**3), ``virial``
        (3x3 native eV) and the verification inputs ``positions`` (wrapped),
        ``unwrapped_positions``, ``cell``, ``pbc`` and ``symbols``.

    Raises:
        GPUMDParseError: If the file does not contain exactly one complete
            frame or a required quantity is missing.
    """
    frames = parse_extxyz_frames(path)
    if len(frames) != 1:
        raise GPUMDParseError(
            f"{path}: expected exactly one SP frame, found {len(frames)}"
        )
    frame = frames[0]
    natoms = len(frame.atoms)
    forces = _require_forces(frame, context=f"{path} SP frame")
    stress_eV_A3 = frame.stress_eV_A3
    stress_source = "header"
    if stress_eV_A3 is None:
        if frame.virial_eV is not None:
            volume = float(frame.atoms.cell.volume)
            if not math.isfinite(volume) or volume <= 0.0:
                raise GPUMDParseError(
                    f"{path}: cannot derive stress from virial with "
                    f"non-positive cell volume {volume!r}"
                )
            stress_eV_A3 = -(frame.virial_eV / volume)
            stress_source = "virial/volume"
        else:
            raise GPUMDParseError(
                f"{path}: SP frame has neither stress= nor virial= header"
            )
    return {
        "energy": frame.energy_eV,
        "energy_per_atom": frame.energy_eV / natoms,
        "forces": forces,
        "stress": voigt_stress(stress_eV_A3),
        "stress_source": stress_source,
        "virial": frame.virial_eV,
        "ase_virial": (
            None if frame.virial_eV is None else -np.asarray(frame.virial_eV)
        ),
        "natoms": natoms,
        "symbols": list(frame.atoms.get_chemical_symbols()),
        "positions": np.asarray(frame.atoms.get_positions(), dtype=float),
        "unwrapped_positions": frame.unwrapped_positions,
        "cell": np.asarray(frame.atoms.cell, dtype=float),
        "pbc": tuple(bool(value) for value in frame.atoms.pbc),
        "positions_convention": "wrapped",
        "step": frame.step,
        "time_fs": frame.time_fs,
    }


def _parse_minimize_stdout(path: Path | None) -> dict[str, Any] | None:
    """Extract the last reported minimiser step and f_max from stdout."""
    if path is None or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(_MINIMIZE_STEP.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    return {
        "nsteps": int(last.group(1)),
        "native_energy_eV": float(last.group(2)),
        "native_fmax_eV_A": float(last.group(3)),
        "reported_steps": [int(match.group(1)) for match in matches],
    }


def _parse_minimize_run_in(path: Path | None) -> dict[str, Any] | None:
    """Read the native minimiser criterion from a sibling ``run.in``."""
    if path is None or not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _MINIMIZE_COMMAND.match(line)
        if match is None:
            continue
        tokens = match.group("body").split()
        if len(tokens) < 3:
            return None
        try:
            tolerance = float(tokens[1])
            max_steps = int(tokens[2])
        except ValueError:
            return None
        return {
            "optimizer": tokens[0].lower(),
            "force_tolerance_eV_A": tolerance,
            "max_steps": max_steps,
            "raw_line": line.strip(),
        }
    return None


def optimization_result(
    path: str | Path,
    *,
    force_tolerance: float | None = None,
    max_steps: int | None = None,
    stdout_path: str | Path | None = None,
    run_in_path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract the final result of a native ``minimize`` run.

    GPUMD's native convergence criterion is the largest absolute Cartesian
    force *component* ``f_max < force_tolerance`` (not ASE's per-atom force
    norm).  The flag is derived from that criterion: the recorded final
    ``f_max`` from stdout and the final dumped forces must both be below the
    tolerance.  When the tolerance is not supplied it is read from the sibling
    ``run.in``; if neither is available ``converged`` is None rather than a
    fabricated True.

    Args:
        path: Extended-XYZ file written after ``minimize`` (final frame used).
        force_tolerance: Native force tolerance in eV/Angstrom.
        max_steps: Native step limit (used to identify a max-steps exit).
        stdout_path: GPUMD stdout log; defaults to a sibling ``stdout.log``.
        run_in_path: ``run.in`` to read the criterion from; defaults to a
            sibling ``run.in``.

    Returns:
        Mapping with at least ``energy``, ``forces``, ``fmax_component``,
        ``fmax_norm``, ``stress``, ``converged`` and ``natoms``.

    Raises:
        GPUMDParseError: If no complete frame or no forces are available.
    """
    source = Path(path)
    frames = parse_extxyz_frames(source)
    if not frames:
        raise GPUMDParseError(f"{source}: no optimisation frames")
    final = frames[-1]
    forces = _require_forces(final, context=f"{source} final frame")
    natoms = len(final.atoms)

    stdout = (
        Path(stdout_path) if stdout_path is not None else source.parent / "stdout.log"
    )
    run_in = Path(run_in_path) if run_in_path is not None else source.parent / "run.in"
    native = _parse_minimize_stdout(stdout)
    settings = _parse_minimize_run_in(run_in)

    tolerance = force_tolerance
    if tolerance is None and settings is not None:
        tolerance = settings["force_tolerance_eV_A"]
    step_limit = max_steps
    if step_limit is None and settings is not None:
        step_limit = settings["max_steps"]

    fmax_component = float(np.max(np.abs(forces)))
    fmax_norm = float(np.max(np.linalg.norm(forces, axis=1)))
    nsteps = None if native is None else native["nsteps"]
    native_fmax = None if native is None else native["native_fmax_eV_A"]

    converged: bool | None
    reason: str
    if tolerance is None:
        converged = None
        reason = "native force tolerance unavailable; refusing to claim convergence"
    else:
        measured = fmax_component
        if native_fmax is not None:
            measured = max(measured, float(native_fmax))
        converged = measured <= float(tolerance)
        reason = (
            f"largest |force component| {measured:.6g} eV/A vs native "
            f"tolerance {float(tolerance):.6g} eV/A"
        )

    stress = None
    if final.stress_eV_A3 is not None:
        stress = voigt_stress(final.stress_eV_A3)
    return {
        "energy": final.energy_eV,
        "energy_per_atom": final.energy_eV / natoms,
        "forces": forces,
        "fmax_component": fmax_component,
        "fmax_norm": fmax_norm,
        "stress": stress,
        "converged": converged,
        "converged_reason": reason,
        "natoms": natoms,
        "nsteps": nsteps,
        "native_fmax_eV_A": native_fmax,
        "force_tolerance_eV_A": tolerance,
        "max_steps": step_limit,
        "max_steps_reached": (
            nsteps is not None
            and step_limit is not None
            and nsteps >= step_limit
            and converged is not True
        ),
        "optimizer": None if settings is None else settings["optimizer"],
        "native": native,
        "step": final.step,
        "time_fs": final.time_fs,
    }


def md_frames_summary(path: str | Path) -> dict[str, Any]:
    """Summarise an MD ``dump_xyz`` trajectory without loading it twice.

    Args:
        path: Extended-XYZ trajectory.

    Returns:
        Mapping with ``nframes``, ``natoms``, ``steps`` (per frame, ``None``
        when GPUMD did not record one), ``times_fs`` (per frame),
        ``unwrapped_available`` and per-quantity availability flags.

    Raises:
        GPUMDParseError: If the file is empty, truncated, or frames disagree on
            the atom count.
    """
    source = Path(path)
    nframes = 0
    natoms: int | None = None
    steps: list[int | None] = []
    times: list[float | None] = []
    unwrapped_available = True
    forces_available = True
    velocities_available = True
    for frame in iter_extxyz_frames(source):
        count = len(frame.atoms)
        if natoms is None:
            natoms = count
        elif count != natoms:
            raise GPUMDParseError(
                f"{source}: frame {nframes} has {count} atoms, expected {natoms}"
            )
        steps.append(frame.step)
        times.append(frame.time_fs)
        unwrapped_available = (
            unwrapped_available
            and frame.unwrapped_positions is not None
            and np.asarray(frame.unwrapped_positions).shape == (count, 3)
        )
        forces_available = forces_available and frame.forces is not None
        velocities_available = velocities_available and frame.velocities is not None
        nframes += 1
    if nframes == 0 or natoms is None:
        raise GPUMDParseError(f"{source}: trajectory contains no frames")
    return {
        "nframes": nframes,
        "natoms": natoms,
        "steps": steps,
        "times_fs": times,
        "unwrapped_available": bool(unwrapped_available),
        "forces_available": bool(forces_available),
        "velocities_available": bool(velocities_available),
    }


def write_normalized_trajectory(
    frames: list[Frame],
    path: str | Path,
    *,
    dt_fs: float | None = None,
    source_path: str | Path | None = None,
) -> Path:
    """Write a deterministic, ASE-readable normalised extended-XYZ trajectory.

    Frames carry ``step``, ``time``, ``energy``, PBC, Lattice and the chosen
    positions.  When unwrapped positions exist they become ``pos`` and the
    wrapped positions are retained as ``wrapped_position``; otherwise ``pos``
    is the wrapped position and the caller must record that MSD analysis is not
    eligible (no replacement is invented here).

    Args:
        frames: Frames parsed from native GPUMD output.
        path: Destination file.
        dt_fs: MD time step, used only to derive a missing frame ``step`` from
            ``time_fs``.
        source_path: Optional raw source recorded in every frame's info.

    Returns:
        The destination path.

    Raises:
        GPUMDParseError: If ``frames`` is empty or ASE cannot write the file.
    """
    if not frames:
        raise GPUMDParseError("Cannot write an empty normalised trajectory")
    try:
        from ase.io import write as ase_write  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - ASE is a hard dependency
        raise GPUMDParseError("ASE is required to write a trajectory") from exc

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output: list[Atoms] = []
    for frame in frames:
        atoms = frame.atoms.copy()
        if frame.unwrapped_positions is not None:
            positions = np.asarray(frame.unwrapped_positions, dtype=float)
            convention = "unwrapped"
            atoms.set_array(
                "wrapped_position",
                np.asarray(frame.atoms.get_positions(), dtype=float),
            )
            atoms.set_array("unwrapped_position", positions)
        else:
            positions = np.asarray(frame.atoms.get_positions(), dtype=float)
            convention = "wrapped"
        atoms.set_positions(positions)
        if frame.forces is not None:
            atoms.set_array("forces", np.asarray(frame.forces, dtype=float))
        if frame.velocities is not None:
            atoms.set_array("vel", np.asarray(frame.velocities, dtype=float))

        step = frame.step
        if step is None and frame.time_fs is not None and dt_fs:
            step = int(round(float(frame.time_fs) / float(dt_fs)))
        if step is not None:
            atoms.info["step"] = int(step)
        if frame.time_fs is not None:
            atoms.info["time"] = float(frame.time_fs)
        atoms.info["energy"] = float(frame.energy_eV)
        atoms.info["positions_convention"] = convention
        if source_path is not None:
            atoms.info["source"] = str(source_path)
        atoms.calc = None
        output.append(atoms)

    try:
        ase_write(str(destination), output, format="extxyz")
    except (OSError, ValueError) as exc:
        raise GPUMDParseError(
            f"Could not write normalised trajectory {destination}: {exc}"
        ) from exc
    return destination
