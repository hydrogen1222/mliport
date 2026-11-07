"""GPUMD input generation: ``model.xyz``, ``run.in`` and NEP metadata.

GPUMD is a *native external process*, not an ASE calculator (PR #61 §C): one
`gpumd` invocation performs a whole SP / optimisation / MD run.  This module
owns the two files that define such a run and never emits a shell fragment.

Design rules that matter scientifically:

* ``model.xyz`` is written only after the structure has been proven to be a
  full 3-D periodic cell.  GPUMD silently wraps coordinates into the cell, so a
  partially periodic or singular cell would silently change the physical
  system.  We fail closed instead.
* Atom order is preserved exactly.  NEP returns per-atom forces in the order it
  read, so any sorting would attribute forces to the wrong atoms.
* Velocities are written in Angstrom/fs, the unit GPUMD documents for the
  ``vel`` property.  ASE stores momenta in its own internal time unit
  (``sqrt(amu * Angstrom**2 / eV)``, i.e. 10.18 fs); the conversion therefore
  includes the ``ase.units.fs`` factor.  See
  :func:`momenta_to_velocities_ang_per_fs`.
* The returned ``run.in`` text depends only on the function arguments, so the
  exact file can be reconstructed from the run-in provenance record.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import units as ase_units

if TYPE_CHECKING:
    from ase import Atoms

__all__ = [
    "ASE_TIME_UNIT_FS",
    "OPTIMIZER_MAPPING",
    "THERMOSTAT_MAPPING",
    "build_model_xyz",
    "build_run_in",
    "momenta_to_velocities_ang_per_fs",
    "parse_nep_model",
    "run_in_provenance",
    "validate_elements",
]

#: One ASE internal time unit expressed in femtoseconds.  ASE momenta are
#: mass * velocity in internal units; converting to GPUMD's Angstrom/fs needs
#: this factor (``ase.units.fs`` is the number of internal time units per fs,
#: so multiplying an internal velocity by it yields Angstrom/fs).
ASE_TIME_UNIT_FS = float(ase_units.fs)

#: The two native GPUMD minimisers.  mliport optimizer names are matched
#: case-insensitively; BFGS/LBFGS have no native GPUMD implementation and are
#: rejected rather than mapped onto a different algorithm.
OPTIMIZER_MAPPING: dict[str, str] = {
    "fire": "fire",
    "sd": "sd",
}

#: NVT thermostat mapping (PR #61 §C).  ``tau_param`` names the constructor
#: argument that supplies the physical coupling time in fs; ``inverse`` marks
#: the Langevin case where ``tau_T = 1 / friction_per_fs``.
THERMOSTAT_MAPPING: dict[str, dict[str, Any]] = {
    "LANGEVIN": {
        "ensemble": "nvt_lan",
        "tau_param": "friction_per_fs",
        "inverse": True,
    },
    "BUSSI": {
        "ensemble": "nvt_bdp",
        "tau_param": "bussi_tau_fs",
        "inverse": False,
    },
    "NHC": {
        "ensemble": "nvt_nhc",
        "tau_param": "nhc_tdamp_fs",
        "inverse": False,
    },
}

_SUPPORTED_CALC_TYPES = frozenset({"sp", "opt", "md"})
_SUPPORTED_ENSEMBLES = frozenset({"NVE", "NVT"})


def _finite_float(value: Any, *, name: str) -> float:
    """Return ``value`` as a finite float or raise ``ValueError``."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return numeric


def _positive_float(value: Any, *, name: str) -> float:
    """Return a strictly positive finite float or raise ``ValueError``."""
    numeric = _finite_float(value, name=name)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return numeric


def _positive_int(value: Any, *, name: str) -> int:
    """Return a strictly positive integer or raise ``ValueError``."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if numeric <= 0 or numeric != value:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return numeric


def _fmt(value: float) -> str:
    """Format a float compactly without losing the supplied value."""
    return format(float(value), ".15g")


def momenta_to_velocities_ang_per_fs(momenta: Any, masses: Any) -> np.ndarray:
    """Convert ASE momenta to GPUMD velocities in Angstrom/fs.

    ASE's ``momenta`` array is in internal units of
    ``sqrt(amu * Angstrom**2 / eV)`` = ``amu * Angstrom / (ASE time unit)``.
    One ASE time unit is roughly 10.18 fs, so ``momenta / masses`` alone is
    *not* Angstrom/fs; multiplying by :data:`ASE_TIME_UNIT_FS` is required.

    Args:
        momenta: Array of shape ``(N, 3)`` in ASE momentum units.
        masses: Array of shape ``(N,)`` in atomic mass units.

    Returns:
        Array of shape ``(N, 3)`` in Angstrom/fs.

    Raises:
        ValueError: If shapes disagree, masses are non-positive, or any value
            is non-finite.
    """
    momenta_array = np.asarray(momenta, dtype=float)
    masses_array = np.asarray(masses, dtype=float)
    if momenta_array.ndim != 2 or momenta_array.shape[1] != 3:
        raise ValueError(
            "momenta must have shape (N, 3), got " f"{momenta_array.shape}"
        )
    if masses_array.shape != (momenta_array.shape[0],):
        raise ValueError(
            "masses must have shape (N,), got "
            f"{masses_array.shape} for {momenta_array.shape[0]} atoms"
        )
    if not np.all(np.isfinite(momenta_array)):
        raise ValueError("momenta contain NaN or Inf")
    if not np.all(np.isfinite(masses_array)) or np.any(masses_array <= 0.0):
        raise ValueError("atomic masses must be finite and > 0")
    velocities = momenta_array / masses_array[:, np.newaxis]
    velocities *= ASE_TIME_UNIT_FS
    if not np.all(np.isfinite(velocities)):
        raise ValueError("converted velocities contain NaN or Inf")
    return velocities


def build_model_xyz(atoms: Atoms, path: str | Path) -> Path:
    """Write a GPUMD ``model.xyz`` for ``atoms`` and return its path.

    The file is extended XYZ with an explicit row-major ``Lattice=`` and
    ``pbc="T T T"``.  When the structure carries momenta the optional
    ``vel:R:3`` property is written in Angstrom/fs (GPUMD's documented unit).
    Atom order is never changed.

    Args:
        atoms: ASE ``Atoms`` with a full 3-D periodic cell.
        path: Destination file.  The caller owns the directory.

    Returns:
        The destination path.

    Raises:
        ValueError: For empty structures, non-finite coordinates, a singular
            cell, zero/negative volume, or partially periodic PBC.
    """
    destination = Path(path)
    symbols = list(atoms.get_chemical_symbols())
    if not symbols:
        raise ValueError("Cannot write a GPUMD model.xyz for an empty structure")

    positions = np.asarray(atoms.get_positions(), dtype=float)
    if positions.shape != (len(symbols), 3):
        raise ValueError(
            f"positions must have shape ({len(symbols)}, 3), got " f"{positions.shape}"
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError("Atomic positions contain NaN or Inf")

    cell = np.asarray(atoms.get_cell(), dtype=float)
    if cell.shape != (3, 3):
        raise ValueError(f"cell must be a 3x3 matrix, got {cell.shape}")
    if not np.all(np.isfinite(cell)):
        raise ValueError("The cell contains NaN or Inf")
    determinant = float(np.linalg.det(cell))
    volume = abs(determinant)
    if not math.isfinite(determinant) or abs(determinant) <= 1e-12:
        raise ValueError(
            "The cell is singular (|det| <= 1e-12); GPUMD needs a full 3-D cell"
        )
    if volume <= 0.0:
        raise ValueError(f"The cell volume must be > 0, got {volume!r}")

    pbc = np.asarray(atoms.pbc, dtype=bool)
    if pbc.shape != (3,):
        raise ValueError(f"pbc must have shape (3,), got {pbc.shape}")
    if not np.all(pbc):
        raise ValueError(
            "GPUMD is a fully periodic code; refusing to write model.xyz for "
            f"pbc={pbc.tolist()}. Set pbc=(True, True, True) explicitly or use "
            "a different engine; mliport never promotes partial PBC to T T T."
        )

    momenta_array = atoms.arrays.get("momenta")
    velocities: np.ndarray | None = None
    if momenta_array is not None:
        velocities = momenta_to_velocities_ang_per_fs(momenta_array, atoms.get_masses())

    properties = "species:S:1:pos:R:3"
    if velocities is not None:
        # GPUMD's reader recognises the property name ``vel`` exactly; an
        # unknown ``velocities`` column would be silently ignored and the
        # ``velocity <T>`` command would then re-initialise the gas instead of
        # reusing the provided state.
        properties += ":vel:R:3"

    lattice = " ".join(_fmt(value) for value in cell.reshape(-1))
    header = f'Lattice="{lattice}" Properties={properties} pbc="T T T"'

    lines = [str(len(symbols)), header]
    for index, symbol in enumerate(symbols):
        columns = [_fmt(value) for value in positions[index]]
        if velocities is not None:
            columns.extend(_fmt(value) for value in velocities[index])
        lines.append(f"{symbol:<2} " + " ".join(columns))

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def validate_elements(atoms: Atoms, nep_element_types: Any) -> None:
    """Reject species the NEP model was not trained on.

    Args:
        atoms: Structure to be evaluated.
        nep_element_types: Element symbols declared by the NEP model.

    Raises:
        ValueError: If the model declares no elements or the structure uses a
            species outside that list.
    """
    model_elements = {str(element).strip() for element in nep_element_types}
    if not model_elements:
        raise ValueError("The NEP model declares no element types")
    structure_elements = set(atoms.get_chemical_symbols())
    unknown = sorted(structure_elements - model_elements)
    if unknown:
        raise ValueError(
            "Structure contains element(s) not present in the NEP model: "
            f"{sorted(unknown)}. Model elements: {sorted(model_elements)}. "
            f"Structure elements: {sorted(structure_elements)}."
        )


def parse_nep_model(path: str | Path) -> dict[str, Any]:
    """Read and validate the NEP identity shared with the ASE runtime.

    Legacy nep4 and versioned GPUMD format_version 1 use independent version
    namespaces; model_type is strictly validated and stored separately.
    The full parameter block is left to the GPUMD executable to interpret.
    """
    from mliport.nep_model_format import read_nep_model_metadata  # noqa: PLC0415

    data = read_nep_model_metadata(path)
    radial = data["cutoff_radial"]
    angular = data["cutoff_angular"]
    if radial is None or angular is None:
        raise ValueError(f"NEP model file has no 'cutoff' line: {path}")
    return {
        "model_format": data["model_token"],
        "model_version": data["legacy_model_version"],
        "format_version": data["format_version"],
        "model_type": data["model_type"],
        "element_types": list(data["element_types"]),
        "cutoffs": {"radial_A": radial, "angular_A": angular},
        "cutoff_by_element_A": data["cutoff_by_element_A"],
        "max_neighbours_radial": data["max_neighbours_radial"],
        "max_neighbours_angular": data["max_neighbours_angular"],
        "zbl": data["zbl"],
        "energy_reference_eV": data["energy_reference_eV"],
        "cutoff_max_A": max(radial, angular),
        "raw_first_line": data["raw_first_line"],
    }


def _resolve_thermostat(
    *,
    dt_fs: float,
    thermostat: str | None,
    friction_per_fs: float | None,
    bussi_tau_fs: float | None,
    nhc_tdamp_fs: float | None,
) -> dict[str, Any]:
    """Resolve an NVT thermostat to GPUMD's ensemble line parameters."""
    key = "" if thermostat is None else str(thermostat).strip().upper()
    if key not in THERMOSTAT_MAPPING:
        raise ValueError(
            f"Unsupported NVT thermostat {thermostat!r}. Choose one of: "
            + ", ".join(sorted(THERMOSTAT_MAPPING))
        )
    mapping = THERMOSTAT_MAPPING[key]
    tau_param = mapping["tau_param"]
    supplied = {
        "friction_per_fs": friction_per_fs,
        "bussi_tau_fs": bussi_tau_fs,
        "nhc_tdamp_fs": nhc_tdamp_fs,
    }[tau_param]
    if supplied is None:
        raise ValueError(f"Thermostat {key} requires {tau_param}; it is not optional")
    value = _positive_float(supplied, name=tau_param)
    tau_t_fs = (1.0 / value) if mapping["inverse"] else value
    t_coup = int(round(tau_t_fs / dt_fs))
    if t_coup < 1:
        raise ValueError(
            f"Thermostat {key} gives T_coup = int(round({tau_t_fs!r} fs / "
            f"{dt_fs!r} fs)) = {t_coup}, but GPUMD requires T_coup >= 1. "
            "Reduce the coupling time or increase the MD time step."
        )
    return {
        "thermostat": key,
        "ensemble": mapping["ensemble"],
        "tau_param": tau_param,
        "tau_T_fs": tau_t_fs,
        "T_coup": t_coup,
    }


def _build(
    *,
    calc_type: str,
    nep_path: str | Path,
    ensemble: str | None,
    temperature: float | None,
    timestep_fs: float | None,
    steps: int | None,
    thermostat: str | None,
    friction_per_fs: float | None,
    bussi_tau_fs: float | None,
    nhc_tdamp_fs: float | None,
    seed: int | None,
    fmax: float | None,
    max_steps: int | None,
    optimizer: str | None,
    dump_interval: int | None,
    traj_filename: str,
    restart_interval: int | None,
    unwrapped: bool,
) -> tuple[str, dict[str, Any]]:
    """Build ``run.in`` text and its argument record (shared implementation)."""
    kind = str(calc_type).strip().lower()
    if kind not in _SUPPORTED_CALC_TYPES:
        raise ValueError(
            f"Unsupported calc_type {calc_type!r}. Choose one of: "
            + ", ".join(sorted(_SUPPORTED_CALC_TYPES))
        )
    resolved_nep = Path(nep_path).expanduser().resolve()
    if not resolved_nep.is_file():
        raise ValueError(f"NEP model file not found: {resolved_nep}")

    provenance: dict[str, Any] = {
        "schema": "mliport.gpumd-run-in/1",
        "calc_type": kind,
        "nep_path": str(nep_path),
        "nep_path_resolved": str(resolved_nep),
        "ensemble": None,
        "temperature": None,
        "timestep_fs": None,
        "steps": None,
        "thermostat": None,
        "friction_per_fs": friction_per_fs,
        "bussi_tau_fs": bussi_tau_fs,
        "nhc_tdamp_fs": nhc_tdamp_fs,
        "seed": None if seed is None else int(seed),
        "fmax": None if fmax is None else float(fmax),
        "max_steps": None if max_steps is None else int(max_steps),
        "optimizer": None if optimizer is None else str(optimizer),
        "dump_interval": None if dump_interval is None else int(dump_interval),
        "traj_filename": str(traj_filename),
        "restart_interval": (
            None if restart_interval is None else int(restart_interval)
        ),
        "unwrapped": bool(unwrapped),
        "thermostat_mapping": None,
        "T_coup": None,
    }

    lines = [f"potential {resolved_nep}"]

    if kind == "sp":
        lines.extend(
            [
                "velocity 1",
                "ensemble nve",
                "time_step 0",
                "dump_xyz 1 sp.xyz force virial precision double",
                "run 1",
            ]
        )
        provenance["traj_filename"] = "sp.xyz"
        return "\n".join(lines) + "\n", provenance

    if kind == "opt":
        if optimizer is None:
            raise ValueError("calc_type='opt' requires an optimizer")
        optimizer_key = str(optimizer).strip().lower()
        if optimizer_key not in OPTIMIZER_MAPPING:
            raise ValueError(
                f"Optimizer {optimizer!r} has no native GPUMD implementation. "
                "Choose one of: fire, sd (case-insensitive)."
            )
        force_tolerance = _positive_float(fmax, name="fmax")
        max_steps_value = _positive_int(max_steps, name="max_steps")
        lines.extend(
            [
                f"minimize {OPTIMIZER_MAPPING[optimizer_key]} "
                f"{_fmt(force_tolerance)} {max_steps_value}",
                "velocity 1",
                "ensemble nve",
                "time_step 0",
                "dump_xyz 1 opt.xyz force virial precision double",
                "run 1",
            ]
        )
        provenance["fmax"] = force_tolerance
        provenance["max_steps"] = max_steps_value
        provenance["optimizer"] = optimizer_key
        provenance["traj_filename"] = "opt.xyz"
        return "\n".join(lines) + "\n", provenance

    # MD
    ensemble_key = "" if ensemble is None else str(ensemble).strip().upper()
    if ensemble_key not in _SUPPORTED_ENSEMBLES:
        raise ValueError(f"Unsupported ensemble {ensemble!r}. Choose one of: NVE, NVT.")
    temperature_value = _positive_float(temperature, name="temperature")
    dt_fs = _positive_float(timestep_fs, name="timestep_fs")
    step_count = _positive_int(steps, name="steps")
    dump_every = _positive_int(
        dump_interval if dump_interval is not None else 1,
        name="dump_interval",
    )
    if seed is not None:
        seed_value = _positive_int(seed, name="seed")
        velocity_line = f"velocity {_fmt(temperature_value)} seed {seed_value}"
    else:
        seed_value = None
        velocity_line = f"velocity {_fmt(temperature_value)}"

    lines.append(velocity_line)
    lines.append(f"time_step {_fmt(dt_fs)}")
    if ensemble_key == "NVE":
        lines.append("ensemble nve")
    else:
        mapping = _resolve_thermostat(
            dt_fs=dt_fs,
            thermostat=thermostat,
            friction_per_fs=friction_per_fs,
            bussi_tau_fs=bussi_tau_fs,
            nhc_tdamp_fs=nhc_tdamp_fs,
        )
        lines.append(
            f"ensemble {mapping['ensemble']} {_fmt(temperature_value)} "
            f"{_fmt(temperature_value)} {mapping['T_coup']}"
        )
        provenance["thermostat_mapping"] = mapping
        provenance["T_coup"] = mapping["T_coup"]
        # Record the user-facing thermostat name at the top level too: the
        # mapping dict carries it, but a reader scanning `provenance["thermostat"]`
        # must not see `null` for an NVT run that clearly has a thermostat.
        provenance["thermostat"] = mapping["thermostat"]

    lines.append(f"dump_thermo {dump_every}")
    dump_xyz = (
        f"dump_xyz {dump_every} {traj_filename} precision double " "force virial mass"
    )
    if unwrapped:
        dump_xyz += " unwrapped_position"
    lines.append(dump_xyz)
    if restart_interval is not None:
        restart_every = _positive_int(restart_interval, name="restart_interval")
        # GPUMD 5.9.1 accepts only the interval here and writes the fixed
        # name ``restart.xyz``.  The file is *not* a reproducible checkpoint
        # on its own (no RNG state), so provenance records it as raw output
        # only.
        lines.append(f"dump_restart {restart_every}")
        provenance["restart_interval"] = restart_every
    lines.append(f"run {step_count}")

    provenance["ensemble"] = ensemble_key
    provenance["temperature"] = temperature_value
    provenance["timestep_fs"] = dt_fs
    provenance["steps"] = step_count
    provenance["dump_interval"] = dump_every
    provenance["seed"] = seed_value
    return "\n".join(lines) + "\n", provenance


def build_run_in(
    *,
    calc_type: str,
    nep_path: str | Path,
    ensemble: str | None = None,
    temperature: float | None = None,
    timestep_fs: float | None = None,
    steps: int | None = None,
    thermostat: str | None = None,
    friction_per_fs: float | None = None,
    bussi_tau_fs: float | None = None,
    nhc_tdamp_fs: float | None = None,
    seed: int | None = None,
    fmax: float | None = None,
    max_steps: int | None = None,
    optimizer: str | None = None,
    dump_interval: int | None = None,
    traj_filename: str = "traj.xyz",
    restart_interval: int | None = None,
    unwrapped: bool = True,
) -> str:
    """Build the exact ``run.in`` text for a native GPUMD calculation.

    The returned text is written verbatim by the caller; the executable is
    launched as an argv list (never via a shell).  All values needed to
    reproduce the file are captured by :func:`run_in_provenance`.

    Args:
        calc_type: ``"sp"``, ``"opt"`` or ``"md"`` (case-insensitive).
        nep_path: NEP model file used by the ``potential`` command.
        ensemble: ``"NVE"`` or ``"NVT"`` (required for MD).
        temperature: Initial/target temperature in K (required for MD).
        timestep_fs: MD time step in fs (required for MD).
        steps: Number of MD steps (required for MD).
        thermostat: NVT thermostat name (``LANGEVIN``/``BUSSI``/``NHC``).
        friction_per_fs: Langevin friction in 1/fs (``tau_T = 1/friction``).
        bussi_tau_fs: Bussi coupling time in fs.
        nhc_tdamp_fs: Nose-Hoover-chain damping time in fs.
        seed: Optional velocity-initialisation seed.
        fmax: Native minimiser force tolerance in eV/Angstrom.
        max_steps: Native minimiser step limit.
        optimizer: ``"fire"`` or ``"sd"`` (case-insensitive).
        dump_interval: ``dump_thermo``/``dump_xyz`` interval in steps.
        traj_filename: Raw trajectory filename for MD.
        restart_interval: Optional ``dump_restart`` interval.
        unwrapped: Request the ``unwrapped_position`` column for MD.

    Returns:
        The ``run.in`` text, ending with a newline.

    Note:
        The verified GPUMD 5.9.1 binary accepts ``dump_restart <interval>``
        only and always writes the fixed name ``restart.xyz``; it rejects the
        two-argument form.  The interval is therefore the only restart value
        emitted, and the raw file is recorded for provenance rather than
        treated as a reproducible checkpoint (it carries no RNG state).

    Raises:
        ValueError: For unsupported calculation types, ensembles, optimizers
            or thermostats, missing required values, or ``T_coup < 1``.
    """
    text, _ = _build(
        calc_type=calc_type,
        nep_path=nep_path,
        ensemble=ensemble,
        temperature=temperature,
        timestep_fs=timestep_fs,
        steps=steps,
        thermostat=thermostat,
        friction_per_fs=friction_per_fs,
        bussi_tau_fs=bussi_tau_fs,
        nhc_tdamp_fs=nhc_tdamp_fs,
        seed=seed,
        fmax=fmax,
        max_steps=max_steps,
        optimizer=optimizer,
        dump_interval=dump_interval,
        traj_filename=traj_filename,
        restart_interval=restart_interval,
        unwrapped=unwrapped,
    )
    return text


def run_in_provenance(
    *,
    calc_type: str,
    nep_path: str | Path,
    ensemble: str | None = None,
    temperature: float | None = None,
    timestep_fs: float | None = None,
    steps: int | None = None,
    thermostat: str | None = None,
    friction_per_fs: float | None = None,
    bussi_tau_fs: float | None = None,
    nhc_tdamp_fs: float | None = None,
    seed: int | None = None,
    fmax: float | None = None,
    max_steps: int | None = None,
    optimizer: str | None = None,
    dump_interval: int | None = None,
    traj_filename: str = "traj.xyz",
    restart_interval: int | None = None,
    unwrapped: bool = True,
) -> dict[str, Any]:
    """Return the argument record (and derived ``T_coup``) for ``run.in``.

    This is the machine-readable companion of :func:`build_run_in`; it records
    the exact arguments plus the derived thermostat mapping and ``T_coup``.
    It raises exactly like :func:`build_run_in` so provenance can never
    describe a file that was not generated.
    """
    _, provenance = _build(
        calc_type=calc_type,
        nep_path=nep_path,
        ensemble=ensemble,
        temperature=temperature,
        timestep_fs=timestep_fs,
        steps=steps,
        thermostat=thermostat,
        friction_per_fs=friction_per_fs,
        bussi_tau_fs=bussi_tau_fs,
        nhc_tdamp_fs=nhc_tdamp_fs,
        seed=seed,
        fmax=fmax,
        max_steps=max_steps,
        optimizer=optimizer,
        dump_interval=dump_interval,
        traj_filename=traj_filename,
        restart_interval=restart_interval,
        unwrapped=unwrapped,
    )
    return provenance
