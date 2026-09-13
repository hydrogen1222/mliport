"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Modified for the mliport project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
Molecular dynamics runner.

Runs MD simulations using ASE's integrators:
- NVT ensemble (Langevin, Bussi/CSVR, or Nose-Hoover-chain dynamics)
- NVE ensemble (Velocity Verlet)

Outputs trajectories in multiple formats.
"""

from __future__ import annotations

import contextlib
import json
import math
import sys
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase import units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixCartesian, FixCom
from ase.md.bussi import Bussi
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize import FIRE

from mliport.config.defaults import BUILTIN_DEFAULTS
from mliport.protocols import CancellationRequested
from mliport.runners.base import BaseRunner
from mliport.runners.md_output import (
    AsyncMDOutputWriter,
    MDFrameSnapshot,
    MDFrameStats,
    MDTrajectorySummary,
)
from mliport.writers.contcar import ContcarWriter
from mliport.writers.outcar import MDOutcarWriter
from mliport.writers.xdatcar import XdatcarWriter
from mliport.writers.json_writer import JsonWriter

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from mliport.protocols import ProgressCallback


class ForceSafetyAbort(RuntimeError):
    """Raised after checkpointing an MD frame that exceeds ``fmax_abort``."""

    def __init__(
        self, *, step: int, max_force: float, atom_index: int, threshold: float
    ):
        self.step = step
        self.max_force = max_force
        self.atom_index = atom_index
        self.threshold = threshold
        super().__init__(
            f"Force safety abort at MD step {step}: atom {atom_index} has "
            f"raw model force |F|={max_force:.6g} eV/Angstrom, exceeding "
            f"fmax_abort={threshold:.6g} eV/Angstrom"
        )


def _finite_float(value: object, *, name: str) -> float:
    """Return a finite real scalar without accepting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return numeric


def _strict_integer(value: object, *, name: str) -> int:
    """Return an integer while rejecting bool and fractional real values."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    raise TypeError(f"{name} must be an integer, got {value!r}")


class MDRunner(BaseRunner):
    """Run molecular dynamics simulations.

    Supports NVT (Langevin, Bussi/CSVR, or Nose-Hoover chain) and NVE
    (Velocity Verlet) ensembles.
    Can optionally reduce large initial atomic forces with a positions-only
    pre-relaxation before MD.

    Example:
        >>> runner = MDRunner(
        ...     calculator,
        ...     ensemble="NVT",
        ...     temperature=300,
        ...     timestep=1.0,
        ...     steps=10000
        ... )
        >>> results = runner.run(atoms)
        >>> print(f"Final temperature: {results['temperature']:.1f} K")
    """

    VALID_ENSEMBLES = {"nvt", "nve"}
    VALID_THERMOSTATS = {"langevin", "bussi", "nhc"}
    VALID_COM_POLICIES = {"none", "initialize_only", "constraint", "auto"}
    # This matrix is deliberately conservative. These combinations are backed
    # by ASE's public constraint-aware paths and local known-answer tests. An
    # unknown or compound constraint must be validated independently before it
    # can be added here.
    CONSTRAINT_CAPABILITIES = {
        "nve": frozenset({"fixcom", "fixatoms", "fixcartesian"}),
        "langevin": frozenset({"fixcom", "fixatoms", "fixcartesian"}),
        "bussi": frozenset({"fixcom", "fixatoms", "fixcartesian"}),
        "nhc": frozenset(),
    }
    COM_POLICY_CAPABILITIES = {
        "nve": frozenset({"none", "constraint"}),
        "langevin": frozenset({"none", "initialize_only", "constraint"}),
        "bussi": frozenset({"none", "constraint"}),
        "nhc": frozenset({"none"}),
    }
    # Every MD step is retained in run.log independently of trajectory saving.
    # LiveRunLogger coalesces the actual filesystem flushes and suppresses these
    # high-frequency records from UI callbacks.
    LOG_INTERVAL_STEPS = 1
    PROGRESS_INTERVAL_STEPS = 100

    def __init__(
        self,
        calculator,
        ensemble: str = "NVT",
        temperature: float = 300.0,
        timestep: float = 1.0,
        steps: int = 1000,
        equilibration_steps: int = 0,
        thermostat: str = "LANGEVIN",
        friction: float = 0.001,
        bussi_tau: float = 1000.0,
        nhc_tdamp: float = 100.0,
        nhc_tchain: int = 3,
        nhc_tloop: int = 1,
        save_interval: int = 10,
        output_dir: Path | str = ".",
        write_outcar: bool = True,
        write_forces: bool = True,
        write_stress: bool = True,
        write_xdatcar: bool = True,
        write_trajectory: bool = True,
        write_json: bool = True,
        verbose: bool = True,
        job_name: str | None = None,
        # NEW: Pre-relaxation options
        pre_relax: bool = True,
        pre_relax_steps: int = 50,
        pre_relax_fmax: float = 0.1,
        # Reproducibility / velocity policy (plan section 5.2 / 5.3).
        seed: int | None = None,
        velocity_policy: str = "auto",
        com_policy: str = "auto",
        pre_relax_mode: str = "none",
        # Explosion-guard threshold for finite (but unsafe) forces.
        fmax_abort: float = BUILTIN_DEFAULTS["safety"]["fmax_abort"],
        log_fn: Any | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        charge: int | None = None,
        spin: int | None = None,
    ):
        """Initialize MD runner.

        Args:
            calculator: UMA calculator wrapper
            ensemble: MD ensemble (NVT or NVE)
            temperature: Temperature in Kelvin
            timestep: Time step in femtoseconds
            steps: Number of production MD steps
            equilibration_steps: Same-ensemble MD steps before production
            thermostat: NVT thermostat (LANGEVIN, BUSSI, or NHC)
            friction: Langevin friction coefficient (1/fs)
            bussi_tau: Bussi/CSVR coupling time in femtoseconds
            nhc_tdamp: Nose-Hoover-chain damping time in femtoseconds
            nhc_tchain: Nose-Hoover chain length
            nhc_tloop: Nose-Hoover thermostat integration substeps
            save_interval: Interval for saving trajectory frames
            output_dir: Directory for output files
            write_outcar: Whether to write OUTCAR file
            write_forces: Whether OUTCAR includes the final force table
            write_stress: Whether OUTCAR includes the final stress tensor
            write_xdatcar: Whether to write XDATCAR file
            write_trajectory: Whether to write ASE trajectory
            write_json: Whether to write JSON results
            verbose: Whether to print progress messages
            job_name: Optional job name for organizing results
            pre_relax: Whether to perform pre-relaxation before MD
            pre_relax_steps: Maximum steps for pre-relaxation
            pre_relax_fmax: Force threshold for pre-relaxation
            seed: Optional random seed used for velocity initialization
            velocity_policy: Existing-velocity handling policy
            com_policy: Center-of-mass handling policy
            pre_relax_mode: Explicit pre-relaxation mode selector
            fmax_abort: Raw model-force abort threshold in eV/Angstrom
            log_fn: Optional callback function for custom log output
            progress_callback: Optional callback for progress events
        """
        # Validate every physical scalar before BaseRunner creates the output
        # directory. Direct Python callers receive the same fail-closed
        # behaviour as schema-mediated CLI/INCAR/queue calls.
        ensemble_value = str(ensemble).lower()
        thermostat_value = str(thermostat).lower()
        velocity_policy_value = str(velocity_policy).lower()
        com_policy_value = str(com_policy).lower()
        pre_relax_mode_value = str(pre_relax_mode).lower()

        if ensemble_value not in self.VALID_ENSEMBLES:
            raise ValueError(
                f"Unknown ensemble: {ensemble}. "
                f"Use one of: {', '.join(sorted(self.VALID_ENSEMBLES))}"
            )
        if velocity_policy_value not in {"auto", "initialize", "preserve"}:
            raise ValueError(
                f"Unknown velocity_policy {velocity_policy!r}. "
                "Use one of: auto, initialize, preserve."
            )
        if com_policy_value not in self.VALID_COM_POLICIES:
            raise ValueError(
                f"Unknown com_policy {com_policy!r}. Use one of: "
                "auto, none, initialize_only, constraint."
            )

        temperature_value = _finite_float(temperature, name="temperature")
        timestep_fs = _finite_float(timestep, name="timestep")
        friction_fs = _finite_float(friction, name="friction")
        bussi_tau_fs = _finite_float(bussi_tau, name="bussi_tau")
        nhc_tdamp_fs = _finite_float(nhc_tdamp, name="nhc_tdamp")
        pre_relax_fmax_value = _finite_float(pre_relax_fmax, name="pre_relax_fmax")
        fmax_abort_value = _finite_float(fmax_abort, name="fmax_abort")
        steps_value = _strict_integer(steps, name="steps")
        equilibration_steps_value = _strict_integer(
            equilibration_steps, name="equilibration_steps"
        )
        nhc_tchain_value = _strict_integer(nhc_tchain, name="nhc_tchain")
        nhc_tloop_value = _strict_integer(nhc_tloop, name="nhc_tloop")
        save_interval_value = _strict_integer(save_interval, name="save_interval")
        pre_relax_steps_value = _strict_integer(pre_relax_steps, name="pre_relax_steps")
        seed_value = None if seed is None else _strict_integer(seed, name="seed")

        if temperature_value < 0:
            raise ValueError("temperature must be >= 0 K")
        if timestep_fs <= 0:
            raise ValueError("timestep must be > 0 fs")
        if steps_value < 0:
            raise ValueError("steps must be >= 0")
        if equilibration_steps_value < 0:
            raise ValueError("equilibration_steps must be >= 0")
        if save_interval_value <= 0:
            raise ValueError("save_interval must be > 0")
        if pre_relax_steps_value < 0:
            raise ValueError("pre_relax_steps must be >= 0")
        if pre_relax_fmax_value <= 0:
            raise ValueError("pre_relax_fmax must be > 0 eV/Angstrom")
        if fmax_abort_value <= 0:
            raise ValueError("fmax_abort must be > 0 eV/Angstrom")
        if seed_value is not None and seed_value < 0:
            raise ValueError("seed must be >= 0")

        if ensemble_value == "nvt":
            if thermostat_value not in self.VALID_THERMOSTATS:
                raise ValueError(
                    f"Unknown thermostat: {thermostat}. Use one of: "
                    f"{', '.join(sorted(self.VALID_THERMOSTATS))}"
                )
            if thermostat_value == "langevin" and friction_fs <= 0:
                raise ValueError("friction must be > 0 fs^-1 for NVT Langevin dynamics")
            if thermostat_value == "bussi":
                if bussi_tau_fs <= 0:
                    raise ValueError("bussi_tau must be > 0 fs for NVT Bussi dynamics")
                if temperature_value <= 0:
                    raise ValueError("temperature must be > 0 K for NVT Bussi dynamics")
            if thermostat_value == "nhc":
                if nhc_tdamp_fs <= 0:
                    raise ValueError("nhc_tdamp must be > 0 fs for NVT NHC dynamics")
                if nhc_tchain_value < 1:
                    raise ValueError("nhc_tchain must be >= 1 for NVT NHC dynamics")
                if nhc_tloop_value < 1:
                    raise ValueError("nhc_tloop must be >= 1 for NVT NHC dynamics")
                if temperature_value <= 0:
                    raise ValueError("temperature must be > 0 K for NVT NHC dynamics")

        # pre_relax_mode is future vocabulary and must not be silently ignored.
        if pre_relax_mode_value != "none":
            raise NotImplementedError(
                f"pre_relax_mode={pre_relax_mode_value!r} is not yet implemented "
                "(Phase 3). Use the legacy pre_relax=True/False (positions-only) "
                "for now."
            )

        super().__init__(
            calculator,
            output_dir,
            verbose,
            job_name,
            log_fn,
            progress_callback,
            cancel_event=cancel_event,
            charge=charge,
            spin=spin,
        )
        self.ensemble = ensemble_value
        self.temperature = temperature_value
        self.timestep = timestep_fs * units.fs  # Convert to ASE units
        self.steps = steps_value
        self.equilibration_steps = equilibration_steps_value
        self.total_steps = self.equilibration_steps + self.steps
        self.thermostat = thermostat_value
        self.friction = friction_fs / units.fs  # Convert to ASE units
        self.bussi_tau = bussi_tau_fs * units.fs
        self.nhc_tdamp = nhc_tdamp_fs * units.fs
        self.nhc_tchain = nhc_tchain_value
        self.nhc_tloop = nhc_tloop_value
        self.save_interval = save_interval_value
        self.write_outcar = write_outcar
        self.write_forces = write_forces
        self.write_stress = write_stress
        self.write_xdatcar = write_xdatcar
        self.write_trajectory = write_trajectory
        self.write_json = write_json

        # MD output contract: ``raw`` is the lossless internal source of truth
        # and ``vasp`` is a syntax-compatible interoperability view.
        self.raw_dir = self.output_dir / "raw"
        self.vasp_dir = self.output_dir / "vasp"

        # NEW: Pre-relaxation settings
        self.pre_relax = pre_relax
        self.pre_relax_steps = pre_relax_steps_value
        self.pre_relax_fmax = pre_relax_fmax_value

        # Reproducibility / velocity policy (plan section 5.2 / 5.3).
        self.seed = seed_value
        # One generator drives *both* the initial Maxwell-Boltzmann draw and
        # every stochastic Langevin kick.  Seeding only the initial velocities
        # does not make an NVT trajectory reproducible.
        self.rng = np.random.default_rng(seed_value)
        self.velocity_policy = velocity_policy_value
        self.com_policy = com_policy_value
        self.com_policy_effective: str | None = None
        self.constraint_types: tuple[str, ...] = ()
        self.md_degrees_of_freedom: int | None = None
        self.pre_relax_mode = pre_relax_mode_value
        self.fmax_abort = fmax_abort_value

        # turbo-mode recommendation only applies to engines that support it
        # (currently only UMA). Other engines return 'default' and ignore it.
        if getattr(calculator, "inference_mode", "default") != "turbo" and hasattr(
            calculator, "VALID_INFERENCE_MODES"
        ):
            self.log(
                "Consider using inference_mode='turbo' for better MD performance",
                level="warning",
            )

    def _integrator_key(self) -> str:
        return "nve" if self.ensemble == "nve" else self.thermostat

    @staticmethod
    def _constraint_kind(constraint: object) -> str | None:
        if isinstance(constraint, FixCom):
            return "fixcom"
        if isinstance(constraint, FixAtoms):
            return "fixatoms"
        if isinstance(constraint, FixCartesian):
            return "fixcartesian"
        return None

    def _validate_atomic_state(self, atoms: Atoms, *, context: str) -> None:
        """Validate the complete atomic state before it reaches an integrator."""
        if len(atoms) == 0:
            raise ValueError(f"{context}: atoms must contain at least one atom")

        positions = np.asarray(atoms.positions, dtype=float)
        if positions.shape != (len(atoms), 3) or not np.all(np.isfinite(positions)):
            raise ValueError(
                f"{context}: positions must have shape ({len(atoms)}, 3) and "
                "contain only finite values"
            )

        cell = np.asarray(atoms.cell, dtype=float)
        if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
            raise ValueError(
                f"{context}: cell must have shape (3, 3) and contain only "
                "finite values"
            )
        pbc = np.asarray(atoms.pbc, dtype=bool)
        periodic_dimensions = int(np.count_nonzero(pbc))
        if (
            periodic_dimensions
            and np.linalg.matrix_rank(cell[pbc]) < periodic_dimensions
        ):
            raise ValueError(
                f"{context}: periodic cell vectors must be non-zero and linearly "
                "independent"
            )

        masses = np.asarray(atoms.get_masses(), dtype=float)
        if masses.shape != (len(atoms),) or not np.all(np.isfinite(masses)):
            raise ValueError(
                f"{context}: masses must have shape ({len(atoms)},) and contain "
                "only finite values"
            )
        if np.any(masses <= 0.0):
            raise ValueError(f"{context}: every atomic mass must be > 0")

        if atoms.has("momenta"):
            momenta = np.asarray(atoms.get_momenta(), dtype=float)
            if momenta.shape != (len(atoms), 3) or not np.all(np.isfinite(momenta)):
                raise ValueError(
                    f"{context}: momenta must have shape ({len(atoms)}, 3) and "
                    "contain only finite values"
                )
            velocities = momenta / masses[:, None]
            if not np.all(np.isfinite(velocities)):
                raise ValueError(
                    f"{context}: velocities must contain only finite values"
                )

    def _degrees_of_freedom(self, atoms: Atoms) -> int:
        try:
            ndof = int(atoms.get_number_of_degrees_of_freedom())
        except NotImplementedError as exc:
            raise ValueError(
                "MD constraint does not report removed degrees of freedom; "
                "this combination is unsupported"
            ) from exc
        if ndof <= 0:
            raise ValueError(
                f"MD has zero or negative degrees of freedom ({ndof} DOF); "
                "thermal initialization and integration are undefined"
            )
        return ndof

    def _validate_constraint_capability(self, atoms: Atoms) -> None:
        constraints = list(atoms.constraints)
        integrator = self._integrator_key()
        effective = self.com_policy_effective
        if effective is None:
            raise RuntimeError(
                "Internal error: COM policy must be resolved before validating "
                "the MD constraint capability"
            )

        if effective not in self.COM_POLICY_CAPABILITIES[integrator]:
            raise ValueError(
                f"{integrator.upper()} does not support "
                f"com_policy={self.com_policy!r}; supported effective policies: "
                f"{', '.join(sorted(self.COM_POLICY_CAPABILITIES[integrator]))}"
            )
        if len(constraints) > 1:
            names = ", ".join(type(item).__name__ for item in constraints)
            raise ValueError(
                f"The compound MD constraint combination [{names}] has not been "
                "independently validated; refusing to integrate"
            )
        if not constraints:
            return

        constraint = constraints[0]
        kind = self._constraint_kind(constraint)
        if integrator == "nhc":
            raise ValueError(
                f"NHC with constraint {type(constraint).__name__} is unsupported: "
                "ASE NHC targets 3N and maintains private thermostat state. "
                "Use an unconstrained structure with com_policy='none'."
            )
        if kind is None or kind not in self.CONSTRAINT_CAPABILITIES[integrator]:
            raise ValueError(
                f"{integrator.upper()} with constraint "
                f"{type(constraint).__name__} has not been independently validated"
            )

    def _configure_com_policy(self, atoms: Atoms) -> None:
        """Apply the requested COM policy without rewriting user constraints."""
        constraints = list(atoms.constraints)
        fixcom_count = sum(isinstance(item, FixCom) for item in constraints)
        if fixcom_count > 1:
            raise ValueError("Repeated FixCom constraints are unsupported")

        if self.com_policy == "none":
            if fixcom_count:
                raise ValueError(
                    "com_policy='none' conflicts with an existing FixCom constraint"
                )
            effective = "none"
        elif self.com_policy == "initialize_only":
            if constraints:
                raise ValueError(
                    "com_policy='initialize_only' is only validated for structures "
                    "without constraints"
                )
            if len(atoms) <= 1:
                raise ValueError(
                    "A single atom has no internal motion after COM initialization; "
                    "use com_policy='none' to retain translational degrees of freedom"
                )
            effective = "initialize_only"
        elif self.com_policy == "constraint":
            if any(not isinstance(item, FixCom) for item in constraints):
                raise ValueError(
                    "com_policy='constraint' cannot append FixCom to existing "
                    "non-COM constraints; this combination is unsupported"
                )
            if len(atoms) <= 1:
                raise ValueError(
                    "A FixCom constraint leaves a single atom with 0 DOF; use "
                    "com_policy='none' to retain translational degrees of freedom"
                )
            if not fixcom_count:
                atoms.set_constraint(FixCom())
            effective = "constraint"
        else:  # auto
            if self._integrator_key() == "nhc" and not constraints:
                raise ValueError(
                    "NHC cannot use automatic COM removal. Select "
                    "com_policy='none' explicitly for unconstrained 3N dynamics."
                )
            if not constraints:
                if len(atoms) <= 1:
                    raise ValueError(
                        "Automatic COM removal is undefined for a single atom; "
                        "select com_policy='none' explicitly"
                    )
                atoms.set_constraint(FixCom())
                effective = "constraint"
            elif fixcom_count:
                effective = "constraint"
            else:
                # A positional constraint usually already breaks translation.
                # Preserve it exactly and do not append a second projector.
                effective = "none"

        self.com_policy_effective = effective
        self._validate_constraint_capability(atoms)
        self.md_degrees_of_freedom = self._degrees_of_freedom(atoms)
        self.constraint_types = tuple(type(item).__name__ for item in atoms.constraints)

    def _stress_observables(self, atoms: Atoms) -> dict[str, Any]:
        """Return explicitly named 3D-bulk stress and pressure observables.

        Calculator/configurational stress excludes the kinetic ideal-gas term;
        total MD stress includes it via ASE. Scalar bulk pressure is only
        physically exposed for fully periodic systems. Molecules and partial-PBC
        systems return unavailable values instead of vacuum-dependent numbers.
        """
        unavailable = {
            "configurational_stress": None,
            "total_stress": None,
            "configurational_pressure_gpa": None,
            "total_pressure_gpa": None,
        }
        if (
            not self.write_stress
            or not getattr(self.calculator, "has_stress", False)
            or not bool(np.asarray(atoms.pbc, dtype=bool).all())
        ):
            return unavailable

        configurational = np.asarray(
            atoms.get_stress(voigt=True, include_ideal_gas=False), dtype=float
        )
        total = np.asarray(
            atoms.get_stress(voigt=True, include_ideal_gas=True), dtype=float
        )
        for name, stress in (
            ("configurational stress", configurational),
            ("total stress", total),
        ):
            if stress.shape != (6,) or not np.all(np.isfinite(stress)):
                raise RuntimeError(
                    f"Non-finite or malformed {name} during MD; aborting before "
                    "invalid values are written to outputs"
                )
        factor = MDOutcarWriter.EV_A3_TO_GPA
        observables = {
            "configurational_stress": configurational,
            "total_stress": total,
            "configurational_pressure_gpa": (
                -float(np.sum(configurational[:3])) / 3.0 * factor
            ),
            "total_pressure_gpa": -float(np.sum(total[:3])) / 3.0 * factor,
        }
        if not all(
            math.isfinite(observables[key])
            for key in ("configurational_pressure_gpa", "total_pressure_gpa")
        ):
            raise RuntimeError(
                "Non-finite pressure derived from stress during MD; aborting "
                "before invalid values are written to outputs"
            )
        return observables

    def _force_views(
        self,
        atoms: Atoms,
        energy: float,
        *,
        context: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return finite raw and constraint-applied force arrays."""
        raw = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
        expected_shape = (len(atoms), 3)
        if raw.shape != expected_shape:
            raise RuntimeError(
                f"Malformed raw forces during {context}: expected "
                f"{expected_shape}, got {raw.shape}"
            )
        # Validate the calculator result before asking a constraint projector
        # to transform it. Projecting Inf/NaN can emit warnings and obscure
        # which layer first violated the finite-state contract.
        self._check_finite(atoms, energy, raw, context=f"{context} (raw forces)")

        applied = np.asarray(atoms.get_forces(apply_constraint=True), dtype=float)
        if applied.shape != expected_shape:
            raise RuntimeError(
                f"Malformed constraint-applied forces during {context}: expected "
                f"{expected_shape}, got {applied.shape}"
            )
        self._check_finite(
            atoms,
            energy,
            applied,
            context=f"{context} (constraint-applied forces)",
        )
        return raw, applied

    def _validate_md_safe_point(
        self,
        atoms: Atoms,
        *,
        energy: float,
        forces_raw: np.ndarray,
        forces_applied: np.ndarray,
        context: str,
    ) -> tuple[float, float, float]:
        """Validate positions, momenta and scalar observables at an MD safe point."""
        self._validate_atomic_state(atoms, context=context)
        self._check_finite(atoms, energy, forces_raw, context=context)
        self._check_finite(atoms, energy, forces_applied, context=context)
        kinetic_energy = float(atoms.get_kinetic_energy())
        temperature = self._calculate_temperature(atoms)
        total_energy = float(energy) + kinetic_energy
        volume = float(atoms.get_volume())
        for name, value in (
            ("kinetic energy", kinetic_energy),
            ("total energy", total_energy),
            ("temperature", temperature),
            ("volume", volume),
        ):
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Non-finite {name} ({value!r}) during {context}; aborting"
                )
        # A zero cell volume is valid for isolated molecules and can also be
        # valid for lower-dimensional PBC. Only a 3D-periodic system requires
        # a non-zero bulk volume.
        if bool(np.asarray(atoms.pbc, dtype=bool).all()) and volume <= 0.0:
            raise RuntimeError(f"Non-positive volume ({volume!r}) during {context}")
        return kinetic_energy, temperature, total_energy

    def _write_artifacts_manifest(
        self,
        *,
        status: str,
        frame_stats: MDFrameStats,
        error: dict[str, Any] | None = None,
    ) -> None:
        """Describe the versioned MD output contract without parsing files."""
        artifacts: dict[str, dict[str, Any]] = {}
        candidates = {
            "resolved_config": (
                self.output_dir / "resolved_config.json",
                "mliport-resolved-config",
            ),
            "trajectory": (self.raw_dir / "trajectory.traj", "ase-traj"),
            "thermodynamics": (self.raw_dir / "md.csv", "csv"),
            "results": (self.raw_dir / "mliport_results.json", "mliport-json"),
            "xdatcar": (self.vasp_dir / "XDATCAR", "vasp-xdatcar"),
            "contcar": (self.vasp_dir / "CONTCAR", "vasp-poscar"),
            "outcar": (self.vasp_dir / "OUTCAR", "mliport-vasp-like-outcar"),
        }
        for name, (path, file_format) in candidates.items():
            if path.exists():
                artifacts[name] = {
                    "path": path.relative_to(self.output_dir).as_posix(),
                    "format": file_format,
                    "bytes": path.stat().st_size,
                }

        timestep_fs = float(self.timestep / units.fs)
        try:
            mliport_version = version("mliport")
        except PackageNotFoundError:
            mliport_version = "unknown"

        dependency_versions = {}
        for distribution in (
            "ase",
            "numpy",
            "torch",
            "fairchem-core",
            "mace-torch",
            "deepmd-kit",
            "tensorpotential",
        ):
            try:
                dependency_versions[distribution] = version(distribution)
            except PackageNotFoundError:
                dependency_versions[distribution] = "unknown"

        manifest = {
            "schema": "mliport.md-artifacts/3",
            "status": status,
            "producer": {"name": "mliport", "version": mliport_version},
            "runtime": {"packages": dependency_versions},
            "model": self.calculator.info(),
            "layout": {
                "raw": "Lossless canonical trajectory and machine-readable data",
                "vasp": "VASP-syntax-compatible interoperability exports",
            },
            "trajectory": {
                "frames": frame_stats.count,
                "first_step": frame_stats.first_step,
                "last_step": frame_stats.last_step,
                "md_timestep_fs": timestep_fs,
                "frame_stride_steps": self.save_interval,
                "frame_interval_fs": timestep_fs * self.save_interval,
                # Legacy aliases remain readable by older consumers. The
                # explicitly named fields above are authoritative.
                "timestep_fs": timestep_fs,
                "save_interval_steps": self.save_interval,
                "saved_interval_fs": timestep_fs * self.save_interval,
                "positions_convention": "unwrapped",
                "positions": "unwrapped Cartesian in trajectory.traj; "
                "unwrapped direct in XDATCAR",
                "equilibration_steps": self.equilibration_steps,
                "production_steps": self.steps,
                "total_steps": self.total_steps,
                "production_start_step": self.equilibration_steps,
                "production_start_frame": frame_stats.production_start_frame,
                "com_policy": self.com_policy,
                "com_policy_effective": self.com_policy_effective,
                "constraints": list(self.constraint_types),
                "degrees_of_freedom": self.md_degrees_of_freedom,
            },
            "units": {
                "time": "fs",
                "length": "angstrom",
                "energy": "eV",
                "force": "eV/angstrom",
                "stress": "eV/angstrom^3",
                "temperature": "K",
            },
            "observables": {
                "forces": "constraint-applied compatibility alias",
                "forces_raw_eV_A": "unprojected calculator forces",
                "forces_applied_eV_A": "ASE constraint-applied forces",
                "fmax_abort": "evaluated from forces_raw_eV_A",
                "configurational_stress": "ASE calculator stress; kinetic term excluded",
                "total_stress": "configurational stress plus ASE ideal-gas kinetic term",
                "configurational_pressure_gpa": "-trace(configurational_stress)/3",
                "total_pressure_gpa": "-trace(total_stress)/3; reported only for 3D PBC",
                "non_3d_pbc_policy": "stress and scalar bulk pressure unavailable",
            },
            "artifacts": artifacts,
        }
        if error is not None:
            manifest["error"] = error
        (self.output_dir / "artifacts.json").write_text(
            json.dumps(manifest, indent=2, default=str, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _calculate_temperature(self, atoms: Atoms) -> float:
        """Instantaneous temperature for logging / trajectory.

        Delegates to ASE's canonical constraint-aware kinetic energy and DOF
        definitions. Constraints that cannot report their removed DOF are
        rejected by the capability gate; there is no approximate fallback.

        Args:
            atoms: ASE Atoms object

        Returns:
            Temperature in Kelvin
        """
        ndof = self._degrees_of_freedom(atoms)
        kinetic_energy = float(atoms.get_kinetic_energy())
        if not math.isfinite(kinetic_energy) or kinetic_energy < 0.0:
            raise RuntimeError(
                f"Non-finite or negative kinetic energy ({kinetic_energy!r}) "
                "during MD"
            )
        temperature = 2.0 * kinetic_energy / (ndof * units.kB)
        if not math.isfinite(temperature):
            raise RuntimeError(f"Non-finite temperature ({temperature!r}) during MD")
        return temperature

    def _validate_preserved_momenta(self, atoms: Atoms) -> None:
        """Reject preserved momenta that conflict with requested policies."""
        momenta = np.asarray(atoms.get_momenta(), dtype=float)
        adjusted = momenta.copy()
        for constraint in atoms.constraints:
            constraint.adjust_momenta(atoms, adjusted)
        if not np.allclose(adjusted, momenta, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError(
                "Existing momenta violate the active MD constraint. Refusing to "
                "silently project a continued state; provide constraint-consistent "
                "momenta or use velocity_policy='initialize'."
            )
        if self.com_policy_effective == "initialize_only":
            masses = np.asarray(atoms.get_masses(), dtype=float)
            stationary = momenta - masses[:, None] * (
                np.sum(momenta, axis=0) / np.sum(masses)
            )
            if not np.allclose(stationary, momenta, rtol=1.0e-12, atol=1.0e-12):
                raise ValueError(
                    "com_policy='initialize_only' would alter the supplied "
                    "momenta, which conflicts with preserving a continued state. "
                    "Provide zero-COM momenta or use "
                    "velocity_policy='initialize'."
                )

    def _initialize_velocities(self, atoms: Atoms) -> None:
        """Initialize velocities at the requested MD temperature.

        Honours ``velocity_policy`` (plan section 5.3):

        * ``auto``       -- initialize only when the structure has no momenta;
                            otherwise preserve existing velocities (restart).
        * ``initialize`` -- always (re-)initialize a Maxwell-Boltzmann distribution.
        * ``preserve``   -- keep existing velocities; raise if there are none.

        When initializing, a seeded numpy Generator (``self.seed``) is forwarded
        to ASE's ``MaxwellBoltzmannDistribution`` so the run is reproducible when
        a seed is recorded (the resolver auto-generates one for MD).
        """
        if self.com_policy_effective is None:
            self._configure_com_policy(atoms)
        self._validate_atomic_state(atoms, context="MD velocity initialization")

        # Presence, not magnitude, defines a continued state. An explicitly stored
        # all-zero momenta array is still a deliberate initial condition.
        has_momenta = atoms.has("momenta")
        policy = self.velocity_policy
        if policy == "preserve":
            if not has_momenta:
                raise ValueError(
                    "velocity_policy='preserve' requires existing velocities, but "
                    "the structure has none. Use velocity_policy='initialize' or "
                    "'auto'."
                )
            self._validate_preserved_momenta(atoms)
            self.log("\nPreserving existing velocities (velocity_policy=preserve)")
            return
        if policy == "auto" and has_momenta:
            self._validate_preserved_momenta(atoms)
            self.log("\nPreserving existing velocities (velocity_policy=auto)")
            return
        # policy == "initialize", or auto with no velocities.
        seed_msg = f" (seed={self.seed})" if self.seed is not None else ""
        self.log(
            f"\nInitializing Maxwell-Boltzmann distribution at "
            f"{self.temperature} K{seed_msg}"
        )
        MaxwellBoltzmannDistribution(
            atoms,
            temperature_K=self.temperature,
            force_temp=True,
            rng=self.rng,
        )
        if self.com_policy_effective == "initialize_only":
            Stationary(atoms, preserve_temperature=True)
        self._validate_atomic_state(atoms, context="initialized MD state")
        self._calculate_temperature(atoms)

    def _ensure_com_constraint(self, atoms: Atoms) -> None:
        """Compatibility wrapper for the explicit COM/capability gate."""
        self._configure_com_policy(atoms)

    def _build_dynamics(self, atoms: Atoms):
        """Build the ASE integrator without coupling it to calculator options."""
        if self.com_policy_effective is None:
            self._configure_com_policy(atoms)
        else:
            self._validate_constraint_capability(atoms)
            self._degrees_of_freedom(atoms)
        if self.ensemble == "nve":
            self.log("Setting up NVE (Velocity Verlet)")
            return VelocityVerlet(atoms, timestep=self.timestep)

        if self.thermostat == "langevin":
            friction_fs = self.friction * units.fs
            self.log(
                "Setting up NVT Langevin dynamics "
                f"(friction={friction_fs:.4f} fs^-1)"
            )
            self.log(
                "Approx. velocity damping time: " f"{1.0 / friction_fs / 1000.0:.6g} ps"
            )
            return Langevin(
                atoms,
                timestep=self.timestep,
                temperature_K=self.temperature,
                friction=self.friction,
                rng=self.rng,
                fixcm=False,
            )

        if self.thermostat == "bussi":
            if np.isclose(atoms.get_kinetic_energy(), 0.0, rtol=0.0, atol=1e-12):
                raise ValueError(
                    "Bussi/CSVR requires non-zero initial kinetic energy. "
                    "Use a positive temperature with velocity_policy='initialize', "
                    "or provide non-zero velocities."
                )
            self.log(
                "Setting up NVT Bussi/CSVR dynamics "
                f"(coupling time={self.bussi_tau / units.fs:.6g} fs)"
            )
            return Bussi(
                atoms,
                timestep=self.timestep,
                temperature_K=self.temperature,
                taut=self.bussi_tau,
                rng=self.rng,
            )

        if self.thermostat == "nhc":
            self.log(
                "Setting up NVT Nose-Hoover-chain dynamics "
                f"(tdamp={self.nhc_tdamp / units.fs:.6g} fs, "
                f"chain={self.nhc_tchain}, substeps={self.nhc_tloop})"
            )
            return NoseHooverChainNVT(
                atoms,
                timestep=self.timestep,
                temperature_K=self.temperature,
                tdamp=self.nhc_tdamp,
                tchain=self.nhc_tchain,
                tloop=self.nhc_tloop,
            )

        raise ValueError(f"Unknown thermostat: {self.thermostat}")

    def _md_provenance(self) -> dict[str, Any]:
        """Return canonical MD settings, including only the active coupling."""
        provenance: dict[str, Any] = {
            "ensemble": self.ensemble.upper(),
            "thermostat": self.thermostat.upper() if self.ensemble == "nvt" else None,
            "temperature": self.temperature,
            "timestep": float(self.timestep / units.fs),
            "steps": self.steps,
            "equilibration_steps": self.equilibration_steps,
            "production_steps": self.steps,
            "total_steps": self.total_steps,
            "seed": self.seed,
            "velocity_policy": self.velocity_policy,
            "com_policy": self.com_policy,
            "com_policy_effective": self.com_policy_effective,
            "constraints": list(self.constraint_types),
            "degrees_of_freedom": self.md_degrees_of_freedom,
            "force_contract": {
                "forces": "constraint-applied compatibility alias",
                "forces_raw_eV_A": "unprojected calculator forces",
                "forces_applied_eV_A": "ASE constraint-applied forces",
                "fmax_abort": "maximum per-atom norm of forces_raw_eV_A",
            },
        }
        if self.ensemble == "nvt" and self.thermostat == "langevin":
            friction_fs = float(self.friction * units.fs)
            provenance.update(
                {
                    "friction_fs^-1": friction_fs,
                    "approx_velocity_damping_time_ps": 1.0 / friction_fs / 1000.0,
                }
            )
        elif self.ensemble == "nvt" and self.thermostat == "bussi":
            provenance["bussi_tau_fs"] = float(self.bussi_tau / units.fs)
        elif self.ensemble == "nvt" and self.thermostat == "nhc":
            provenance.update(
                {
                    "nhc_tdamp_fs": float(self.nhc_tdamp / units.fs),
                    "nhc_tchain": self.nhc_tchain,
                    "nhc_tloop": self.nhc_tloop,
                }
            )
        return provenance

    def _outcar_md_settings(self) -> dict[str, Any]:
        """Translate canonical provenance to human-readable OUTCAR labels."""
        provenance = self._md_provenance()
        settings: dict[str, Any] = {
            "Ensemble": provenance["ensemble"],
            "Thermostat": provenance["thermostat"],
            "Target temperature (K)": provenance["temperature"],
            "Time step (fs)": provenance["timestep"],
            "MD steps": provenance["steps"],
            "Equilibration steps": provenance["equilibration_steps"],
            "Production steps": provenance["production_steps"],
            "Total integrated steps": provenance["total_steps"],
            "Save interval (steps)": self.save_interval,
            "Random seed": provenance["seed"],
            "Velocity policy": provenance["velocity_policy"],
            "Pre-relaxed": self.pre_relax,
        }
        if "friction_fs^-1" in provenance:
            settings["Friction (1/fs)"] = provenance["friction_fs^-1"]
            settings["Approx. velocity damping time (ps)"] = provenance[
                "approx_velocity_damping_time_ps"
            ]
        elif "bussi_tau_fs" in provenance:
            settings["Bussi coupling time (fs)"] = provenance["bussi_tau_fs"]
        elif "nhc_tdamp_fs" in provenance:
            settings.update(
                {
                    "NHC damping time (fs)": provenance["nhc_tdamp_fs"],
                    "NHC chain length": provenance["nhc_tchain"],
                    "NHC thermostat substeps": provenance["nhc_tloop"],
                }
            )
        return settings

    def _pre_relax_structure(self, atoms: Atoms) -> Atoms:
        """Perform a short positions-only relaxation to reduce large forces.

        This does not optimize the cell and therefore does not in general
        remove cell stress or guarantee a local minimum.

        This can lower risky initial forces, but a short capped run does not
        ensure that the structure reaches a local minimum.

        Args:
            atoms: ASE Atoms object

        Returns:
            Relaxed Atoms object
        """
        self._emit_progress(
            "running",
            "Pre-relaxing structure...",
            step=0,
            total_steps=self.pre_relax_steps,
        )
        self.log("\n" + "=" * 60)
        self.log("PRE-RELAXATION PHASE")
        self.log("=" * 60)
        self.log("Reducing large initial atomic forces before MD...")
        self.log(f"Target fmax: {self.pre_relax_fmax} eV/Å")
        self.log(f"Max steps: {self.pre_relax_steps}")

        if atoms.calc is None:
            raise RuntimeError("Pre-relaxation requires the validated MD calculator")

        # Use FIRE optimizer for robust relaxation
        optimizer = FIRE(atoms, logfile=None)

        # Attach cancellation check to the optimizer
        def _check_cancel():
            if self._is_cancelled():
                self.log("\nCancellation requested during pre-relaxation")
                raise CancellationRequested("Pre-relaxation cancelled by user")

        optimizer.attach(_check_cancel, interval=1)

        def _check_pre_relax_state() -> None:
            context = f"MD pre-relaxation step {optimizer.nsteps}"
            self._validate_atomic_state(atoms, context=context)
            energy = float(atoms.get_potential_energy())
            self._force_views(atoms, energy, context=context)

        optimizer.attach(_check_pre_relax_state, interval=1)

        # Track initial energy
        e_init = float(atoms.get_potential_energy())
        self._force_views(atoms, e_init, context="MD pre-relaxation initial state")
        self.log(f"Initial energy: {e_init:.6f} eV")

        # Run optimization.  A backend/neighbor-list failure here is not a
        # recoverable "partially relaxed" result: continuing into MD would
        # merely hide the original error and can launch from an unsafe state.
        # Fail closed, matching safety.pre_relax_failure=abort.
        try:
            optimizer.run(fmax=self.pre_relax_fmax, steps=self.pre_relax_steps)

            e_final = float(atoms.get_potential_energy())
            _, forces_applied = self._force_views(
                atoms, e_final, context="MD pre-relaxation final state"
            )
            delta_e = e_final - e_init

            self.log(f"Final energy: {e_final:.6f} eV")
            self.log(
                f"Energy change: {delta_e:.6f} eV ({delta_e / len(atoms):.6f} eV/atom)"
            )

            if optimizer.converged():
                self.log("✓ Pre-relaxation converged")
            else:
                final_fmax = float(np.max(np.linalg.norm(forces_applied, axis=1)))
                raise RuntimeError(
                    "Pre-relaxation did not converge after "
                    f"{optimizer.nsteps} steps (final fmax={final_fmax:.6g} "
                    "eV/Angstrom); production MD was not started. Increase "
                    "pre_relax_steps, loosen pre_relax_fmax with scientific "
                    "justification, or disable pre_relax explicitly."
                )

        except CancellationRequested:
            self.log("Pre-relaxation cancelled by user")
            raise  # Re-raise to stop the entire MD run
        except RuntimeError as e:
            if "Pre-relaxation did not converge" in str(e):
                raise
            raise RuntimeError(
                "Pre-relaxation failed at step "
                f"{optimizer.nsteps}; MD was not started because the structure "
                "may be unsafe."
            ) from e
        except Exception as e:
            raise RuntimeError(
                "Pre-relaxation failed at step "
                f"{optimizer.nsteps}; MD was not started because the structure "
                "may be unsafe."
            ) from e

        self.log("=" * 60)

        return atoms

    def run(self, atoms: Atoms) -> dict[str, Any]:
        """Run MD simulation.

        Args:
            atoms: ASE Atoms object

        Returns:
            Dictionary with results including final temperature and trajectory
        """
        if (
            self.pre_relax
            and atoms.has("momenta")
            and self.velocity_policy in {"auto", "preserve"}
        ):
            raise ValueError(
                "Pre-relaxation changes positions and is incompatible with "
                f"velocity_policy={self.velocity_policy!r} on a structure that "
                "already contains momenta. Continuing from coordinates/momenta is "
                "not a strict trajectory restart. Use pre_relax=False to retain "
                "that supplied state, or start a new trajectory with "
                "velocity_policy='initialize'."
            )

        # Copy and validate before model loading or creation of raw/vasp output
        # trees. The capability preflight uses a second copy because auto COM
        # policy may add FixCom; pre-relaxation must see exactly the user's
        # original positional constraints.
        atoms = atoms.copy()
        self._validate_atomic_state(atoms, context="MD input atoms")
        atoms = self._prepare_atoms(atoms)
        self._validate_atomic_state(atoms, context="prepared MD atoms")
        contract_atoms = atoms.copy()
        self._configure_com_policy(contract_atoms)
        if contract_atoms.has("momenta") and self.velocity_policy in {
            "auto",
            "preserve",
        }:
            self._validate_preserved_momenta(contract_atoms)

        self.print_header("MOLECULAR DYNAMICS")
        self._emit_progress("loading_model", "Loading model and preparing structure...")

        # Print settings
        self.log(f"Ensemble:         {self.ensemble.upper()}")
        self.log(
            "Thermostat:       "
            + (self.thermostat.upper() if self.ensemble == "nvt" else "none")
        )
        self.log(f"Temperature:      {self.temperature} K")
        self.log(f"Time step:        {self.timestep / units.fs} fs")
        self.log(f"Equilibration:    {self.equilibration_steps} steps")
        self.log(f"Production:       {self.steps} steps")
        self.log(f"Total MD steps:   {self.total_steps}")
        self.log(f"Save interval:    {self.save_interval} steps (trajectory frames)")
        self.log(f"COM policy:       {self.com_policy}")
        log_step_unit = "step" if self.LOG_INTERVAL_STEPS == 1 else "steps"
        self.log(
            f"Thermodynamic log interval: {self.LOG_INTERVAL_STEPS} "
            f"{log_step_unit} (buffered disk flush)"
        )
        self.log(f"Pre-relaxation:   {'Yes' if self.pre_relax else 'No'}")

        # Setup calculator
        calc = self._get_calculator()
        atoms.calc = calc

        # NEW: Pre-relaxation step
        if self.pre_relax:
            atoms = self._pre_relax_structure(atoms)

        self._validate_atomic_state(atoms, context="MD state after pre-relaxation")
        self._configure_com_policy(atoms)
        self.log(
            f"Effective COM:     {self.com_policy_effective} "
            f"({self.md_degrees_of_freedom} DOF)"
        )

        # Initialize a thermal velocity distribution for both NVT and NVE.
        self._initialize_velocities(atoms)

        # Validate the first model-evaluated state before an ASE integrator can
        # project forces internally and before any formal MD artifact is
        # created. Subsequent states are checked by the step observer.
        initial_energy = float(atoms.get_potential_energy())
        initial_forces_raw, initial_forces_applied = self._force_views(
            atoms, initial_energy, context="MD initial state"
        )
        self._validate_md_safe_point(
            atoms,
            energy=initial_energy,
            forces_raw=initial_forces_raw,
            forces_applied=initial_forces_applied,
            context="MD initial state",
        )
        self._stress_observables(atoms)

        # Setup integrator. Thermostat settings stay in MDRunner and are never
        # forwarded to the backend calculator.
        if self.ensemble == "nvt":
            self.log(
                "Note: under NVT the total energy E = PE + KE is NOT conserved "
                "(the thermostat exchanges heat); monitor T instead."
            )
        dyn = self._build_dynamics(atoms)

        for directory in (self.raw_dir, self.vasp_dir):
            directory.mkdir(parents=True, exist_ok=True)

        # The live calculator and mutable Atoms object stay on the MD thread.
        # Saved frames are detached into CPU-only snapshots and submitted to a
        # small bounded queue; the writer thread owns every output file handle.
        traj_path = self.raw_dir / "trajectory.traj"
        md_csv_path = self.raw_dir / "md.csv"
        xdatcar_writer = XdatcarWriter() if self.write_xdatcar else None
        xdatcar_path = self.vasp_dir / "XDATCAR"
        md_outcar_writer = MDOutcarWriter() if self.write_outcar else None
        if md_outcar_writer is not None:
            md_outcar_writer.write_header(
                atoms,
                self.vasp_dir / "OUTCAR",
                task_name=self.calculator.task,
                metadata=self.calculator.info(),
                settings=self._outcar_md_settings(),
            )
        output_stream = AsyncMDOutputWriter(
            trajectory_path=traj_path,
            csv_path=md_csv_path,
            write_trajectory=self.write_trajectory,
            xdatcar_writer=xdatcar_writer,
            xdatcar_path=xdatcar_path,
            outcar_writer=md_outcar_writer,
        )
        # Run MD
        self.log("\nStarting MD simulation...")
        self._emit_progress(
            "running",
            f"Starting {self.ensemble.upper()} MD simulation...",
            step=0,
            total_steps=self.total_steps,
        )
        start_time = time.time()

        def print_progress():
            """Print progress and save trajectory."""
            # Surface writer failures promptly even on steps that do not save
            # a frame; continuing an MD run with broken output is not allowed.
            output_stream.raise_if_failed()
            # Check for cooperative cancellation first
            if self._is_cancelled():
                raise CancellationRequested("MD simulation cancelled by user")

            step = dyn.nsteps
            phase = "equilibration" if step < self.equilibration_steps else "production"

            pe = float(atoms.get_potential_energy())
            forces_raw, forces_applied = self._force_views(
                atoms, pe, context=f"MD step {step}"
            )
            ke, temp, total_e = self._validate_md_safe_point(
                atoms,
                energy=pe,
                forces_raw=forces_raw,
                forces_applied=forces_applied,
                context=f"MD step {step}",
            )

            # The safety threshold is deliberately based on unprojected model
            # forces. A frozen atom can carry a large physical reaction force;
            # constraint projection must never hide it from the safety gate.
            raw_force_magnitudes = np.linalg.norm(forces_raw, axis=1)
            applied_force_magnitudes = np.linalg.norm(forces_applied, axis=1)
            atom_index = int(np.argmax(raw_force_magnitudes))
            max_force = float(raw_force_magnitudes[atom_index])
            max_force_applied = float(np.max(applied_force_magnitudes))
            force_abort = max_force > self.fmax_abort

            # Save regular trajectory frames and always checkpoint the unsafe
            # frame before raising a force-safety abort.
            if step % self.save_interval == 0 or force_abort:
                time_fs = float(step * self.timestep / units.fs)
                stress_data = self._stress_observables(atoms)
                configurational_stress = stress_data["configurational_stress"]
                total_stress = stress_data["total_stress"]
                volume = float(atoms.get_volume())
                frame_data = {
                    "step": int(step),
                    "time_fs": time_fs,
                    "phase": phase,
                    "energy": float(pe),
                    "kinetic_energy": float(ke),
                    "total_energy": float(total_e),
                    "temperature": float(temp),
                    "volume": volume,
                    "max_force_raw_eV_A": max_force,
                    "max_force_applied_eV_A": max_force_applied,
                    "configurational_stress": (
                        np.array(configurational_stress, dtype=float, copy=True)
                        if configurational_stress is not None
                        else None
                    ),
                    "total_stress": (
                        np.array(total_stress, dtype=float, copy=True)
                        if total_stress is not None
                        else None
                    ),
                    "configurational_pressure_gpa": stress_data[
                        "configurational_pressure_gpa"
                    ],
                    "total_pressure_gpa": stress_data["total_pressure_gpa"],
                }
                snapshot = atoms.copy()
                snapshot.info["mliport_step"] = int(step)
                snapshot.info["mliport_time_fs"] = time_fs
                snapshot.info["mliport_phase"] = phase
                snapshot.info["forces_raw_eV_A"] = forces_raw.tolist()
                snapshot.info["forces_applied_eV_A"] = forces_applied.tolist()
                snapshot.info["forces_alias"] = "constraint-applied"
                single_point_results: dict[str, Any] = {
                    "energy": float(pe),
                    # Preserve the historical ASE ``get_forces()`` behaviour:
                    # the compatibility alias is constraint-applied. The two
                    # explicit force views above are authoritative.
                    "forces": np.array(forces_applied, dtype=float, copy=True),
                }
                if configurational_stress is not None:
                    single_point_results["stress"] = np.array(
                        configurational_stress, dtype=float, copy=True
                    )
                snapshot.calc = SinglePointCalculator(snapshot, **single_point_results)
                output_stream.submit(
                    MDFrameSnapshot(
                        atoms=snapshot,
                        summary=frame_data,
                        outcar_forces=(
                            np.array(forces_applied, dtype=float, copy=True)
                            if self.write_forces
                            else None
                        ),
                    )
                )

            if force_abort:
                raise ForceSafetyAbort(
                    step=step,
                    max_force=max_force,
                    atom_index=atom_index,
                    threshold=self.fmax_abort,
                )
            # Retain one thermodynamic record per step, independently of the
            # trajectory save interval. The live run logger batches filesystem
            # flushes and does not emit these records as UI render events.
            if step % self.LOG_INTERVAL_STEPS == 0 or step == self.total_steps:
                self.log_buffered(
                    f"Step {step:6d}/{self.total_steps}: "
                    f"E = {total_e:12.4f} eV, T = {temp:6.1f} K "
                    f"[{phase}]"
                )

            # Keep UI/progress-callback updates throttled; emitting hundreds
            # of thousands of render events would not add useful information.
            if step % self.PROGRESS_INTERVAL_STEPS == 0 or step == self.total_steps:
                self._emit_progress(
                    "running",
                    f"Step {step:6d}/{self.total_steps}: "
                    f"E = {total_e:12.4f} eV, T = {temp:6.1f} K "
                    f"[{phase}]",
                    step=step,
                    total_steps=self.total_steps,
                    extra={
                        "energy": float(pe),
                        "temperature": float(temp),
                        "total_energy": float(total_e),
                    },
                )

        dyn.attach(print_progress, interval=1)

        output_closed = False

        def close_output_stream(primary_error: BaseException | None = None) -> None:
            """Drain requested frames, reporting output failure as fatal."""
            nonlocal output_closed
            if output_closed:
                return
            output_closed = True
            try:
                output_stream.close()
            except Exception as output_error:
                self.log(
                    f"\nMD output failed while draining buffered frames: "
                    f"{output_error}",
                    level="error",
                )
                if md_outcar_writer is not None:
                    md_outcar_writer.finalize(status="failed")
                self._write_artifacts_manifest(
                    status="failed",
                    frame_stats=output_stream.stats,
                    error={
                        "type": "output_error",
                        "message": str(output_error),
                    },
                )
                if primary_error is not None:
                    raise output_error from primary_error
                raise

        run_finished_normally = False
        try:
            dyn.run(self.total_steps)
            run_finished_normally = True
        except ForceSafetyAbort as exc:
            self.log(f"\nMD aborted by force safety threshold: {exc}", level="error")
            close_output_stream(exc)
            ContcarWriter().write_with_energy(
                atoms,
                self.vasp_dir / "CONTCAR",
                energy=float(atoms.get_potential_energy()),
                forces=atoms.get_forces(apply_constraint=True),
            )
            if md_outcar_writer is not None:
                md_outcar_writer.finalize(status="aborted")
            self._write_artifacts_manifest(
                status="aborted",
                frame_stats=output_stream.stats,
                error={
                    "type": "force_safety_abort",
                    "message": str(exc),
                    "step": exc.step,
                    "atom_index": exc.atom_index,
                    "max_force_raw_eV_A": exc.max_force,
                    # Compatibility alias retained for artifact readers from
                    # schema v2; its meaning is now explicitly raw.
                    "max_force_eV_A": exc.max_force,
                    "fmax_abort_eV_A": exc.threshold,
                },
            )
            raise
        except CancellationRequested as exc:
            self.log("\n⚠️ MD simulation cancelled by user")
            close_output_stream(exc)
            if md_outcar_writer is not None:
                md_outcar_writer.finalize(status="cancelled")
            self._write_artifacts_manifest(
                status="cancelled", frame_stats=output_stream.stats
            )
            raise  # Re-raise to propagate cancellation
        except KeyboardInterrupt as exc:
            # Ctrl-C is cancellation, not a crash: drain the writer thread
            # (it is non-daemon and would otherwise block process exit) and
            # publish a cancelled status before propagating (review R08).
            self.log("\n⚠️ MD simulation interrupted by user (KeyboardInterrupt)")
            close_output_stream(exc)
            if md_outcar_writer is not None:
                md_outcar_writer.finalize(status="cancelled")
            self._write_artifacts_manifest(
                status="cancelled", frame_stats=output_stream.stats
            )
            raise
        except Exception as e:
            self.log(f"\n❌ MD simulation failed: {e}", level="error")
            close_output_stream(e)
            if md_outcar_writer is not None:
                md_outcar_writer.finalize(status="failed")
            self._write_artifacts_manifest(
                status="failed", frame_stats=output_stream.stats
            )
            raise
        finally:
            # Backstop for BaseException paths that no handler above
            # enumerated (KeyboardInterrupt raised inside a handler,
            # SystemExit, ...): the non-daemon writer thread must always be
            # drained and the manifest must never stay "running" (R08).
            if not run_finished_normally and not output_closed:
                exc_type = sys.exc_info()[0]
                terminal = (
                    "cancelled"
                    if exc_type in (KeyboardInterrupt, CancellationRequested)
                    else "failed"
                )
                self.log(
                    f"\nMD run ended abnormally ({exc_type.__name__ if exc_type else 'unknown'}); "
                    f"closing output with status {terminal}",
                    level="error",
                )
                try:
                    close_output_stream()
                except Exception as cleanup_error:  # noqa: BLE001
                    self.log(
                        f"MD output cleanup failed: {cleanup_error}", level="error"
                    )
                if md_outcar_writer is not None:
                    md_outcar_writer.finalize(status=terminal)
                with contextlib.suppress(Exception):
                    self._write_artifacts_manifest(
                        status=terminal, frame_stats=output_stream.stats
                    )

        close_output_stream()
        md_time = time.time() - start_time

        final_energy = float(atoms.get_potential_energy())
        final_forces_raw, final_forces_applied = self._force_views(
            atoms, final_energy, context="MD final structure"
        )
        _, final_temp, _ = self._validate_md_safe_point(
            atoms,
            energy=final_energy,
            forces_raw=final_forces_raw,
            forces_applied=final_forces_applied,
            context="MD final structure",
        )
        final_stress_data = self._stress_observables(atoms)

        self.log(f"\nMD simulation completed in {md_time:.2f} s")
        self.log(f"Final temperature: {final_temp:.1f} K")
        self.log(f"Final energy: {final_energy:.6f} eV")

        # Build results
        results = {
            "energy": final_energy,
            # ``forces`` remains the historical constraint-applied alias.
            "forces": final_forces_applied,
            "forces_raw_eV_A": final_forces_raw,
            "forces_applied_eV_A": final_forces_applied,
            **final_stress_data,
            "temperature": final_temp,
            "md_steps": self.total_steps,
            "equilibration_steps": self.equilibration_steps,
            "production_steps": self.steps,
            "production_start_step": self.equilibration_steps,
            "production_start_frame": output_stream.stats.production_start_frame,
            "timestep_fs": float(self.timestep / units.fs),
            "save_interval": self.save_interval,
            "ensemble": self.ensemble.upper(),
            "thermostat": self.thermostat.upper() if self.ensemble == "nvt" else None,
            "target_temperature": self.temperature,
            "seed": self.seed,
            "velocity_policy": self.velocity_policy,
            "md_provenance": self._md_provenance(),
            "time": md_time,
            "trajectory": MDTrajectorySummary(md_csv_path, output_stream.stats.count),
            "trajectory_frame_count": output_stream.stats.count,
            "trajectory_path": str(traj_path) if self.write_trajectory else None,
            "md_csv_path": str(md_csv_path),
            "pre_relaxed": self.pre_relax,
        }

        # Write outputs (XDATCAR was already streamed during the run)
        self._emit_progress("writing_output", "Writing trajectory and output files...")
        if md_outcar_writer is not None:
            md_outcar_writer.finalize(
                status="completed",
                md_time_s=md_time,
                final_energy=float(final_energy),
                final_temperature=float(final_temp),
            )
        self._write_outputs(atoms, results)
        self._write_artifacts_manifest(
            status="completed", frame_stats=output_stream.stats
        )

        # Print summary
        self._write_summary(results, atoms)
        self._emit_progress(
            "done",
            f"MD complete. Final T = {final_temp:.1f} K",
            extra={"energy": float(final_energy), "temperature": float(final_temp)},
        )

        return results

    def _write_outputs(
        self,
        atoms: Atoms,
        results: dict[str, Any],
    ) -> None:
        """Write output files.

        XDATCAR, OUTCAR, trajectory.traj, and md.csv are streamed during the
        run (see :meth:`run`), so this writes the JSON summary and final
        VASP-syntax-compatible CONTCAR.

        Args:
            atoms: ASE Atoms object
            results: Results dictionary
        """
        metadata = self.calculator.info()
        metadata["pre_relaxed"] = self.pre_relax

        if self.write_outcar and (self.vasp_dir / "OUTCAR").exists():
            self.log(f"VASP-like OUTCAR written to: {self.vasp_dir / 'OUTCAR'}")

        # XDATCAR + md.csv were streamed frame-by-frame during the run;
        # log their paths if anything was written.
        if self.write_xdatcar and (self.vasp_dir / "XDATCAR").exists():
            self.log(f"XDATCAR written to: {self.vasp_dir / 'XDATCAR'}")
        if (self.raw_dir / "md.csv").exists():
            self.log(f"md.csv written to: {self.raw_dir / 'md.csv'}")
        if (self.raw_dir / "trajectory.traj").exists():
            self.log(
                f"Canonical trajectory written to: "
                f"{self.raw_dir / 'trajectory.traj'}"
            )

        # Write JSON
        if self.write_json:
            json_path = self.raw_dir / "mliport_results.json"
            writer = JsonWriter()
            json_metadata = metadata.copy() if metadata else {}
            if self.job_name:
                json_metadata["job_name"] = self.job_name
            writer.write(
                atoms,
                results,
                json_path,
                mode="md",
                metadata=json_metadata,
            )
            self.log(f"JSON results written to: {json_path}")

        # Write final structure
        contcar_path = self.vasp_dir / "CONTCAR"
        export_atoms = atoms.copy()
        # FixCom is an integrator detail, not a VASP selective-dynamics flag.
        # Keeping it makes ASE emit a redundant "Selective dynamics" section
        # with T/T/T on every ion. Preserve any real user constraints.
        export_atoms.set_constraint(
            [
                constraint
                for constraint in export_atoms.constraints
                if not isinstance(constraint, FixCom)
            ]
        )
        ContcarWriter().write(
            export_atoms,
            contcar_path,
            comment="mliport MD final structure",
            direct=True,
        )
        self.log(f"CONTCAR written to: {contcar_path}")
