# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Geometry optimization runner.

Runs geometry optimization with support for:
- Position optimization
- Cell optimization (constant pressure)
- Multiple optimizers (FIRE, BFGS, LBFGS)
- Symmetry preservation
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING
import threading

import numpy as np
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE, LBFGS

from mliport.runners.base import BaseRunner
from mliport.protocols import CancellationRequested
from mliport.writers.contcar import ContcarWriter
from mliport.writers.json_writer import JsonWriter
from mliport.writers.oszicar import OszicarWriter
from mliport.writers.outcar import OutcarWriter

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from mliport.protocols import ProgressCallback


class OptimizationRunner(BaseRunner):
    """Run geometry optimization calculations.

    Optimizes atomic positions and optionally cell parameters
    until forces converge below threshold.

    Example:
        >>> runner = OptimizationRunner(
        ...     calculator,
        ...     fmax=0.05,
        ...     max_steps=500,
        ...     cell_opt=True
        ... )
        >>> results = runner.run(atoms)
        >>> print(f"Converged: {results['converged']}")
        >>> print(f"Steps: {results['nsteps']}")
    """

    OPTIMIZERS = {
        "fire": FIRE,
        "bfgs": BFGS,
        "lbfgs": LBFGS,
    }

    def __init__(
        self,
        calculator,
        fmax: float = 0.05,
        max_steps: int = 500,
        optimizer: str = "FIRE",
        cell_opt: bool = False,
        fix_symmetry: bool = False,
        output_dir: Path | str = ".",
        write_outcar: bool = True,
        write_forces: bool = True,
        write_stress: bool = True,
        write_oszicar: bool = True,
        write_json: bool = True,
        trajectory_interval: int = 1,
        verbose: bool = True,
        job_name: str | None = None,
        log_fn: Any | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        charge: int | None = None,
        spin: int | None = None,
    ):
        """Initialize optimization runner.

        Args:
            calculator: UMA calculator wrapper
            fmax: Force convergence threshold (eV/Å)
            max_steps: Maximum optimization steps
            optimizer: Optimizer algorithm (FIRE, BFGS, LBFGS)
            cell_opt: Whether to optimize cell parameters
            fix_symmetry: Whether to preserve symmetry
            output_dir: Directory for output files
            write_outcar: Whether to write OUTCAR file
            write_forces: Whether OUTCAR includes the final force table
            write_stress: Whether OUTCAR includes the final stress tensor
            write_oszicar: Whether to write OSZICAR file
            write_json: Whether to write JSON results
            trajectory_interval: Interval for saving trajectory frames
            verbose: Whether to print progress messages
            job_name: Optional job name for organizing results
            log_fn: Optional callback function for custom log output
            progress_callback: Optional callback for progress events
        """
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
        self.fmax = fmax
        self.max_steps = max_steps
        self.optimizer_name = optimizer.lower()
        self.cell_opt = cell_opt
        self.fix_symmetry = fix_symmetry
        self.write_outcar = write_outcar
        self.write_forces = write_forces
        self.write_stress = write_stress
        self.write_oszicar = write_oszicar
        self.write_json = write_json
        self.trajectory_interval = trajectory_interval

        # Validate optimizer
        if self.optimizer_name not in self.OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer: {optimizer}. "
                f"Use one of: {', '.join(self.OPTIMIZERS.keys())}"
            )

        # Check cell_opt requirements
        if cell_opt and calculator.task in {"omol", "molecule"}:
            raise ValueError(
                "Cell optimization is not physically defined for an isolated "
                "molecule. Use positions-only optimization."
            )
        elif cell_opt and not calculator.has_stress:
            raise ValueError(
                "Cell optimization requires calculator stress support; use "
                "positions-only optimization for this model/task."
            )

    def run(self, atoms: Atoms) -> dict[str, Any]:
        """Run geometry optimization.

        Args:
            atoms: ASE Atoms object

        Returns:
            Dictionary with results including converged status and steps
        """
        if self.cell_opt and not bool(np.asarray(atoms.pbc, dtype=bool).all()):
            raise ValueError(
                "Cell optimization currently requires full 3D periodicity "
                "(pbc=True in all directions). Partial-PBC and non-periodic "
                "cell relaxation are not supported; use positions-only "
                "optimization."
            )

        self.print_header("GEOMETRY OPTIMIZATION")
        self._emit_progress("loading_model", "Loading model and preparing structure...")

        # Print settings
        self.log(f"Optimizer:        {self.optimizer_name.upper()}")
        self.log(f"Convergence:      fmax = {self.fmax} eV/Å")
        self.log(f"Max steps:        {self.max_steps}")
        self.log(f"Cell optimization: {'Yes' if self.cell_opt else 'No'}")
        self.log(f"Fix symmetry:     {'Yes' if self.fix_symmetry else 'No'}")

        # Prepare atoms
        atoms = self._prepare_atoms(atoms)

        # Apply symmetry constraint
        if self.fix_symmetry:
            self.log("Applying symmetry constraint")
            constraints = list(atoms.constraints)
            constraints.append(FixSymmetry(atoms))
            atoms.set_constraint(constraints)

        # Setup calculator
        calc = self._get_calculator()
        atoms.calc = calc

        # Setup optimizer
        optimizer_class = self.OPTIMIZERS[self.optimizer_name]

        if self.cell_opt:
            # Use cell filter for cell optimization
            opt_atoms = FrechetCellFilter(atoms)
            logfile = str(self.output_dir / "optimization.log")
        else:
            opt_atoms = atoms
            logfile = str(self.output_dir / "optimization.log")

        opt = optimizer_class(opt_atoms, logfile=logfile)
        self._emit_progress(
            "running", "Starting optimization...", step=0, total_steps=self.max_steps
        )

        # Setup trajectory tracking
        trajectory = []

        def trajectory_callback():
            """Callback to track optimization progress."""
            # Check for cooperative cancellation first
            if self._is_cancelled():
                raise CancellationRequested("Optimization cancelled by user")

            step = opt.nsteps
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            # Abort on NaN/inf before it pollutes the trajectory/outputs.
            self._check_finite(
                atoms, energy, forces, context=f"optimization step {step}"
            )

            trajectory.append(
                {
                    "step": step,
                    "energy": energy,
                    "forces": forces.copy(),
                    "natoms": len(atoms),
                }
            )

            # Print progress every 10 steps
            if step % 10 == 0 or step == 1:
                force_mags = np.linalg.norm(forces, axis=1)
                fmax_current = np.max(force_mags)
                self._emit_progress(
                    "running",
                    f"Step {step:4d}: E = {energy:12.6f} eV, fmax = {fmax_current:.6f} eV/A",
                    step=step,
                    total_steps=self.max_steps,
                    extra={"energy": float(energy), "fmax": float(fmax_current)},
                )
                self.log(
                    f"Step {step:4d}: E = {energy:12.6f} eV, fmax = {fmax_current:.6f} eV/Å"
                )

        opt.attach(trajectory_callback)

        # Run optimization
        self.log("\nStarting optimization...")
        start_time = time.time()

        try:
            opt.run(fmax=self.fmax, steps=self.max_steps)
            converged = opt.converged()
        except CancellationRequested:
            self.log("\nOptimization cancelled by user")
            converged = False
            raise  # Re-raise to stop the run
        except Exception as e:
            self.log(f"Optimization failed: {e}", level="error")
            converged = False
            # A non-finite (NaN/inf) abort from _check_finite must propagate
            # immediately: by then the structure may hold NaN positions that
            # crash the model on re-evaluation, and we must never write NaN
            # results. Other optimizer failures fall through to the graceful
            # "return partial results" path below.
            if isinstance(e, RuntimeError) and "Non-finite" in str(e):
                raise

        opt_time = time.time() - start_time

        # Get final results
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        self._check_finite(atoms, energy, forces, context="optimization final")
        stress = None
        if self.calculator.has_stress and atoms.pbc.any():
            stress = atoms.get_stress()
            if not np.all(np.isfinite(stress)):
                raise RuntimeError(
                    "Non-finite stress during optimization final; aborting "
                    "before NaN is written to outputs."
                )

        self.log(f"\nOptimization finished in {opt.nsteps} steps")
        self.log(f"Converged: {'Yes' if converged else 'No'}")

        # Build results
        results = {
            "energy": energy,
            "forces": forces,
            "stress": stress,
            "nsteps": opt.nsteps,
            "converged": converged,
            "fmax": self.fmax,
            "time": opt_time,
            "trajectory": trajectory,
        }

        # Write outputs
        self._emit_progress("writing_output", "Writing output files...")
        self._write_outputs(atoms, results, trajectory)

        # Print summary
        self._write_summary(results, atoms)
        self._emit_progress(
            "done",
            f"Optimization {'converged' if converged else 'not converged'} in {opt.nsteps} steps",
            extra={
                "energy": float(energy),
                "converged": converged,
                "nsteps": opt.nsteps,
            },
        )

        return results

    def _write_outputs(
        self,
        atoms: Atoms,
        results: dict[str, Any],
        trajectory: list,
    ) -> None:
        """Write output files.

        Args:
            atoms: ASE Atoms object
            results: Results dictionary
            trajectory: Optimization trajectory
        """
        metadata = self.calculator.info()

        # Write OUTCAR
        if self.write_outcar:
            outcar_path = self.output_dir / "OUTCAR"
            writer = OutcarWriter()
            outcar_results = dict(results)
            if not self.write_forces:
                outcar_results.pop("forces", None)
            if not self.write_stress:
                outcar_results.pop("stress", None)
            writer.write(
                atoms,
                outcar_results,
                outcar_path,
                mode="optimization",
                task_name=self.calculator.task,
                metadata=metadata,
            )
            self.log(f"OUTCAR written to: {outcar_path}")

        # Write OSZICAR
        if self.write_oszicar and trajectory:
            oszicar_path = self.output_dir / "OSZICAR"
            writer = OszicarWriter()
            writer.write(
                oszicar_path,
                trajectory,
                fmax_target=self.fmax,
            )
            self.log(f"OSZICAR written to: {oszicar_path}")

        # Write JSON
        if self.write_json:
            json_path = self.output_dir / "mliport_results.json"
            writer = JsonWriter()
            json_metadata = metadata.copy() if metadata else {}
            if self.job_name:
                json_metadata["job_name"] = self.job_name
            writer.write(
                atoms,
                results,
                json_path,
                mode="optimization",
                metadata=json_metadata,
            )
            self.log(f"JSON results written to: {json_path}")

        # Write CONTCAR (optimized structure)
        contcar_path = self.output_dir / "CONTCAR"
        writer = ContcarWriter()
        writer.write_with_energy(
            atoms,
            contcar_path,
            energy=results["energy"],
            forces=results["forces"],
        )
        self.log(f"CONTCAR written to: {contcar_path}")
