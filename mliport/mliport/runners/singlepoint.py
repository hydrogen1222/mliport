# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Single point calculation runner.

Runs single point energy, force, and stress calculations.
Outputs results in multiple formats (OUTCAR, JSON).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING
import threading

import numpy as np

from mliport.runners.base import BaseRunner
from mliport.writers.contcar import ContcarWriter
from mliport.writers.json_writer import JsonWriter
from mliport.writers.outcar import OutcarWriter

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from mliport.protocols import ProgressCallback


class SinglePointRunner(BaseRunner):
    """Run single point calculations.

    Calculates energy, forces, and optionally stress for a given structure.

    Example:
        >>> runner = SinglePointRunner(calculator, output_dir="results")
        >>> results = runner.run(atoms)
        >>> print(f"Energy: {results['energy']:.4f} eV")
    """

    def __init__(
        self,
        calculator,
        output_dir: Path | str = ".",
        write_outcar: bool = True,
        write_forces: bool = True,
        write_stress: bool = True,
        write_json: bool = True,
        write_contcar: bool = True,
        verbose: bool = True,
        job_name: str | None = None,
        log_fn: Any | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        charge: int | None = None,
        spin: int | None = None,
    ):
        """Initialize single point runner.

        Args:
            calculator: UMA calculator wrapper
            output_dir: Directory for output files
            write_outcar: Whether to write OUTCAR file
            write_forces: Whether OUTCAR includes the force table
            write_stress: Whether OUTCAR includes the stress tensor
            write_json: Whether to write JSON results
            write_contcar: Whether to write CONTCAR file
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
        self.write_outcar = write_outcar
        self.write_forces = write_forces
        self.write_stress = write_stress
        self.write_json = write_json
        self.write_contcar = write_contcar

    def run(self, atoms: Atoms) -> dict[str, Any]:
        """Run single point calculation.

        Args:
            atoms: ASE Atoms object

        Returns:
            Dictionary with results (energy, forces, stress, time)
        """
        self.print_header("SINGLE POINT CALCULATION")
        self._emit_progress("loading_model", "Loading model and preparing structure...")

        # Prepare atoms (check PBC, cell, etc.)
        atoms = self._prepare_atoms(atoms)

        # Log structure info for debugging
        self.log(f"Cell: {' x '.join([f'{x:.4f}' for x in atoms.cell.lengths()])} Å")
        self.log(f"PBC: {atoms.pbc}")
        self.log(f"Volume: {atoms.cell.volume:.2f} Å³")

        # Validate structure: a *periodic* system needs a real, non-zero
        # volume cell; a non-periodic molecule does not (the cell is only a
        # bounding box and is absent for e.g. .xyz input, which is normal for
        # the isolated gas molecule in an adsorption-energy calculation).
        if atoms.pbc.any() and atoms.cell.volume <= 0:
            raise ValueError(
                "Invalid cell: zero or negative volume for a periodic "
                "system. Check input structure."
            )

        # Setup calculator
        calc = self._get_calculator()
        atoms.calc = calc
        self._emit_progress("running", "Calculating energy and forces...")

        # Run calculation
        self.log("Calculating energy and forces...")
        start_time = time.time()

        # Run calculation with error handling
        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
        except ValueError as e:
            error_msg = str(e)
            if "No edges found" in error_msg:
                raise RuntimeError(
                    "\n" + "=" * 70 + "\n"
                    "CALCULATION FAILED: No edges found in structure\n"
                    "=" * 70 + "\n\n"
                    "The model could not build a neighbor list for your structure.\n\n"
                    "Common causes:\n"
                    "  1. Atoms are too far apart (>6 Å cutoff)\n"
                    "  2. Cell is too large or has wrong PBC settings\n"
                    "  3. Structure is not periodic but should be (or vice versa)\n\n"
                    "Debug information:\n"
                    f"  Cell lengths: {atoms.cell.lengths()}\n"
                    f"  Cell volume: {atoms.cell.volume:.2f} Å³\n"
                    f"  PBC: {atoms.pbc}\n"
                    f"  Number of atoms: {len(atoms)}\n\n"
                    "Suggestions:\n"
                    "  - Check that the input structure file is correct\n"
                    "  - For bulk materials, ensure cell is not too large\n"
                    "  - Try the original POSCAR format instead of CIF\n"
                    "=" * 70
                ) from e
            raise

        # Abort early on NaN/inf so it never reaches the output files.
        self._check_finite(atoms, energy, forces, context="single point")

        # Get stress if supported. Stress is only physically defined for
        # periodic systems; MACE advertises "stress" even for non-periodic
        # molecules, so gate on PBC to avoid a meaningless / erroring tensor.
        stress = None
        if self.calculator.has_stress and atoms.pbc.any():
            self._emit_progress(
                "running", "Calculating stress...", extra={"energy": float(energy)}
            )
            self.log("Calculating stress...")
            stress = atoms.get_stress()
            if not np.all(np.isfinite(stress)):
                raise RuntimeError(
                    "Non-finite stress during single point; aborting before "
                    "NaN is written to outputs."
                )

        calc_time = time.time() - start_time

        self.log(f"Energy: {energy:.6f} eV")
        self.log(f"Calculation completed in {calc_time:.2f} s")

        # Build results
        results = {
            "energy": energy,
            "forces": forces,
            "stress": stress,
            "time": calc_time,
        }

        # Write outputs
        self._emit_progress("writing_output", "Writing output files...")
        self._write_outputs(atoms, results)

        # Print summary
        self._write_summary(results, atoms)
        self._emit_progress(
            "done",
            "Calculation complete",
            extra={
                "energy": float(energy),
                "time": calc_time,
            },
        )

        return results

    def _write_outputs(self, atoms: Atoms, results: dict[str, Any]) -> None:
        """Write output files.

        Args:
            atoms: ASE Atoms object
            results: Results dictionary
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
                mode="single_point",
                task_name=self.calculator.task,
                metadata=metadata,
            )
            self.log(f"OUTCAR written to: {outcar_path}")

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
                mode="single_point",
                metadata=json_metadata,
            )
            self.log(f"JSON results written to: {json_path}")

        # Write CONTCAR (same as input for SP, but included for consistency)
        if self.write_contcar:
            contcar_path = self.output_dir / "CONTCAR"
            writer = ContcarWriter()
            writer.write_with_energy(
                atoms,
                contcar_path,
                energy=results["energy"],
                forces=results["forces"],
            )
            self.log(f"CONTCAR written to: {contcar_path}")
