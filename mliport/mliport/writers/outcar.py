# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
VASP-style OUTCAR writer.

Generates detailed output files similar to VASP's OUTCAR format
for familiar visualization and analysis workflows.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import numpy as np

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms


class OutcarWriter:
    """Write calculation results in VASP OUTCAR-like format.

    Provides detailed output with structure, energies, forces, stress,
    and timing information in a format familiar to VASP users.

    Example:
        >>> writer = OutcarWriter()
        >>> writer.write(atoms, results, Path("OUTCAR"), mode="single_point")
    """

    def __init__(self):
        """Initialize OUTCAR writer."""
        self.lines: list[str] = []

    def write(
        self,
        atoms: Atoms,
        results: dict[str, Any],
        output_path: Path | str,
        mode: str = "single_point",
        task_name: str = "omat",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write results to OUTCAR file.

        Args:
            atoms: ASE Atoms object
            results: Results dictionary from calculation
            output_path: Output file path
            mode: Calculation mode (single_point, optimization, md)
            task_name: Task name (omat, omol, etc.)
            metadata: Additional metadata to include
        """
        output_path = Path(output_path)
        self.lines = []

        # Generate content
        self._write_header()
        self._write_system_info(atoms, task_name, mode)
        self._write_model_info(metadata)
        self._write_input_structure(atoms)
        self._write_results(atoms, results)
        if mode == "optimization":
            self._write_optimization_info(results)
        elif mode == "md":
            self._write_md_info(results)
        self._write_timing(results)
        self._write_footer()

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))

    def _write_header(self) -> None:
        """Write file header."""
        self.lines.extend(
            [
                "=" * 80,
                " MLIP CALCULATION RESULTS".center(80),
                " (mliport - backend-neutral MLIP workflows)".center(80),
                "=" * 80,
                "",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
            ]
        )

    def _write_system_info(self, atoms: Atoms, task_name: str, mode: str) -> None:
        """Write system information section."""
        formula = atoms.get_chemical_formula()
        symbols = atoms.get_chemical_symbols()
        atom_counts = Counter(symbols)
        atom_summary = ", ".join([f"{k}: {v}" for k, v in sorted(atom_counts.items())])

        self.lines.extend(
            [
                "-" * 80,
                " SYSTEM INFORMATION",
                "-" * 80,
                "",
                f"Formula:           {formula}",
                f"Number of atoms:   {len(atoms)}",
                f"Atom types:        {atom_summary}",
                f"Task:              {task_name}",
                f"Calculation mode:  {mode}",
                "",
            ]
        )

    def _write_model_info(self, metadata: dict[str, Any] | None) -> None:
        """Write model information section."""
        self.lines.extend(
            [
                "-" * 80,
                " MODEL INFORMATION",
                "-" * 80,
                "",
            ]
        )

        if metadata:
            if "model_path" in metadata:
                self.lines.append(f"Model path:        {metadata['model_path']}")
            if "device" in metadata:
                self.lines.append(f"Device:            {metadata['device']}")
            if "inference_mode" in metadata:
                self.lines.append(f"Inference mode:    {metadata['inference_mode']}")
            if "implemented_properties" in metadata:
                props = ", ".join(metadata["implemented_properties"])
                self.lines.append(f"Properties:        {props}")

        self.lines.append("")

    def _write_input_structure(self, atoms: Atoms) -> None:
        """Write input structure section."""
        cell = atoms.cell
        positions = atoms.positions
        symbols = atoms.get_chemical_symbols()

        self.lines.extend(
            [
                "-" * 80,
                " INPUT STRUCTURE",
                "-" * 80,
                "",
                "Lattice vectors (Å):",
            ]
        )

        for i in range(3):
            self.lines.append(
                f"  {cell[i][0]:12.6f}  {cell[i][1]:12.6f}  {cell[i][2]:12.6f}"
            )

        self.lines.extend(
            [
                "",
                f"Cell lengths (Å):    {cell.lengths()[0]:.6f}  {cell.lengths()[1]:.6f}  {cell.lengths()[2]:.6f}",
                f"Cell angles (°):     {cell.angles()[0]:.6f}  {cell.angles()[1]:.6f}  {cell.angles()[2]:.6f}",
                f"Volume (Å³):         {atoms.get_volume():.6f}",
                "",
                "Atomic positions (Cartesian, Å):",
                f"{'Atom':>6} {'Type':>6} {'x':>12} {'y':>12} {'z':>12}",
                "-" * 60,
            ]
        )

        for i, (symbol, pos) in enumerate(zip(symbols, positions)):
            self.lines.append(
                f"{i + 1:>6} {symbol:>6} {pos[0]:>12.6f} {pos[1]:>12.6f} {pos[2]:>12.6f}"
            )

        self.lines.append("")

    def _write_results(self, atoms: Atoms, results: dict[str, Any]) -> None:
        """Write calculation results section."""
        energy = results.get("energy")
        forces = results.get("forces")
        stress = results.get("stress")

        self.lines.extend(
            [
                "-" * 80,
                " ENERGY",
                "-" * 80,
                "",
            ]
        )

        if energy is not None:
            self.lines.extend(
                [
                    f"Total energy:       {energy:16.8f} eV",
                    f"Energy per atom:    {energy / len(atoms):16.8f} eV/atom",
                    "",
                ]
            )

        if forces is not None:
            self.lines.extend(
                [
                    "-" * 80,
                    " FORCES (eV/Å)",
                    "-" * 80,
                    "",
                    f"{'Atom':>6} {'Type':>6} {'Fx':>12} {'Fy':>12} {'Fz':>12} {'|F|':>12}",
                    "-" * 70,
                ]
            )

            symbols = atoms.get_chemical_symbols()
            max_force = 0.0
            max_force_idx = 0
            rms_force = 0.0

            for i in range(len(atoms)):
                fx, fy, fz = forces[i]
                force_mag = np.linalg.norm(forces[i])
                self.lines.append(
                    f"{i + 1:>6} {symbols[i]:>6} {fx:>12.6f} {fy:>12.6f} {fz:>12.6f} {force_mag:>12.6f}"
                )

                if force_mag > max_force:
                    max_force = force_mag
                    max_force_idx = i
                rms_force += force_mag**2

            rms_force = np.sqrt(rms_force / len(atoms))

            self.lines.extend(
                [
                    "",
                    f"Maximum force:      {max_force:12.6f} eV/Å on atom {max_force_idx + 1} ({symbols[max_force_idx]})",
                    f"RMS force:          {rms_force:12.6f} eV/Å",
                    "",
                ]
            )

        if stress is not None:
            self.lines.extend(
                [
                    "-" * 80,
                    " STRESS TENSOR",
                    "-" * 80,
                    "",
                    "Stress (eV/Å³):",
                    f"{'':>12} {'xx':>12} {'yy':>12} {'zz':>12} {'yz':>12} {'xz':>12} {'xy':>12}",
                    f"{'Voigt':>12} {stress[0]:>12.6f} {stress[1]:>12.6f} {stress[2]:>12.6f} "
                    f"{stress[3]:>12.6f} {stress[4]:>12.6f} {stress[5]:>12.6f}",
                    "",
                    "Stress (GPa):",
                ]
            )

            # Convert to GPa (1 eV/Å³ = 160.2177 GPa)
            stress_gpa = np.array(stress) * 160.2177
            self.lines.append(
                f"{'Voigt':>12} {stress_gpa[0]:>12.6f} {stress_gpa[1]:>12.6f} {stress_gpa[2]:>12.6f} "
                f"{stress_gpa[3]:>12.6f} {stress_gpa[4]:>12.6f} {stress_gpa[5]:>12.6f}"
            )

            # Pressure
            pressure = -(stress[0] + stress[1] + stress[2]) / 3.0 * 160.2177
            self.lines.extend(
                [
                    "",
                    f"Pressure:           {pressure:12.6f} GPa",
                    "",
                ]
            )

    def _write_optimization_info(self, results: dict[str, Any]) -> None:
        """Write optimization-specific information."""
        nsteps = results.get("nsteps")
        converged = results.get("converged")
        fmax = results.get("fmax")

        self.lines.extend(
            [
                "-" * 80,
                " OPTIMIZATION",
                "-" * 80,
                "",
            ]
        )

        if nsteps is not None:
            self.lines.append(f"Steps taken:        {nsteps}")
        if converged is not None:
            status = "Yes" if converged else "No"
            self.lines.append(f"Converged:          {status}")
        if fmax is not None:
            self.lines.append(f"Final fmax:         {fmax:.6f} eV/Å")

        self.lines.append("")

    def _write_md_info(self, results: dict[str, Any]) -> None:
        """Write MD-specific information."""
        steps = results.get("md_steps")
        temperature = results.get("temperature")

        self.lines.extend(
            [
                "-" * 80,
                " MOLECULAR DYNAMICS",
                "-" * 80,
                "",
            ]
        )

        if steps is not None:
            self.lines.append(f"MD steps:           {steps}")
        if temperature is not None:
            self.lines.append(f"Temperature:        {temperature:.2f} K")

        self.lines.append("")

    def _write_timing(self, results: dict[str, Any]) -> None:
        """Write timing information."""
        calc_time = results.get("time", 0.0)

        self.lines.extend(
            [
                "-" * 80,
                " TIMING",
                "-" * 80,
                "",
                f"Calculation time:   {calc_time:.2f} s",
                f"                     ({calc_time / 60:.2f} min)"
                if calc_time > 60
                else "",
            ]
        )

        # Remove empty line if no minutes
        if calc_time <= 60:
            self.lines.pop()

        self.lines.append("")

    def _write_footer(self) -> None:
        """Write file footer."""
        self.lines.extend(
            [
                "=" * 80,
                " END OF MLIP CALCULATION",
                "=" * 80,
            ]
        )


class MDOutcarWriter:
    """Stream a documented VASP-like subset for an MLIP MD trajectory.

    This deliberately is *not* presented as a native VASP OUTCAR: an MLIP
    calculation has no electronic SCF, POTCAR, Fermi level, or VASP thermostat
    state.  Familiar position/force tables make the
    text useful to humans and simple downstream readers without inventing
    electronic-structure data.
    """

    EV_A3_TO_GPA = 160.21766208

    def __init__(self) -> None:
        self.output_path: Path | None = None
        self.configuration_index = 0
        self._finished = False
        self._stream: TextIO | None = None

    def write_header(
        self,
        atoms: Atoms,
        output_path: Path | str,
        *,
        task_name: str,
        metadata: dict[str, Any] | None,
        settings: dict[str, Any],
    ) -> None:
        """Create the file and write static model/system/MD information."""
        self.close_stream()
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.configuration_index = 0
        self._finished = False

        atom_counts = Counter(atoms.get_chemical_symbols())
        summary = "  ".join(
            f"{symbol}={count}" for symbol, count in sorted(atom_counts.items())
        )
        model = metadata or {}
        lines = [
            "=" * 100,
            " MLIPORT VASP-LIKE MOLECULAR DYNAMICS OUTPUT".center(100),
            " NOT A NATIVE VASP OUTCAR: no electronic/SCF data are implied".center(100),
            "=" * 100,
            "",
            f"Generated:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Format contract:    mliport.vasp-like-outcar.md/3",
            f"Formula:            {atoms.get_chemical_formula()}",
            f"Number of ions:     {len(atoms)}",
            f"Ion counts:         {summary}",
            f"Task:               {task_name}",
        ]

        for key, label in (
            ("model_type", "Model type"),
            ("model_path", "Model path"),
            ("device", "Device"),
            ("inference_mode", "Inference mode"),
        ):
            if model.get(key) is not None:
                lines.append(f"{label + ':':<20}{model[key]}")

        lines.extend(["", " MD SETTINGS", "-" * 100])
        for key, value in settings.items():
            lines.append(f"{key + ':':<20}{value}")
        lines.extend(
            [
                "",
                "Units: positions=Angstrom, forces=eV/Angstrom, energy=eV, ",
                "       stress=eV/Angstrom^3 and GPa, time=fs",
                "Stress convention: ASE Voigt order xx yy zz yz xz xy.",
                "  configurational = calculator stress (no kinetic term)",
                "  total MD stress = configurational + ideal-gas kinetic term",
                "  scalar pressures = -trace(stress)/3 (3D PBC only)",
                "Force table: ASE constraint-applied forces; raw model forces are",
                "  preserved separately in raw/trajectory.traj and results JSON.",
                "",
            ]
        )
        self.output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def open_stream(self) -> None:
        """Keep the initialized MD OUTCAR open for efficient frame appends."""
        if self.output_path is None:
            raise RuntimeError("write_header() must be called before open_stream()")
        if self._finished:
            raise RuntimeError("Cannot open a finished MD OUTCAR")
        self.close_stream()
        self._stream = self.output_path.open(
            "a", encoding="utf-8", buffering=1024 * 1024
        )

    def flush(self) -> None:
        """Flush a persistent append stream, if one is active."""
        if self._stream is not None:
            self._stream.flush()

    def close_stream(self) -> None:
        """Flush and close a persistent append stream."""
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
        self._stream = None

    def append_frame(
        self,
        atoms: Atoms,
        *,
        step: int,
        time_fs: float,
        potential_energy: float,
        kinetic_energy: float,
        total_energy: float,
        temperature: float,
        forces: np.ndarray | None,
        configurational_stress: np.ndarray | None,
        total_stress: np.ndarray | None,
    ) -> None:
        """Append one saved ionic configuration and its observables."""
        if self.output_path is None:
            raise RuntimeError("write_header() must be called before append_frame()")
        if self._finished:
            raise RuntimeError("Cannot append a frame after finalize()")

        self.configuration_index += 1
        lines = [
            "-" * 100,
            f" Ionic step {step:12d}   "
            f"Saved configuration {self.configuration_index:8d}   "
            f"Time = {time_fs:16.8f} fs",
            "-" * 100,
            "",
            " direct lattice vectors                 reciprocal lattice vectors",
        ]
        reciprocal = atoms.cell.reciprocal()
        for vector, rec_vector in zip(atoms.cell, reciprocal):
            lines.append(
                " "
                + "".join(f"{value:13.7f}" for value in vector)
                + "    "
                + "".join(f"{value:13.7f}" for value in rec_vector)
            )

        lines.append("")
        if forces is not None:
            force_rows = np.asarray(forces, dtype=float)
            if force_rows.shape != (len(atoms), 3) or not np.all(
                np.isfinite(force_rows)
            ):
                raise ValueError("MD OUTCAR forces must be a finite (N, 3) array")
            lines.extend(
                [
                    " POSITION                         CONSTRAINT-APPLIED-FORCE (eV/Angst)",
                    " " + "-" * 91,
                ]
            )
            for position, force in zip(atoms.positions, force_rows):
                lines.append(
                    " "
                    + "".join(f"{value:14.8f}" for value in position)
                    + "   "
                    + "".join(f"{value:14.8f}" for value in force)
                )
        else:
            lines.extend(
                [" POSITION (Angstrom; force output disabled)", " " + "-" * 91]
            )
            for position in atoms.positions:
                lines.append(" " + "".join(f"{value:14.8f}" for value in position))
        lines.extend(
            [
                " " + "-" * 91,
                "",
                f"  free  energy   MLIPORT-TOTEN  = {potential_energy:20.10f} eV",
                f"  kinetic energy EKIN         = {kinetic_energy:20.10f} eV",
                f"  total energy   ETOTAL       = {total_energy:20.10f} eV",
                f"  temperature    T            = {temperature:20.8f} K",
                f"  volume of cell               = {atoms.get_volume():20.10f} Angstrom^3",
            ]
        )

        if configurational_stress is not None and total_stress is not None:
            for label, stress in (
                ("configurational", configurational_stress),
                ("total MD", total_stress),
            ):
                voigt = np.asarray(stress, dtype=float).reshape(6)
                stress_gpa = voigt * self.EV_A3_TO_GPA
                pressure = -float(np.sum(stress_gpa[:3])) / 3.0
                tensor = np.array(
                    [
                        [voigt[0], voigt[5], voigt[4]],
                        [voigt[5], voigt[1], voigt[3]],
                        [voigt[4], voigt[3], voigt[2]],
                    ]
                )
                lines.extend(
                    [
                        "",
                        f"  {label} stress tensor (ASE convention, eV/Angstrom^3):",
                        *[
                            "    " + "".join(f"{value:16.9f}" for value in row)
                            for row in tensor
                        ],
                        f"  {label} stress Voigt xx yy zz yz xz xy (GPa):",
                        "    " + "".join(f"{value:16.8f}" for value in stress_gpa),
                        f"  {label} pressure = {pressure:16.8f} GPa",
                    ]
                )
        else:
            lines.extend(
                [
                    "",
                    "  3D bulk stress/pressure: unavailable or disabled",
                ]
            )
        lines.append("")

        payload = "\n".join(lines) + "\n"
        if self._stream is not None:
            self._stream.write(payload)
        else:
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(payload)

    def finalize(
        self,
        *,
        status: str,
        md_time_s: float | None = None,
        final_energy: float | None = None,
        final_temperature: float | None = None,
    ) -> None:
        """Finish the stream with an explicit MLIP calculation summary."""
        if self.output_path is None or self._finished:
            return
        self.close_stream()
        lines = [
            "=" * 100,
            " MLIPORT MD SUMMARY",
            "=" * 100,
            f"Status:              {status}",
            f"Saved configurations:{self.configuration_index:12d}",
        ]
        if final_energy is not None:
            lines.append(f"Final potential E:  {final_energy:20.10f} eV")
        if final_temperature is not None:
            lines.append(f"Final temperature:  {final_temperature:20.8f} K")
        if md_time_s is not None:
            lines.append(f"MD wall time:       {md_time_s:20.6f} s")
        lines.extend(["", " END OF MLIPORT VASP-LIKE MD OUTPUT", "=" * 100, ""])
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        self._finished = True
