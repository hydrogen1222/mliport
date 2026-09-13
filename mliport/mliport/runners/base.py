# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified for the mliport project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
"""
Base runner class for MLIP calculations.

Provides common functionality for all calculation runners,
including output directory management and logging.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mliport.protocols import CancellationRequested, ProgressEvent
from mliport.timing import RunTiming, append_timing_to_outputs, timing_log_lines

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator

    from mliport.base_calculator import BaseMLIPCalculator
    from mliport.protocols import ProgressCallback


class BaseRunner(ABC):
    """Base class for all calculation runners.

    Provides common infrastructure for running calculations,
    managing output, and handling errors.

    Attributes:
        calculator: UMA calculator wrapper
        output_dir: Directory for output files
        verbose: Whether to print verbose output
        job_name: Optional job name for the calculation

    Example:
        >>> class MyRunner(BaseRunner):
        ...     def run(self, atoms):
        ...         # Implementation
        ...         pass
    """

    def __init__(
        self,
        calculator: BaseMLIPCalculator,
        output_dir: Path | str = ".",
        verbose: bool = True,
        job_name: str | None = None,
        log_fn: Any | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        charge: int | None = None,
        spin: int | None = None,
    ):
        """Initialize base runner.

        Args:
            calculator: UMA calculator wrapper
            output_dir: Directory for output files
            verbose: Whether to print progress messages
            job_name: Optional job name for organizing results
            log_fn: Optional callback function for custom log output
            progress_callback: Optional callback for progress events
            charge: Optional total molecular charge override
            spin: Optional molecular spin metadata override. UMA omol treats
                this as spin multiplicity (2S+1).
        """
        self.calculator = calculator
        self.job_name = job_name
        self.verbose = verbose
        self.log_fn = log_fn
        self.progress_callback = progress_callback
        self._cancel_event = (
            cancel_event  # threading.Event for cooperative cancellation
        )
        self._timing: RunTiming | None = None
        self._pending_done_event: ProgressEvent | None = None
        self.charge = charge
        self.spin = spin

        # Build output directory path
        base_dir = Path(output_dir)
        if job_name:
            self.output_dir = base_dir / job_name
        else:
            self.output_dir = base_dir

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        atoms: Atoms,
        started_at: float | None = None,
    ) -> dict[str, Any]:
        """Run with uniform timing, final logging, and output-file metadata."""
        self._timing = RunTiming(
            started_at=time.perf_counter() if started_at is None else started_at
        )
        self._pending_done_event = None

        try:
            results = self.run(atoms)
        except CancellationRequested:
            timing = self._timing.finish()
            self.log(
                f"Run cancelled after {timing['total_elapsed_time_s']:.2f} s",
                level="warning",
            )
            raise
        except Exception:
            timing = self._timing.finish()
            self.log(
                "Run ended before completion after "
                f"{timing['total_elapsed_time_s']:.2f} s",
                level="error",
            )
            raise

        timing = self._timing.finish()
        results["timing"] = timing

        try:
            append_timing_to_outputs(self.output_dir, timing)
        except Exception as exc:
            self.log(
                f"Could not add timing to output files: {exc}",
                level="warning",
            )

        for line in timing_log_lines(timing):
            self.log(line)

        done_event = self._pending_done_event or ProgressEvent(
            phase="done",
            message="Calculation complete",
        )
        done_extra = dict(done_event.extra or {})
        done_extra["timing"] = timing
        if self.progress_callback is not None:
            self.progress_callback(
                ProgressEvent(
                    phase=done_event.phase,
                    message=done_event.message,
                    step=done_event.step,
                    total_steps=done_event.total_steps,
                    extra=done_extra,
                )
            )

        return results

    def log(self, message: str, level: str = "info") -> None:
        """Print log message if verbose mode or log callback is enabled.

        Args:
            message: Message to print
            level: Log level (info, warning, error)
        """
        if not self.verbose and not self.log_fn:
            return

        prefix = {
            "info": "  ",
            "warning": "! ",
            "error": "ERROR: ",
        }.get(level, "  ")

        formatted_msg = f"{prefix}{message}"

        if self.log_fn:
            self.log_fn(formatted_msg, level)
        if self.verbose:
            print(formatted_msg)

    def log_buffered(self, message: str, level: str = "info") -> None:
        """Record a high-frequency message with coalesced file flushing.

        ``LiveRunLogger`` exposes ``write_buffered`` for this path. Generic log
        callbacks retain the historical callable contract and receive the
        message normally.
        """
        if not self.verbose and not self.log_fn:
            return

        prefix = {
            "info": "  ",
            "warning": "! ",
            "error": "ERROR: ",
        }.get(level, "  ")
        formatted_msg = f"{prefix}{message}"

        if self.log_fn:
            buffered_writer = getattr(self.log_fn, "write_buffered", None)
            if callable(buffered_writer):
                buffered_writer(formatted_msg, level)
            else:
                self.log_fn(formatted_msg, level)
        if self.verbose:
            print(formatted_msg)

    def print_header(self, title: str) -> None:
        """Print section header.

        Args:
            title: Section title
        """
        if self.verbose or self.log_fn:
            msg = f"\n{'-' * 80}\n {title}\n{'-' * 80}"
            if self.log_fn:
                self.log_fn(msg, "info")
            if self.verbose:
                print(msg)

    def _emit_progress(
        self,
        phase: str,
        message: str,
        step: int | None = None,
        total_steps: int | None = None,
        extra: dict | None = None,
    ) -> None:
        """Emit a progress event to the callback if registered.

        Args:
            phase: Current phase (loading_model, running, writing_output, done, error)
            message: Human-readable description
            step: Current step number
            total_steps: Total expected steps
            extra: Optional phase-specific data
        """
        if self._timing is not None:
            self._timing.observe_phase(phase)

        event = ProgressEvent(
            phase=phase,
            message=message,
            step=step,
            total_steps=total_steps,
            extra=extra,
        )
        if phase == "done" and self._timing is not None:
            self._pending_done_event = event
            return
        if self.progress_callback is not None:
            self.progress_callback(event)

    def _is_cancelled(self) -> bool:
        """Check whether cooperative cancellation has been requested.

        Returns True if a cancel_event was provided and has been set,
        indicating the caller wants the calculation to stop early.
        """
        return self._cancel_event is not None and self._cancel_event.is_set()

    @abstractmethod
    def run(self, atoms: Atoms) -> dict[str, Any]:
        """Run calculation on structure.

        Args:
            atoms: ASE Atoms object

        Returns:
            Dictionary with calculation results
        """

    def _prepare_atoms(self, atoms: Atoms) -> Atoms:
        """Prepare atoms for calculation.

        Handles task-specific preparation like setting charge/spin for molecules
        and ensuring PBC is correctly set for periodic systems.

        Args:
            atoms: Input ASE Atoms object

        Returns:
            Prepared Atoms object
        """
        # Copy first so the task-specific PBC / cell / info / center() changes
        # below never mutate the caller's Atoms object. Every runner shares
        # this helper (single-point / optimisation / MD), so guarding it here
        # protects all of them; MDRunner.run() additionally copies to shield
        # pre-relaxation and velocity initialisation.
        atoms = atoms.copy()
        task = self.calculator.task

        # Explicit CLI/TUI/INCAR values override metadata embedded in the
        # structure. If omitted, preserve the structure value and let the
        # task-specific defaults below fill only missing fields.
        if self.charge is not None:
            atoms.info["charge"] = self.charge
        if self.spin is not None:
            atoms.info["spin"] = self.spin

        # Periodic vs molecular task classification. UMA tasks (omat/oc20/...)
        # and the generic 'bulk' are periodic; 'omol'/'molecule' are not.
        PERIODIC_TASKS = {"omat", "oc20", "oc25", "odac", "omc", "bulk"}
        MOLECULAR_TASKS = {"omol", "molecule"}

        pbc = np.asarray(atoms.pbc, dtype=bool)

        # The task selects a physical boundary condition; it must agree with
        # the structure instead of rewriting it behind the user's back.
        if task in PERIODIC_TASKS:
            if not pbc.all():
                raise ValueError(
                    f"Task {task!r} requires full 3D periodicity, but the input "
                    f"PBC is {pbc.tolist()}. Set PBC explicitly in the structure "
                    "or choose a molecular task; mliport will not change it."
                )
            # Log cell info for debugging
            cell = atoms.cell
            if cell.rank == 3 and cell.volume > 0:
                self.log(
                    f"Cell: {cell.lengths()[0]:.4f} x {cell.lengths()[1]:.4f} x {cell.lengths()[2]:.4f} Å"
                )
            else:
                raise ValueError("Invalid cell: zero volume. Check input structure.")
        elif task in MOLECULAR_TASKS:
            if pbc.any():
                raise ValueError(
                    f"Task {task!r} requires a nonperiodic structure, but the "
                    f"input PBC is {pbc.tolist()}. Disable PBC explicitly in "
                    "the structure; mliport will not change it."
                )
            # A non-periodic molecule still needs a bounding-box cell for
            # output writers (CONTCAR/OUTCAR call atoms.get_volume()) and for
            # logging. ASE's .xyz reader leaves no cell (rank 0); add a vacuum
            # box. For pbc=False this never affects energies/forces -- it only
            # fixes output formatting (e.g. the isolated gas in an adsorption
            # energy calculation).
            if atoms.cell.rank < 3:
                atoms.center(vacuum=6.0)
                self.log("Added a 6 Å vacuum cell for the non-periodic molecule.")
            if task == "omol":
                # UMA omol reads atoms.info["charge"] (total charge) and
                # atoms.info["spin"] (spin *multiplicity*, 1 = singlet).
                # FAIRChemCalculator already defaults these to 0 / 1, but set
                # them explicitly so the run is deterministic.
                atoms.info.setdefault("charge", 0)
                atoms.info.setdefault("spin", 1)
            else:
                # Generic molecular engines (MACE/DPA/GRACE). MACE reads
                # atoms.info["spin"] as *total spin* (number of unpaired
                # electrons, 0 = singlet) -- NOT multiplicity -- so injecting a
                # blanket default spin=1 would wrongly force a doublet radical
                # on spin-enabled MACE models. Only default the total charge.
                atoms.info.setdefault("charge", 0)
        else:
            raise ValueError(f"Unknown task {task!r}; refusing to infer PBC semantics.")

        return atoms

    def _get_calculator(self) -> Calculator:
        """Get ASE calculator instance.

        Returns:
            ASE Calculator
        """
        return self.calculator.get_calculator()

    def _check_finite(
        self,
        atoms: Atoms,
        energy: float,
        forces: Any | None = None,
        *,
        context: str = "calculation",
    ) -> None:
        """Abort the run if energy/forces are non-finite (NaN/inf).

        MLIPs can return NaN when atoms get too close or leave the training
        distribution. Without this guard ASE integrators propagate NaN to the
        end and write "successful" NaN results. Non-finite values are always
        fatal, so this raises unconditionally (plan safety.abort_on_nan).
        """
        import numpy as np  # noqa: PLC0415

        if not np.isfinite(energy):
            raise RuntimeError(
                f"Non-finite energy ({energy!r}) during {context}; aborting "
                "before NaN is written to outputs."
            )
        if forces is not None:
            arr = np.asarray(forces, dtype=float)
            if arr.size and not np.all(np.isfinite(arr)):
                fmax = (
                    float(np.max(np.linalg.norm(arr, axis=1)))
                    if arr.ndim == 2
                    else float("nan")
                )
                raise RuntimeError(
                    f"Non-finite forces during {context} (max|F|={fmax}); "
                    "aborting before NaN is written to outputs."
                )

    def _write_summary(self, results: dict[str, Any], atoms: Atoms) -> None:
        """Write calculation summary to stdout.

        Args:
            results: Results dictionary
            atoms: ASE Atoms object
        """
        if not self.verbose:
            return

        energy = results.get("energy")
        forces = results.get("forces")

        print()
        print("=" * 80)
        if self.job_name:
            print(f" SUMMARY - {self.job_name}")
        else:
            print(" SUMMARY")
        print("=" * 80)

        if energy is not None:
            print(f"Total energy:     {energy:16.8f} eV")
            print(f"Energy per atom:  {energy / len(atoms):16.8f} eV/atom")

        if forces is not None:
            import numpy as np

            force_mags = np.linalg.norm(forces, axis=1)
            print(f"Max force:        {np.max(force_mags):16.8f} eV/Å")
            print(f"RMS force:        {np.sqrt(np.mean(force_mags**2)):16.8f} eV/Å")

        if "stress" in results and results["stress"] is not None:
            import numpy as np

            stress = results["stress"]
            pressure = -(stress[0] + stress[1] + stress[2]) / 3.0 * 160.2177
            print(f"Pressure:         {pressure:16.8f} GPa")

        calc_time = results.get("time")
        if calc_time is not None:
            print(f"Calculation time: {calc_time:16.2f} s")

        print("=" * 80)
