"""ASE-backed fixed-cell NEB/CI-NEB runner with one shared calculator."""

from __future__ import annotations

import threading

import numpy as np
from ase.mep import NEB
from ase.optimize import FIRE

from mliport.neb.prepare import BandInput, interpolate_lifted_band
from mliport.neb.results import NEBResult
from mliport.neb.schema import (
    NEBEndpointNotConvergedError,
    NEBOptions,
    NEBPreparationError,
)
from mliport.protocols import CancellationRequested, ProgressEvent


class NEBRunner:
    """Run a prepared band serially with one calculator instance.

    File output and model loading remain owned by the public workflow. The
    runner exposes only trusted complete-band checkpoint callbacks.
    """

    def __init__(
        self,
        calculator,
        options: NEBOptions | None = None,
        *,
        verbose: bool = True,
        progress_callback=None,
        cancel_event: threading.Event | None = None,
        checkpoint_callback=None,
    ):
        self.calculator = calculator
        self.options = options or NEBOptions()
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.cancel_event = cancel_event
        self.checkpoint_callback = checkpoint_callback
        self.last_band: tuple | None = None

    def _emit(self, message: str, *, step: int | None = None, stage: str = "running"):
        if self.verbose:
            print(message)
        if self.progress_callback is not None:
            self.progress_callback(
                ProgressEvent(
                    phase=stage,
                    message=message,
                    step=step,
                    total_steps=self.options.max_steps,
                )
            )

    def _is_cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def _get_calculator(self):
        getter = getattr(self.calculator, "get_calculator", None)
        return getter() if callable(getter) else self.calculator

    @staticmethod
    def _fmax(forces: np.ndarray) -> float:
        values = np.asarray(forces, dtype=float)
        if values.ndim != 3 or values.shape[-1] != 3:
            raise NEBPreparationError("force snapshot has invalid shape")
        if not np.all(np.isfinite(values)):
            raise NEBPreparationError("NEB force snapshot contains NaN or Inf")
        return float(np.linalg.norm(values, axis=2).max()) if values.size else 0.0

    def _physical_snapshot(self, images, *, context: str):
        """Copy and validate raw calculator energy/forces for every image."""
        energies = []
        physical_forces = []
        for image in images:
            energy = float(image.get_potential_energy())
            # ASE's NEB optimizer uses constraint-applied forces. The result
            # snapshot separately records the raw calculator gradient so a
            # frozen atom's reaction force is not silently replaced by zero.
            forces = np.asarray(
                image.get_forces(apply_constraint=False), dtype=float
            ).copy()
            if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
                raise NEBPreparationError(f"{context} energy/force contains NaN or Inf")
            energies.append(energy)
            physical_forces.append(forces)
        force_array = np.asarray(physical_forces, dtype=float)
        if self._fmax(force_array) > self.options.force_abort_eV_A:
            raise NEBPreparationError(
                f"Raw physical {context} force exceeds force_abort_eV_A; "
                "stopping before the band can be published"
            )
        return np.asarray(energies, dtype=float), force_array

    def _snapshot(self, images, *, neb_forces: np.ndarray | None = None):
        energies, force_array = self._physical_snapshot(images, context="NEB")
        applied_array = np.asarray(
            [
                np.asarray(image.get_forces(apply_constraint=True), dtype=float).copy()
                for image in images
            ],
            dtype=float,
        )
        self._fmax(applied_array)
        if neb_forces is None:
            neb_array = np.empty((0, len(images), 3), dtype=float)
            max_neb = float("nan")
        else:
            interior = np.asarray(neb_forces, dtype=float).reshape(
                len(images) - 2, len(images[0]), 3
            )
            neb_array = np.zeros((len(images), len(images[0]), 3), dtype=float)
            neb_array[1:-1] = interior
            max_neb = self._fmax(neb_array[1:-1])
        return (
            energies,
            force_array,
            applied_array,
            neb_array,
            max_neb,
        )

    def _validate_or_relax_endpoints(self, images: list) -> None:
        if self.options.endpoint_policy == "validate":
            for index, image in enumerate((images[0], images[-1])):
                energy = float(image.get_potential_energy())
                forces = np.asarray(image.get_forces(), dtype=float)
                raw_forces = np.asarray(
                    image.get_forces(apply_constraint=False), dtype=float
                )
                if not np.isfinite(energy):
                    raise NEBPreparationError(f"Endpoint {index} energy is not finite")
                fmax = self._fmax(forces[None, ...])
                if self._fmax(raw_forces[None, ...]) > self.options.force_abort_eV_A:
                    raise NEBPreparationError(
                        f"Endpoint {index} raw force exceeds force_abort_eV_A"
                    )
                if fmax > self.options.endpoint_fmax_eV_A:
                    raise NEBEndpointNotConvergedError(
                        f"Endpoint {index} fmax={fmax:g} eV/A exceeds "
                        f"endpoint_fmax_eV_A={self.options.endpoint_fmax_eV_A:g}"
                    )
            return
        for index, image in enumerate((images[0], images[-1])):
            optimizer = FIRE(image, maxstep=self.options.maxstep_A, logfile=None)
            for _ in optimizer.irun(
                fmax=self.options.endpoint_fmax_eV_A,
                steps=self.options.endpoint_steps,
            ):
                if self._is_cancelled():
                    raise CancellationRequested(
                        f"NEB endpoint {index} relaxation cancelled by user"
                    )
                self._physical_snapshot([image], context=f"endpoint {index} relaxation")
            if not optimizer.converged():
                raise NEBEndpointNotConvergedError(
                    f"Endpoint {index} did not converge within endpoint_steps"
                )

    def _optimize_stage(self, neb: NEB, *, stage: str, fmax: float, steps: int):
        # Gate raw calculator outputs before ASE can use them in tangent or
        # spring-force arithmetic. Subsequent yielded states are checked below
        # before the optimizer is allowed to take its next step.
        self._physical_snapshot(neb.images[1:-1], context=f"NEB {stage}")
        optimizer = FIRE(neb, maxstep=self.options.maxstep_A, logfile=None)
        for _ in optimizer.irun(fmax=fmax, steps=steps):
            if self._is_cancelled():
                raise CancellationRequested(f"NEB {stage} cancelled by user")
            # ASE optimizes constraint-applied forces. Check the raw physical
            # state at every accepted optimizer state so a frozen high force,
            # NaN, or Inf cannot be hidden until the final result snapshot.
            self._physical_snapshot(neb.images[1:-1], context=f"NEB {stage}")
            if self.checkpoint_callback is not None:
                self.checkpoint_callback(stage, int(optimizer.nsteps), neb.images)
        neb_forces = np.asarray(neb.get_forces(), dtype=float).copy()
        if not np.all(np.isfinite(neb_forces)):
            raise NEBPreparationError(f"NEB {stage} produced a non-finite band force")
        max_neb = self._fmax(
            neb_forces.reshape(len(neb.images) - 2, len(neb.images[0]), 3)
        )
        stage_result = {
            "stage": stage,
            "steps": int(optimizer.nsteps),
            "converged_by_full_band_force": bool(max_neb <= fmax),
            "max_neb_force_eV_A": max_neb,
            "climbing_image_index": int(neb.imax) if neb.climb else None,
        }
        return bool(max_neb <= fmax), stage_result, neb_forces

    def run(self, band: BandInput) -> NEBResult:
        """Run ordinary NEB, optionally followed by a fresh CI-NEB FIRE stage."""

        images = [image.copy() for image in band.images]
        calc = self._get_calculator()
        if calc is None:
            raise NEBPreparationError("A calculator is required for NEB")
        if self.options.endpoint_policy == "relax" and band.source == "provided_band":
            raise NEBPreparationError(
                "endpoint_policy='relax' cannot rewrite an explicit band; "
                "provide endpoints or use endpoint_policy='validate'"
            )
        for image in images:
            image.calc = calc
        resume_has_optimized_band = band.source == "checkpoint_optimized_band"
        if resume_has_optimized_band:
            # Geometry resume must preserve the saved interior path. Endpoints
            # in an optimized-band checkpoint have already passed the original
            # endpoint stage; only re-run the raw physical safety gate here.
            self._physical_snapshot(images, context="resumed NEB band")
        else:
            self._validate_or_relax_endpoints(images)
        if self.options.endpoint_policy == "relax" and not resume_has_optimized_band:
            images = interpolate_lifted_band(images[0], images[-1], self.options)
            for image in images:
                image.calc = calc

        neb = NEB(
            images,
            k=self.options.spring_eV_A2,
            climb=False,
            parallel=False,
            allow_shared_calculator=True,
            method=self.options.method,
            remove_rotation_and_translation=False,
        )
        stages: list[dict] = []
        converged, stage_result, neb_forces = self._optimize_stage(
            neb,
            stage="neb_pre" if self.options.climb else "neb",
            fmax=self.options.pre_fmax_eV_A
            if self.options.climb
            else self.options.fmax_eV_A,
            steps=self.options.pre_max_steps
            if self.options.climb
            else self.options.max_steps,
        )
        stages.append(stage_result)
        if self.options.climb and converged:
            neb.climb = True
            # A new FIRE instance is intentional: CI changes the force field.
            converged, stage_result, neb_forces = self._optimize_stage(
                neb,
                stage="ci_neb",
                fmax=self.options.fmax_eV_A,
                steps=self.options.max_steps,
            )
            stages.append(stage_result)
        (
            energies,
            physical_forces,
            constraint_applied_forces,
            band_forces,
            max_neb,
        ) = self._snapshot(images, neb_forces=neb_forces)
        final_band = []
        for image in images:
            snapshot = image.copy()
            snapshot.calc = None
            final_band.append(snapshot)
        self.last_band = tuple(final_band)
        climbing_index = int(neb.imax) if neb.climb else None
        climbing_fmax = (
            float(np.linalg.norm(physical_forces[climbing_index], axis=1).max())
            if climbing_index is not None
            else None
        )
        return NEBResult(
            status="completed" if converged else "not_converged",
            converged=converged,
            energies_eV=energies,
            physical_forces_eV_A=physical_forces,
            constraint_applied_forces_eV_A=constraint_applied_forces,
            neb_forces_eV_A=band_forces,
            stages=stages,
            climbing_image_index=climbing_index,
            max_neb_force_eV_A=max_neb,
            climbing_physical_fmax_eV_A=climbing_fmax,
        )
