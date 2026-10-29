"""Confsearch search engine: cycles, MD segments, quench, Metropolis (spec 7).

The engine is calculator-agnostic: it takes any ASE calculator with energy and
forces, which is what makes the synthetic-model end-to-end tests possible.  The
model identity, gradient pre-flight and CLI/API wiring live one layer above
(phase P3).

Flow (spec 7.1)::

    prepare starts -> stage "start_quench" (quench every start)
    -> for cycle c = 1..cs_cycles:
           for replica r = 0..n-1 (round robin, skip finished/failed):
               run cycle (r, c); register its minima; checkpoint
    -> consolidation -> refinement of top-K -> ranking, exports, diagnostics

One model instance is used for the whole search.  Every stochastic draw comes
from a per-replica ``SeedSequence`` stream (section 7.5), and every checkpoint
stores the bit-generator state before the next unit of work, so an interrupted
search resumes bitwise-identically.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import units
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from mliport.confsearch import status as cs_status
from mliport.confsearch.checkpoint import (
    CheckpointData,
    ConfsearchCheckpointStore,
)
from mliport.confsearch.errors import (
    ConfsearchInternalError,
    ConfsearchPreparationError,
)
from mliport.confsearch.geometry import wrap_positions
from mliport.confsearch.graphs import (
    ClassificationReference,
    broken_reference_bonds,
    classify_minimum,
    covalent_bonds,
)
from mliport.confsearch.options import ConfsearchOptions
from mliport.confsearch.quench import QuenchResult, quench
from mliport.confsearch.regions import (
    Regions,
    SurfaceGeometry,
    mobile_indices,
)
from mliport.confsearch.registry import (
    MinimumOrigin,
    MinimumRecord,
    MinimumRegistry,
)
from mliport.confsearch.results import build_results
from mliport.runners.md_output import (
    AsyncMDOutputWriter,
    MDFrameSnapshot,
)
from mliport.runners.safety import (
    evaluate_force_abort,
    force_magnitudes,
    force_views,
    validate_safe_point,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ase import Atoms

#: Sampling distance for the "unique minima within 0.2 eV" discovery column.
DISCOVERY_WINDOW_EV = 0.2


def _format_duration(seconds: float) -> str:
    """``h:mm:ss`` for log lines."""
    total = max(0, round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


@dataclass(frozen=True, slots=True)
class PreparedStart:
    """One validated start structure plus its regions and surface facts."""

    index: int
    atoms: Atoms
    regions: Regions
    surface: SurfaceGeometry
    label: str
    placement: Any = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "n_atoms": len(self.atoms),
            "regions": {
                "adsorbate": self.regions.adsorbate.tolist(),
                "slab": self.regions.slab.tolist(),
                "fixed": self.regions.fixed.tolist(),
                "relax": self.regions.relax.tolist(),
                "fix_below_added": list(self.regions.fix_below_added),
                "warnings": list(self.regions.warnings),
            },
            "surface": {
                "z_top_A": self.surface.z_top,
                "z_min_A": self.surface.z_min,
                "vacuum_gap_A": self.surface.vacuum_gap,
                "adsorbate_centroid_z_A": self.surface.adsorbate_centroid_z,
            },
            "placement": (
                self.placement.to_dict() if self.placement is not None else None
            ),
        }


@dataclass
class ReplicaState:
    """One independent search chain."""

    index: int
    start: PreparedStart
    md_rng: np.random.Generator
    mc_rng: np.random.Generator
    status: str = "active"
    current: Atoms | None = None
    current_energy: float | None = None
    reference: ClassificationReference | None = None
    start_minimum_energy: float | None = None
    cycle_count: int = 0
    cycles_completed: list[int] = field(default_factory=list)

    def to_checkpoint(self) -> dict:
        return {
            "index": self.index,
            "status": self.status,
            "cycle_count": self.cycle_count,
            "cycles_completed": list(self.cycles_completed),
            "current_energy_eV": self.current_energy,
            "start_minimum_energy_eV": self.start_minimum_energy,
            "current": (
                None
                if self.current is None
                else {
                    "numbers": [int(z) for z in self.current.numbers],
                    "cell_A": np.asarray(self.current.cell, dtype=float).tolist(),
                    "pbc": [bool(value) for value in self.current.pbc],
                    "positions_A": np.asarray(
                        self.current.get_positions(), dtype=float
                    ).tolist(),
                }
            ),
            "reference_start_positions_A": (
                None
                if self.reference is None
                else np.asarray(self.reference.start_positions, dtype=float).tolist()
            ),
        }


class _MDAbort(RuntimeError):
    """Internal signal: the MD segment stopped with a recorded status."""

    def __init__(self, status: str, message: str, frame: Atoms | None = None):
        super().__init__(message)
        self.status = status
        self.frame = frame


class SearchEngine:
    """Run one confsearch search (or resume one) with a single model instance."""

    def __init__(
        self,
        *,
        calculator,
        options: ConfsearchOptions,
        starts: Sequence[PreparedStart],
        output_dir: str | Path,
        seed: int,
        model_record: dict | None = None,
        gradient_preflight: dict | None = None,
        resume: CheckpointData | None = None,
        resume_fingerprint: dict | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        if not starts:
            raise ConfsearchPreparationError("confsearch needs at least one start")
        self.calculator = calculator
        self.options = options
        self.starts = list(starts)
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        self.model_record = dict(model_record or {})
        self.gradient_preflight = dict(gradient_preflight or {})
        self.resume = resume
        self.resume_fingerprint = dict(resume_fingerprint or {})
        self._log_lines: list[str] = []
        self._log_callback = log
        self.warnings: list[str] = []
        self.started = time.time()

        self.store = ConfsearchCheckpointStore(self.output_dir)
        self.registry: MinimumRegistry
        self.replicas: list[ReplicaState]
        self.discovery: list[dict] = []
        self.baseline: dict = {}
        self.cumulative_md_ps = 0.0
        self.quench_index = 0
        self.completed_units = 0
        self.stage = "prepared"
        self._reference_graphs: tuple[Any, Any] | None = None
        self._initialize_state()

    # -- logging ----------------------------------------------------------

    def log(self, message: str) -> None:
        self._log_lines.append(message)
        if self._log_callback is not None:
            self._log_callback(message)

    def log_lines(self) -> list[str]:
        return list(self._log_lines)

    # -- state ------------------------------------------------------------

    def _initialize_state(self) -> None:
        adsorbate = np.asarray(self.starts[0].regions.adsorbate, dtype=int)
        fixed = np.asarray(self.starts[0].regions.fixed, dtype=int)
        self.registry = MinimumRegistry(
            options=self.options, adsorbate=adsorbate, fixed=fixed
        )
        seed_sequence = np.random.SeedSequence(self.seed)
        placement_ss, *replica_sequences = seed_sequence.spawn(1 + len(self.starts))
        self.placement_seed_sequence = placement_ss
        self.replicas = []
        for index, start in enumerate(self.starts):
            md_ss, mc_ss = replica_sequences[index].spawn(2)
            self.replicas.append(
                ReplicaState(
                    index=index,
                    start=start,
                    md_rng=np.random.default_rng(md_ss),
                    mc_rng=np.random.default_rng(mc_ss),
                )
            )
        if fixed.size == 0:
            self.warnings.append(
                "F is empty: the geometric identity rule G is disabled for this "
                "search, so dedup relies on the fingerprint alone"
            )

    def _reference_identity(self) -> tuple[Any, Any]:
        """Graphs shared by every start (section 8.1); fail closed on mismatch."""
        if self._reference_graphs is not None:
            return self._reference_graphs
        reference: tuple[Any, Any] | None = None
        for start in self.starts:
            regions = start.regions
            graphs = (
                covalent_bonds(
                    start.atoms,
                    regions.adsorbate,
                    bond_scale=self.options.cs_bond_scale,
                    ionic_symbols=self.options.ionic_symbols,
                ),
                covalent_bonds(
                    start.atoms,
                    regions.slab,
                    bond_scale=self.options.cs_bond_scale,
                    ionic_symbols=self.options.ionic_symbols,
                ),
            )
            if reference is None:
                reference = graphs
                continue
            if (
                graphs[0].bonds != reference[0].bonds
                or graphs[1].bonds != reference[1].bonds
            ):
                raise ConfsearchPreparationError(
                    f"Start {start.index} has a different covalent graph than "
                    "start 0; all starts must describe the same molecule on the "
                    "same slab (section 8.1)"
                )
        assert reference is not None
        self._reference_graphs = reference
        return reference

    # -- checkpoint -------------------------------------------------------

    def _rng_states(self) -> dict:
        return {
            "replicas": [
                {
                    "index": replica.index,
                    "md": replica.md_rng.bit_generator.state,
                    "mc": replica.mc_rng.bit_generator.state,
                }
                for replica in self.replicas
            ],
        }

    def _write_checkpoint(self) -> None:
        data = CheckpointData(
            unit_index=self.completed_units,
            stage=self.stage,
            replicas=[replica.to_checkpoint() for replica in self.replicas],
            rng_states=self._rng_states(),
            registry=self.registry.to_dict(),
            discovery=list(self.discovery),
            baseline=dict(self.baseline),
            cumulative_md_ps=self.cumulative_md_ps,
            quench_index=self.quench_index,
        )
        self.store.write(data, self.resume_fingerprint)

    def _restore(self) -> None:
        assert self.resume is not None
        self.completed_units = int(self.resume.unit_index)
        self.stage = str(self.resume.stage)
        self.discovery = list(self.resume.discovery)
        self.baseline = dict(self.resume.baseline)
        self.cumulative_md_ps = float(self.resume.cumulative_md_ps)
        self.quench_index = int(self.resume.quench_index)
        adsorbate = np.asarray(self.starts[0].regions.adsorbate, dtype=int)
        fixed = np.asarray(self.starts[0].regions.fixed, dtype=int)
        self.registry = MinimumRegistry.from_dict(
            self.resume.registry,
            options=self.options,
            adsorbate=adsorbate,
            fixed=fixed,
        )
        states = {
            int(entry["index"]): entry
            for entry in (self.resume.rng_states or {}).get("replicas", [])
        }
        for entry in self.resume.replicas:
            index = int(entry["index"])
            replica = self.replicas[index]
            replica.status = str(entry.get("status", "active"))
            replica.cycle_count = int(entry.get("cycle_count", 0))
            replica.cycles_completed = [
                int(value) for value in entry.get("cycles_completed", [])
            ]
            replica.current_energy = entry.get("current_energy_eV")
            replica.start_minimum_energy = entry.get("start_minimum_energy_eV")
            geometry = entry.get("current")
            if geometry is not None:
                from ase import Atoms  # noqa: PLC0415

                replica.current = Atoms(
                    numbers=np.asarray(geometry["numbers"], dtype=int),
                    positions=np.asarray(geometry["positions_A"], dtype=float),
                    cell=np.asarray(geometry["cell_A"], dtype=float),
                    pbc=np.asarray(geometry["pbc"], dtype=bool),
                )
            reference_positions = entry.get("reference_start_positions_A")
            if reference_positions is not None:
                adsorbate_graph, slab_graph = self._reference_identity()
                replica.reference = ClassificationReference(
                    adsorbate_graph=adsorbate_graph,
                    slab_graph=slab_graph,
                    start_positions=np.asarray(reference_positions, dtype=float),
                )
            if index in states:
                replica.md_rng.bit_generator.state = states[index]["md"]
                replica.mc_rng.bit_generator.state = states[index]["mc"]
        # Raw files of the interrupted attempt must survive (AGENTS.md rule 11).
        for replica in self.replicas:
            if replica.status != "active":
                continue
            cycle = replica.cycle_count + 1
            archived = self.store.archive_incomplete(replica=replica.index, cycle=cycle)
            if archived is not None:
                self.log(f"Archived interrupted attempt to {archived.name}")

    # -- plan -------------------------------------------------------------

    def _plan(self) -> list[tuple[str, int, int]]:
        units: list[tuple[str, int, int]] = [("start", 0, 0)]
        for cycle in range(1, int(self.options.cs_cycles) + 1):
            for replica in self.replicas:
                units.append(("cycle", replica.index, cycle))
        return units

    # -- public entry point -----------------------------------------------

    def run(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.resume is not None:
            self._restore()
        self._reference_identity()
        self._log_budget()
        plan = self._plan()
        n_cycle_units = sum(1 for kind, _, _ in plan if kind == "cycle")
        session_start = time.time()
        session_cycle_units = 0
        for index, (kind, replica_index, cycle) in enumerate(plan):
            if index < self.completed_units:
                continue
            if kind == "start":
                self._run_start_stage()
                self.stage = "start_quench_done"
            else:
                self._run_cycle_unit(replica_index, cycle)
                self.stage = "cycles"
                session_cycle_units += 1
            self.completed_units = index + 1
            self._write_checkpoint()
            if kind == "cycle":
                self._log_progress(
                    done=sum(1 for k, _, _ in plan[: index + 1] if k == "cycle"),
                    total=n_cycle_units,
                    session_units=session_cycle_units,
                    session_seconds=time.time() - session_start,
                )
        return self._finish()

    def _log_progress(
        self, *, done: int, total: int, session_units: int, session_seconds: float
    ) -> None:
        """One progress line per cycle unit: count, best so far, elapsed, ETA.

        The ETA extrapolates the mean wall time of the cycle units run in this
        process (a resumed run does not count the units of the earlier one).
        """
        valid = [
            unique.energy_eV
            for unique in self.registry.uniques(cs_status.VALID)
            if math.isfinite(unique.energy_eV)
        ]
        best = f"{min(valid):.6f} eV" if valid else "none"
        eta = (
            _format_duration(session_seconds / session_units * (total - done))
            if session_units
            else "unknown"
        )
        self.log(
            f"progress: {done}/{total} cycles done, {len(valid)} unique valid "
            f"minima, best {best}, elapsed {_format_duration(session_seconds)}, "
            f"ETA {eta}"
        )

    # -- budget / diagnostics ---------------------------------------------

    def _log_budget(self) -> None:
        options = self.options
        n_replicas = len(self.replicas)
        md_steps = n_replicas * options.total_md_steps_per_cycle * options.cs_cycles
        self.log(
            "confsearch budget: "
            f"{n_replicas} replicas x {options.cs_cycles} cycles, "
            f"{md_steps} MD steps, <= {n_replicas * options.total_quench_upper_bound} "
            "quenches"
        )

    def _discovery_row(self, record: MinimumRecord) -> dict:
        valid = [
            unique
            for unique in self.registry.uniques(cs_status.VALID)
            if math.isfinite(unique.energy_eV)
        ]
        energies = [unique.energy_eV for unique in valid]
        best = min(energies) if energies else None
        within = (
            sum(1 for energy in energies if best is not None and energy - best <= 0.2)
            if best is not None
            else 0
        )
        energy = float(record.energy_eV)
        return {
            "cumulative_md_ps": float(self.cumulative_md_ps),
            "quench_index": int(record.quench_index),
            "unique_valid": len(valid),
            "unique_valid_within_0p2eV": int(within),
            "current_best_eV": best,
            "energy_eV": energy if math.isfinite(energy) else None,
            "status": record.status,
        }

    # -- start stage ------------------------------------------------------

    def _run_start_stage(self) -> None:
        self.log("stage: quench every start")
        adsorbate_graph, slab_graph = self._reference_identity()
        for replica in self.replicas:
            start = replica.start
            result = quench(
                start.atoms,
                self.calculator,
                self.options,
                fixed_indices=start.regions.fixed,
            )
            reference = ClassificationReference(
                adsorbate_graph=adsorbate_graph,
                slab_graph=slab_graph,
                start_positions=np.asarray(result.atoms.get_positions(), dtype=float),
            )
            record = self._register(
                result,
                regions=start.regions,
                reference=reference,
                origin=MinimumOrigin(replica=replica.index, cycle=0, source="start"),
                md_ps=0.0,
            )
            replica.reference = reference
            replica.start_minimum_energy = record.energy_eV
            if record.status != cs_status.VALID:
                replica.status = cs_status.FAILED_START
                replica.current = None
                self.log(
                    f"replica {replica.index}: start quench is {record.status}; "
                    "marked failed_start (no cycles)"
                )
                continue
            replica.current = record.atoms.copy()
            replica.current_energy = record.energy_eV
            self.log(
                f"replica {replica.index}: start minimum {record.energy_eV:.6f} eV"
            )
        valid = [unique.energy_eV for unique in self.registry.uniques(cs_status.VALID)]
        self.baseline = {
            "stage": "start_quench",
            "n_replicas": len(self.replicas),
            "n_surviving": sum(
                1 for replica in self.replicas if replica.status == "active"
            ),
            "best_energy_eV": min(valid) if valid else None,
        }
        if not any(replica.status == "active" for replica in self.replicas):
            raise ConfsearchPreparationError(
                "No replica survived the start-quench stage; the search cannot "
                "continue (section 7.1)"
            )

    # -- cycles -----------------------------------------------------------

    def _run_cycle_unit(self, replica_index: int, cycle: int) -> None:
        replica = self.replicas[replica_index]
        if replica.status != "active" or replica.current is None:
            return
        self._run_cycle(replica, cycle)
        replica.cycle_count = max(replica.cycle_count, cycle)
        replica.cycles_completed.append(cycle)

    def _run_cycle(self, replica: ReplicaState, cycle: int) -> None:
        options = self.options
        regions = replica.start.regions
        x0 = replica.current
        assert x0 is not None
        z_top = float(np.max(wrap_positions(x0)[regions.slab, 2]))
        mobile = mobile_indices(x0, regions, options, z_top=z_top)
        fixed_md = np.setdiff1d(np.arange(len(x0)), mobile)

        md_atoms = x0.copy()
        md_atoms.set_constraint(FixAtoms(indices=fixed_md) if fixed_md.size else None)
        md_atoms.calc = self.calculator
        self._initialize_velocities(md_atoms, mobile, replica.md_rng)

        raw_dir = (
            self.output_dir
            / "raw"
            / f"replica_{replica.index:03d}"
            / f"cycle_{cycle:03d}"
        )
        raw_dir.mkdir(parents=True, exist_ok=True)
        writer = AsyncMDOutputWriter(
            trajectory_path=raw_dir / "trajectory.traj",
            csv_path=raw_dir / "md.csv",
            write_trajectory=True,
            xdatcar_writer=None,
            xdatcar_path=raw_dir / "XDATCAR",
            outcar_writer=None,
        )

        dynamics = Langevin(
            md_atoms,
            timestep=float(options.timestep) * units.fs,
            temperature_K=float(options.cs_temperature_high),
            friction=float(options.friction) / units.fs,
            rng=replica.md_rng,
            fixcm=False,
        )
        snapshots: list[Atoms] = []
        snapshot_steps = set(options.snapshot_steps)
        abort_status: str | None = None
        abort_message = ""
        schedule: list[dict] = []

        def observe() -> None:
            nonlocal abort_status, abort_message
            step = int(dynamics.nsteps)
            phase = "hold" if step <= options.n_hold else "cool"
            try:
                energy = float(md_atoms.get_potential_energy())
                raw, applied = force_views(
                    md_atoms, energy, context=f"confsearch MD step {step}"
                )
                kinetic, temperature, total = validate_safe_point(
                    md_atoms,
                    energy=energy,
                    forces_raw=raw,
                    forces_applied=applied,
                    temperature_fn=lambda atoms: float(atoms.get_temperature()),
                    context=f"confsearch MD step {step}",
                )
            except (RuntimeError, ValueError) as exc:
                abort_status = cs_status.MD_NONFINITE_ABORT
                abort_message = str(exc)
                raise _MDAbort(abort_status, abort_message, md_atoms.copy()) from exc

            gate = evaluate_force_abort(raw, threshold=options.fmax_abort)
            frame = self._frame_summary(
                step=step,
                phase=phase,
                energy=energy,
                kinetic=kinetic,
                total=total,
                temperature=temperature,
                volume=float(md_atoms.get_volume()),
                raw=raw,
                applied=applied,
                gate_max=gate.max_force_raw,
            )
            keep = (
                step % int(options.cs_save_interval) == 0
                or step in snapshot_steps
                or step == options.total_md_steps_per_cycle
                or gate.abort
            )
            if keep:
                # The raw trajectory keeps every requested frame, including the
                # one that triggers an abort.
                writer.submit(
                    MDFrameSnapshot(
                        atoms=md_atoms.copy(), summary=frame, outcar_forces=None
                    )
                )
            if step in snapshot_steps:
                # Check the integrity *before* keeping the frame: a frame that
                # already broke the adsorbate must not be quenched (spec 7.3
                # step 9 quenches only snapshots taken before the stop).
                self._check_md_integrity(replica, md_atoms, step)
            if gate.abort:
                abort_status = cs_status.MD_FORCE_ABORT
                abort_message = (
                    f"raw force |F|={gate.max_force_raw:.6g} eV/A on atom "
                    f"{gate.atom_index} exceeds fmax_abort={options.fmax_abort}"
                )
                raise _MDAbort(abort_status, abort_message, md_atoms.copy())
            if step in snapshot_steps:
                snapshots.append(md_atoms.copy())

        dynamics.attach(observe, interval=1)
        total_steps = int(options.total_md_steps_per_cycle)
        heartbeat = max(1, total_steps // 4)
        md_started = time.time()

        def heartbeat_log() -> None:
            step = int(dynamics.nsteps)
            if step == 0 or step % heartbeat != 0 or step >= total_steps:
                return
            rate = step / max(time.time() - md_started, 1e-9)
            self.log(
                f"replica {replica.index} cycle {cycle}: MD step {step}/"
                f"{total_steps} ({'hold' if step <= options.n_hold else 'cool'}, "
                f"T={float(md_atoms.get_temperature()):.0f} K, "
                f"{rate:.1f} steps/s)"
            )

        dynamics.attach(heartbeat_log, interval=1)
        self.log(
            f"replica {replica.index} cycle {cycle}: MD {total_steps} steps "
            f"({options.cs_temperature_high:g} K hold, cool to "
            f"{options.cs_temperature_low:g} K), {len(mobile)} mobile atoms"
        )
        steps_done = 0
        try:
            dynamics.run(int(options.n_hold))
            steps_done += int(options.n_hold)
            remaining = int(options.n_cool)
            while remaining > 0:
                block = min(int(options.cs_ramp_interval), remaining)
                target = self._cool_temperature(steps_done, options)
                dynamics.set_temperature(temperature_K=target)
                schedule.append(
                    {
                        "step": int(dynamics.nsteps),
                        "target_temperature_K": float(target),
                    }
                )
                dynamics.run(block)
                steps_done += block
                remaining -= block
            # Spec 7.3 step 7: the end-of-cycle frame gets the same early
            # integrity check as the hold-phase snapshots.
            self._check_md_integrity(replica, md_atoms, steps_done)
        except _MDAbort as exc:
            abort_status = exc.status
            abort_message = str(exc)
            # The abort frame lives in the raw trajectory only; it is never
            # quenched (spec 7.3 step 9).
        finally:
            try:
                writer.close()
            except Exception as exc:  # pragma: no cover - writer failures are fatal
                raise ConfsearchInternalError(
                    f"Raw MD output failed for replica {replica.index} cycle {cycle}: {exc}"
                ) from exc

        self.cumulative_md_ps += steps_done * float(options.timestep) / 1000.0

        # Quench the snapshots (chronological), then the end-of-cycle frame.
        end_frame = md_atoms.copy()
        quench_targets: list[tuple[str, int | None, Atoms]] = [
            ("snapshot", index, snapshot) for index, snapshot in enumerate(snapshots)
        ]
        if abort_status is None:
            quench_targets.append(("end", None, end_frame))

        end_record: MinimumRecord | None = None
        uniques_before = len(self.registry.uniques(cs_status.VALID))
        quench_statuses: dict[str, int] = {}
        for source, snapshot_index, frame in quench_targets:
            result = quench(
                frame,
                self.calculator,
                options,
                fixed_indices=regions.fixed,
            )
            record = self._register(
                result,
                regions=regions,
                reference=replica.reference,
                origin=MinimumOrigin(
                    replica=replica.index,
                    cycle=cycle,
                    source=source,
                    snapshot_index=snapshot_index,
                ),
                md_ps=self.cumulative_md_ps,
            )
            quench_statuses[record.status] = quench_statuses.get(record.status, 0) + 1
            if source == "end":
                end_record = record

        cycle_payload = {
            "replica": replica.index,
            "cycle": cycle,
            "status": abort_status or "completed",
            "message": abort_message,
            "steps_run": int(steps_done),
            "n_hold": int(options.n_hold),
            "n_cool": int(options.n_cool),
            "snapshot_steps": list(options.snapshot_steps),
            "temperature_schedule": schedule,
            "md_time_ps": steps_done * float(options.timestep) / 1000.0,
            "cumulative_md_ps": self.cumulative_md_ps,
            "mobile_atoms": mobile.tolist(),
        }
        (raw_dir / "cycle.json").write_text(
            json.dumps(cycle_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        new_uniques = len(self.registry.uniques(cs_status.VALID)) - uniques_before
        quench_summary = (
            ", ".join(
                f"{count} {name}" for name, count in sorted(quench_statuses.items())
            )
            or "none"
        )
        if abort_status is not None:
            self.log(
                f"replica {replica.index} cycle {cycle}: {abort_status} "
                f"({abort_message}); quenched {len(quench_targets)} earlier "
                f"snapshot(s) [{quench_summary}], {new_uniques} new unique; "
                "current minimum unchanged"
            )
            return

        previous = replica.current_energy
        accepted = self._metropolis(replica, end_record)
        end_text = (
            f"end minimum {end_record.energy_eV:.6f} eV ({end_record.status})"
            if end_record is not None and math.isfinite(end_record.energy_eV)
            else "no usable end minimum"
        )
        if accepted is None:
            decision = "not eligible, current minimum unchanged"
        elif accepted:
            decision = (
                f"accepted (dE = {end_record.energy_eV - previous:+.4f} eV)"
                if previous is not None
                else "accepted"
            )
        else:
            decision = (
                f"rejected (dE = {end_record.energy_eV - previous:+.4f} eV)"
                if previous is not None
                else "rejected"
            )
        self.log(
            f"replica {replica.index} cycle {cycle}: quenched "
            f"{len(quench_targets)} [{quench_summary}], {new_uniques} new unique; "
            f"{end_text}, {decision}"
        )

    def _cool_temperature(self, steps_done: int, options: ConfsearchOptions) -> float:
        """Linear T(t) from cs_temperature_high to cs_temperature_low."""
        n_cool = max(1, options.n_cool)
        fraction = min(1.0, max(0.0, (steps_done - options.n_hold) / n_cool))
        return float(
            options.cs_temperature_high
            + (options.cs_temperature_low - options.cs_temperature_high) * fraction
        )

    def _check_md_integrity(
        self, replica: ReplicaState, atoms: Atoms, step: int
    ) -> None:
        reference = replica.reference
        if reference is None:
            return
        broken = broken_reference_bonds(
            atoms,
            reference.adsorbate_graph,
            break_scale=self.options.cs_bond_break_scale,
            ionic_symbols=self.options.ionic_symbols,
        )
        if broken:
            raise _MDAbort(
                cs_status.MD_INTEGRITY_ABORT,
                f"adsorbate bond(s) {list(broken)} broke beyond "
                f"{self.options.cs_bond_break_scale}x the covalent-radii sum at "
                f"step {step}",
                atoms.copy(),
            )

    def _initialize_velocities(
        self, atoms: Atoms, mobile: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Maxwell-Boltzmann over the mobile DOF; fixed atoms keep zero momenta."""
        MaxwellBoltzmannDistribution(
            atoms,
            temperature_K=float(self.options.cs_temperature_high),
            force_temp=True,
            rng=rng,
        )
        fixed = np.setdiff1d(np.arange(len(atoms)), mobile)
        momenta = np.asarray(atoms.get_momenta(), dtype=float)
        if fixed.size:
            momenta[fixed] = 0.0
            atoms.set_momenta(momenta)
        ndof = int(atoms.get_number_of_degrees_of_freedom())
        kinetic = float(atoms.get_kinetic_energy())
        target = 0.5 * ndof * units.kB * float(self.options.cs_temperature_high)
        if kinetic > 0.0:
            atoms.set_momenta(momenta * math.sqrt(target / kinetic))

    def _frame_summary(
        self,
        *,
        step: int,
        phase: str,
        energy: float,
        kinetic: float,
        total: float,
        temperature: float,
        volume: float,
        raw: np.ndarray,
        applied: np.ndarray,
        gate_max: float,
    ) -> dict:
        return {
            "step": int(step),
            "time_fs": float(step * float(self.options.timestep)),
            "phase": phase,
            "energy": float(energy),
            "kinetic_energy": float(kinetic),
            "total_energy": float(total),
            "temperature": float(temperature),
            "volume": float(volume),
            "max_force_raw_eV_A": float(gate_max),
            "max_force_applied_eV_A": float(np.max(force_magnitudes(applied))),
            "configurational_stress": None,
            "total_stress": None,
            "configurational_pressure_gpa": None,
            "total_pressure_gpa": None,
        }

    # -- registration -----------------------------------------------------

    def _register(
        self,
        result: QuenchResult,
        *,
        regions: Regions,
        reference: ClassificationReference | None,
        origin: MinimumOrigin,
        md_ps: float,
    ) -> MinimumRecord:
        self.quench_index += 1
        if not result.usable:
            classification = None
        else:
            assert reference is not None
            classification = classify_minimum(
                result.atoms,
                regions=regions,
                options=self.options,
                reference=reference,
            )
        if classification is None:
            from mliport.confsearch.graphs import MinimumClassification  # noqa: PLC0415

            classification = MinimumClassification(
                status=result.status,
                flags=(),
                adsorbate_slab_bonds=(),
                reacted_bonds={},
                min_adsorbate_slab_distance=None,
                min_self_image_distance=None,
                min_vacuum_gap=None,
                max_slab_displacement_ionic=None,
                max_slab_displacement_nonionic=None,
                ionic_coordination=(),
            )
        record = MinimumRecord(
            record_id=self.registry.new_record_id(),
            energy_eV=float(result.energy_eV),
            atoms=result.atoms,
            classification=classification,
            origin=origin,
            quench_index=self.quench_index,
            cumulative_md_ps=float(md_ps),
            fmax_eV_A=float(result.fmax_eV_A),
            steps=int(result.steps),
        )
        self.registry.register(record)
        self.discovery.append(self._discovery_row(record))
        return record

    def _metropolis(
        self, replica: ReplicaState, end_record: MinimumRecord | None
    ) -> bool | None:
        """Metropolis update on the end-of-cycle minimum (spec 7.3 step 11).

        Returns whether the proposal was accepted, or ``None`` when there was no
        eligible proposal (no RNG draw happens then).
        """
        if end_record is None or replica.current_energy is None:
            return None
        eligible = end_record.status == cs_status.VALID or (
            end_record.status == cs_status.REACTED
            and self.options.cs_reaction_policy == "include"
        )
        if not eligible:
            return None
        delta = float(end_record.energy_eV) - float(replica.current_energy)
        # The uniform draw happens every time an eligible proposal exists, so
        # the stream stays aligned (section 7.3 step 11).
        draw = float(replica.mc_rng.random())
        accept = delta <= 0.0
        if not accept:
            exponent = -delta / (units.kB * float(self.options.cs_accept_temperature))
            accept = draw < math.exp(exponent)
        if accept:
            replica.current = end_record.atoms.copy()
            replica.current_energy = float(end_record.energy_eV)
        return accept

    # -- completion -------------------------------------------------------

    def _write_run_context(self) -> None:
        """Write run_context.json when the caller (workflow) has not already.

        Spec 11.1 lists it in every run directory; a direct engine call must be
        self-contained enough for `mliport confsearch export RUN` to work.
        """
        import dataclasses  # noqa: PLC0415

        from mliport.confsearch.revisions import CS_SCIENTIFIC_REVISION  # noqa: PLC0415
        from mliport.neb.io import atomic_write_json  # noqa: PLC0415

        path = self.output_dir / "run_context.json"
        if path.is_file():
            return
        regions = self.starts[0].regions
        system_atoms = self.starts[0].atoms
        payload = {
            "schema": "mliport.confsearch-run-context/1",
            # Spec 11.4 revision 5: the merge comparability key needs the atomic
            # numbers in order, the cell and the PBC (index sets alone are not
            # enough).  The workflow records the electronic state as well; a
            # direct engine call leaves it null.
            "system": {
                "numbers": [int(z) for z in system_atoms.numbers],
                "cell_A": np.asarray(system_atoms.cell, dtype=float).tolist(),
                "pbc": [bool(value) for value in system_atoms.pbc],
                "fixed_positions_A": [
                    [float(value) for value in position]
                    for position in np.asarray(
                        system_atoms.get_positions(), dtype=float
                    )[regions.fixed]
                ],
            },
            "electronic_state": {"charge": None, "spin": None},
            "scientific_revision": CS_SCIENTIFIC_REVISION,
            "seed": int(self.seed),
            "resume": self.resume is not None,
            "model": self.model_record,
            "gradient_preflight": self.gradient_preflight,
            "regions": {
                "adsorbate": regions.adsorbate.tolist(),
                "slab": regions.slab.tolist(),
                "fixed": regions.fixed.tolist(),
                "relax": regions.relax.tolist(),
            },
            "budget": {
                "replicas": len(self.replicas),
                "cycles": int(self.options.cs_cycles),
                "md_steps_per_cycle": int(self.options.total_md_steps_per_cycle),
                "md_steps_total": int(
                    len(self.replicas)
                    * self.options.total_md_steps_per_cycle
                    * self.options.cs_cycles
                ),
                "quench_upper_bound": int(
                    len(self.replicas) * self.options.total_quench_upper_bound
                ),
            },
            "options": {
                item.name: _jsonable(getattr(self.options, item.name))
                for item in dataclasses.fields(self.options)
                if not item.name.startswith("_")
            },
            "starts": [start.to_dict() for start in self.starts],
            "warnings": list(self.warnings),
        }
        atomic_write_json(path, payload)

    def _finish(self) -> dict:
        self.registry.consolidate()
        self._write_run_context()
        references = [replica.reference for replica in self.replicas]
        return build_results(
            regions=self.starts[0].regions,
            references=[item for item in references if item is not None],
            calculator=self.calculator,
            registry=self.registry,
            options=self.options,
            replicas=self.replicas,
            discovery=self.discovery,
            baseline=self.baseline,
            warnings=self.warnings,
            output_dir=self.output_dir,
            model_record=self.model_record,
            gradient_preflight=self.gradient_preflight,
            cumulative_md_ps=self.cumulative_md_ps,
            seed=self.seed,
            starts=[start.to_dict() for start in self.starts],
        )


def run_confsearch(
    calculator,
    options: ConfsearchOptions,
    *,
    starts: Sequence[PreparedStart],
    output_dir: str | Path,
    seed: int,
    model_record: dict | None = None,
    gradient_preflight: dict | None = None,
    resume: CheckpointData | None = None,
    resume_fingerprint: dict | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Run (or resume) one search and return its ``search_results.json`` content."""
    engine = SearchEngine(
        calculator=calculator,
        options=options,
        starts=starts,
        output_dir=output_dir,
        seed=seed,
        model_record=model_record,
        gradient_preflight=gradient_preflight,
        resume=resume,
        resume_fingerprint=resume_fingerprint,
        log=log,
    )
    return engine.run()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = ["PreparedStart", "ReplicaState", "SearchEngine", "run_confsearch"]
