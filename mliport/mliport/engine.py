"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Modified for the mliport project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
Unified execution engine for MLIP calculations.

Provides CalculationEngine as the single entry point for CLI, TUI, and API.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from mliport.base_calculator import BaseMLIPCalculator
from mliport.config.defaults import BUILTIN_DEFAULTS
from mliport.logger import LiveRunLogger, follow_log_command
from mliport.protocols import CancellationRequested, ProgressCallback, ProgressEvent
from mliport.timing import RunTiming, append_timing_to_outputs, timing_log_lines

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Any, Literal

    from ase import Atoms


@dataclass
class EngineConfig:
    """Unified configuration for all calculation types and interfaces.

    Fields:
        calc_type: sp, opt, md, neb, or batch.
        model_path: Path to model checkpoint/file.
        model_type: MLIP engine (uma, mace, dpa, grace).
        task: Task type. UMA: omat/omol/...; others: bulk/molecule.
        device: cpu, cuda, gpu or cuda:N.
        inference_mode: default or turbo (UMA only; non-default values are
            rejected for other engines).
        output_dir: Directory for output files.
        job_name: Optional job name.
        calculator_options: Engine-specific options that reach the underlying
            ASE calculator (e.g. MACE ``default_dtype``/``head``, UMA
            ``inference_mode``/``torch_num_threads``). Plan section 11.1.
        run_options: Calc-type-specific parameters (fmax, temperature, ...).
        settings: Resolved output/global settings consumed by the engine.
        options: *Deprecated* untyped bag. Kept for backward compatibility;
            routed into run/calculator options on first use with a
            DeprecationWarning. New code should use the two fields above.
        torch_num_threads: CPU intra-op thread count. The historical field name
            is retained for compatibility; it controls PyTorch for
            UMA/MACE and DPA PyTorch models, and TensorFlow for GRACE. Legacy
            DPA TensorFlow ``.pb`` models use their DeepMD backend settings.
        activation_checkpointing: GPU memory saving (UMA; overrides
            inference_mode preset).
        strict_config: When True, unknown option keys raise instead of warn
            (plan section 10).
        detach: If True, submit as background job.
    """

    calc_type: Literal["sp", "opt", "md", "neb", "batch"]
    model_path: Path
    # No implicit backend: EngineConfig() without a model_type fails closed.
    model_type: str = ""
    # Task default follows the backend (UMA: omat; others: explicit PBC bulk).
    task: str = ""
    device: str = "cpu"
    inference_mode: str = "default"
    output_dir: Path = field(default_factory=lambda: Path("./results"))
    job_name: str | None = None
    calculator_options: dict = field(default_factory=dict)
    run_options: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    # Deprecated; kept for backward compatibility (plan section 11).
    options: dict = field(default_factory=dict)
    torch_num_threads: int | None = None
    activation_checkpointing: bool | None = None
    strict_config: bool = False
    detach: bool = False

    def __post_init__(self) -> None:
        from mliport.backend_selection import (  # noqa: PLC0415
            UMA_ALIASES,
            default_task_for,
            require_model_type,
        )

        self.model_type = require_model_type(self.model_type)
        task = str(self.task).strip().lower()
        self.task = task if task else default_task_for(self.model_type)
        if self.model_type not in UMA_ALIASES and self.task not in {"bulk", "molecule"}:
            raise ValueError(
                f"Invalid task {self.task!r} for engine {self.model_type!r}. "
                "Non-UMA engines require the explicit PBC semantic 'bulk' or "
                "'molecule'."
            )

    @classmethod
    def from_resolved(cls, resolved: Any) -> EngineConfig:
        """Build an EngineConfig from a :class:`ResolvedConfig`.

        This is the bridge between the layered config resolver and the
        execution engine: CLI/API build a resolved config and hand it here so
        the engine receives cleanly split calculator/run options.
        """
        run_options = dict(resolved.run_options)
        # Safety options live in ResolvedConfig.settings so they do not get
        # mistaken for arbitrary runner kwargs.  Copy the one guard currently
        # implemented by MDRunner/NEBRunner into their typed option bags.
        # Without this bridge, [safety] fmax_abort would be recorded in
        # resolved_config.json but silently have no effect on execution.
        if resolved.calc_type in {"md", "neb"} and "fmax_abort" in resolved.settings:
            run_options.setdefault("fmax_abort", resolved.settings["fmax_abort"])

        return cls(
            calc_type=resolved.calc_type,
            model_path=Path(resolved.model_path) if resolved.model_path else Path(""),
            model_type=resolved.model_type,
            task=resolved.task,
            device=resolved.device,
            inference_mode=resolved.inference_mode,
            calculator_options=dict(resolved.calculator_options),
            run_options=run_options,
            settings=dict(resolved.settings),
            torch_num_threads=(
                resolved.calculator_options.get("torch_num_threads")
                or resolved.settings.get("torch_num_threads")
            ),
            activation_checkpointing=resolved.calculator_options.get(
                "activation_checkpointing"
            ),
            strict_config=resolved.strict,
        )


class CalculationEngine:
    """Unified execution engine for MLIP calculations.

    Use CalculationEngine.from_config() to create an instance, then
    call run(), run_async(), run_neb(), or run_batch().
    """

    VALID_CALC_TYPES: ClassVar[set[str]] = {"sp", "opt", "md", "neb", "batch"}

    def __init__(self, config: EngineConfig):
        self.config = config
        self._validate()

    @classmethod
    def from_config(cls, config: EngineConfig) -> CalculationEngine:
        return cls(config)

    def _validate(self) -> None:
        if self.config.calc_type not in self.VALID_CALC_TYPES:
            raise ValueError(
                f"Unknown calc_type '{self.config.calc_type}'. "
                f"Must be one of: {', '.join(self.VALID_CALC_TYPES)}"
            )
        # Validate option keys against the central schema (plan section 10).
        # In strict mode unknown / mistyped keys raise; otherwise they warn,
        # preserving the historical non-fatal behaviour.
        from mliport.config.schema import get_schema  # noqa: PLC0415

        schema = get_schema()
        merged: dict[str, Any] = {}
        merged.update(self._effective_calculator_options())
        merged.update(self._effective_run_options())
        errors = schema.validate_dict(
            merged,
            strict=self.config.strict_config,
            context=f"calc_type={self.config.calc_type}",
        )
        if self.config.strict_config and errors:
            raise ValueError(
                "Strict config validation failed:\n  - " + "\n  - ".join(errors)
            )
        for err in errors:
            warnings.warn(err)
        # Non-strict mode: still warn about unknown keys (with a typo hint) so
        # silent typos remain visible, matching the historical behaviour.
        if not self.config.strict_config:
            for key in merged:
                if schema.resolve(key) is None:
                    suggestion = schema.suggest(key)
                    hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
                    warnings.warn(
                        f"Unknown option {key!r} for "
                        f"calc_type '{self.config.calc_type}'.{hint}"
                    )

    # Calculator-option keys that may live in the legacy ``options`` dict and
    # must be routed to ``calculator_options`` for backward compatibility.
    _CALC_OPTION_KEYS = {
        "default_dtype",
        "head",
        "inference_mode",
        "torch_num_threads",
        "activation_checkpointing",
        "gpu_memory_growth",
        "gpu_memory_limit_mb",
        "neighbor_cache",
        "neighbor_skin",
    }

    def _effective_calculator_options(self) -> dict:
        """Return calculator options, merging legacy ``options`` calc keys."""
        opts = dict(self.config.calculator_options)
        if self.config.options:
            for key in self._CALC_OPTION_KEYS:
                if key in self.config.options and key not in opts:
                    opts[key] = self.config.options[key]
        return opts

    def _effective_run_options(self) -> dict:
        """Return run options, falling back to the legacy ``options`` dict."""
        if self.config.run_options:
            return dict(self.config.run_options)
        if self.config.options:
            warnings.warn(
                "EngineConfig.options is deprecated; use calculator_options "
                "and run_options instead (plan section 11).",
                DeprecationWarning,
                stacklevel=2,
            )
            return {
                k: v
                for k, v in self.config.options.items()
                if k not in self._CALC_OPTION_KEYS
            }
        return {}

    def _create_calculator(self) -> BaseMLIPCalculator:
        from mliport.calculators.factory import CalculatorFactory  # noqa: PLC0415

        calc_opts = self._effective_calculator_options()
        # Apply a user-selected thread limit to every PyTorch backend before
        # its model is imported/constructed. UMA also receives the value in its
        # InferenceSettings below.
        if (
            self.config.torch_num_threads is not None
            and self.config.model_type.lower() in {"uma", "fairchem", "mace", "dpa"}
        ):
            import torch  # noqa: PLC0415

            torch.set_num_threads(self.config.torch_num_threads)
        # The UMA-only top-level fields feed the calculator for UMA engines;
        # they are intentionally NOT forwarded to MACE/DPA/GRACE (which would
        # otherwise trigger a cross-engine warning in the factory).
        if self.config.model_type.lower() in {"uma", "fairchem"}:
            calc_opts.setdefault("inference_mode", self.config.inference_mode)
            if self.config.torch_num_threads is not None:
                calc_opts.setdefault("torch_num_threads", self.config.torch_num_threads)
            if self.config.activation_checkpointing is not None:
                calc_opts.setdefault(
                    "activation_checkpointing", self.config.activation_checkpointing
                )
        elif (
            self.config.model_type.lower() == "grace"
            and self.config.torch_num_threads is not None
        ):
            calc_opts.setdefault("cpu_threads", self.config.torch_num_threads)
        if self.config.model_type.lower() == "grace":
            # Polite TensorFlow allocation is the safe package default. A user
            # can additionally set gpu_memory_limit_mb for hard isolation.
            calc_opts.setdefault("gpu_memory_growth", True)
        # MACE uses an accuracy-first float64 default for every calculation type.
        # A higher config layer may explicitly opt into float32 for performance.
        if self.config.model_type.lower() == "mace":
            calc_opts.setdefault("default_dtype", "float64")
        return CalculatorFactory.create(
            model_type=self.config.model_type,
            model_path=self.config.model_path,
            device=self.config.device,
            task=self.config.task,
            strict=self.config.strict_config,
            **calc_opts,
        )

    @property
    def output_dir(self) -> Path:
        """Final output directory, including an optional job-name subdirectory."""
        if self.config.job_name:
            return self.config.output_dir / self.config.job_name
        return self.config.output_dir

    @property
    def run_log_path(self) -> Path:
        """Path of the continuously flushed log for this run."""
        return (self.output_dir / "run.log").resolve()

    def _create_runner(
        self, calculator, progress_callback=None, log_fn=None, cancel_event=None
    ):
        from mliport.runners.md import MDRunner  # noqa: PLC0415
        from mliport.runners.optimization import OptimizationRunner  # noqa: PLC0415
        from mliport.runners.singlepoint import SinglePointRunner  # noqa: PLC0415

        opts = self._effective_run_options()
        common = dict(
            calculator=calculator,
            output_dir=self.config.output_dir,
            charge=opts.get("charge"),
            spin=opts.get("spin"),
            write_outcar=self.config.settings.get("write_outcar", True),
            write_forces=self.config.settings.get("write_forces", True),
            write_stress=self.config.settings.get("write_stress", True),
            write_json=self.config.settings.get("write_json", True),
            verbose=False,
            job_name=self.config.job_name,
            log_fn=log_fn,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

        if self.config.calc_type == "sp":
            return SinglePointRunner(**common)
        elif self.config.calc_type == "opt":
            return OptimizationRunner(
                fmax=opts.get("fmax", 0.05),
                max_steps=opts.get("max_steps", 500),
                optimizer=opts.get("optimizer", "FIRE"),
                cell_opt=opts.get("cell_opt", False),
                fix_symmetry=opts.get("fix_symmetry", False),
                **common,
            )
        elif self.config.calc_type == "md":
            return MDRunner(
                ensemble=opts.get("ensemble", "NVT"),
                temperature=opts.get("temperature", 300.0),
                timestep=opts.get("timestep", 1.0),
                steps=opts.get("steps", 1000),
                equilibration_steps=opts.get("equilibration_steps", 0),
                thermostat=opts.get("thermostat", "LANGEVIN"),
                friction=opts.get("friction", 0.001),
                bussi_tau=opts.get("bussi_tau", 1000.0),
                nhc_tdamp=opts.get("nhc_tdamp", 100.0),
                nhc_tchain=opts.get("nhc_tchain", 3),
                nhc_tloop=opts.get("nhc_tloop", 1),
                save_interval=opts.get("save_interval", 10),
                # NVE is energy-conserving: pre-relaxing first moves the
                # structure to a 0 K minimum and changes the conserved-energy
                # baseline, so default it OFF for NVE (still ON for NVT to avoid
                # explosions). An explicit user setting always wins.
                pre_relax=opts.get(
                    "pre_relax",
                    str(opts.get("ensemble", "NVT")).upper() != "NVE",
                ),
                pre_relax_steps=opts.get("pre_relax_steps", 50),
                pre_relax_fmax=opts.get("pre_relax_fmax", 0.1),
                seed=opts.get("seed"),
                velocity_policy=opts.get("velocity_policy", "auto"),
                com_policy=opts.get("com_policy", "auto"),
                pre_relax_mode=opts.get("pre_relax_mode", "none"),
                fmax_abort=opts.get(
                    "fmax_abort", BUILTIN_DEFAULTS["safety"]["fmax_abort"]
                ),
                write_trajectory=self.config.settings.get("write_trajectory", True),
                write_xdatcar=self.config.settings.get("write_xdatcar", True),
                **common,
            )
        else:
            raise ValueError(f"Unknown calc_type: {self.config.calc_type}")

    def run(
        self,
        atoms: Atoms,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        log_fn: Any | None = None,
        started_at: float | None = None,
    ) -> dict[str, Any]:
        """Run calculation synchronously.

        Args:
            atoms: ASE Atoms object.
            progress_callback: Optional callback for progress events.
            cancel_event: Optional threading.Event for cooperative cancellation.
            log_fn: Optional callback receiving live log messages.
            started_at: Monotonic timestamp when the user requested the run.

        Returns:
            Results dictionary.
        """
        with LiveRunLogger(self.run_log_path, callback=log_fn) as run_logger:
            run_logger(f"Output directory: {self.output_dir.resolve()}")
            run_logger(f"Live log: {self.run_log_path}")
            run_logger(f"Follow live output: {follow_log_command(self.run_log_path)}")
            run_logger(
                f"Loading {self.config.model_type.upper()} model on "
                f"{self.config.device}: {self.config.model_path}"
            )
            if progress_callback is not None:
                progress_callback(
                    ProgressEvent(
                        phase="loading_model",
                        message="Loading model and preparing calculation...",
                    )
                )

            calculator = self._create_calculator()
            runner = self._create_runner(
                calculator,
                progress_callback=progress_callback,
                log_fn=run_logger,
                cancel_event=cancel_event,
            )
            try:
                return runner.execute(atoms, started_at=started_at)
            except CancellationRequested as exc:
                run_logger(f"Calculation cancelled: {exc}", "warning")
                raise
            except Exception as exc:
                run_logger(f"Calculation failed: {exc}", "error")
                raise

    def run_neb(
        self,
        resolved: Any,
        *,
        initial: Atoms | None = None,
        final: Atoms | None = None,
        resume: str | Path | None = None,
        atom_map=None,
        image_shifts=None,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        log_fn: Any | None = None,
    ) -> dict[str, Any]:
        """Run the public NEB workflow with one shared model instance."""
        if self.config.calc_type != "neb" or resolved.calc_type != "neb":
            raise ValueError("run_neb requires calc_type='neb'")

        from mliport.neb.workflow import (  # noqa: PLC0415
            _preflight_new_output,
            _run_directory_lock,
            _run_neb_workflow_locked,
            checkpoint_run_directory,
            options_from_resolved,
        )

        # Reject invalid scientific options before importing/loading a backend.
        options_from_resolved(resolved)
        output = self.output_dir.resolve()
        if resume is None:
            _preflight_new_output(output)
        elif output != checkpoint_run_directory(resume):
            raise ValueError(
                "Resume output differs from the checkpoint's original run directory"
            )

        # Own the run directory before opening/truncating its log or loading a
        # potentially large model. This closes the race where two frontends
        # could both load a model before the workflow-level lock was acquired.
        with _run_directory_lock(output):
            with LiveRunLogger(self.run_log_path, callback=log_fn) as run_logger:
                run_logger(f"Output directory: {output}")
                run_logger(f"Live log: {self.run_log_path}")
                run_logger(
                    f"Follow live output: {follow_log_command(self.run_log_path)}"
                )
                run_logger(
                    f"Loading {self.config.model_type.upper()} model on "
                    f"{self.config.device}: {self.config.model_path}"
                )
                if progress_callback is not None:
                    progress_callback(
                        ProgressEvent(
                            phase="loading_model",
                            message=(
                                "Loading one model instance for serial NEB images..."
                            ),
                        )
                    )
                calculator = self._create_calculator()
                try:
                    result = _run_neb_workflow_locked(
                        calculator,
                        resolved,
                        output_dir=output,
                        initial=initial,
                        final=final,
                        resume=resume,
                        atom_map=atom_map,
                        image_shifts=image_shifts,
                        verbose=False,
                        progress_callback=progress_callback,
                        cancel_event=cancel_event,
                    )
                except CancellationRequested as exc:
                    run_logger(f"NEB calculation cancelled: {exc}", "warning")
                    raise
                except Exception as exc:
                    run_logger(f"NEB calculation failed: {exc}", "error")
                    raise
                if progress_callback is not None:
                    progress_callback(
                        ProgressEvent(
                            phase="done",
                            message=(
                                f"NEB workflow finished with status {result['status']}"
                            ),
                        )
                    )
                return result

    async def run_async(
        self,
        atoms: Atoms,
        log_fn: Any | None = None,
        started_at: float | None = None,
    ) -> AsyncIterator[ProgressEvent]:
        """Run calculation asynchronously, yielding progress events.

        The actual computation runs in a thread pool; progress events
        are bridged back to the asyncio loop via a queue.

        Cooperative cancellation: when the async generator is cancelled,
        a threading.Event is set so the worker thread can stop early.
        The finally block waits for the thread with a timeout to avoid
        blocking indefinitely on long-running simulations.

        Args:
            atoms: ASE Atoms object.
            log_fn: Optional callback receiving live log messages.
            started_at: Monotonic timestamp when the user requested the run.

        Yields:
            ProgressEvent at each phase transition.
        """
        queue: asyncio.Queue[ProgressEvent | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        cancel_event = threading.Event()

        def progress_callback(event: ProgressEvent) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def blocking_work() -> dict[str, Any]:
            try:
                return self.run(
                    atoms,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                    log_fn=log_fn,
                    started_at=started_at,
                )
            except CancellationRequested as exc:
                # Normal cancellation — emit a "cancelled" event, not "error"
                event = ProgressEvent(
                    phase="cancelled",
                    message=str(exc),
                )
                loop.call_soon_threadsafe(queue.put_nowait, event)
                raise
            except Exception as exc:
                event = ProgressEvent(
                    phase="error",
                    message=f"Calculation failed: {exc}",
                )
                loop.call_soon_threadsafe(queue.put_nowait, event)
                raise

        task = loop.run_in_executor(None, blocking_work)
        _cancelled = False

        try:
            while True:
                event = await queue.get()
                yield event
                if event.phase in ("done", "error", "cancelled"):
                    break
        except asyncio.CancelledError:
            _cancelled = True
            # Signal the worker thread to stop cooperatively
            cancel_event.set()
            # Attempt to cancel the future (best-effort; won't stop a running thread)
            task.cancel()
            raise  # Re-raise so the caller's CancelledError handler runs
        finally:
            if _cancelled:
                # Wait for the worker thread with a generous timeout.
                # The cancel_event will cause runners to bail out at their next
                # check-point (every MD step or optimization step).
                with contextlib.suppress(
                    asyncio.TimeoutError, asyncio.CancelledError, Exception
                ):
                    await asyncio.wait_for(task, timeout=30.0)
            else:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    def run_batch(
        self,
        files: list[Path],
        progress_callback: ProgressCallback | None = None,
        log_fn: Any | None = None,
        started_at: float | None = None,
    ) -> dict[str, Any]:
        """Run batch calculation on multiple structure files.

        Args:
            files: List of structure file paths.
            progress_callback: Optional callback for progress events.
            log_fn: Optional callback receiving live log messages.
            started_at: Monotonic timestamp when the user requested the run.

        Returns:
            Batch summary dictionary.
        """
        from mliport.runners.batch import BatchRunner  # noqa: PLC0415

        timing = RunTiming(
            started_at=time.perf_counter() if started_at is None else started_at
        )
        with LiveRunLogger(self.run_log_path, callback=log_fn) as run_logger:
            run_logger(f"Output directory: {self.output_dir.resolve()}")
            run_logger(f"Live log: {self.run_log_path}")
            run_logger(f"Follow live output: {follow_log_command(self.run_log_path)}")
            run_logger(
                f"Loading {self.config.model_type.upper()} model on "
                f"{self.config.device}: {self.config.model_path}"
            )
            if progress_callback is not None:
                progress_callback(
                    ProgressEvent(
                        phase="loading_model",
                        message="Loading model for batch calculation...",
                    )
                )

            calculator = self._create_calculator()
            calculator.get_calculator()
            timing.mark_compute_started()

            opts = self._effective_run_options()
            runner = BatchRunner(
                calculator,
                calc_type=opts.get("sub_calc_type", "sp"),
                output_dir=self.config.output_dir,
                parallel=opts.get("parallel", False),
                max_workers=opts.get("max_workers", 1),
                verbose=False,
                job_name=self.config.job_name,
                write_forces=self.config.settings.get("write_forces", True),
                write_stress=self.config.settings.get("write_stress", True),
                write_json=self.config.settings.get("write_json", True),
                charge=opts.get("charge"),
                spin=opts.get("spin"),
                progress_callback=progress_callback,
                log_fn=run_logger,
            )
            try:
                summary = runner.run_from_files(files)
            except Exception as exc:
                run_logger(f"Batch calculation failed: {exc}", "error")
                raise

            timing.mark_compute_finished()
            timing_values = timing.finish()
            summary["timing"] = timing_values
            append_timing_to_outputs(self.output_dir, timing_values)
            for line in timing_log_lines(timing_values):
                run_logger(line)
            if progress_callback is not None:
                progress_callback(
                    ProgressEvent(
                        phase="done",
                        message="Batch calculation complete",
                        extra={"timing": timing_values},
                    )
                )
            return summary
