"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Run screen for persistent background calculations with live output.
"""

from __future__ import annotations

import json
import shlex
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ase.io import read
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import Button, Log, ProgressBar, Static

from mliport.jobs import JobManager
from mliport.queue import build_mliport_command

if TYPE_CHECKING:
    from textual.app import ComposeResult


class RunScreen(Screen):
    """Launch and monitor a calculation that survives screen/TUI exit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._job_manager = JobManager()
        self._job_id: str | None = None
        self._refresh_timer: Timer | None = None
        self._displayed_log = ""

    def compose(self) -> ComposeResult:
        calc_type = self.app.get_config("calc_type", "sp")
        structure = self.app.get_config("structure_file", "Not set")
        yield Container(
            Static(f"Running: {calc_type.upper()}", id="title"),
            Static(f"Structure: {structure}", id="subtitle"),
            Static("Progress:"),
            ProgressBar(total=None, id="progress-bar"),
            Static("Starting background job...", id="status-text"),
            Static("Log Output:", classes="section-header"),
            Log(id="run-log"),
            Horizontal(
                Button("Cancel", variant="error", id="cancel-btn"),
                Button("◀ Back (Keep Running)", id="back-btn"),
                id="action-buttons",
            ),
            id="run-container",
        )

    def on_mount(self) -> None:
        self.log_widget = self.query_one("#run-log", Log)
        self.progress = self.query_one("#progress-bar", ProgressBar)
        self.status = self.query_one("#status-text", Static)
        calc_type = self.app.get_config("calc_type", "sp")
        try:
            if calc_type == "neb" and self.app.get_config("neb_resume"):
                # Resume locks the original run directory; the band geometry
                # comes from the checkpoint, not the structure input.
                from mliport.neb.io import resolve_checkpoint_path

                checkpoint_dir = resolve_checkpoint_path(
                    str(self.app.get_config("neb_resume"))
                )
                atoms = read(checkpoint_dir / "band.traj")
            else:
                atoms = read(self.app.get_config("structure_file"))
            display_name = self._make_display_name()
            self._job_id = self._job_manager.new_job_id()
            command = self._build_command()
            # Submit to the queue as PENDING: the scheduler (``mliport queue
            # start`` or the Jobs screen button) promotes it to RUNNING when a
            # slot is free. On a single-GPU machine this naturally serialises
            # several submitted jobs.
            self._job_manager.enqueue(
                job_id=self._job_id,
                calc_type=calc_type,
                structure=self.app.get_config("structure_file"),
                formula=atoms.get_chemical_formula(),
                natoms=len(atoms),
                device=self.app.get_config("device", "cpu"),
                cmd=command,
                display_name=display_name,
            )
            log_path = self._job_manager._log_file(self._job_id).resolve()
            self.log_widget.write(
                f"Queued job: {display_name} [{self._job_id}] (PENDING)\n"
            )
            self.log_widget.write(f"Live log: {log_path}\n")
            self.log_widget.write(
                f"Follow live output: tail -f {shlex.quote(str(log_path))}\n"
            )
            self.log_widget.write(
                "The queue scheduler starts it when a slot is free; "
                "if it is not running, start it with: mliport queue start\n"
            )
            self.status.update("Queued (PENDING) — safe to go Back or exit TUI")
            self._refresh_timer = self.set_interval(0.5, self._refresh_job)
        except Exception as exc:
            self.log_widget.write(f"ERROR: Could not queue the job: {exc}\n")
            self.status.update("Failed to queue")
            self.progress.update(total=100, progress=0)
            self.query_one("#cancel-btn", Button).disabled = True

    def on_unmount(self) -> None:
        """Stop only the UI refresh; the background process keeps running."""
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
            self._refresh_timer = None

    def _make_display_name(self) -> str:
        configured = self.app.get_config("job_name")
        base = configured or (
            f"{self.app.get_config('calc_type', 'job')}-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        display_name = str(base).replace("/", "_").replace("\\", "_")
        if configured is None:
            self.app.update_config("job_name", display_name)
        return display_name

    def _build_command(self) -> list[str]:
        """Build the argv for the background calculation via the shared
        queue helper, so the TUI and `mliport queue submit` cannot drift."""
        calc_type = self.app.get_config("calc_type", "sp")
        model_type = self.app.get_config("model_type", "uma")
        structure_file = self.app.get_config("structure_file")
        model_file = self.app.get_config("model_file")
        resume_path = self.app.get_config("neb_resume") if calc_type == "neb" else None
        # Guard against a missing model: previously an unset model_file was
        # stringified into the argv list as the literal ``--model None``, which
        # made the child process fail later with a confusing "model not found"
        # error instead of surfacing the real problem in the TUI.
        if not model_file and not resume_path:
            raise ValueError("No model file configured; go back and set it.")
        if not structure_file and not resume_path:
            raise ValueError("No structure file configured; go back and set it.")
        options: dict = {}
        for key in (
            "charge",
            "spin",
            "inference_mode",
            "activation_checkpointing",
            "torch_num_threads",
            "default_dtype",
            "head",
        ):
            value = self.app.get_config(key)
            if value is not None:
                options[key] = value
        if model_type == "grace":
            options["gpu_memory_growth"] = self.app.get_config(
                "gpu_memory_growth", True
            )
            memory_limit = self.app.get_config("gpu_memory_limit_mb")
            if memory_limit is not None:
                options["gpu_memory_limit_mb"] = memory_limit
            options["neighbor_cache"] = self.app.get_config("neighbor_cache", True)
            options["neighbor_skin"] = self.app.get_config("neighbor_skin", 1.5)
        if calc_type == "opt":
            for key in (
                "fmax",
                "max_steps",
                "optimizer",
                "cell_opt",
                "fix_symmetry",
            ):
                value = self.app.get_config(key)
                if value is not None:
                    options[key] = value
        elif calc_type == "md":
            for key in (
                "ensemble",
                "temperature",
                "timestep",
                "steps",
                "equilibration_steps",
                "save_interval",
                "pre_relax",
                "pre_relax_steps",
                "pre_relax_fmax",
                "velocity_policy",
                "com_policy",
                "fmax_abort",
                "seed",
            ):
                value = self.app.get_config(key)
                if value is not None:
                    options[key] = value
            if str(options.get("ensemble", "NVT")).upper() == "NVT":
                thermostat = str(self.app.get_config("thermostat", "LANGEVIN")).upper()
                options["thermostat"] = thermostat
                active_keys = {
                    "LANGEVIN": ("friction",),
                    "BUSSI": ("bussi_tau",),
                    "NHC": ("nhc_tdamp", "nhc_tchain", "nhc_tloop"),
                }[thermostat]
                for key in active_keys:
                    value = self.app.get_config(key)
                    if value is not None:
                        options[key] = value
        elif calc_type == "neb":
            for key in (
                "n_intermediate_images",
                "climb",
                "neb_method",
                "neb_interpolation",
                "path_convention",
                "neb_spring",
                "fmax",
                "max_steps",
                "neb_pre_fmax",
                "neb_pre_max_steps",
                "neb_maxstep",
                "endpoint_policy",
                "endpoint_fmax",
                "endpoint_steps",
                "idpp_fmax",
                "idpp_steps",
                "idpp_mic",
                "neb_min_distance",
                "checkpoint_interval",
                "fmax_abort",
                "allow_unvalidated_neb",
            ):
                value = self.app.get_config(key)
                if value is not None:
                    options[key] = value

        return build_mliport_command(
            calc_type=calc_type,
            structure=None if resume_path else structure_file,
            model=None if resume_path else model_file,
            model_type=model_type,
            task=None if resume_path else self.app.get_config("task", "omat"),
            device=self.app.get_config("device", "cpu"),
            output_dir=self.app.get_config("output_dir", "./results"),
            job_name=self._job_id,
            options=options,
            initial=None if resume_path else structure_file,
            final=None if resume_path else self.app.get_config("neb_final"),
            resume=resume_path,
            atom_map=None if resume_path else self.app.get_config("neb_atom_map"),
            image_shifts=(
                None if resume_path else self.app.get_config("neb_image_shifts")
            ),
        )

    def _refresh_job(self) -> None:
        if self._job_id is None:
            return
        log_text = self._job_manager.tail_log(self._job_id, lines=500)
        if log_text != self._displayed_log:
            if log_text.startswith(self._displayed_log):
                new_text = log_text[len(self._displayed_log) :]
            else:
                self.log_widget.clear()
                new_text = log_text
            for line in new_text.splitlines():
                self.log_widget.write(f"{line}\n")
            self._displayed_log = log_text

        data = self._job_manager.get_job(self._job_id)
        if data is None:
            return
        status = data.get("status", "unknown")
        if status == "pending":
            self.status.update("Pending — waiting for a scheduler slot")
        elif status in {"done", "failed", "cancelled"}:
            self.progress.update(total=100, progress=100 if status == "done" else 0)
            self.status.update(status.capitalize())
            self.query_one("#cancel-btn", Button).disabled = True
            if self._refresh_timer is not None:
                self._refresh_timer.stop()
                self._refresh_timer = None
        if self.app.get_config("calc_type") == "neb":
            self._refresh_neb_status(status)

    def _neb_artifacts_path(self) -> Path | None:
        """artifacts.json of this job's NEB run directory, if it exists."""
        if self._job_id is None:
            return None
        run_root = Path(str(self.app.get_config("output_dir", "./results")))
        artifacts_path = run_root / str(self._job_id) / "artifacts.json"
        return artifacts_path if artifacts_path.is_file() else None

    def _refresh_neb_status(self, job_status: str) -> None:
        """Overlay engine artifacts (artifacts.json, checkpoints) on the status."""
        if job_status == "pending":
            return  # Base pending text is already accurate.
        artifacts_path = self._neb_artifacts_path()
        if artifacts_path is None:
            return
        try:
            artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
        except Exception:
            return
        engine_status = str(artifacts.get("status", ""))

        if job_status in {"done", "not_converged", "failed", "cancelled"}:
            self.status.update(self._neb_result_summary(artifacts, engine_status))
            return

        # Still running: report the stage/step recorded by the latest
        # complete-band checkpoint written by the engine itself.
        detail = ""
        checkpoint = (artifacts.get("artifacts") or {}).get("checkpoint") or {}
        latest = checkpoint.get("path")
        if latest:
            try:
                metadata = json.loads(
                    (Path(str(latest)) / "checkpoint.json").read_text(encoding="utf-8")
                )
                stage = str(metadata.get("stage", "running"))
                detail = f" — stage {stage}, step {metadata.get('stage_step', 0)}"
                # Report the budget of THIS stage separately; a single
                # combined percentage would be invented (review R09).
                options = (
                    metadata.get("resume_fingerprint", {}).get("neb_options", {}) or {}
                )
                if stage == "neb_pre":
                    budget = options.get("pre_max_steps")
                    if budget:
                        detail += f" (pre-CI budget {budget} steps)"
                elif stage in {"neb", "ci_neb"}:
                    budget = options.get("max_steps")
                    if budget:
                        detail += f" (CI budget {budget} steps)"
            except Exception:
                detail = " — checkpoint written"
        self.status.update(f"NEB {engine_status or 'running'}{detail}")

    def _neb_result_summary(self, artifacts: dict, engine_status: str) -> str:
        """Human summary for a finished NEB job (converged/not/failed)."""
        result_path = (artifacts.get("artifacts") or {}).get("result")
        if result_path:
            try:
                result = json.loads(Path(str(result_path)).read_text(encoding="utf-8"))
            except Exception:
                result = None
            if isinstance(result, dict):
                if result.get("converged"):
                    summary = (
                        "Completed — converged; sampled forward barrier "
                        f"{result.get('barrier_forward_sampled_eV')} eV"
                    )
                    if result.get("climbing_image_index") is not None:
                        summary += f", climbing image {result['climbing_image_index']}"
                    summary += (
                        f", max NEB force {result.get('max_neb_force_eV_A')} eV/Å"
                    )
                    return summary
                if engine_status == "not_converged":
                    return (
                        "Completed — NOT converged within max_steps; resume "
                        "from the latest checkpoint to continue"
                    )
        if engine_status == "cancelled":
            return "Cancelled — resume from the latest checkpoint to continue"
        if engine_status == "failed":
            return "Failed — see the log and run_context.json for the reason"
        return f"Finished ({engine_status or 'unknown'})"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "cancel-btn" and self._job_id is not None:
            if self._job_manager.kill_job(self._job_id):
                self.status.update("Cancelled")
                event.button.disabled = True
            else:
                self.notify("Job is no longer running", severity="warning")
