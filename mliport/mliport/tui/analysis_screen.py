"""Progressively disclosed TUI for Analysis v2 tasks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from textual import work
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Select, Static, Switch

if TYPE_CHECKING:
    from typing import Any, ClassVar

    from textual.app import ComposeResult


class AnalysisScreen(Screen):
    """Analyze an existing run without exposing every advanced parameter at once."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="config-main"):
            yield Container(
                Static("Analysis v2 — Existing Run", id="title"),
                id="config-header",
            )
            with VerticalScroll(id="config-scroll"):
                yield Label("Run Directory or Trajectory:")
                yield Input(
                    placeholder="e.g. ./results/LGPS-800K",
                    id="analysis-run-input",
                )
                yield Static(
                    "Trajectory range is read from metadata. Production-only is the "
                    "default; no automatic equilibration detection is performed.",
                    id="analysis-range-summary",
                )
                yield Label(
                    "Positions Convention Override:",
                    id="analysis-positions-convention-label",
                )
                yield Select(
                    options=[
                        ("Auto / metadata", "auto"),
                        ("wrapped", "wrapped"),
                        ("unwrapped", "unwrapped"),
                        ("unknown", "unknown"),
                    ],
                    value="auto",
                    id="analysis-positions-convention-select",
                )
                yield Label(
                    "Frame Interval Override (fs):",
                    id="analysis-frame-interval-label",
                )
                yield Input(value="", id="analysis-frame-interval-input")
                yield Label("Analysis Task:")
                yield Select(
                    options=[
                        ("Validate trajectory", "validate"),
                        ("Thermodynamics", "thermo"),
                        ("RDF + coordination", "rdf"),
                        ("MSD diagnostic", "msd"),
                        ("Transport estimate (kinisi)", "transport"),
                        ("Mobile-ion density", "density"),
                        ("GEMDAT advanced", "electrolyte"),
                        ("VACF", "vacf"),
                        ("Velocity spectrum advanced", "spectrum"),
                    ],
                    value="validate",
                    id="analysis-task-select",
                )
                yield Horizontal(
                    Label("Include equilibration:"),
                    Switch(value=False, id="analysis-include-equilibration"),
                    id="analysis-equilibration-row",
                )

                yield Label("Mobile Species:", id="analysis-mobile-label")
                yield Input(value="Li", id="analysis-mobile-input")
                yield Label("Drift Correction:", id="analysis-drift-label")
                yield Select(
                    options=[
                        ("None (explicit default)", "none"),
                        ("Nonmobile framework", "nonmobile"),
                    ],
                    value="none",
                    id="analysis-drift-select",
                )
                yield Static(
                    "For solid-electrolyte transport, nonmobile framework drift is "
                    "often appropriate, but must be selected explicitly.",
                    id="analysis-drift-note",
                )

                yield Label("RDF Center Species:", id="analysis-rdf-center-label")
                yield Input(value="Li", id="analysis-rdf-center-input")
                yield Label("RDF Neighbor Species:", id="analysis-rdf-neighbor-label")
                yield Input(value="S", id="analysis-rdf-neighbor-input")
                yield Label(
                    "RDF r_max (A; blank = safe maximum):", id="analysis-rmax-label"
                )
                yield Input(value="", id="analysis-rmax-input")
                yield Label(
                    "Coordination Cutoff (A; explicit):", id="analysis-cn-label"
                )
                yield Input(value="", id="analysis-cn-input")

                yield Label("Axes:", id="analysis-axes-label")
                yield Input(value="xyz", id="analysis-axes-input")
                yield Label("Ionic Charge (e):", id="analysis-charge-label")
                yield Input(
                    value="",
                    placeholder="required, e.g. +1",
                    id="analysis-charge-input",
                )
                yield Label("Fit Start (ps):", id="analysis-fit-label")
                yield Input(value="", id="analysis-fit-input")
                yield Label(
                    "MSD Diagnostic Fit Start (ps; optional):",
                    id="analysis-msd-fit-start-label",
                )
                yield Input(value="", id="analysis-msd-fit-start-input")
                yield Label(
                    "MSD Diagnostic Fit Stop (ps; optional):",
                    id="analysis-msd-fit-stop-label",
                )
                yield Input(value="", id="analysis-msd-fit-stop-input")
                yield Label(
                    "Lag Step (ps; blank = kinisi default):",
                    id="analysis-lag-step-label",
                )
                yield Input(value="", id="analysis-lag-step-input")
                yield Label(
                    "Lag Stop (ps; blank = kinisi default):",
                    id="analysis-lag-stop-label",
                )
                yield Input(value="", id="analysis-lag-stop-input")
                yield Label(
                    "Temperature (K; blank = metadata):",
                    id="analysis-temperature-label",
                )
                yield Input(value="", id="analysis-temperature-input")
                yield Horizontal(
                    Label("Collective Conductivity:"),
                    Switch(value=False, id="analysis-collective-switch"),
                    id="analysis-collective-row",
                )
                yield Label("kinisi Random Seed:", id="analysis-seed-label")
                yield Input(value="0", id="analysis-seed-input")
                yield Label(
                    "kinisi Parser Memory Limit (GiB):",
                    id="analysis-parser-memory-label",
                )
                yield Input(value="4.0", id="analysis-parser-memory-input")

                yield Label("Density Grid Spacing (A):", id="analysis-spacing-label")
                yield Input(value="0.25", id="analysis-spacing-input")

                yield Label("Site Structure (CIF):", id="analysis-sites-label")
                yield Input(value="", id="analysis-sites-input")
                yield Label("Jump Dimensions:", id="analysis-jump-label")
                yield Select(
                    options=[("1", 1), ("2", 2), ("3", 3)],
                    value=3,
                    id="analysis-jump-select",
                )
                yield Label("Percolation Axes:", id="analysis-percolation-label")
                yield Input(value="xyz", id="analysis-percolation-input")

                yield Static("", id="analysis-status")
            with Horizontal(id="button-bar"):
                yield Button("◀ Back", id="analysis-back-btn")
                yield Button("Run Analysis", variant="success", id="analysis-run-btn")

    def on_mount(self) -> None:
        self._update_task_fields("validate")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "analysis-task-select":
            self._update_task_fields(str(event.value))

    def _set_display(self, selectors: tuple[str, ...], visible: bool) -> None:
        for selector in selectors:
            self.query_one(selector).display = visible

    def _update_task_fields(self, task: str) -> None:
        ranged = task not in {"validate", "transport", "electrolyte"}
        self.query_one("#analysis-equilibration-row").display = ranged
        source_overrides = task in {"validate", "msd", "transport"}
        self._set_display(
            (
                "#analysis-positions-convention-label",
                "#analysis-positions-convention-select",
                "#analysis-frame-interval-label",
                "#analysis-frame-interval-input",
            ),
            source_overrides,
        )
        mobile = task in {
            "msd",
            "transport",
            "density",
            "electrolyte",
            "vacf",
            "spectrum",
        }
        self.query_one("#analysis-mobile-label", Label).update(
            "Species:" if task in {"vacf", "spectrum"} else "Mobile Species:"
        )
        self._set_display(("#analysis-mobile-label", "#analysis-mobile-input"), mobile)
        drift = task in {"msd", "transport"}
        self._set_display(
            (
                "#analysis-drift-label",
                "#analysis-drift-select",
                "#analysis-drift-note",
            ),
            drift,
        )
        rdf = task == "rdf"
        self._set_display(
            (
                "#analysis-rdf-center-label",
                "#analysis-rdf-center-input",
                "#analysis-rdf-neighbor-label",
                "#analysis-rdf-neighbor-input",
                "#analysis-rmax-label",
                "#analysis-rmax-input",
                "#analysis-cn-label",
                "#analysis-cn-input",
            ),
            rdf,
        )
        transport = task == "transport"
        self._set_display(
            (
                "#analysis-charge-label",
                "#analysis-charge-input",
                "#analysis-fit-label",
                "#analysis-fit-input",
                "#analysis-lag-step-label",
                "#analysis-lag-step-input",
                "#analysis-lag-stop-label",
                "#analysis-lag-stop-input",
                "#analysis-temperature-label",
                "#analysis-temperature-input",
                "#analysis-collective-row",
                "#analysis-seed-label",
                "#analysis-seed-input",
                "#analysis-parser-memory-label",
                "#analysis-parser-memory-input",
            ),
            transport,
        )
        msd = task == "msd"
        self._set_display(
            (
                "#analysis-msd-fit-start-label",
                "#analysis-msd-fit-start-input",
                "#analysis-msd-fit-stop-label",
                "#analysis-msd-fit-stop-input",
            ),
            msd,
        )
        axes = task in {"msd", "transport"}
        self._set_display(("#analysis-axes-label", "#analysis-axes-input"), axes)
        density = task == "density"
        self._set_display(
            ("#analysis-spacing-label", "#analysis-spacing-input"), density
        )
        electrolyte = task == "electrolyte"
        self._set_display(
            (
                "#analysis-sites-label",
                "#analysis-sites-input",
                "#analysis-jump-label",
                "#analysis-jump-select",
                "#analysis-percolation-label",
                "#analysis-percolation-input",
            ),
            electrolyte,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "analysis-back-btn":
            self.app.pop_screen()
        elif event.button.id == "analysis-run-btn":
            self._start_analysis()

    def _parameters(self, task: str) -> dict[str, Any]:
        parameters: dict[str, Any] = {}
        if task in {"validate", "msd", "transport"}:
            convention = str(
                self.query_one("#analysis-positions-convention-select", Select).value
            )
            if convention != "auto":
                parameters["positions_convention"] = convention
            frame_interval = self.query_one(
                "#analysis-frame-interval-input", Input
            ).value.strip()
            if frame_interval:
                value = float(frame_interval)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("Frame interval must be positive")
                parameters["frame_interval_fs"] = value
        if task not in {"validate", "transport", "electrolyte"}:
            parameters["include_equilibration"] = self.query_one(
                "#analysis-include-equilibration", Switch
            ).value
        mobile = self.query_one("#analysis-mobile-input", Input).value.strip()
        if task in {"msd", "transport", "density", "electrolyte"}:
            if not mobile:
                raise ValueError("Mobile species is required")
            parameters["mobile_species"] = mobile
        elif task in {"vacf", "spectrum"}:
            parameters["species"] = mobile or None
        if task in {"msd", "transport"}:
            parameters["drift_reference"] = str(
                self.query_one("#analysis-drift-select", Select).value
            )
            parameters["axes" if task == "msd" else "dimensions"] = self.query_one(
                "#analysis-axes-input", Input
            ).value.strip()
        if task == "msd":
            fit_start = self.query_one(
                "#analysis-msd-fit-start-input", Input
            ).value.strip()
            fit_stop = self.query_one(
                "#analysis-msd-fit-stop-input", Input
            ).value.strip()
            if bool(fit_start) != bool(fit_stop):
                raise ValueError(
                    "MSD diagnostic fit start and stop must be provided together"
                )
            if fit_start:
                parameters["fit_start_ps"] = float(fit_start)
                parameters["fit_stop_ps"] = float(fit_stop)
                if (
                    not math.isfinite(parameters["fit_start_ps"])
                    or not math.isfinite(parameters["fit_stop_ps"])
                    or parameters["fit_start_ps"] < 0
                    or parameters["fit_stop_ps"] <= parameters["fit_start_ps"]
                ):
                    raise ValueError("MSD fit stop must be greater than fit start >= 0")
        if task == "rdf":
            parameters["center_species"] = self.query_one(
                "#analysis-rdf-center-input", Input
            ).value.strip()
            parameters["neighbor_species"] = self.query_one(
                "#analysis-rdf-neighbor-input", Input
            ).value.strip()
            rmax = self.query_one("#analysis-rmax-input", Input).value.strip()
            cutoff = self.query_one("#analysis-cn-input", Input).value.strip()
            if rmax:
                parameters["r_max_A"] = float(rmax)
            if cutoff:
                parameters["cn_cutoff_A"] = float(cutoff)
        elif task == "transport":
            charge = self.query_one("#analysis-charge-input", Input).value.strip()
            fit = self.query_one("#analysis-fit-input", Input).value.strip()
            if not charge or not fit:
                raise ValueError("Transport requires explicit charge and fit start")
            charge_value = float(charge)
            fit_value = float(fit)
            if not math.isfinite(charge_value) or charge_value == 0:
                raise ValueError("Ionic charge must be finite and non-zero")
            if not math.isfinite(fit_value) or fit_value < 0:
                raise ValueError("Fit start must be finite and non-negative")
            parameters["ionic_charge_e"] = charge_value
            parameters["fit_start_ps"] = fit_value
            lag_step = self.query_one("#analysis-lag-step-input", Input).value.strip()
            lag_stop = self.query_one("#analysis-lag-stop-input", Input).value.strip()
            if bool(lag_step) != bool(lag_stop):
                raise ValueError(
                    "Transport lag step and lag stop must be provided together"
                )
            if lag_step:
                parameters["lag_step_ps"] = float(lag_step)
                parameters["lag_stop_ps"] = float(lag_stop)
                if (
                    not math.isfinite(parameters["lag_step_ps"])
                    or not math.isfinite(parameters["lag_stop_ps"])
                    or parameters["lag_step_ps"] <= 0
                    or parameters["lag_stop_ps"] <= 0
                ):
                    raise ValueError("Transport lag step and lag stop must be positive")
            temperature = self.query_one(
                "#analysis-temperature-input", Input
            ).value.strip()
            if temperature:
                parameters["temperature_K"] = float(temperature)
                if (
                    not math.isfinite(parameters["temperature_K"])
                    or parameters["temperature_K"] <= 0
                ):
                    raise ValueError("Temperature must be positive")
            parameters["collective_conductivity"] = self.query_one(
                "#analysis-collective-switch", Switch
            ).value
            parameters["random_seed"] = int(
                self.query_one("#analysis-seed-input", Input).value
            )
            parser_limit = float(
                self.query_one("#analysis-parser-memory-input", Input).value
            )
            if not math.isfinite(parser_limit) or parser_limit <= 0:
                raise ValueError("kinisi parser memory limit must be positive")
            parameters["parser_memory_limit_gib"] = parser_limit
        elif task == "density":
            parameters["spacing_A"] = float(
                self.query_one("#analysis-spacing-input", Input).value
            )
        elif task == "electrolyte":
            sites = self.query_one("#analysis-sites-input", Input).value.strip()
            if not sites:
                raise ValueError("GEMDAT analysis requires an explicit site CIF")
            parameters["sites_path"] = sites
            parameters["jump_dimensions"] = int(
                self.query_one("#analysis-jump-select", Select).value
            )
            parameters["percolation_axes"] = self.query_one(
                "#analysis-percolation-input", Input
            ).value.strip()
        return parameters

    def _start_analysis(self) -> None:
        source = self.query_one("#analysis-run-input", Input).value.strip()
        if not source:
            self.notify("Specify a run directory or trajectory", severity="error")
            return
        if not Path(source).expanduser().exists():
            self.notify(f"Analysis source not found: {source}", severity="error")
            return
        task = str(self.query_one("#analysis-task-select", Select).value)
        try:
            parameters = self._parameters(task)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        self.query_one("#analysis-status", Static).update("Running analysis...")
        self.query_one("#analysis-run-btn", Button).disabled = True
        self._run_analysis_worker(source, task, parameters)

    @work(thread=True, exclusive=True)
    def _run_analysis_worker(
        self, source: str, task: str, parameters: dict[str, Any]
    ) -> None:
        from mliport.analysis.runner import run_analysis
        from mliport.analysis.schema import AnalysisRequest

        try:
            result = run_analysis(
                AnalysisRequest(task=task, source=source, parameters=parameters)
            )
        except Exception as exc:
            self.app.call_from_thread(self._finish_analysis, None, str(exc))
            return
        self.app.call_from_thread(self._finish_analysis, result, None)

    def _finish_analysis(
        self, result: dict[str, Any] | None, error: str | None
    ) -> None:
        self.query_one("#analysis-run-btn", Button).disabled = False
        status = self.query_one("#analysis-status", Static)
        if error is not None:
            status.update(f"Analysis failed: {error}")
            self.notify(error, severity="error")
            return
        status.update(f"Completed: {result['output_dir']}")
        self.notify(f"Analysis saved to {result['output_dir']}")
