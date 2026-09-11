# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified for the mlipx project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
"""Configuration screen for mlipx TUI."""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read as ase_read
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Select,
    Static,
    Switch,
)

if TYPE_CHECKING:
    from typing import ClassVar

    from textual.app import ComposeResult

from mlipx.tui.run_screen import RunScreen


class ConfigScreen(Screen):
    """Configuration screen for setting up calculation parameters."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "back", "Back"),
        ("up", "scroll_up", "Scroll Up"),
        ("down", "scroll_down", "Scroll Down"),
        ("pageup", "page_up", "Page Up"),
        ("pagedown", "page_down", "Page Down"),
    ]

    CSS = """
    #structure-status, #model-status {
        margin-top: 0;
        margin-bottom: 1;
        padding-left: 1;
    }
    .status-ok {
        color: $success;
        text-style: bold;
    }
    .status-error {
        color: $error;
        text-style: italic;
    }
    .switch-row {
        height: 3;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the configuration screen."""
        calc_type = self.app.get_config("calc_type", "sp")

        # Main container fills the screen
        with Container(id="config-main"):
            # Header at top
            yield Container(
                Static(f"Configuration: {calc_type.upper()}", id="title"),
                id="config-header",
            )

            # Scrollable content area
            with VerticalScroll(id="config-scroll"):
                # File paths section
                yield Static("📁 File Paths", classes="section-title")

                yield Label("Structure File:")
                yield Input(
                    placeholder="e.g., structure.cif or POSCAR (relative paths supported)",
                    value=str(self.app.get_config("structure_file", "") or ""),
                    id="structure-input",
                )
                yield Static("", id="structure-status")

                yield Label("Model File:")
                yield Input(
                    placeholder="e.g., uma-s-1p2p1.pt (relative paths supported)",
                    value=str(self.app.get_config("model_file", "") or ""),
                    id="model-input",
                )
                yield Static("", id="model-status")

                yield Label("Output Directory:")
                yield Input(
                    value=self.app.get_config("output_dir", "./results"),
                    id="output-input",
                )

                yield Label("Job Name (optional):")
                yield Input(
                    placeholder="e.g., structure_01",
                    value=str(self.app.get_config("job_name", "") or ""),
                    id="job-name-input",
                )

                # Model engine + task selection
                yield Static("⚙️  Model Engine & Task", classes="section-title")

                yield Label("Model Engine:")
                yield Select(
                    options=[
                        ("UMA (FAIRChem)", "uma"),
                        ("MACE", "mace"),
                        ("DPA (DeepMD-kit)", "dpa"),
                        ("GRACE", "grace"),
                    ],
                    value=self.app.get_config("model_type", "uma"),
                    id="model-type-select",
                )

                yield Label("Task Type:", id="task-type-label")
                yield Select(
                    options=self._task_options(),
                    value=self._default_task_value(),
                    id="task-select",
                )

                charge = self.app.get_config("charge")
                yield Static("", id="electronic-state-note")
                yield Label("Total Charge:", id="charge-label")
                yield Input(
                    value="" if charge is None else str(charge),
                    placeholder="e.g., 0, -1, or 1",
                    id="charge-input",
                )

                spin = self.app.get_config("spin")
                yield Label("Spin:", id="spin-label")
                yield Input(
                    value="" if spin is None else str(spin),
                    placeholder="e.g., 1",
                    id="spin-input",
                )

                yield Label("Device:")
                yield Input(
                    value=str(self.app.get_config("device", "cpu")),
                    placeholder="cpu, cuda, or cuda:N (for example cuda:1)",
                    id="device-input",
                )

                yield Static("🧠 Backend & Resource Options", classes="section-title")
                yield Static(
                    "",
                    id="engine-options-note",
                )

                yield Label("MACE Precision:", id="dtype-label")
                yield Select(
                    options=[
                        ("float64 (accuracy default)", "float64"),
                        ("float32 (faster)", "float32"),
                    ],
                    value=self.app.get_config("default_dtype", "float64"),
                    id="dtype-select",
                )

                yield Label(
                    "MACE/DPA Model Head or Branch (optional):",
                    id="head-label",
                )
                yield Input(
                    value=str(self.app.get_config("head", "") or ""),
                    placeholder="e.g., default or Domains_SSE_PBE",
                    id="head-input",
                )

                yield Label("UMA Inference Mode:", id="inference-mode-label")
                yield Select(
                    options=[("default", "default"), ("turbo", "turbo")],
                    value=self.app.get_config("inference_mode", "default"),
                    id="inference-mode-select",
                )

                threads = self.app.get_config("torch_num_threads")
                yield Label("CPU Threads (blank = backend/system default):")
                yield Input(
                    value="" if threads is None else str(threads),
                    placeholder="e.g., 4",
                    id="torch-threads-input",
                )

                grace_limit = self.app.get_config("gpu_memory_limit_mb")
                yield Label(
                    "GRACE GPU Memory Limit MiB (blank = on-demand growth):",
                    id="grace-memory-limit-label",
                )
                yield Input(
                    value="" if grace_limit is None else str(grace_limit),
                    placeholder="e.g., 6144 when sharing a GPU",
                    id="grace-memory-limit-input",
                )

                yield Horizontal(
                    Label("GRACE Neighbour Cache:", id="grace-neighbor-cache-label"),
                    Switch(
                        value=bool(self.app.get_config("neighbor_cache", True)),
                        id="grace-neighbor-cache-switch",
                    ),
                    id="grace-neighbor-cache-row",
                    classes="switch-row",
                )
                yield Label(
                    "GRACE Neighbour Skin (Å):",
                    id="grace-neighbor-skin-label",
                )
                yield Input(
                    value=str(self.app.get_config("neighbor_skin", 1.5)),
                    placeholder="positive length, e.g., 1.5",
                    id="grace-neighbor-skin-input",
                )

                checkpointing = self.app.get_config("activation_checkpointing")
                checkpointing_value = (
                    "auto"
                    if checkpointing is None
                    else ("on" if checkpointing else "off")
                )
                yield Label(
                    "UMA Activation Checkpointing:",
                    id="activation-checkpointing-label",
                )
                yield Select(
                    options=[
                        ("Auto (from inference mode)", "auto"),
                        ("On (lower VRAM)", "on"),
                        ("Off (faster, more VRAM)", "off"),
                    ],
                    value=checkpointing_value,
                    id="activation-checkpointing-select",
                )

                # Calculation-specific options
                yield Static("🔧 Calculation Options", classes="section-title")
                yield from self._calc_options(calc_type)

                # Add some bottom padding
                yield Static("")

            # Action buttons fixed at bottom
            with Horizontal(id="button-bar"):
                yield Button("◀ Back", id="back-btn")
                yield Button("🚀 Run", variant="success", id="run-btn")

    # UMA-specific tasks vs generic bulk/molecule for other engines
    UMA_TASKS = [
        ("Inorganic Materials (omat)", "omat"),
        ("Molecules (omol)", "omol"),
        ("Catalysis OC20 (oc20)", "oc20"),
        ("Catalysis OC25 (oc25)", "oc25"),
        ("MOFs (odac)", "odac"),
        ("Molecular Crystals (omc)", "omc"),
    ]
    GENERIC_TASKS = [
        ("Bulk / Periodic (bulk)", "bulk"),
        ("Molecule (molecule)", "molecule"),
    ]

    def _current_model_type(self) -> str:
        """Model type from the select widget (or app config fallback)."""
        try:
            sel = self.query_one("#model-type-select", Select)
            if sel.value:
                return str(sel.value)
        except Exception:
            pass
        return str(self.app.get_config("model_type", "uma"))

    def _task_options(self):
        """Task options for the current model engine."""
        return (
            self.UMA_TASKS
            if self._current_model_type() == "uma"
            else self.GENERIC_TASKS
        )

    def _default_task_value(self) -> str:
        """Default task value for the current model engine."""
        current = str(self.app.get_config("task", "omat"))
        valid = {v for _, v in self._task_options()}
        return (
            current
            if current in valid
            else ("omat" if self._current_model_type() == "uma" else "bulk")
        )

    def _current_task(self) -> str:
        """Task from the select widget (or app config fallback)."""
        try:
            sel = self.query_one("#task-select", Select)
            if sel.value:
                return str(sel.value)
        except Exception:
            pass
        return str(self.app.get_config("task", self._default_task_value()))

    def _calc_options(self, calc_type: str):
        """Generate calculation-specific options."""
        if calc_type == "opt":
            yield Label("Force Threshold (eV/Å):")
            yield Input(value=str(self.app.get_config("fmax", 0.05)), id="fmax-input")

            yield Label("Max Steps:")
            yield Input(
                value=str(self.app.get_config("max_steps", 500)), id="max-steps-input"
            )

            yield Label("Optimizer:")
            yield Select(
                options=[("FIRE", "FIRE"), ("BFGS", "BFGS"), ("LBFGS", "LBFGS")],
                value=self.app.get_config("optimizer", "FIRE"),
                id="optimizer-select",
            )

            yield Horizontal(
                Label("Cell Optimization:"),
                Switch(value=self.app.get_config("cell_opt", False), id="cell-opt"),
                classes="switch-row",
            )
            yield Horizontal(
                Label("Preserve Crystal Symmetry:"),
                Switch(
                    value=self.app.get_config("fix_symmetry", False),
                    id="fix-symmetry",
                ),
                classes="switch-row",
            )

        elif calc_type == "md":
            current_ensemble = self.app.get_config("ensemble", "NVT").upper()
            yield Label("Ensemble:")
            yield RadioSet(
                RadioButton("NVT", id="nvt", value=current_ensemble == "NVT"),
                RadioButton("NVE", id="nve", value=current_ensemble == "NVE"),
                id="ensemble-radio",
            )

            yield Label("Temperature (K):")
            yield Input(
                value=str(self.app.get_config("temperature", 300.0)),
                id="temp-input",
            )

            yield Label("Time Step (fs):")
            yield Input(
                value=str(self.app.get_config("timestep", 1.0)),
                id="timestep-input",
            )

            yield Label("Production Steps:")
            yield Input(
                value=str(self.app.get_config("steps", 1000)),
                id="steps-input",
            )

            yield Label("Equilibration Steps (same ensemble):")
            yield Input(
                value=str(self.app.get_config("equilibration_steps", 0)),
                id="equilibration-steps-input",
            )

            yield Label("Save Interval:")
            yield Input(
                value=str(self.app.get_config("save_interval", 10)),
                id="save-interval-input",
            )

            yield Label("Thermostat:", id="thermostat-label")
            yield Select(
                options=[
                    ("Langevin", "LANGEVIN"),
                    ("Bussi-CSVR", "BUSSI"),
                    ("Nosé-Hoover Chain", "NHC"),
                ],
                value=self.app.get_config("thermostat", "LANGEVIN"),
                id="thermostat-select",
            )

            yield Label("Friction γ (fs^-1):", id="friction-label")
            yield Input(
                value=str(self.app.get_config("friction", 0.001)),
                id="friction-input",
            )

            yield Label("Coupling time τT (fs):", id="bussi-tau-label")
            yield Input(
                value=str(self.app.get_config("bussi_tau", 1000.0)),
                id="bussi-tau-input",
            )

            yield Label("Thermostat time tdamp (fs):", id="nhc-tdamp-label")
            yield Input(
                value=str(self.app.get_config("nhc_tdamp", 100.0)),
                id="nhc-tdamp-input",
            )
            yield Label("Chain length:", id="nhc-tchain-label")
            yield Input(
                value=str(self.app.get_config("nhc_tchain", 3)),
                id="nhc-tchain-input",
            )
            yield Label("Thermostat substeps:", id="nhc-tloop-label")
            yield Input(
                value=str(self.app.get_config("nhc_tloop", 1)),
                id="nhc-tloop-input",
            )

            # Unset (None) means "engine ensemble-aware default": display the
            # value the engine will actually apply for the selected ensemble.
            pre_relax_config = self.app.get_config("pre_relax")
            if pre_relax_config is None:
                pre_relax_config = (
                    str(self.app.get_config("ensemble", "NVT")).upper() != "NVE"
                )
            yield Horizontal(
                Label("Pre-relaxation:"),
                Switch(
                    value=pre_relax_config,
                    id="pre-relax",
                ),
                classes="switch-row",
            )

            yield Label("Pre-relaxation Max Steps:")
            yield Input(
                value=str(self.app.get_config("pre_relax_steps", 50)),
                id="pre-relax-steps-input",
            )

            yield Label("Pre-relaxation Force Threshold (eV/Å):")
            yield Input(
                value=str(self.app.get_config("pre_relax_fmax", 0.1)),
                id="pre-relax-fmax-input",
            )

            seed = self.app.get_config("seed")
            yield Label("Random Seed (blank = auto-generate and record):")
            yield Input(
                value="" if seed is None else str(seed),
                placeholder="e.g., 42",
                id="seed-input",
            )

            yield Label("Velocity Policy:")
            velocity_policy_config = self.app.get_config("velocity_policy")
            if velocity_policy_config is None:
                # Unset (None) means "MDRunner default (auto)"; display it.
                velocity_policy_config = "auto"
            yield Select(
                options=[
                    ("Auto", "auto"),
                    ("Initialize / overwrite", "initialize"),
                    ("Preserve existing velocities", "preserve"),
                ],
                value=velocity_policy_config,
                id="velocity-policy-select",
            )

            yield Label("Center-of-mass Policy:")
            yield Select(
                options=[
                    ("Auto (fail closed when ambiguous)", "auto"),
                    ("None", "none"),
                    ("Initialization only (Langevin only)", "initialize_only"),
                    ("Explicit FixCom constraint", "constraint"),
                ],
                value=self.app.get_config("com_policy", "auto"),
                id="com-policy-select",
            )

            yield Label("Force-safety Abort Threshold (eV/Å):")
            yield Input(
                value=str(self.app.get_config("fmax_abort", 20.0)),
                id="fmax-abort-input",
            )

        elif calc_type == "neb":
            from mlipx.config.defaults import BUILTIN_DEFAULTS

            neb_defaults = BUILTIN_DEFAULTS.get("neb", {})

            def neb_hint(key: str) -> str:
                value = neb_defaults.get(key)
                return "default: -" if value is None else f"default: {value}"

            def neb_number(key: str, config_key: str | None = None) -> Input:
                stored = self.app.get_config(config_key or key)
                return Input(
                    value="" if stored is None else str(stored),
                    placeholder=neb_hint(key),
                    id=f"{config_key or key}-input",
                )

            def neb_bool(config_key: str, widget_id: str) -> Switch:
                stored = self.app.get_config(config_key)
                default = neb_defaults.get(config_key, False)
                return Switch(
                    value=bool(default if stored is None else stored),
                    id=widget_id,
                )

            yield Static("🧭 Endpoints & Mapping", classes="section-title")

            yield Label("Final Endpoint File:")
            yield Input(
                placeholder="e.g., final.vasp (relative paths supported)",
                value=str(self.app.get_config("neb_final", "") or ""),
                id="final-structure-input",
            )
            yield Static("", id="final-structure-status")

            resume = self.app.get_config("neb_resume")
            yield Label("Resume From (checkpoint dir, checkpoints root, or run dir):")
            yield Input(
                placeholder="blank = new run from endpoints",
                value=str(resume or ""),
                id="neb-resume-input",
            )

            yield Label("Atom Map (0-based final indices in initial-atom order):")
            yield Input(
                placeholder="blank = atoms already in the same order",
                value=str(self.app.get_config("neb_atom_map_text", "") or ""),
                id="atom-map-input",
            )

            yield Label("Image Shifts (per atom: rows ';', lattice ','):")
            yield Input(
                placeholder="e.g., 0,0,0;1,0,0 (blank = automatic)",
                value=str(self.app.get_config("neb_image_shifts_text", "") or ""),
                id="image-shifts-input",
            )

            yield Static("", id="neb-preview")

            yield Static("🛤️  NEB Path", classes="section-title")

            yield Label("Intermediate Images (endpoints are additional):")
            yield neb_number("n_intermediate_images")

            yield Label("Interpolation:")
            _stored = self.app.get_config("neb_interpolation")
            yield Select(
                options=[("Linear", "linear"), ("IDPP", "idpp")],
                value=str(_stored)
                if _stored is not None
                else str(neb_defaults.get("neb_interpolation", "linear")),
                id="neb-interpolation-select",
            )

            yield Horizontal(
                Label("Climbing Image (CI-NEB):"),
                neb_bool("climb", "neb-climb-switch"),
                classes="switch-row",
            )

            yield Label("Spring Constant (eV/Å²):")
            yield neb_number("neb_spring")

            yield Label("Path Convention:")
            _stored = self.app.get_config("path_convention")
            yield Select(
                options=[
                    ("Minimum image (mic)", "mic"),
                    ("Unwrapped", "unwrapped"),
                ],
                value=str(_stored)
                if _stored is not None
                else str(neb_defaults.get("path_convention", "mic")),
                id="neb-path-convention-select",
            )

            yield Label("Minimum Interatomic Distance (Å):")
            yield neb_number("neb_min_distance")

            yield Label("IDPP Fmax (eV/Å):", id="idpp_fmax-label")
            yield neb_number("idpp_fmax")
            yield Label("IDPP Steps:", id="idpp_steps-label")
            yield neb_number("idpp_steps")
            yield Horizontal(
                Label("IDPP Minimum-Image Convention:", id="idpp_mic-label"),
                neb_bool("idpp_mic", "idpp-mic-switch"),
                classes="switch-row",
            )

            yield Static("🧱 Endpoint Policy", classes="section-title")

            yield Label("Endpoint Policy:")
            _stored = self.app.get_config("endpoint_policy")
            yield Select(
                options=[
                    ("Validate (fail if endpoints move)", "validate"),
                    ("Relax endpoints first", "relax"),
                ],
                value=str(_stored)
                if _stored is not None
                else str(neb_defaults.get("endpoint_policy", "validate")),
                id="endpoint-policy-select",
            )

            yield Label("Endpoint Fmax (eV/Å):")
            yield neb_number("endpoint_fmax")

            yield Label("Endpoint Steps:")
            yield neb_number("endpoint_steps")

            yield Static("🎯 Convergence & Safety", classes="section-title")

            yield Label("CI/NEB Fmax (eV/Å):")
            yield neb_number("fmax")

            yield Label("Max Steps:")
            yield neb_number("max_steps")

            yield Label("Pre-stage Fmax (eV/Å) (CI-NEB only):")
            yield neb_number("neb_pre_fmax")

            yield Label("Pre-stage Max Steps (CI-NEB only):")
            yield neb_number("neb_pre_max_steps")

            yield Label("Optimizer Max Step (Å):")
            yield neb_number("neb_maxstep")

            yield Label("Checkpoint Interval (steps):")
            yield neb_number("checkpoint_interval")

            yield Label("Force-safety Abort Threshold (eV/Å):")
            yield neb_number("fmax_abort")

            yield Horizontal(
                Label("Allow unvalidated NEB (skip engine evidence check):"),
                neb_bool("allow_unvalidated_neb", "allow-unvalidated-neb-switch"),
                classes="switch-row",
            )
            yield Static(
                "NEB runs are checked against the installed validation records; "
                "unvalidated engines require an explicit opt-in here.",
                id="neb-validation-note",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "back-btn":
            self.app.pop_screen()

        elif button_id == "run-btn":
            self._save_and_run()

    def _validate_path(self, input_id: str, value: str) -> bool:
        """Validate input path and update its status label.

        Returns True if valid/exists, False otherwise.
        """
        status_id = f"#{input_id.replace('-input', '-status')}"
        try:
            status_widget = self.query_one(status_id, Static)
        except Exception:
            return False

        val_stripped = value.strip()
        if not val_stripped:
            status_widget.update(
                "[!] Please specify a path (relative paths supported, e.g. ./structure.cif)"
            )
            status_widget.set_classes("status-error")
            return False

        path = Path(val_stripped)
        resolved_path = path.resolve()

        if resolved_path.exists():
            status_widget.update(f"[OK] Found: {resolved_path}")
            status_widget.set_classes("status-ok")
            return True
        else:
            status_widget.update(f"[NOT FOUND] Checked: {resolved_path}")
            status_widget.set_classes("status-error")
            return False

    def on_select_changed(self, event: Select.Changed) -> None:
        """Refresh engine- and task-dependent controls."""
        if event.select.id == "model-type-select":
            self.app.update_config("model_type", event.value)
            task_select = self.query_one("#task-select", Select)
            task_select.set_options(self._task_options())
            task_select.value = self._default_task_value()
            self._update_engine_option_states()
        elif event.select.id == "task-select":
            self._update_molecular_option_states()
        elif event.select.id == "thermostat-select":
            self._update_thermostat_option_states()
        elif event.select.id == "neb-interpolation-select":
            self._update_neb_option_states()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Refresh thermostat controls when NVT/NVE changes."""
        if event.radio_set.id == "ensemble-radio":
            self._update_thermostat_option_states()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle live path validation on input change."""
        if event.input.id in (
            "structure-input",
            "model-input",
            "final-structure-input",
        ):
            self._validate_path(event.input.id, event.value)
        if self.app.get_config("calc_type") == "neb" and event.input.id in (
            "structure-input",
            "final-structure-input",
            "atom-map-input",
            "image-shifts-input",
            "neb-resume-input",
        ):
            self._update_neb_option_states()

    def on_mount(self) -> None:
        """Initialize and validate paths on mount."""
        # Initial validation of pre-filled values
        structure_input = self.query_one("#structure-input", Input)
        model_input = self.query_one("#model-input", Input)
        self._validate_path("structure-input", structure_input.value)
        self._validate_path("model-input", model_input.value)
        self._update_engine_option_states()
        calc_type = self.app.get_config("calc_type")
        if calc_type == "md":
            self._update_thermostat_option_states()
        elif calc_type == "neb":
            final_input = self.query_one("#final-structure-input", Input)
            self._validate_path("final-structure-input", final_input.value)
            self._update_neb_option_states()

    def _update_engine_option_states(self) -> None:
        """Show only controls consumed by the selected backend."""
        model_type = self._current_model_type()
        is_uma = model_type == "uma"
        is_mace = model_type == "mace"
        is_grace = model_type == "grace"
        has_head = model_type in {"mace", "dpa"}

        task_label = self.query_one("#task-type-label", Label)
        task_label.update("Task Type:" if is_uma else "System Type:")

        for selector in ("#dtype-label", "#dtype-select"):
            self.query_one(selector).display = is_mace
        for selector in ("#head-label", "#head-input"):
            self.query_one(selector).display = has_head
        for selector in ("#inference-mode-label", "#inference-mode-select"):
            self.query_one(selector).display = is_uma
        for selector in (
            "#activation-checkpointing-label",
            "#activation-checkpointing-select",
        ):
            self.query_one(selector).display = is_uma
        for selector in (
            "#grace-memory-limit-label",
            "#grace-memory-limit-input",
            "#grace-neighbor-cache-row",
            "#grace-neighbor-skin-label",
            "#grace-neighbor-skin-input",
        ):
            self.query_one(selector).display = is_grace

        self.query_one("#dtype-select", Select).disabled = not is_mace
        self.query_one("#head-input", Input).disabled = not has_head
        self.query_one("#inference-mode-select", Select).disabled = not is_uma
        self.query_one("#activation-checkpointing-select", Select).disabled = not is_uma
        self.query_one("#torch-threads-input", Input).disabled = False
        self.query_one("#grace-memory-limit-input", Input).disabled = not is_grace
        self.query_one("#grace-neighbor-cache-switch", Switch).disabled = not is_grace
        self.query_one("#grace-neighbor-skin-input", Input).disabled = not is_grace

        head_label = self.query_one("#head-label", Label)
        if model_type == "mace":
            head_label.update("MACE Model Head (required for multi-head models):")
        elif model_type == "dpa":
            head_label.update("DPA Model Branch (required for multi-task models):")

        note = self.query_one("#engine-options-note", Static)
        if is_uma:
            note.update(
                "UMA inference mode and activation checkpointing are shown below. "
                "CPU Threads controls PyTorch intra-op threads."
            )
        elif model_type == "mace":
            note.update(
                "Only MACE options are shown below. CPU Threads controls "
                "PyTorch intra-op threads."
            )
        elif model_type == "dpa":
            note.update(
                "Only DPA options are shown below. CPU Threads controls "
                "PyTorch intra-op threads for PyTorch models; legacy TensorFlow "
                ".pb models use their TensorFlow backend settings."
            )
        else:
            note.update(
                "GRACE uses TensorFlow on-demand GPU allocation by default. "
                "Set a hard MiB limit below when sharing a GPU. CPU Threads "
                "controls TensorFlow intra-op threads."
            )
        self._update_molecular_option_states()

    def _update_thermostat_option_states(self) -> None:
        """Show only the coupling parameters active for this MD method."""
        ensemble_radio = self.query_one("#ensemble-radio", RadioSet)
        pressed = ensemble_radio.pressed_button
        is_nvt = pressed is None or pressed.id == "nvt"
        thermostat_select = self.query_one("#thermostat-select", Select)
        thermostat = str(thermostat_select.value or "LANGEVIN").upper()

        for selector in ("#thermostat-label", "#thermostat-select"):
            self.query_one(selector).display = is_nvt
        thermostat_select.disabled = not is_nvt

        groups = {
            "LANGEVIN": ("#friction-label", "#friction-input"),
            "BUSSI": ("#bussi-tau-label", "#bussi-tau-input"),
            "NHC": (
                "#nhc-tdamp-label",
                "#nhc-tdamp-input",
                "#nhc-tchain-label",
                "#nhc-tchain-input",
                "#nhc-tloop-label",
                "#nhc-tloop-input",
            ),
        }
        for name, selectors in groups.items():
            active = is_nvt and thermostat == name
            for selector in selectors:
                widget = self.query_one(selector)
                widget.display = active
                if isinstance(widget, Input):
                    widget.disabled = not active

    def _update_molecular_option_states(self) -> None:
        """Show charge/spin controls only for a molecular task."""
        model_type = self._current_model_type()
        task = self._current_task()
        is_molecular = task in {"omol", "molecule"}
        for selector in (
            "#electronic-state-note",
            "#charge-label",
            "#charge-input",
            "#spin-label",
            "#spin-input",
        ):
            self.query_one(selector).display = is_molecular

        if not is_molecular:
            return

        note = self.query_one("#electronic-state-note", Static)
        spin_label = self.query_one("#spin-label", Label)
        spin_input = self.query_one("#spin-input", Input)
        if model_type == "uma":
            note.update(
                "UMA omol electronic state (blank = charge 0, spin multiplicity 1)."
            )
            spin_label.update("Spin Multiplicity (2S+1):")
            spin_input.placeholder = "1 = singlet, 2 = doublet, 3 = triplet"
        else:
            note.update(
                "Optional molecular metadata; whether it affects predictions depends "
                "on the selected model."
            )
            spin_label.update("Spin Metadata (model-specific):")
            spin_input.placeholder = "blank = do not inject spin metadata"

    def _update_neb_option_states(self) -> None:
        """Toggle IDPP-only controls and refresh the endpoint preview."""
        interpolation_select = self.query_one("#neb-interpolation-select", Select)
        is_idpp = str(interpolation_select.value or "linear") == "idpp"
        for selector in (
            "#idpp_fmax-label",
            "#idpp_fmax-input",
            "#idpp_steps-label",
            "#idpp_steps-input",
        ):
            widget = self.query_one(selector)
            widget.display = is_idpp
            if isinstance(widget, Input):
                widget.disabled = not is_idpp
        mic_row = self.query_one("#idpp_mic-label", Label).parent
        mic_row.display = is_idpp
        self.query_one("#idpp-mic-switch", Switch).disabled = not is_idpp
        self._update_neb_preview()

    def _parse_neb_atom_map_text(self) -> tuple[list[int] | None, str | None]:
        """Parse the atom-map input; returns (value, error)."""
        text = self.query_one("#atom-map-input", Input).value.strip()
        if not text:
            return None, None
        try:
            values = [int(part) for part in text.split(",")]
        except ValueError:
            return None, "Atom map must be comma-separated integers"
        return values, None

    def _parse_neb_image_shifts_text(
        self,
    ) -> tuple[list[list[int]] | None, str | None]:
        """Parse the image-shifts input; returns (value, error)."""
        text = self.query_one("#image-shifts-input", Input).value.strip()
        if not text:
            return None, None
        rows: list[list[int]] = []
        for row in text.split(";"):
            parts = row.strip().split(",")
            if len(parts) != 3:
                return None, "Each image-shift row needs 3 lattice integers"
            try:
                rows.append([int(part) for part in parts])
            except ValueError:
                return None, "Image shifts must be integers"
        return rows, None

    def _update_neb_preview(self) -> None:
        """Show an endpoint consistency summary (formula, constraints, motion)."""
        try:
            preview = self.query_one("#neb-preview", Static)
        except Exception:
            return
        resume_text = self.query_one("#neb-resume-input", Input).value.strip()
        preview.set_classes("")
        if resume_text:
            preview.update(
                "Resume mode: endpoints, atom map, and image shifts are ignored."
            )
            return
        initial_text = self.query_one("#structure-input", Input).value.strip()
        final_text = self.query_one("#final-structure-input", Input).value.strip()
        if not initial_text or not final_text:
            preview.update("")
            return
        try:
            initial = ase_read(initial_text)
            final = ase_read(final_text)
        except Exception as exc:
            preview.update(f"[!] Could not read endpoints: {exc}")
            preview.set_classes("status-error")
            return
        lines = [
            f"Initial: {initial.get_chemical_formula()} ({len(initial)} atoms)",
            f"Final:   {final.get_chemical_formula()} ({len(final)} atoms)",
        ]
        atom_map, map_error = self._parse_neb_atom_map_text()
        if map_error:
            lines.append(f"[!] {map_error}")
            preview.update("\n".join(lines))
            preview.set_classes("status-error")
            return
        image_shifts, shifts_error = self._parse_neb_image_shifts_text()
        if shifts_error:
            lines.append(f"[!] {shifts_error}")
            preview.update("\n".join(lines))
            preview.set_classes("status-error")
            return
        fixed_initial = 0
        for constraint in initial.constraints:
            if isinstance(constraint, FixAtoms):
                fixed_initial += len(constraint.index)
        lines.append(f"Fixed atoms (initial): {fixed_initial}")
        from mlipx.neb.prepare import (  # noqa: PLC0415
            NEBPreparationError,
            prepare_endpoint_geometry,
        )

        path_convention = str(
            self.query_one("#neb-path-convention-select", Select).value or "mic"
        )
        # Shared with the production band preparation: the preview reports the
        # same mapped/shifted/MIC-resolved displacement the NEB will run with
        # (RC-05), instead of a raw wrapped-coordinate difference.
        try:
            endpoint = prepare_endpoint_geometry(
                initial,
                final,
                atom_map=atom_map,
                image_shifts=image_shifts,
                path_convention=path_convention,
            )
        except NEBPreparationError as exc:
            lines.append(f"[!] {exc}")
            preview.update("\n".join(lines))
            preview.set_classes("status-error")
            return
        displacements = endpoint.intended_displacement
        norms = (displacements**2).sum(axis=1) ** 0.5
        max_index = int(norms.argmax())
        lines.append(
            f"Preview displacement: max {norms[max_index]:.3f} Å "
            f"at atom {max_index} ({initial.get_chemical_symbols()[max_index]})"
        )
        segments = self._neb_preview_segment_count()
        if segments is not None:
            lines.append(
                f"Per-segment estimate ({segments} segments): "
                f"{norms[max_index] / segments:.3f} Å"
            )
        if endpoint.winding_present:
            shifted_atoms = int(
                np.count_nonzero(np.asarray(endpoint.image_shifts).any(axis=1))
            )
            lines.append(
                f"Winding: present (explicit lattice shifts on "
                f"{shifted_atoms} atoms)"
            )
        else:
            lines.append("Winding: none")
        preview.update("\n".join(lines))

    def _neb_preview_segment_count(self) -> int | None:
        """Band segment count implied by the configured image count, if set."""
        try:
            text = self.query_one("#n_intermediate_images-input", Input).value.strip()
        except Exception:
            return None
        if not text:
            return None
        try:
            n_intermediate = int(text)
        except ValueError:
            return None
        if n_intermediate < 1:
            return None
        return n_intermediate + 1

    def _save_and_run(self) -> None:
        """Save configuration and run calculation."""
        run_started_at = time.perf_counter()

        calc_type = self.app.get_config("calc_type")

        # Get file paths
        structure = self.query_one("#structure-input", Input).value
        model = self.query_one("#model-input", Input).value
        output = self.query_one("#output-input", Input).value

        # NEB resume mode: the checkpoint carries the endpoints and model.
        neb_resume_path: Path | None = None
        resume_mode = False
        if calc_type == "neb":
            resume_text = self.query_one("#neb-resume-input", Input).value.strip()
            resume_mode = bool(resume_text)
            if resume_mode:
                neb_resume_path = Path(resume_text).resolve()

        if not structure and not resume_mode:
            self.notify("Please specify a structure file", severity="error")
            return

        if not model and not resume_mode:
            self.notify("Please specify a model file", severity="error")
            return

        # Resolve paths to absolute paths
        structure_path = (
            Path(structure.strip()).resolve() if structure.strip() else None
        )
        model_path = Path(model.strip()).resolve() if model.strip() else None
        output_path = Path(output.strip()).resolve()

        if structure_path is not None and not structure_path.exists():
            self.notify(f"Structure file not found: {structure_path}", severity="error")
            return

        if model_path is not None and not model_path.exists():
            self.notify(f"Model file not found: {model_path}", severity="error")
            return

        if resume_mode and not neb_resume_path.exists():
            self.notify(f"Resume path not found: {neb_resume_path}", severity="error")
            return

        # Save absolute paths to config
        self.app.update_config(
            "structure_file", str(structure_path) if structure_path else None
        )
        self.app.update_config("model_file", str(model_path) if model_path else None)
        self.app.update_config("output_dir", str(output_path))
        if calc_type == "neb":
            self.app.update_config(
                "neb_resume", str(neb_resume_path) if resume_mode else None
            )

        # Get job name
        job_name = self.query_one("#job-name-input", Input).value
        if job_name:
            self.app.update_config("job_name", job_name)
        else:
            self.app.update_config("job_name", None)

        # Get model engine, task and device
        model_type_select = self.query_one("#model-type-select", Select)
        if model_type_select.value:
            self.app.update_config("model_type", model_type_select.value)
        model_type = str(self.app.get_config("model_type", "uma"))

        task_select = self.query_one("#task-select", Select)
        if task_select.value:
            self.app.update_config("task", task_select.value)
        task = str(self.app.get_config("task", "omat"))

        if task in {"omol", "molecule"}:
            charge_text = self.query_one("#charge-input", Input).value.strip()
            spin_text = self.query_one("#spin-input", Input).value.strip()
            try:
                charge = int(charge_text) if charge_text else None
            except ValueError:
                self.notify("Total charge must be an integer", severity="error")
                return
            try:
                spin = int(spin_text) if spin_text else None
            except ValueError:
                self.notify("Spin must be an integer", severity="error")
                return
            if charge is not None and not -100 <= charge <= 100:
                self.notify(
                    "Total charge must be between -100 and 100", severity="error"
                )
                return
            if spin is not None and not 0 <= spin <= 100:
                self.notify("Spin must be between 0 and 100", severity="error")
                return
            self.app.update_config("charge", charge)
            self.app.update_config("spin", spin)
        else:
            # Do not leak electronic-state values from a previous molecular
            # setup after the user switches back to a periodic task.
            self.app.update_config("charge", None)
            self.app.update_config("spin", None)

        device = self.query_one("#device-input", Input).value.strip().lower()
        if not re.fullmatch(r"(?:cpu|gpu|cuda(?::\d+)?)", device):
            self.notify(
                "Device must be cpu, gpu, cuda, or cuda:N (for example cuda:1)",
                severity="error",
            )
            return
        self.app.update_config("device", device)

        # Backend/resource options. Disabled controls are intentionally not
        # forwarded to an incompatible backend.
        if model_type == "mace":
            dtype_select = self.query_one("#dtype-select", Select)
            self.app.update_config("default_dtype", str(dtype_select.value))
        if model_type in {"mace", "dpa"}:
            head = self.query_one("#head-input", Input).value.strip()
            self.app.update_config("head", head or None)
        if model_type == "uma":
            inference_select = self.query_one("#inference-mode-select", Select)
            self.app.update_config("inference_mode", str(inference_select.value))
            checkpoint_select = self.query_one(
                "#activation-checkpointing-select", Select
            )
            checkpoint_value = str(checkpoint_select.value)
            self.app.update_config(
                "activation_checkpointing",
                None if checkpoint_value == "auto" else checkpoint_value == "on",
            )
        threads_text = self.query_one("#torch-threads-input", Input).value.strip()
        if threads_text:
            try:
                threads = int(threads_text)
            except ValueError:
                self.notify("CPU threads must be a positive integer", severity="error")
                return
            if threads < 1:
                self.notify("CPU threads must be at least 1", severity="error")
                return
            self.app.update_config("torch_num_threads", threads)
        else:
            self.app.update_config("torch_num_threads", None)
        if model_type == "grace":
            limit_text = self.query_one(
                "#grace-memory-limit-input", Input
            ).value.strip()
            if limit_text:
                try:
                    memory_limit = int(limit_text)
                except ValueError:
                    self.notify(
                        "GRACE GPU memory limit must be a positive integer",
                        severity="error",
                    )
                    return
                if memory_limit < 1:
                    self.notify(
                        "GRACE GPU memory limit must be at least 1 MiB",
                        severity="error",
                    )
                    return
                if device == "cpu":
                    self.notify(
                        "GRACE GPU memory limit requires a CUDA/GPU device",
                        severity="error",
                    )
                    return
                self.app.update_config("gpu_memory_limit_mb", memory_limit)
            else:
                self.app.update_config("gpu_memory_limit_mb", None)
            self.app.update_config("gpu_memory_growth", True)
            skin_text = self.query_one(
                "#grace-neighbor-skin-input", Input
            ).value.strip()
            try:
                neighbor_skin = float(skin_text)
            except ValueError:
                self.notify(
                    "GRACE neighbour skin must be a positive number in Å",
                    severity="error",
                )
                return
            if not math.isfinite(neighbor_skin) or neighbor_skin <= 0:
                self.notify(
                    "GRACE neighbour skin must be finite and greater than 0 Å",
                    severity="error",
                )
                return
            self.app.update_config(
                "neighbor_cache",
                self.query_one("#grace-neighbor-cache-switch", Switch).value,
            )
            self.app.update_config("neighbor_skin", neighbor_skin)

        # Get calculation-specific options
        calc_type = self.app.get_config("calc_type")

        if calc_type == "opt":
            try:
                fmax = float(self.query_one("#fmax-input", Input).value)
                self.app.update_config("fmax", fmax)
            except ValueError:
                pass

            try:
                max_steps = int(self.query_one("#max-steps-input", Input).value)
                self.app.update_config("max_steps", max_steps)
            except ValueError:
                pass

            optimizer_select = self.query_one("#optimizer-select", Select)
            if optimizer_select.value:
                self.app.update_config("optimizer", optimizer_select.value)

            cell_opt = self.query_one("#cell-opt", Switch)
            self.app.update_config("cell_opt", cell_opt.value)
            fix_symmetry = self.query_one("#fix-symmetry", Switch)
            self.app.update_config("fix_symmetry", fix_symmetry.value)

        elif calc_type == "md":
            ensemble_radio = self.query_one("#ensemble-radio", RadioSet)
            pressed_ensemble = ensemble_radio.pressed_button
            self.app.update_config(
                "ensemble",
                pressed_ensemble.id.upper() if pressed_ensemble else "NVT",
            )

            try:
                temp = float(self.query_one("#temp-input", Input).value)
            except ValueError:
                self.notify("Temperature must be a number", severity="error")
                return
            if not math.isfinite(temp) or temp < 0:
                self.notify(
                    "Temperature must be a finite value >= 0 K", severity="error"
                )
                return
            self.app.update_config("temperature", temp)

            try:
                timestep = float(self.query_one("#timestep-input", Input).value)
            except ValueError:
                self.notify("Time step must be a number", severity="error")
                return
            if not math.isfinite(timestep) or timestep <= 0:
                self.notify("Time step must be a finite value > 0 fs", severity="error")
                return
            self.app.update_config("timestep", timestep)

            try:
                steps = int(self.query_one("#steps-input", Input).value)
            except ValueError:
                self.notify("Steps must be an integer", severity="error")
                return
            if steps < 0:
                self.notify("Steps must be >= 0", severity="error")
                return
            self.app.update_config("steps", steps)

            try:
                equilibration_steps = int(
                    self.query_one("#equilibration-steps-input", Input).value
                )
            except ValueError:
                self.notify("Equilibration steps must be an integer", severity="error")
                return
            if equilibration_steps < 0:
                self.notify("Equilibration steps must be >= 0", severity="error")
                return
            self.app.update_config("equilibration_steps", equilibration_steps)

            try:
                save_interval = int(self.query_one("#save-interval-input", Input).value)
            except ValueError:
                self.notify("Save interval must be an integer", severity="error")
                return
            if save_interval < 1:
                self.notify("Save interval must be >= 1", severity="error")
                return
            self.app.update_config("save_interval", save_interval)

            thermostat_select = self.query_one("#thermostat-select", Select)
            thermostat = str(thermostat_select.value).upper()
            self.app.update_config("thermostat", thermostat)

            if (
                self.app.get_config("ensemble") == "NVT"
                and thermostat in {"BUSSI", "NHC"}
                and temp <= 0
            ):
                self.notify(
                    f"{thermostat} requires a temperature greater than 0 K",
                    severity="error",
                )
                return

            if self.app.get_config("ensemble") == "NVT" and thermostat == "LANGEVIN":
                try:
                    friction = float(self.query_one("#friction-input", Input).value)
                except ValueError:
                    self.notify("Langevin friction must be a number", severity="error")
                    return
                if not math.isfinite(friction) or friction <= 0:
                    self.notify(
                        "Langevin friction must be a finite value > 0 fs^-1",
                        severity="error",
                    )
                    return
                self.app.update_config("friction", friction)
            elif self.app.get_config("ensemble") == "NVT" and thermostat == "BUSSI":
                try:
                    bussi_tau = float(self.query_one("#bussi-tau-input", Input).value)
                except ValueError:
                    self.notify(
                        "Bussi coupling time must be a number", severity="error"
                    )
                    return
                if not math.isfinite(bussi_tau) or bussi_tau <= 0:
                    self.notify(
                        "Bussi coupling time must be a finite value > 0 fs",
                        severity="error",
                    )
                    return
                self.app.update_config("bussi_tau", bussi_tau)
            elif self.app.get_config("ensemble") == "NVT" and thermostat == "NHC":
                try:
                    nhc_tdamp = float(self.query_one("#nhc-tdamp-input", Input).value)
                except ValueError:
                    self.notify(
                        "NHC thermostat time must be a number", severity="error"
                    )
                    return
                if not math.isfinite(nhc_tdamp) or nhc_tdamp <= 0:
                    self.notify(
                        "NHC thermostat time must be a finite value > 0 fs",
                        severity="error",
                    )
                    return
                try:
                    nhc_tchain = int(self.query_one("#nhc-tchain-input", Input).value)
                except ValueError:
                    self.notify("NHC chain length must be an integer", severity="error")
                    return
                try:
                    nhc_tloop = int(self.query_one("#nhc-tloop-input", Input).value)
                except ValueError:
                    self.notify(
                        "NHC thermostat substeps must be an integer", severity="error"
                    )
                    return
                if nhc_tchain < 1:
                    self.notify("NHC chain length must be >= 1", severity="error")
                    return
                if nhc_tloop < 1:
                    self.notify(
                        "NHC thermostat substeps must be >= 1", severity="error"
                    )
                    return
                self.app.update_config("nhc_tdamp", nhc_tdamp)
                self.app.update_config("nhc_tchain", nhc_tchain)
                self.app.update_config("nhc_tloop", nhc_tloop)

            pre_relax = self.query_one("#pre-relax", Switch)
            self.app.update_config("pre_relax", pre_relax.value)

            try:
                pre_relax_steps = int(
                    self.query_one("#pre-relax-steps-input", Input).value
                )
            except ValueError:
                self.notify(
                    "Pre-relaxation max steps must be an integer", severity="error"
                )
                return
            if pre_relax_steps < 0:
                self.notify("Pre-relaxation max steps must be >= 0", severity="error")
                return
            self.app.update_config("pre_relax_steps", pre_relax_steps)

            try:
                pre_relax_fmax = float(
                    self.query_one("#pre-relax-fmax-input", Input).value
                )
            except ValueError:
                self.notify(
                    "Pre-relaxation force threshold must be a number",
                    severity="error",
                )
                return
            if not math.isfinite(pre_relax_fmax) or pre_relax_fmax <= 0:
                self.notify(
                    "Pre-relaxation force threshold must be a finite value > 0",
                    severity="error",
                )
                return
            self.app.update_config("pre_relax_fmax", pre_relax_fmax)

            seed_text = self.query_one("#seed-input", Input).value.strip()
            if seed_text:
                try:
                    seed = int(seed_text)
                except ValueError:
                    self.notify(
                        "Random seed must be a non-negative integer",
                        severity="error",
                    )
                    return
                if seed < 0:
                    self.notify(
                        "Random seed must be a non-negative integer",
                        severity="error",
                    )
                    return
                self.app.update_config("seed", seed)
            else:
                self.app.update_config("seed", None)

            velocity_select = self.query_one("#velocity-policy-select", Select)
            self.app.update_config("velocity_policy", str(velocity_select.value))

            com_select = self.query_one("#com-policy-select", Select)
            com_policy = str(com_select.value)
            ensemble = str(self.app.get_config("ensemble", "NVT")).upper()
            if thermostat == "NHC" and ensemble == "NVT" and com_policy != "none":
                self.notify(
                    "NHC requires COM policy 'none' and an unconstrained structure",
                    severity="error",
                )
                return
            if com_policy == "initialize_only" and (
                ensemble == "NVE" or thermostat in {"BUSSI", "NHC"}
            ):
                self.notify(
                    "Initialization-only COM removal is supported only by "
                    "NVT Langevin",
                    severity="error",
                )
                return
            self.app.update_config("com_policy", com_policy)

            try:
                fmax_abort = float(self.query_one("#fmax-abort-input", Input).value)
            except ValueError:
                self.notify(
                    "Force-safety abort threshold must be a number",
                    severity="error",
                )
                return
            if not math.isfinite(fmax_abort) or fmax_abort <= 0:
                self.notify(
                    "Force-safety abort threshold must be a finite value > 0",
                    severity="error",
                )
                return
            self.app.update_config("fmax_abort", fmax_abort)

        if calc_type == "neb":
            self.app.update_config(
                "neb_interpolation",
                str(self.query_one("#neb-interpolation-select", Select).value),
            )
            self.app.update_config(
                "climb", self.query_one("#neb-climb-switch", Switch).value
            )
            self.app.update_config(
                "path_convention",
                str(self.query_one("#neb-path-convention-select", Select).value),
            )
            self.app.update_config(
                "endpoint_policy",
                str(self.query_one("#endpoint-policy-select", Select).value),
            )
            self.app.update_config(
                "idpp_mic", self.query_one("#idpp-mic-switch", Switch).value
            )
            self.app.update_config(
                "allow_unvalidated_neb",
                self.query_one("#allow-unvalidated-neb-switch", Switch).value,
            )

            int_fields = (
                ("n_intermediate_images", 1),
                ("max_steps", 1),
                ("neb_pre_max_steps", 0),
                ("endpoint_steps", 0),
                ("idpp_steps", 0),
                ("checkpoint_interval", 1),
            )
            for key, minimum in int_fields:
                text = self.query_one(f"#{key}-input", Input).value.strip()
                if not text:
                    continue
                try:
                    parsed = int(text)
                except ValueError:
                    self.notify(
                        f"{key.replace('_', ' ').capitalize()} must be an integer",
                        severity="error",
                    )
                    return
                if parsed < minimum:
                    self.notify(
                        f"{key.replace('_', ' ').capitalize()} must be >= {minimum}",
                        severity="error",
                    )
                    return
                self.app.update_config(key, parsed)

            float_fields = (
                "neb_spring",
                "neb_min_distance",
                "fmax",
                "neb_pre_fmax",
                "neb_maxstep",
                "endpoint_fmax",
                "idpp_fmax",
                "fmax_abort",
            )
            for key in float_fields:
                text = self.query_one(f"#{key}-input", Input).value.strip()
                if not text:
                    continue
                try:
                    parsed = float(text)
                except ValueError:
                    self.notify(
                        f"{key.replace('_', ' ').capitalize()} must be a number",
                        severity="error",
                    )
                    return
                if not math.isfinite(parsed) or parsed <= 0:
                    self.notify(
                        f"{key.replace('_', ' ').capitalize()} must be finite and > 0",
                        severity="error",
                    )
                    return
                self.app.update_config(key, parsed)

            if resume_mode:
                # The checkpoint owns the endpoints and mapping; the inputs
                # above stay in the form but are not forwarded.
                self.app.update_config("neb_final", None)
                self.app.update_config("neb_atom_map", None)
                self.app.update_config("neb_image_shifts", None)
            else:
                final_text = self.query_one(
                    "#final-structure-input", Input
                ).value.strip()
                if not final_text:
                    self.notify(
                        "Please specify a final endpoint file", severity="error"
                    )
                    return
                final_path = Path(final_text).resolve()
                if not final_path.exists():
                    self.notify(
                        f"Final endpoint not found: {final_path}", severity="error"
                    )
                    return
                atom_map, map_error = self._parse_neb_atom_map_text()
                if map_error:
                    self.notify(map_error, severity="error")
                    return
                image_shifts, shifts_error = self._parse_neb_image_shifts_text()
                if shifts_error:
                    self.notify(shifts_error, severity="error")
                    return
                try:
                    initial_atoms = ase_read(structure_path)
                    ase_read(final_path)  # fail closed if unreadable
                except Exception as exc:
                    self.notify(f"Could not read endpoints: {exc}", severity="error")
                    return
                if atom_map is not None and (
                    len(atom_map) != len(initial_atoms)
                    or sorted(atom_map) != list(range(len(initial_atoms)))
                ):
                    self.notify(
                        "Atom map must be a 0-based permutation of the initial "
                        "atoms (one final index per initial atom)",
                        severity="error",
                    )
                    return
                if image_shifts is not None and len(image_shifts) != len(initial_atoms):
                    self.notify(
                        "Image shifts need one 3-component row per initial atom",
                        severity="error",
                    )
                    return
                path_convention = str(self.app.get_config("path_convention", "mic"))
                if image_shifts is not None and path_convention != "unwrapped":
                    self.notify(
                        "Image shifts require path convention 'unwrapped'",
                        severity="error",
                    )
                    return
                self.app.update_config("neb_final", str(final_path))
                self.app.update_config("neb_atom_map", atom_map)
                self.app.update_config("neb_image_shifts", image_shifts)
        # TUI calculations are launched as persistent background processes.
        self.app.update_config("detach", True)
        self.app.update_config("run_started_at", run_started_at)

        # Go to run screen. Push a fresh instance so on_mount/on_compose re-run
        # for this run's config (the named "run" screen is cached by Textual).
        self.app.push_screen(RunScreen())

    def action_back(self) -> None:
        """Go back to main screen."""
        self.app.pop_screen()

    def action_scroll_up(self) -> None:
        """Scroll up in the config panel."""
        scroll = self.query_one("#config-scroll", VerticalScroll)
        scroll.scroll_up()

    def action_scroll_down(self) -> None:
        """Scroll down in the config panel."""
        scroll = self.query_one("#config-scroll", VerticalScroll)
        scroll.scroll_down()

    def action_page_up(self) -> None:
        """Page up in the config panel."""
        scroll = self.query_one("#config-scroll", VerticalScroll)
        scroll.scroll_page_up()

    def action_page_down(self) -> None:
        """Page down in the config panel."""
        scroll = self.query_one("#config-scroll", VerticalScroll)
        scroll.scroll_page_down()
