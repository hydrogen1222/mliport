"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Main TUI Application for mliport.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Footer, Header, Static

if TYPE_CHECKING:
    from typing import Any, ClassVar

# Screen imports - done here to avoid circular imports
from mliport.tui.analysis_screen import AnalysisScreen
from mliport.tui.config_screen import ConfigScreen
from mliport.tui.jobs_screen import JobsScreen
from mliport.tui.main_screen import MainScreen, TemplateScreen
from mliport.tui.run_screen import RunScreen


class MliportApp(App):
    """mliport TUI Application.

    Provides an interactive terminal interface for running MLIP calculations.

    Example:
        >>> from mliport.tui import MliportApp
        >>> app = MliportApp()
        >>> app.run()
    """

    CSS = """
    #main-container {
        width: 100%;
        height: auto;
        border: solid $primary;
        padding: 1;
    }

    MainScreen {
        align: center middle;
    }

    #title {
        text-align: center;
        text-style: bold;
        color: $accent;
        height: auto;
    }

    #subtitle {
        text-align: center;
        color: $text-muted;
        height: auto;
    }

    /* Config Screen Styles */
    #config-main {
        width: 100%;
        height: 100%;
        layout: vertical;
    }

    #config-header {
        height: auto;
        padding: 1;
        border-bottom: solid $primary;
    }

    #config-scroll {
        width: 100%;
        height: 1fr;
        padding: 1 2;
    }

    .section-title {
        text-style: bold;
        color: $accent;
        margin-top: 1;
        margin-bottom: 1;
    }

    Input {
        margin-bottom: 1;
    }

    Label {
        margin-top: 1;
    }

    Select {
        margin-bottom: 1;
    }

    RadioSet {
        margin-bottom: 1;
    }

    Switch {
        margin-left: 2;
    }

    #button-bar {
        height: auto;
        padding: 1;
        border-top: solid $primary;
        align: center middle;
    }

    #button-bar Button {
        margin: 0 1;
    }

    /* Run Screen */
    #run-container {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #progress-bar {
        width: 100%;
        height: 3;
    }

    #status-text {
        text-align: center;
        height: auto;
    }

    #run-log {
        width: 100%;
        height: 1fr;
        border: solid $surface-lighten-1;
        padding: 1;
        overflow-y: auto;
    }

    /* General */
    Button {
        width: auto;
    }

    Button.success {
        background: $success;
    }

    Button.error {
        background: $error;
    }
    """

    BINDINGS: ClassVar[list] = [
        Binding("q", "quit", "Quit", show=True),
        Binding("escape", "back", "Back", show=True),
        Binding("f1", "help", "Help", show=True),
    ]

    def __init__(self, **kwargs):
        """Initialize TUI application."""
        super().__init__(**kwargs)

        # Shared configuration state
        self.config: dict[str, Any] = {
            "calc_type": "sp",  # sp, opt, md
            "structure_file": None,
            "model_file": None,
            "model_type": "uma",
            "task": "omat",
            "device": "cpu",
            # Molecular electronic state. None preserves structure metadata;
            # UMA omol then falls back to charge=0 / spin multiplicity=1.
            "charge": None,
            "spin": None,
            # Backend/resource options
            "inference_mode": "default",
            "torch_num_threads": None,
            "activation_checkpointing": None,
            "gpu_memory_growth": True,
            "gpu_memory_limit_mb": None,
            "neighbor_cache": True,
            "neighbor_skin": 1.5,
            "default_dtype": "float64",
            "head": None,
            "output_dir": "./results",
            "job_name": None,
            # OPT options
            "fmax": 0.05,
            "max_steps": 500,
            "optimizer": "FIRE",
            "cell_opt": False,
            "fix_symmetry": False,
            # MD options
            "ensemble": "NVT",
            "temperature": 300.0,
            "timestep": 1.0,
            "steps": 1000,
            "equilibration_steps": 0,
            "thermostat": "LANGEVIN",
            "friction": 0.001,
            "bussi_tau": 1000.0,
            "nhc_tdamp": 100.0,
            "nhc_tchain": 3,
            "nhc_tloop": 1,
            "save_interval": 10,
            # ``pre_relax`` has NO TUI default: its default is ensemble-aware
            # (off for NVE, on for NVT) and is applied by the engine, mirroring
            # the resolver (config/defaults.py). A blanket True here would make
            # TUI-submitted NVE runs silently pre-relax while CLI/API runs do
            # not -- an interface-equivalence violation. ``None`` means "let
            # the engine decide"; the config screen derives the display value.
            "pre_relax": None,
            "pre_relax_steps": 50,
            "pre_relax_fmax": 0.1,
            "seed": None,
            # ``velocity_policy`` has NO TUI default either: the MDRunner
            # default is "auto" and the TUI must not turn an implicit
            # default into an explicit override (interface equivalence).
            "velocity_policy": None,
            "com_policy": "auto",
            "fmax_abort": 20.0,
            # Batch options
            "pattern": "*.cif",
        }

        # Current screen tracking (note: don't use screen_stack - it's a Textual property)
        self.current_screen_name: str = "main"

    def compose(self) -> ComposeResult:
        """Compose the main UI."""
        yield Header(show_clock=True)
        yield Container(
            Static("mliport - MLIP eXtended", id="title"),
            Static("Interactive Configuration Interface", id="subtitle"),
            id="main-container",
        )
        yield Footer()

    SCREENS: ClassVar[dict] = {
        "main": MainScreen,
        "config": ConfigScreen,
        "run": RunScreen,
        "template": TemplateScreen,
        "jobs": JobsScreen,
        "analysis": AnalysisScreen,
    }

    def on_mount(self) -> None:
        """Handle app mount."""
        self.push_screen("main")

    def action_back(self) -> None:
        """Navigate back to previous screen."""
        # Use Textual's screen stack - pop if more than one screen
        if len(self.screen_stack) > 1:
            self.pop_screen()

    def action_help(self) -> None:
        """Show help screen."""
        self.notify(
            "Navigation:\n"
            "- Arrow keys: Navigate\n"
            "- Enter: Select\n"
            "- ESC: Back\n"
            "- Q: Quit",
            title="Help",
            timeout=10,
        )

    def update_config(self, key: str, value: Any) -> None:
        """Update configuration value."""
        self.config[key] = value

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
