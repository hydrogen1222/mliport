# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Main screen for mliport TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import (
    Button,
    ListItem,
    ListView,
    Static,
)

from mliport.tui.config_screen import ConfigScreen

if TYPE_CHECKING:
    from textual.app import ComposeResult


class MainScreen(Screen):
    """Main menu screen for selecting calculation type."""

    def compose(self) -> ComposeResult:
        """Compose the main screen."""
        yield Container(
            Static("Select Calculation Type", id="title"),
            Static("Choose the type of calculation to run", id="subtitle"),
            ListView(
                ListItem(
                    Static(
                        "📊 Single Point (SP)\n   Calculate energy, forces, and stress"
                    ),
                    id="sp",
                ),
                ListItem(
                    Static(
                        "🔧 Geometry Optimization (OPT)\n   Optimize atomic positions"
                    ),
                    id="opt",
                ),
                ListItem(
                    Static("🌡️  Molecular Dynamics (MD)\n   Run NVT/NVE simulations"),
                    id="md",
                ),
                ListItem(
                    Static(
                        "🛤️  Nudged Elastic Band (NEB)\n   "
                        "Find minimum energy paths and barriers"
                    ),
                    id="neb",
                ),
                ListItem(
                    Static(
                        "📈 Analyze Existing Run\n   Validated structure and ion transport"
                    ),
                    id="analysis",
                ),
                ListItem(
                    Static("💼 Background Jobs\n   View/manage running calculations"),
                    id="jobs",
                ),
                ListItem(
                    Static("📝 Generate Template\n   Create INCAR template file"),
                    id="template",
                ),
                ListItem(Static("❌ Exit\n   Quit the application"), id="exit"),
                id="calc-type-list",
            ),
            id="main-container",
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle selection from list view."""
        item_id = event.item.id

        if item_id == "exit":
            self.app.exit()
            return

        if item_id == "template":
            self.app.push_screen("template")
            return

        if item_id == "jobs":
            self.app.push_screen("jobs")
            return

        if item_id == "analysis":
            self.app.push_screen("analysis")
            return

        # Update config and go to config screen.
        # Push a fresh ConfigScreen instance each time (rather than the cached
        # named screen) so compose() re-runs and renders the inputs matching the
        # newly selected calc_type. The named screen in SCREENS is cached after
        # first use, which left stale opt inputs when switching to md, etc.
        self.app.update_config("calc_type", item_id)
        self.app.push_screen(ConfigScreen())


class TemplateScreen(Screen):
    """Template generation screen."""

    def compose(self) -> ComposeResult:
        """Compose the template screen."""
        yield Container(
            Static("Generate INCAR Template", id="title"),
            Static("Select template type:", id="subtitle"),
            ListView(
                ListItem(Static("Single Point (SP)"), id="sp"),
                ListItem(Static("Nudged Elastic Band (NEB)"), id="neb"),
                ListItem(Static("Geometry Optimization (OPT)"), id="opt"),
                ListItem(Static("Molecular Dynamics (MD)"), id="md"),
                id="template-list",
            ),
            Horizontal(
                Button("◀ Back", id="back-btn"),
                Button("Generate", variant="success", id="generate-btn"),
            ),
            id="main-container",
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle template type selection."""
        self.selected_type = event.item.id

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "back-btn":
            self.app.pop_screen()

        elif button_id == "generate-btn":
            self._generate_template()

    def _generate_template(self) -> None:
        """Write a backend-neutral INCAR template from the single defaults source.

        The template deliberately selects no backend (``MODEL_TYPE =
        REQUIRED``); a template that silently defaults to one model would
        contradict mliport's explicit backend-selection contract.
        """
        from mliport.config.defaults import build_incar_default  # noqa: PLC0415

        template_type = getattr(self, "selected_type", "sp")
        if template_type not in {"sp", "opt", "md", "neb"}:
            template_type = "sp"
        filename = f"INCAR.{template_type}.template"
        try:
            with open(filename, "w", encoding="utf-8") as handle:
                handle.write(build_incar_default(template_type))
            self.app.notify(f"Template saved to {filename}", title="Success")
        except Exception as e:
            self.app.notify(f"Error: {e}", title="Error", severity="error")
