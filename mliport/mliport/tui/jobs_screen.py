"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Jobs screen for managing background calculations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import Button, DataTable, Input, Log, Static

from mliport.jobs import JobManager

if TYPE_CHECKING:
    from textual.app import ComposeResult


class JobsScreen(Screen):
    """Screen for viewing and managing background jobs."""

    BINDINGS: ClassVar[list] = [
        ("escape", "back", "Back"),
        ("c", "cancel_job", "Cancel Job"),
        ("d", "delete_job", "Delete"),
        ("p", "pause_selected_job", "Pause Selected Job"),
        ("r", "refresh", "Refresh"),
        ("u", "resume_selected_job", "Resume Selected Job"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._job_manager = JobManager()
        self._jobs_refresh_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Container(
            Static("Background Jobs", id="title"),
            Static("Manage running, queued and completed calculations", id="subtitle"),
            Static("", id="sched-status"),
            Horizontal(
                Static("Concurrency:", id="sched-label"),
                Input(
                    value="1",
                    id="max-concurrent-input",
                    placeholder="1",
                ),
                Button("Start Scheduler", id="start-sched-btn"),
                Button("Stop Scheduler", id="stop-sched-btn"),
                Button("Pause Queue", id="pause-queue-btn"),
                Button("Resume Queue", id="resume-queue-btn"),
                id="scheduler-bar",
            ),
            DataTable(id="jobs-table"),
            Horizontal(
                Button("Pause Job", id="pause-job-btn"),
                Button("Resume Job", id="resume-job-btn"),
                Button("Cancel Job", variant="error", id="cancel-job-btn"),
                Button("Delete", id="delete-btn"),
                Button("Refresh", id="refresh-btn"),
                Button("Back", id="back-btn"),
                id="jobs-button-bar",
            ),
            id="jobs-main",
        )

    def on_mount(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        if not table.columns:
            table.add_columns(
                "Name", "Run ID", "Status", "Type", "Formula", "Atoms", "Device"
            )
        self._refresh_scheduler_status()
        self._refresh_table()
        self._jobs_refresh_timer = self.set_interval(
            2.0,
            self._refresh_all,
            name="jobs-auto-refresh",
        )

    def on_unmount(self) -> None:
        if self._jobs_refresh_timer is not None:
            self._jobs_refresh_timer.stop()
            self._jobs_refresh_timer = None

    def _refresh_all(self) -> None:
        """Refresh scheduler status + job table (auto-refresh tick)."""
        self._refresh_scheduler_status()
        self._refresh_table()

    def _refresh_scheduler_status(self) -> None:
        """Show whether the queue scheduler is alive."""
        try:
            from mliport.queue import scheduler_status

            status = scheduler_status(jobs_dir=self._job_manager.jobs_dir)
        except Exception:
            status = {"running": False, "pid": None}
        widget = self.query_one("#sched-status", Static)
        if status["running"] and status.get("paused"):
            widget.update(
                f"[yellow]Ⅱ Queue PAUSED (scheduler PID {status['pid']})[/]"
                " — running jobs continue; pending jobs wait"
            )
        elif status["running"]:
            widget.update(f"[green]● Scheduler RUNNING (PID {status['pid']})[/]")
        else:
            suffix = " — queue is paused" if status.get("paused") else ""
            widget.update(
                "[yellow]○ Scheduler not running[/] — queued jobs wait until "
                f"you start it (or use: mliport queue start){suffix}"
            )

    def _refresh_table(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        selected_job_id: str | None = None
        if table.row_count and table.cursor_row < table.row_count:
            selected_job_id = str(table.get_row_at(table.cursor_row)[1])
        table.clear()
        jobs = self._job_manager.list_jobs()
        status_icons = {
            "running": "●",
            "done": "✓",
            "not_converged": "◐",
            "failed": "✗",
            "cancelled": "⊘",
            "pending": "○",
            "paused": "Ⅱ",
        }
        for job in jobs:
            job_id = str(job.get("job_id", "unknown"))
            status = str(job.get("status", "unknown"))
            icon = status_icons.get(status, "?")
            table.add_row(
                str(job.get("display_name", job_id)),
                job_id,
                f"{icon} {status}",
                job.get("calc_type", ""),
                job.get("formula", ""),
                str(job.get("natoms", "")),
                job.get("device", ""),
                key=job_id,
            )
        if selected_job_id is not None and selected_job_id in table.rows:
            table.move_cursor(row=table.get_row_index(selected_job_id), scroll=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "back-btn":
            self.app.pop_screen()
        elif button_id == "refresh-btn":
            self._refresh_all()
        elif button_id == "start-sched-btn":
            self._start_scheduler()
        elif button_id == "stop-sched-btn":
            self._stop_scheduler()
        elif button_id == "pause-queue-btn":
            self._pause_queue()
        elif button_id == "resume-queue-btn":
            self._resume_queue()
        elif button_id == "pause-job-btn":
            self._pause_selected_job()
        elif button_id == "resume-job-btn":
            self._resume_selected_job()
        elif button_id == "cancel-job-btn":
            self._cancel_selected_job()
        elif button_id == "delete-btn":
            self._delete_selected_job()

    def _start_scheduler(self) -> None:
        """Launch the background queue scheduler (concurrency from the input)."""
        from mliport.queue import start_scheduler

        try:
            raw = self.query_one("#max-concurrent-input", Input).value.strip()
            max_concurrent = int(raw) if raw else 1
        except ValueError:
            self.app.notify(
                "Concurrency must be a positive integer",
                title="Error",
                severity="error",
            )
            return
        if max_concurrent < 1:
            self.app.notify(
                "Concurrency must be at least 1",
                title="Error",
                severity="error",
            )
            return
        try:
            pid = start_scheduler(
                jobs_dir=self._job_manager.jobs_dir,
                max_concurrent=max_concurrent,
            )
        except Exception as exc:
            self.app.notify(f"Could not start scheduler: {exc}", severity="error")
            return
        self.app.notify(
            f"Scheduler started (PID {pid}, max_concurrent={max_concurrent})",
            title="Scheduler",
        )
        self._refresh_scheduler_status()

    def _stop_scheduler(self) -> None:
        """Stop the background queue scheduler."""
        from mliport.queue import stop_scheduler

        stopped = stop_scheduler(jobs_dir=self._job_manager.jobs_dir)
        if stopped:
            self.app.notify("Scheduler stopped.", title="Scheduler")
        else:
            self.app.notify("No scheduler was running.", title="Scheduler")
        self._refresh_scheduler_status()

    def _pause_queue(self) -> None:
        """Pause only pending-job dispatch; do not touch running workers."""
        from mliport.queue import pause_scheduler

        if pause_scheduler(jobs_dir=self._job_manager.jobs_dir):
            self.app.notify(
                "Pending jobs paused; running jobs continue.", title="Queue"
            )
        else:
            self.app.notify("Queue is already paused.", title="Queue")
        self._refresh_scheduler_status()

    def _resume_queue(self) -> None:
        """Allow the scheduler to launch pending jobs again."""
        from mliport.queue import resume_scheduler

        if resume_scheduler(jobs_dir=self._job_manager.jobs_dir):
            self.app.notify("Pending jobs can now be launched.", title="Queue")
        else:
            self.app.notify("Queue was not paused.", title="Queue")
        self._refresh_scheduler_status()

    def _selected_job_id(self) -> str | None:
        """Return the job ID under the table cursor, if any."""
        table = self.query_one("#jobs-table", DataTable)
        if table.cursor_row is None or table.cursor_row >= table.row_count:
            return None
        row = table.get_row_at(table.cursor_row)
        return str(row[1]) if row else None

    def _pause_selected_job(self) -> None:
        """Pause only the selected PENDING job."""
        from mliport.queue import pause_pending_job

        job_id = self._selected_job_id()
        if job_id is None:
            self.app.notify("Select a pending job first.", title="Job")
            return
        job = self._job_manager.get_job(job_id)
        if not job or job.get("status") != "pending":
            self.app.notify(
                f"Only pending jobs can be paused: {job_id}",
                title="Job",
                severity="error",
            )
            return
        if pause_pending_job(self._job_manager.jobs_dir, job_id):
            self.app.notify(
                f"Paused {job_id}; other pending jobs are unchanged.", title="Job"
            )
            self._refresh_table()

    def _resume_selected_job(self) -> None:
        """Resume only the selected PAUSED job."""
        from mliport.queue import resume_paused_job

        job_id = self._selected_job_id()
        if job_id is None:
            self.app.notify("Select a paused job first.", title="Job")
            return
        job = self._job_manager.get_job(job_id)
        if not job or job.get("status") != "paused":
            self.app.notify(
                f"Only paused jobs can be resumed: {job_id}",
                title="Job",
                severity="error",
            )
            return
        if resume_paused_job(self._job_manager.jobs_dir, job_id):
            self.app.notify(f"Resumed {job_id}; it is pending again.", title="Job")
            self._refresh_table()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Show job detail/log on selection."""
        row_key = event.row_key
        if row_key is not None:
            job_id = str(event.row_key.value)
            log_text = self._job_manager.tail_log(job_id, lines=200)
            self.app.push_screen(JobDetailScreen(job_id, log_text))

    def _cancel_selected_job(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        if table.cursor_row is not None and table.cursor_row < table.row_count:
            job_id = self._selected_job_id()
            if job_id is None:
                return
            ok = self._job_manager.kill_job(job_id)
            if ok:
                self.app.notify(f"Cancelled job: {job_id}", title="OK")
            else:
                self.app.notify(
                    f"Failed to cancel: {job_id}", title="Error", severity="error"
                )
            self._refresh_table()

    def _delete_selected_job(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        if table.cursor_row is not None and table.cursor_row < table.row_count:
            job_id = self._selected_job_id()
            if job_id is not None and self._job_manager.delete_job(job_id):
                self._refresh_table()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_cancel_job(self) -> None:
        self._cancel_selected_job()

    def action_delete_job(self) -> None:
        self._delete_selected_job()

    def action_refresh(self) -> None:
        self._refresh_table()

    def action_pause_queue(self) -> None:
        self._pause_queue()

    def action_resume_queue(self) -> None:
        self._resume_queue()

    def action_pause_selected_job(self) -> None:
        self._pause_selected_job()

    def action_resume_selected_job(self) -> None:
        self._resume_selected_job()


class JobDetailScreen(Screen):
    """Screen for viewing job log output."""

    BINDINGS: ClassVar[list] = [("escape", "back", "Back")]

    def __init__(self, job_id: str, log_text: str, **kwargs):
        super().__init__(**kwargs)
        self.job_id = job_id
        self.log_text = log_text

    def compose(self) -> ComposeResult:
        yield Container(
            Static(f"Job: {self.job_id}", id="title"),
            Log(id="job-detail-log"),
            Button("Back", id="back-btn"),
            id="job-detail-container",
        )

    def on_mount(self) -> None:
        log = self.query_one("#job-detail-log", Log)
        text = self.log_text or "(no log output)"
        log.write("\n".join(text.splitlines()))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()

    def action_back(self) -> None:
        self.app.pop_screen()
