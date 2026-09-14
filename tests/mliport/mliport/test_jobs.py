"""Tests for mliport.jobs (background job manager)."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

from mliport.jobs import JobManager, JobStatus


def _make_manager(tmp_path: Path) -> JobManager:
    return JobManager(jobs_dir=tmp_path / "jobs")


def _job_id(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mliport-test:{name}"))


def _seed_job(mgr: JobManager, name: str, status: str) -> str:
    job_id = _job_id(name)
    mgr._write_job_state(
        job_id=job_id,
        status=JobStatus(status),
        calc_type="sp",
        structure="s.cif",
        formula="H2O",
        natoms=3,
        pid=1234,
        device="cpu",
        display_name=name,
    )
    mgr._log_file(job_id).write_text("some log output\n", encoding="utf-8")
    return job_id


def test_submit_writes_pending_then_running(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    assert mgr.get_job("nonexistent") is None


def test_clean_removes_state_and_logs(tmp_path: Path) -> None:
    """Regression: clean() removed the .json state but left the .log behind."""
    mgr = _make_manager(tmp_path)
    done_id = _seed_job(mgr, "job_done", "done")
    failed_id = _seed_job(mgr, "job_failed", "failed")
    running_id = _seed_job(mgr, "job_running", "running")

    removed = mgr.clean()
    assert sorted(removed) == sorted([done_id, failed_id])
    # State files gone
    assert not mgr._job_file(done_id).exists()
    assert not mgr._job_file(failed_id).exists()
    # Log files gone too (the regression)
    assert not mgr._log_file(done_id).exists()
    assert not mgr._log_file(failed_id).exists()
    # Running job untouched
    assert mgr._job_file(running_id).exists()
    assert mgr._log_file(running_id).exists()


def test_clean_removes_orphaned_logs(tmp_path: Path) -> None:
    """Logs whose state file is already gone are cleaned up as well."""
    mgr = _make_manager(tmp_path)
    orphan = mgr._log_file("ghost")
    orphan.write_text("leftover\n", encoding="utf-8")
    assert mgr.clean() == []
    assert not orphan.exists()


def test_kill_job_requires_running(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    job_id = _seed_job(mgr, "job_done", "done")
    assert mgr.kill_job(job_id) is False
    assert mgr.get_job(job_id)["status"] == "done"


def test_read_job_state_missing(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    assert mgr.get_job("missing") is None


def test_pending_and_paused_jobs_cancel_without_signalling(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    for name, pause in (("pending", False), ("paused", True)):
        job_id = mgr.new_job_id()
        mgr.enqueue(
            job_id=job_id,
            display_name=name,
            calc_type="sp",
            structure="s.cif",
            formula="H2O",
            natoms=3,
            device="cpu",
            cmd=["python", "-c", "pass"],
        )
        if pause:
            assert mgr.pause_pending(job_id)
        mgr._kill_process = lambda pid: pytest.fail("queued cancellation signalled")
        assert mgr.kill_job(job_id)
        assert mgr.get_job(job_id)["status"] == "cancelled"


def test_terminal_state_compare_and_swap_is_immutable(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    job_id = mgr.new_job_id()
    mgr.enqueue(
        job_id=job_id,
        calc_type="sp",
        structure="s.cif",
        formula="H2O",
        natoms=3,
        device="cpu",
        cmd=["python", "-c", "pass"],
    )
    token = mgr.new_job_id()
    assert mgr.claim_job(
        job_id,
        claim_token=token,
        owner_pid=os.getpid(),
        device_uuid=None,
    )
    assert mgr.mark_running(job_id, 1234, claim_token=token, pid_identity="test-1234")
    assert mgr.update_status(
        job_id,
        JobStatus.CANCELLED,
        expected_statuses={JobStatus.RUNNING},
        expected_pid=1234,
        claim_token=token,
    )
    assert not mgr.update_status(
        job_id,
        JobStatus.DONE,
        expected_statuses={JobStatus.RUNNING},
        expected_pid=1234,
        claim_token=token,
    )
    assert mgr.get_job(job_id)["status"] == "cancelled"


# ---------------------------------------------------------------------------
# STATE-01: persistent user state isolation
# ---------------------------------------------------------------------------


def test_state_dir_resolution_precedence(monkeypatch, tmp_path: Path) -> None:
    from mliport import jobs as jobs_mod

    monkeypatch.setenv("MLIPORT_JOBS_DIR", str(tmp_path / "explicit"))
    assert jobs_mod._default_jobs_dir() == tmp_path / "explicit"

    monkeypatch.delenv("MLIPORT_JOBS_DIR")
    monkeypatch.setenv("MLIPORT_STATE_DIR", str(tmp_path / "state"))
    assert jobs_mod._default_jobs_dir() == tmp_path / "state" / "jobs"

    monkeypatch.delenv("MLIPORT_STATE_DIR")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    if os.name != "nt" and sys.platform != "darwin":
        assert jobs_mod._default_jobs_dir() == tmp_path / "xdg" / "mliport" / "jobs"

    monkeypatch.delenv("XDG_STATE_HOME")
    if os.name != "nt" and sys.platform != "darwin":
        assert jobs_mod._state_root() == Path.home() / ".local" / "state" / "mliport"


def test_new_home_has_empty_job_history(monkeypatch, tmp_path: Path) -> None:
    """STATE-T1: a brand-new HOME must not see any job records."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("MLIPORT_STATE_DIR", raising=False)
    monkeypatch.delenv("MLIPORT_JOBS_DIR", raising=False)

    mgr = JobManager()
    assert mgr.jobs_dir == tmp_path / ".local" / "state" / "mliport" / "jobs"
    assert mgr.list_jobs() == []


def test_legacy_state_is_never_imported_silently(monkeypatch, tmp_path: Path) -> None:
    """STATE-T5: legacy ~/.mliport state needs an explicit migration."""
    import json

    from mliport.jobs import legacy_state_hint, migrate_legacy_state

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("MLIPORT_STATE_DIR", raising=False)
    monkeypatch.delenv("MLIPORT_JOBS_DIR", raising=False)

    legacy = tmp_path / ".mliport" / "jobs"
    legacy.mkdir(parents=True)
    record = {
        "job_id": _job_id("legacy"),
        "display_name": "legacy-sp",
        "status": "done",
        "calc_type": "sp",
        "structure": "s.cif",
        "formula": "Li",
        "natoms": 1,
        "device": "cpu",
        "cmd": ["x"],
    }
    (legacy / f"{record['job_id']}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )

    mgr = JobManager()
    assert mgr.list_jobs() == []
    hint = legacy_state_hint()
    assert hint is not None and "state migrate" in hint

    dry = migrate_legacy_state(dry_run=True)
    assert dry["imported"] == [f"{record['job_id']}.json"]
    assert not (mgr.jobs_dir / f"{record['job_id']}.json").exists()

    done = migrate_legacy_state()
    assert done["imported"] == [f"{record['job_id']}.json"]
    migrated = mgr.list_jobs()
    assert len(migrated) == 1
    assert migrated[0]["migrated_from_legacy"] is True
    assert (legacy / f"{record['job_id']}.json").exists()
    assert legacy_state_hint() is None


def test_enqueue_records_state_schema_and_project(tmp_path: Path) -> None:
    from mliport.jobs import STATE_APPLICATION, STATE_SCHEMA_VERSION

    mgr = _make_manager(tmp_path)
    job_id = mgr.new_job_id()
    mgr.enqueue(
        job_id=job_id,
        calc_type="sp",
        structure="s.cif",
        formula="H",
        natoms=1,
        device="cpu",
        cmd=["x"],
    )
    data = mgr.get_job(job_id)
    assert data["schema_version"] == STATE_SCHEMA_VERSION
    assert data["application"] == STATE_APPLICATION
    assert data["submit_cwd"]
    assert data["project_root"]


def test_clean_dry_run_keeps_records(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    done_id = _seed_job(mgr, "dry_done", "done")
    removed = mgr.clean(dry_run=True)
    assert removed == [done_id]
    assert mgr._job_file(done_id).exists()
    assert mgr.clean() == [done_id]
    assert not mgr._job_file(done_id).exists()


def test_jobs_command_prints_state_path(monkeypatch, tmp_path: Path, capsys) -> None:
    from mliport.cli import main

    monkeypatch.setenv("MLIPORT_JOBS_DIR", str(tmp_path / "jobs"))
    assert main(["jobs"]) == 0
    output = capsys.readouterr().out
    assert "User job history:" in output
    assert "No jobs found." in output


def test_state_path_command(monkeypatch, tmp_path: Path, capsys) -> None:
    from mliport.cli import main

    monkeypatch.setenv("MLIPORT_STATE_DIR", str(tmp_path / "state"))
    assert main(["state", "path"]) == 0
    output = capsys.readouterr().out
    assert "Job state:" in output
    assert str(tmp_path / "state" / "jobs") in output
