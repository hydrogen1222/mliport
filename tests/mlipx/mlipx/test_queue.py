"""Tests for the Slurm-like job queue (queue.py + JobManager queue methods)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from mlipx.jobs import JobManager
from mlipx.queue import (
    QueueScheduler,
    build_mlipx_command,
    parse_task_file,
    pause_pending_job,
    pause_scheduler,
    queue_paused,
    resume_paused_job,
    resume_scheduler,
    scheduler_pid_file,
    scheduler_status,
    start_scheduler,
    stop_scheduler,
    submit_task_file,
)

# ---------------------------------------------------------------------------
# build_mlipx_command
# ---------------------------------------------------------------------------


def test_build_command_base() -> None:
    cmd = build_mlipx_command(
        "sp",
        "/s/a.cif",
        "/m/m.pt",
        model_type="uma",
        task="omat",
        device="cuda:0",
        output_dir="/o",
        job_name="j1",
        python="/venv/bin/python",
    )
    assert cmd[0] == "/venv/bin/python"
    assert cmd[:3] == ["/venv/bin/python", "-m", "mlipx.cli"]
    assert "sp" in cmd and "/s/a.cif" in cmd
    assert "--model" in cmd and "/m/m.pt" in cmd
    assert "--task" in cmd and "omat" in cmd
    assert "--name" in cmd and "j1" in cmd


def test_build_command_opt_options() -> None:
    cmd = build_mlipx_command(
        "opt",
        "s.cif",
        "m.pt",
        options={
            "fmax": 0.02,
            "max_steps": 100,
            "optimizer": "BFGS",
            "cell_opt": True,
            "fix_symmetry": False,
        },
    )
    assert "--fmax" in cmd and "0.02" in cmd
    assert "--max-steps" in cmd and "100" in cmd
    assert "--optimizer" in cmd and "BFGS" in cmd
    assert "--cell-opt" in cmd and "--no-cell-opt" not in cmd
    assert "--no-fix-symmetry" in cmd


def test_build_command_md_options() -> None:
    cmd = build_mlipx_command(
        "md",
        "s.cif",
        "m.pt",
        options={
            "ensemble": "NVT",
            "temperature": 500.0,
            "steps": 100,
            "thermostat": "NHC",
            "nhc_tdamp": 150.0,
            "nhc_tchain": 4,
            "nhc_tloop": 2,
            "pre_relax": False,
            "seed": 42,
        },
    )
    assert "--temp" in cmd and "500.0" in cmd
    assert "--steps" in cmd and "100" in cmd
    assert "--thermostat" in cmd and "NHC" in cmd
    assert "--nhc-tdamp" in cmd and "150.0" in cmd
    assert "--nhc-tchain" in cmd and "4" in cmd
    assert "--nhc-tloop" in cmd and "2" in cmd
    assert "--no-pre-relax" in cmd
    assert "--seed" in cmd and "42" in cmd


def test_build_command_forwards_molecular_electronic_state() -> None:
    cmd = build_mlipx_command(
        "sp",
        "molecule.xyz",
        "uma.pt",
        model_type="uma",
        task="omol",
        options={"charge": -1, "spin": 2},
    )
    assert cmd[cmd.index("--charge") + 1] == "-1"
    assert cmd[cmd.index("--spin") + 1] == "2"


def test_build_command_engine_option_isolation() -> None:
    """UMA-only options must not leak into a GRACE task, MACE dtype must not
    leak into UMA, etc."""
    grace = build_mlipx_command(
        "md",
        "s.cif",
        "m",
        model_type="grace",
        options={
            "inference_mode": "turbo",
            "activation_checkpointing": True,
            "torch_num_threads": 4,
            "head": "x",
            "default_dtype": "float64",
            "gpu_memory_growth": True,
            "gpu_memory_limit_mb": 6144,
        },
    )
    assert "--inference-mode" not in grace
    assert "--activation-checkpointing" not in grace
    assert "--head" not in grace and "--dtype" not in grace
    assert "--cpu-threads" in grace and "4" in grace  # threads apply to all
    assert "--gpu-memory-growth" in grace
    assert grace[grace.index("--gpu-memory-limit-mb") + 1] == "6144"

    mace = build_mlipx_command(
        "sp",
        "s.cif",
        "m",
        model_type="mace",
        options={"default_dtype": "float64", "head": "h1", "inference_mode": "turbo"},
    )
    assert "--dtype" in mace and "float64" in mace
    assert "--head" in mace and "h1" in mace
    assert "--inference-mode" not in mace
    assert "--gpu-memory-limit-mb" not in mace

    dpa = build_mlipx_command(
        "sp", "s.cif", "m", model_type="dpa", options={"head": "branch1"}
    )
    assert "--head" in dpa and "branch1" in dpa


# ---------------------------------------------------------------------------
# parse_task_file
# ---------------------------------------------------------------------------


def _write_tasks(tmp_path: Path, tasks: list[dict], max_conc: int = 1) -> Path:
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps({"max_concurrent": max_conc, "tasks": tasks}),
        encoding="utf-8",
    )
    return path


def _sample_task(tmp_path: Path, name: str = "t1", **overrides) -> dict:
    struct = tmp_path / "s.cif"
    struct.write_text("dummy", encoding="utf-8")
    model = tmp_path / "m.pt"
    model.write_text("dummy", encoding="utf-8")
    task = {
        "name": name,
        "calc_type": "opt",
        "structure": str(struct),
        "model": str(model),
        "model_type": "uma",
        "device": "cuda:0",
    }
    task.update(overrides)
    return task


def test_parse_task_file_ok(tmp_path: Path) -> None:
    path = _write_tasks(tmp_path, [_sample_task(tmp_path)])
    parsed = parse_task_file(path)
    assert parsed["max_concurrent"] == 1
    task = parsed["tasks"][0]
    assert task["calc_type"] == "opt"
    assert task["model_type"] == "uma"
    assert task["device"] == "cuda:0"


def test_task_paths_are_frozen_relative_to_task_file(tmp_path: Path) -> None:
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir()
    (spec_dir / "inputs").mkdir()
    (spec_dir / "models").mkdir()
    (spec_dir / "bin").mkdir()
    (spec_dir / "inputs/s.cif").write_text("dummy", encoding="utf-8")
    (spec_dir / "models/m.pt").write_text("dummy", encoding="utf-8")
    python = spec_dir / "bin/python"
    python.write_text("", encoding="utf-8")
    path = _write_tasks(
        spec_dir,
        [
            {
                "calc_type": "md",
                "structure": "inputs/s.cif",
                "model": "models/m.pt",
                "python": "bin/python",
                "output_dir": "outputs/run",
                "options": {
                    "temperature": "3D2",
                    "steps": "1",
                    "pre_relax": "0",
                },
            }
        ],
    )

    task = parse_task_file(path)["tasks"][0]

    assert task["structure"] == str((spec_dir / "inputs/s.cif").resolve())
    assert task["model"] == str((spec_dir / "models/m.pt").resolve())
    assert task["python"] == str(python.resolve())
    assert task["output_dir"] == str((spec_dir / "outputs/run").resolve())
    assert task["options"] == {
        "temperature": 300.0,
        "steps": 1,
        "pre_relax": False,
    }


def test_parse_task_file_max_concurrent(tmp_path: Path) -> None:
    path = _write_tasks(tmp_path, [_sample_task(tmp_path)], max_conc=2)
    assert parse_task_file(path)["max_concurrent"] == 2
    bad = _write_tasks(tmp_path, [_sample_task(tmp_path)], max_conc=0)
    with pytest.raises(ValueError, match="max_concurrent"):
        parse_task_file(bad)
    bad2 = _write_tasks(tmp_path, [_sample_task(tmp_path)], max_conc="x")
    with pytest.raises(ValueError, match="max_concurrent"):
        parse_task_file(bad2)


@pytest.mark.parametrize("value", [True, 1.5, "1.5"])
def test_parse_task_file_rejects_non_integer_concurrency(
    tmp_path: Path, value: object
) -> None:
    path = _write_tasks(tmp_path, [_sample_task(tmp_path)])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["max_concurrent"] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="max_concurrent"):
        parse_task_file(path)


def test_parse_task_file_rejects_unknown_options(tmp_path: Path) -> None:
    path = _write_tasks(
        tmp_path,
        [_sample_task(tmp_path, options={"TEMPRATURE": 999})],
    )
    with pytest.raises(ValueError, match="TEMPRATURE|temprature"):
        parse_task_file(path)


def test_parse_task_file_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        parse_task_file(tmp_path / "nope.json")


def test_parse_task_file_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_task_file(path)


def test_parse_task_file_bad_calc_type(tmp_path: Path) -> None:
    path = _write_tasks(tmp_path, [_sample_task(tmp_path, calc_type="batch")])
    with pytest.raises(ValueError, match="calc_type"):
        parse_task_file(path)


def test_parse_task_file_missing_structure(tmp_path: Path) -> None:
    task = _sample_task(tmp_path)
    task["structure"] = str(tmp_path / "gone.cif")
    path = _write_tasks(tmp_path, [task])
    with pytest.raises(ValueError, match="structure"):
        parse_task_file(path)


def test_parse_task_file_duplicate_names(tmp_path: Path) -> None:
    path = _write_tasks(
        tmp_path,
        [_sample_task(tmp_path, name="dup"), _sample_task(tmp_path, name="dup")],
    )
    with pytest.raises(ValueError, match="duplicate"):
        parse_task_file(path)


def test_parse_task_file_bad_python(tmp_path: Path) -> None:
    path = _write_tasks(tmp_path, [_sample_task(tmp_path, python="/no/such/python")])
    with pytest.raises(ValueError, match="python"):
        parse_task_file(path)


def test_parse_task_file_options_reject_structural_keys(tmp_path: Path) -> None:
    path = _write_tasks(tmp_path, [_sample_task(tmp_path, options={"calc_type": "md"})])
    with pytest.raises(ValueError, match="structural"):
        parse_task_file(path)


# ---------------------------------------------------------------------------
# JobManager queue methods
# ---------------------------------------------------------------------------


@pytest.fixture()
def mgr(tmp_path: Path) -> JobManager:
    return JobManager(jobs_dir=tmp_path / "jobs")


def _enqueue(mgr: JobManager, job_id: str, device: str = "cpu") -> None:
    mgr.enqueue(
        job_id=job_id,
        calc_type="sp",
        structure="s.cif",
        formula="H2O",
        natoms=3,
        device=device,
        cmd=[sys.executable, "-c", "pass"],
    )


def test_enqueue_writes_pending(mgr: JobManager) -> None:
    _enqueue(mgr, "job1")
    data = mgr.get_job("job1")
    assert data["status"] == "pending"
    assert data["cmd"] == [sys.executable, "-c", "pass"]
    assert data["pid"] == 0


def test_next_pending_fifo(mgr: JobManager) -> None:
    _enqueue(mgr, "first")
    time.sleep(0.01)
    _enqueue(mgr, "second")
    assert mgr.next_pending()["job_id"] == "first"
    mgr.mark_running("first", 999)
    assert mgr.next_pending()["job_id"] == "second"
    mgr.mark_running("second", 1000)
    assert mgr.next_pending() is None


def test_count_by_status(mgr: JobManager) -> None:
    _enqueue(mgr, "a")
    _enqueue(mgr, "b")
    mgr.mark_running("a", 1)
    assert mgr.count_by_status("pending") == 1
    assert mgr.count_by_status("running") == 1


def test_queue_summary(mgr: JobManager) -> None:
    _enqueue(mgr, "a")
    _enqueue(mgr, "b")
    mgr.mark_running("a", 1)
    summary = mgr.queue_summary()
    assert summary["pending"] == 1
    assert summary["running"] == 1


def test_mark_running_missing_job(mgr: JobManager) -> None:
    assert mgr.mark_running("ghost", 1) is False


# ---------------------------------------------------------------------------
# QueueScheduler
# ---------------------------------------------------------------------------


def _queued_cmd(
    mgr: JobManager, job_id: str, marker_file: Path, sleep: float = 0.0
) -> None:
    """Queue a job whose command writes a marker file and exits."""
    code = (
        f"import time,sys; time.sleep({sleep}); "
        f"open({str(marker_file)!r},'w').write({job_id!r})"
    )
    mgr.enqueue(
        job_id=job_id,
        calc_type="sp",
        structure="s.cif",
        formula="X",
        natoms=1,
        device="cpu",
        cmd=[sys.executable, "-c", code],
    )


def test_scheduler_run_once_launches_pending(tmp_path: Path) -> None:
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    _queued_cmd(mgr, "j1", tmp_path / "marker1")
    scheduler = QueueScheduler(jobs_dir=mgr.jobs_dir, max_concurrent=1)
    launched = scheduler.run_once()
    assert launched == 1
    assert mgr.get_job("j1")["status"] == "running"
    assert mgr.get_job("j1")["pid"] > 0
    assert mgr.count_by_status("pending") == 0


def test_scheduler_concurrency_limit(tmp_path: Path) -> None:
    """max_concurrent=1: the second queued job must stay PENDING."""
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    _queued_cmd(mgr, "j1", tmp_path / "m1", sleep=1.0)
    _queued_cmd(mgr, "j2", tmp_path / "m2", sleep=1.0)
    scheduler = QueueScheduler(jobs_dir=mgr.jobs_dir, max_concurrent=1)
    assert scheduler.run_once() == 1
    assert mgr.get_job("j1")["status"] == "running"
    assert mgr.get_job("j2")["status"] == "pending"
    # Waiting for j1 to finish, then the scheduler picks j2 and runs it to
    # completion (max_concurrent=1 serialises them).
    deadline = time.time() + 15
    while time.time() < deadline:
        scheduler.run_once()
        if mgr.get_job("j2")["status"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert mgr.get_job("j2")["status"] in ("done", "failed"), mgr.get_job("j2")
    assert (tmp_path / "m1").exists()
    assert (tmp_path / "m2").exists()


def test_scheduler_pause_keeps_pending_jobs_until_resume(tmp_path: Path) -> None:
    """Pausing dispatch leaves running work alone and blocks the next job."""
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    _queued_cmd(mgr, "j1", tmp_path / "m1", sleep=0.6)
    _queued_cmd(mgr, "j2", tmp_path / "m2")
    scheduler = QueueScheduler(jobs_dir=mgr.jobs_dir, max_concurrent=1)

    assert scheduler.run_once() == 1
    assert mgr.get_job("j1")["status"] == "running"
    assert mgr.get_job("j2")["status"] == "pending"
    assert pause_scheduler(mgr.jobs_dir) is True
    assert queue_paused(mgr.jobs_dir) is True

    deadline = time.time() + 15
    while time.time() < deadline and mgr.get_job("j1")["status"] == "running":
        time.sleep(0.1)
    assert mgr.get_job("j1")["status"] == "done"
    assert scheduler.run_once() == 0
    assert mgr.get_job("j2")["status"] == "pending"
    assert not (tmp_path / "m2").exists()

    assert resume_scheduler(mgr.jobs_dir) is True
    assert queue_paused(mgr.jobs_dir) is False
    assert scheduler.run_once() == 1
    deadline = time.time() + 15
    while time.time() < deadline:
        if mgr.get_job("j2")["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert mgr.get_job("j2")["status"] == "done"
    assert (tmp_path / "m2").exists()


def test_scheduler_pause_resume_are_idempotent(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    JobManager(jobs_dir=jobs_dir)
    assert pause_scheduler(jobs_dir) is True
    assert pause_scheduler(jobs_dir) is False
    assert resume_scheduler(jobs_dir) is True
    assert resume_scheduler(jobs_dir) is False


def test_pause_one_pending_job_does_not_block_other_pending_jobs(
    tmp_path: Path,
) -> None:
    """A paused job is skipped while other pending jobs continue FIFO dispatch."""
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    _queued_cmd(mgr, "j1", tmp_path / "m1", sleep=0.6)
    _queued_cmd(mgr, "j2", tmp_path / "m2")
    _queued_cmd(mgr, "j3", tmp_path / "m3")
    scheduler = QueueScheduler(jobs_dir=mgr.jobs_dir, max_concurrent=1)

    assert scheduler.run_once() == 1
    assert pause_pending_job(mgr.jobs_dir, "j2") is True
    assert mgr.get_job("j2")["status"] == "paused"
    assert mgr.get_job("j3")["status"] == "pending"

    deadline = time.time() + 15
    while time.time() < deadline:
        scheduler.run_once()
        if mgr.get_job("j3")["status"] in ("running", "done", "failed"):
            break
        time.sleep(0.1)
    assert mgr.get_job("j3")["status"] in ("running", "done", "failed")
    assert not (tmp_path / "m2").exists()

    assert resume_paused_job(mgr.jobs_dir, "j2") is True
    assert mgr.get_job("j2")["status"] == "pending"
    deadline = time.time() + 15
    while time.time() < deadline:
        scheduler.run_once()
        if mgr.get_job("j2")["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert mgr.get_job("j2")["status"] == "done"
    assert (tmp_path / "m2").exists()


def test_scheduler_max_concurrent_two(tmp_path: Path) -> None:
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    _queued_cmd(mgr, "j1", tmp_path / "m1", sleep=1.0)
    _queued_cmd(mgr, "j2", tmp_path / "m2", sleep=1.0)
    scheduler = QueueScheduler(jobs_dir=mgr.jobs_dir, max_concurrent=2)
    assert scheduler.run_once() == 2
    assert mgr.count_by_status("running") == 2
    deadline = time.time() + 15
    while time.time() < deadline:
        if mgr.count_by_status("running") == 0:
            break
        time.sleep(0.2)
    assert mgr.count_by_status("done") == 2


def test_scheduler_marks_no_command_job_failed(tmp_path: Path) -> None:
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    mgr.enqueue(
        job_id="empty",
        calc_type="sp",
        structure="s.cif",
        formula="X",
        natoms=1,
        device="cpu",
        cmd=[],
    )
    scheduler = QueueScheduler(jobs_dir=mgr.jobs_dir, max_concurrent=1)
    scheduler.run_once()
    assert mgr.get_job("empty")["status"] == "failed"


def test_scheduler_reaps_dead_worker_pid(tmp_path: Path) -> None:
    """A RUNNING job whose worker is gone is marked FAILED and its slot freed.

    Regression: a worker that died without updating its job state (kill -9,
    backend crash) previously stayed RUNNING forever, permanently occupying a
    concurrency slot and blocking the queue."""
    import subprocess

    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    mgr.enqueue(
        job_id="zombie",
        calc_type="sp",
        structure="s.cif",
        formula="X",
        natoms=1,
        device="cpu",
        cmd=[sys.executable, "-c", "pass"],
    )
    # Record a PID that no longer exists (spawn one and let it exit).
    probe = subprocess.Popen([sys.executable, "-c", "pass"])
    probe.wait()
    mgr.mark_running("zombie", probe.pid)

    # A second, healthy job must be launchable after the dead one is reaped.
    _queued_cmd(mgr, "healthy", tmp_path / "m1")
    scheduler = QueueScheduler(jobs_dir=mgr.jobs_dir, max_concurrent=1)
    assert scheduler.run_once() == 1
    assert mgr.get_job("zombie")["status"] == "failed"
    assert mgr.get_job("healthy")["status"] == "running"


def test_scheduler_job_finishes_done(tmp_path: Path) -> None:
    """End-to-end: queued job -> scheduler -> worker -> DONE."""
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    _queued_cmd(mgr, "j1", tmp_path / "m1")
    scheduler = QueueScheduler(jobs_dir=mgr.jobs_dir, max_concurrent=1)
    scheduler.run_once()
    deadline = time.time() + 15
    while time.time() < deadline:
        status = mgr.get_job("j1")["status"]
        if status in ("done", "failed"):
            break
        time.sleep(0.2)
    assert mgr.get_job("j1")["status"] == "done", mgr.get_job("j1")
    assert (tmp_path / "m1").exists()


def test_scheduler_run_forever_stop_file(tmp_path: Path) -> None:
    import threading

    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    _queued_cmd(mgr, "j1", tmp_path / "m1")
    stop_file = tmp_path / "stop"
    scheduler = QueueScheduler(
        jobs_dir=mgr.jobs_dir, max_concurrent=1, poll_interval=0.5
    )

    def _stop_later():
        time.sleep(1.2)
        stop_file.write_text("stop", encoding="utf-8")

    threading.Thread(target=_stop_later, daemon=True).start()
    scheduler.run_forever(stop_file=stop_file)
    # j1 must have been processed before the stop file appeared
    assert mgr.get_job("j1")["status"] in ("done", "failed")


# ---------------------------------------------------------------------------
# submit_task_file
# ---------------------------------------------------------------------------


def test_submit_task_file_enqueues(tmp_path: Path) -> None:
    path = _write_tasks(tmp_path, [_sample_task(tmp_path, name="jobA")])
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    job_ids, max_conc = submit_task_file(mgr, path)
    assert job_ids == ["jobA"]
    assert max_conc == 1
    data = mgr.get_job("jobA")
    assert data["status"] == "pending"
    assert data["calc_type"] == "opt"
    assert data["device"] == "cuda:0"
    # cmd must carry the interpreter + mlipx.cli invocation
    assert data["cmd"][:3] == [sys.executable, "-m", "mlipx.cli"]


def test_submit_task_file_multiple(tmp_path: Path) -> None:
    tasks = [
        _sample_task(tmp_path, name="opt1", calc_type="opt"),
        _sample_task(
            tmp_path,
            name="md1",
            calc_type="md",
            model_type="grace",
            options={"temperature": 400.0},
        ),
    ]
    path = _write_tasks(tmp_path, tasks, max_conc=2)
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    job_ids, max_conc = submit_task_file(mgr, path)
    assert job_ids == ["opt1", "md1"]
    assert max_conc == 2
    md_cmd = mgr.get_job("md1")["cmd"]
    assert "--model-type" in md_cmd and "grace" in md_cmd
    assert "--temp" in md_cmd and "400.0" in md_cmd


# ---------------------------------------------------------------------------
# Scheduler start/stop/status
# ---------------------------------------------------------------------------


def test_start_stop_status_scheduler(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    JobManager(jobs_dir=jobs_dir)  # create dirs
    assert scheduler_status(jobs_dir)["running"] is False

    pid = start_scheduler(jobs_dir=jobs_dir, max_concurrent=1, poll_interval=1.0)
    try:
        status = scheduler_status(jobs_dir)
        assert status["running"] is True
        assert status["pid"] == pid
        # starting again while alive must fail
        with pytest.raises(RuntimeError, match="already running"):
            start_scheduler(jobs_dir=jobs_dir)
    finally:
        assert stop_scheduler(jobs_dir) is True
    assert stop_scheduler(jobs_dir) is False  # already stopped
    assert scheduler_status(jobs_dir)["running"] is False
    assert not scheduler_pid_file(jobs_dir).exists()


def test_scheduler_daemon_processes_queue(tmp_path: Path) -> None:
    """End-to-end through the real daemon process: enqueue two tasks, start
    the detached scheduler, wait for both to finish, stop the scheduler."""
    jobs_dir = tmp_path / "jobs"
    mgr = JobManager(jobs_dir=jobs_dir)
    _queued_cmd(mgr, "j1", tmp_path / "m1", sleep=0.3)
    _queued_cmd(mgr, "j2", tmp_path / "m2", sleep=0.3)

    pid = start_scheduler(jobs_dir=jobs_dir, max_concurrent=1, poll_interval=0.5)
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            summary = mgr.queue_summary()
            if summary["done"] == 2:
                break
            time.sleep(0.3)
        summary = mgr.queue_summary()
        assert summary["done"] == 2, summary
        assert (tmp_path / "m1").exists() and (tmp_path / "m2").exists()
    finally:
        stop_scheduler(jobs_dir=jobs_dir)
