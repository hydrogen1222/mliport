"""Tests for the Slurm-like job queue (queue.py + JobManager queue methods)."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
from ase import Atoms

from mliport.jobs import JobManager, JobStatus
from mliport.neb.io import NEBCheckpointStore
from mliport.queue import (
    QueueScheduler,
    build_mliport_command,
    parse_task_file,
    pause_pending_job,
    pause_scheduler,
    queue_paused,
    resolve_device_uuid,
    resume_paused_job,
    resume_scheduler,
    scheduler_pid_file,
    scheduler_status,
    start_scheduler,
    stop_scheduler,
    submit_task_file,
)

# ---------------------------------------------------------------------------
# build_mliport_command
# ---------------------------------------------------------------------------


def test_build_command_base() -> None:
    cmd = build_mliport_command(
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
    assert cmd[:3] == ["/venv/bin/python", "-m", "mliport.cli"]
    assert "sp" in cmd and "/s/a.cif" in cmd
    assert "--model" in cmd and "/m/m.pt" in cmd
    assert "--task" in cmd and "omat" in cmd
    assert "--name" in cmd and "j1" in cmd


def test_build_command_opt_options() -> None:
    cmd = build_mliport_command(
        "opt",
        "s.cif",
        "m.pt",
        model_type="mace",
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
    cmd = build_mliport_command(
        "md",
        "s.cif",
        "m.pt",
        model_type="mace",
        options={
            "ensemble": "NVT",
            "temperature": 500.0,
            "steps": 100,
            "thermostat": "NHC",
            "nhc_tdamp": 150.0,
            "nhc_tchain": 4,
            "nhc_tloop": 2,
            "com_policy": "none",
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
    assert "--com-policy" in cmd and "none" in cmd
    assert "--no-pre-relax" in cmd
    assert "--seed" in cmd and "42" in cmd


def test_build_command_neb_uses_two_endpoints_and_serial_job_flags() -> None:
    cmd = build_mliport_command(
        "neb",
        None,
        "model.pt",
        model_type="mace",
        output_dir="out",
        initial="initial.vasp",
        final="final.vasp",
        atom_map=[1, 0],
        image_shifts=[[0, 0, 0], [1, 0, 0]],
        options={
            "n_intermediate_images": 3,
            "climb": True,
            "checkpoint_interval": 2,
        },
    )

    assert cmd[3] == "neb"
    assert cmd[cmd.index("--initial") + 1] == "initial.vasp"
    assert cmd[cmd.index("--final") + 1] == "final.vasp"
    assert cmd[cmd.index("--images") + 1] == "3"
    assert "--climb" in cmd
    assert cmd[cmd.index("--checkpoint-interval") + 1] == "2"
    assert cmd[cmd.index("--atom-map") + 1] == "1,0"
    assert cmd[cmd.index("--image-shifts") + 1] == "0,0,0;1,0,0"


def test_build_command_forwards_molecular_electronic_state() -> None:
    cmd = build_mliport_command(
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
    grace = build_mliport_command(
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

    mace = build_mliport_command(
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

    dpa = build_mliport_command(
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


def test_parse_neb_task_freezes_both_endpoints(tmp_path: Path) -> None:
    for name in ("initial.vasp", "final.vasp", "model.pt"):
        (tmp_path / name).write_text("dummy", encoding="utf-8")
    path = _write_tasks(
        tmp_path,
        [
            {
                "calc_type": "neb",
                "initial": "initial.vasp",
                "final": "final.vasp",
                "model": "model.pt",
                "model_type": "mace",
                "task": "bulk",
                "device": "cpu",
                "output_dir": "out",
                "atom_map": [1, 0],
                "image_shifts": [[0, 0, 0], [1, 0, 0]],
                "options": {"NEB_IMAGES": "3", "NEB_CLIMB": "1"},
            }
        ],
    )

    task = parse_task_file(path)["tasks"][0]

    assert task["initial"] == str((tmp_path / "initial.vasp").resolve())
    assert task["final"] == str((tmp_path / "final.vasp").resolve())
    assert task["atom_map"] == [1, 0]
    assert task["image_shifts"] == [[0, 0, 0], [1, 0, 0]]
    assert task["options"] == {"n_intermediate_images": 3, "climb": True}


def test_parse_neb_resume_restores_device_and_model_for_gpu_lease(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    run = tmp_path / "run"
    images = [
        Atoms("H", positions=[[x, 0, 0]], cell=[4, 4, 4], pbc=True)
        for x in (0.5, 1.0, 1.5)
    ]
    NEBCheckpointStore(run).write(
        images,
        run_id=str(uuid.uuid4()),
        attempt_id=str(uuid.uuid4()),
        stage="neb",
        stage_step=1,
        resume_fingerprint={},
        resume_fingerprint_sha256="test",
        resolved_config={
            "model_type": "mace",
            "model_path": str(model),
            "task": "bulk",
            "device": "cpu",
            "inference_mode": "default",
            "calculator_options": {},
            "run_options": {},
            "settings": {},
        },
    )
    path = _write_tasks(
        tmp_path,
        [{"calc_type": "neb", "resume": str(run)}],
    )

    task = parse_task_file(path)["tasks"][0]
    command = build_mliport_command(
        task["calc_type"],
        task["structure"],
        task["model"],
        model_type=task["model_type"],
        task=task["task"],
        device=task["device"],
        output_dir=task["output_dir"],
        job_name="queue-id",
        options=task["options"],
        resume=task["resume"],
    )

    assert task["model"] == str(model.resolve())
    assert task["device"] == "cpu"
    assert command[command.index("--resume") + 1] == str(run.resolve())
    assert "--output" not in command
    assert "--name" not in command


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
                "model_type": "mace",
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


def test_parse_task_file_allows_duplicate_display_names(tmp_path: Path) -> None:
    path = _write_tasks(
        tmp_path,
        [_sample_task(tmp_path, name="dup"), _sample_task(tmp_path, name="dup")],
    )
    parsed = parse_task_file(path)
    assert [task["name"] for task in parsed["tasks"]] == ["dup", "dup"]


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


def _job_id(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mliport-queue-test:{name}"))


def _enqueue(mgr: JobManager, name: str, device: str = "cpu") -> str:
    job_id = _job_id(name)
    mgr.enqueue(
        job_id=job_id,
        display_name=name,
        calc_type="sp",
        structure="s.cif",
        formula="H2O",
        natoms=3,
        device=device,
        cmd=[sys.executable, "-c", "pass"],
    )
    return job_id


def _mark_running(mgr: JobManager, name: str, pid: int) -> None:
    """Fabricate a RUNNING record without a real worker process.

    Production ``mark_running`` captures the /proc start tick (and fails
    closed if it cannot); synthetic PIDs have no such identity, so the test
    pins one explicitly through the documented seam.
    """
    job_id = _job_id(name)
    token = mgr.new_job_id()
    assert mgr.claim_job(
        job_id,
        claim_token=token,
        owner_pid=os.getpid(),
        device_uuid=None,
    )
    assert mgr.mark_running(job_id, pid, claim_token=token, pid_identity=f"test-{pid}")


def test_enqueue_writes_pending(mgr: JobManager) -> None:
    job_id = _enqueue(mgr, "job1")
    data = mgr.get_job(job_id)
    assert uuid.UUID(data["job_id"])
    assert data["display_name"] == "job1"
    assert data["run_id"] == data["job_id"]
    assert data["status"] == "pending"
    assert data["cmd"] == [sys.executable, "-c", "pass"]
    assert data["pid"] == 0


def test_next_pending_fifo(mgr: JobManager) -> None:
    _enqueue(mgr, "first")
    time.sleep(0.01)
    _enqueue(mgr, "second")
    assert mgr.next_pending()["job_id"] == _job_id("first")
    _mark_running(mgr, "first", 999)
    assert mgr.next_pending()["job_id"] == _job_id("second")
    _mark_running(mgr, "second", 1000)
    assert mgr.next_pending() is None


def test_count_by_status(mgr: JobManager) -> None:
    _enqueue(mgr, "a")
    _enqueue(mgr, "b")
    _mark_running(mgr, "a", 1)
    assert mgr.count_by_status("pending") == 1
    assert mgr.count_by_status("running") == 1


def test_queue_summary(mgr: JobManager) -> None:
    _enqueue(mgr, "a")
    _enqueue(mgr, "b")
    _mark_running(mgr, "a", 1)
    summary = mgr.queue_summary()
    assert summary["pending"] == 1
    assert summary["running"] == 1


def test_mark_running_missing_job(mgr: JobManager) -> None:
    assert mgr.mark_running(_job_id("ghost"), 1, claim_token=mgr.new_job_id()) is False


def test_duplicate_internal_id_is_rejected_without_truncating_log(
    mgr: JobManager,
) -> None:
    job_id = _enqueue(mgr, "same name")
    mgr._log_file(job_id).write_text("preserve me\n", encoding="utf-8")
    before = mgr.get_job(job_id)

    with pytest.raises(FileExistsError, match="already exists"):
        mgr.enqueue(
            job_id=job_id,
            display_name="replacement",
            calc_type="sp",
            structure="other.cif",
            formula="X",
            natoms=1,
            device="cpu",
            cmd=[sys.executable, "-c", "pass"],
        )

    assert mgr._log_file(job_id).read_text(encoding="utf-8") == "preserve me\n"
    assert mgr.get_job(job_id) == before


def test_duplicate_display_names_receive_distinct_run_ids(mgr: JobManager) -> None:
    ids = []
    for _ in range(2):
        job_id = mgr.new_job_id()
        mgr.enqueue(
            job_id=job_id,
            display_name="repeatable",
            calc_type="sp",
            structure="s.cif",
            formula="X",
            natoms=1,
            device="cpu",
            cmd=[sys.executable, "-c", "pass"],
        )
        ids.append(job_id)

    assert ids[0] != ids[1]
    assert [mgr.get_job(job_id)["display_name"] for job_id in ids] == [
        "repeatable",
        "repeatable",
    ]


def test_two_schedulers_can_only_claim_a_job_once(tmp_path: Path) -> None:
    first = JobManager(jobs_dir=tmp_path / "jobs")
    second = JobManager(jobs_dir=first.jobs_dir)
    job_id = first.new_job_id()
    first.enqueue(
        job_id=job_id,
        display_name="atomic",
        calc_type="sp",
        structure="s.cif",
        formula="X",
        natoms=1,
        device="cpu",
        cmd=[sys.executable, "-c", "pass"],
    )

    def claim(manager: JobManager) -> bool:
        return (
            manager.claim_job(
                job_id,
                claim_token=manager.new_job_id(),
                owner_pid=os.getpid(),
                device_uuid=None,
            )
            is not None
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(claim, (first, second)))

    assert sorted(outcomes) == [False, True]
    assert first.get_job(job_id)["status"] == "claimed"


def test_gpu_uuid_lease_blocks_aliases_of_the_same_device(mgr: JobManager) -> None:
    first_id = _enqueue(mgr, "gpu-a", device="cuda:0")
    second_id = _enqueue(mgr, "gpu-b", device="gpu")
    first_token = mgr.new_job_id()
    second_token = mgr.new_job_id()

    assert mgr.claim_job(
        first_id,
        claim_token=first_token,
        owner_pid=os.getpid(),
        device_uuid="GPU-physical-0",
    )
    assert (
        mgr.claim_job(
            second_id,
            claim_token=second_token,
            owner_pid=os.getpid(),
            device_uuid="GPU-physical-0",
        )
        is None
    )
    assert mgr.claim_job(
        second_id,
        claim_token=second_token,
        owner_pid=os.getpid(),
        device_uuid="GPU-physical-1",
    )


def test_stale_claim_recovery_releases_device_lease(
    mgr: JobManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = _enqueue(mgr, "stale", device="cuda:0")
    token = mgr.new_job_id()
    assert mgr.claim_job(
        job_id,
        claim_token=token,
        owner_pid=os.getpid(),
        device_uuid="GPU-stale",
    )
    monkeypatch.setattr("mliport.jobs._pid_alive", lambda pid: False)

    assert mgr.recover_stale_claims() == [job_id]
    recovered = mgr.get_job(job_id)
    assert recovered["status"] == "pending"
    assert recovered["device_uuid"] is None


def test_cuda_visible_devices_is_resolved_to_physical_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    completed = Mock(
        returncode=0,
        stdout="1, GPU-one\n3, GPU-three\n",
        stderr="",
    )
    monkeypatch.setattr("mliport.queue.subprocess.run", Mock(return_value=completed))

    assert resolve_device_uuid("cuda:0") == "GPU-three"
    assert resolve_device_uuid("cuda:1") == "GPU-one"


# ---------------------------------------------------------------------------
# QueueScheduler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, 1.5])
def test_scheduler_rejects_non_integer_concurrency(
    tmp_path: Path, value: object
) -> None:
    with pytest.raises((TypeError, ValueError), match="max_concurrent"):
        QueueScheduler(jobs_dir=tmp_path / "jobs", max_concurrent=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
def test_scheduler_rejects_non_positive_or_non_finite_poll_interval(
    tmp_path: Path, value: float
) -> None:
    with pytest.raises(ValueError, match="poll_interval"):
        QueueScheduler(jobs_dir=tmp_path / "jobs", poll_interval=value)


def _queued_cmd(
    mgr: JobManager, name: str, marker_file: Path, sleep: float = 0.0
) -> str:
    """Queue a job whose command writes a marker file and exits."""
    job_id = _job_id(name)
    code = (
        f"import time,sys; time.sleep({sleep}); "
        f"open({str(marker_file)!r},'w').write({name!r})"
    )
    mgr.enqueue(
        job_id=job_id,
        display_name=name,
        calc_type="sp",
        structure="s.cif",
        formula="X",
        natoms=1,
        device="cpu",
        cmd=[sys.executable, "-c", code],
    )
    return job_id


def test_scheduler_run_once_launches_pending(tmp_path: Path) -> None:
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    _queued_cmd(mgr, "j1", tmp_path / "marker1")
    scheduler = QueueScheduler(jobs_dir=mgr.jobs_dir, max_concurrent=1)
    launched = scheduler.run_once()
    assert launched == 1
    assert mgr.get_job(_job_id("j1"))["status"] == "running"
    assert mgr.get_job(_job_id("j1"))["pid"] > 0
    assert mgr.count_by_status("pending") == 0


def test_running_job_cancellation_signals_verified_worker_identity(
    tmp_path: Path,
) -> None:
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    job_id = _queued_cmd(mgr, "cancel-running", tmp_path / "marker", sleep=10.0)
    scheduler = QueueScheduler(jobs_dir=mgr.jobs_dir, max_concurrent=1)
    assert scheduler.run_once() == 1
    assert mgr.get_job(job_id)["pid_identity"] is not None

    assert mgr.kill_job(job_id)

    assert mgr.get_job(job_id)["status"] == "cancelled"
    assert not (tmp_path / "marker").exists()


def test_scheduler_concurrency_limit(tmp_path: Path) -> None:
    """max_concurrent=1: the second queued job must stay PENDING."""
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    _queued_cmd(mgr, "j1", tmp_path / "m1", sleep=1.0)
    _queued_cmd(mgr, "j2", tmp_path / "m2", sleep=1.0)
    scheduler = QueueScheduler(jobs_dir=mgr.jobs_dir, max_concurrent=1)
    assert scheduler.run_once() == 1
    assert mgr.get_job(_job_id("j1"))["status"] == "running"
    assert mgr.get_job(_job_id("j2"))["status"] == "pending"
    # Waiting for j1 to finish, then the scheduler picks j2 and runs it to
    # completion (max_concurrent=1 serialises them).
    deadline = time.time() + 15
    while time.time() < deadline:
        scheduler.run_once()
        if mgr.get_job(_job_id("j2"))["status"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert mgr.get_job(_job_id("j2"))["status"] in ("done", "failed"), mgr.get_job(
        _job_id("j2")
    )
    assert (tmp_path / "m1").exists()
    assert (tmp_path / "m2").exists()


def test_scheduler_pause_keeps_pending_jobs_until_resume(tmp_path: Path) -> None:
    """Pausing dispatch leaves running work alone and blocks the next job."""
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    _queued_cmd(mgr, "j1", tmp_path / "m1", sleep=0.6)
    _queued_cmd(mgr, "j2", tmp_path / "m2")
    scheduler = QueueScheduler(jobs_dir=mgr.jobs_dir, max_concurrent=1)

    assert scheduler.run_once() == 1
    assert mgr.get_job(_job_id("j1"))["status"] == "running"
    assert mgr.get_job(_job_id("j2"))["status"] == "pending"
    assert pause_scheduler(mgr.jobs_dir) is True
    assert queue_paused(mgr.jobs_dir) is True

    deadline = time.time() + 15
    while time.time() < deadline and mgr.get_job(_job_id("j1"))["status"] == "running":
        time.sleep(0.1)
    assert mgr.get_job(_job_id("j1"))["status"] == "done"
    assert scheduler.run_once() == 0
    assert mgr.get_job(_job_id("j2"))["status"] == "pending"
    assert not (tmp_path / "m2").exists()

    assert resume_scheduler(mgr.jobs_dir) is True
    assert queue_paused(mgr.jobs_dir) is False
    assert scheduler.run_once() == 1
    deadline = time.time() + 15
    while time.time() < deadline:
        if mgr.get_job(_job_id("j2"))["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert mgr.get_job(_job_id("j2"))["status"] == "done"
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
    assert pause_pending_job(mgr.jobs_dir, _job_id("j2")) is True
    assert mgr.get_job(_job_id("j2"))["status"] == "paused"
    assert mgr.get_job(_job_id("j3"))["status"] == "pending"

    deadline = time.time() + 15
    while time.time() < deadline:
        scheduler.run_once()
        if mgr.get_job(_job_id("j3"))["status"] in ("running", "done", "failed"):
            break
        time.sleep(0.1)
    assert mgr.get_job(_job_id("j3"))["status"] in ("running", "done", "failed")
    assert not (tmp_path / "m2").exists()

    assert resume_paused_job(mgr.jobs_dir, _job_id("j2")) is True
    assert mgr.get_job(_job_id("j2"))["status"] == "pending"
    deadline = time.time() + 15
    while time.time() < deadline:
        scheduler.run_once()
        if mgr.get_job(_job_id("j2"))["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert mgr.get_job(_job_id("j2"))["status"] == "done"
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
    job_id = _job_id("empty")
    mgr._write_job_state(
        job_id=job_id,
        status=JobStatus.PENDING,
        calc_type="sp",
        structure="s.cif",
        formula="X",
        natoms=1,
        pid=0,
        device="cpu",
        cmd=[],
        display_name="empty",
    )
    scheduler = QueueScheduler(jobs_dir=mgr.jobs_dir, max_concurrent=1)
    scheduler.run_once()
    assert mgr.get_job(job_id)["status"] == "failed"


def test_scheduler_reaps_dead_worker_pid(tmp_path: Path) -> None:
    """A RUNNING job whose worker is gone is marked FAILED and its slot freed.

    Regression: a worker that died without updating its job state (kill -9,
    backend crash) previously stayed RUNNING forever, permanently occupying a
    concurrency slot and blocking the queue."""
    import subprocess

    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    zombie_id = _job_id("zombie")
    mgr.enqueue(
        job_id=zombie_id,
        display_name="zombie",
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
    _mark_running(mgr, "zombie", probe.pid)

    # A second, healthy job must be launchable after the dead one is reaped.
    _queued_cmd(mgr, "healthy", tmp_path / "m1")
    scheduler = QueueScheduler(jobs_dir=mgr.jobs_dir, max_concurrent=1)
    assert scheduler.run_once() == 1
    assert mgr.get_job(zombie_id)["status"] == "failed"
    assert mgr.get_job(_job_id("healthy"))["status"] == "running"


def test_scheduler_job_finishes_done(tmp_path: Path) -> None:
    """End-to-end: queued job -> scheduler -> worker -> DONE."""
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    _queued_cmd(mgr, "j1", tmp_path / "m1")
    scheduler = QueueScheduler(jobs_dir=mgr.jobs_dir, max_concurrent=1)
    scheduler.run_once()
    deadline = time.time() + 15
    while time.time() < deadline:
        status = mgr.get_job(_job_id("j1"))["status"]
        if status in ("done", "failed"):
            break
        time.sleep(0.2)
    assert mgr.get_job(_job_id("j1"))["status"] == "done", mgr.get_job(_job_id("j1"))
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
    assert mgr.get_job(_job_id("j1"))["status"] in ("done", "failed")


# ---------------------------------------------------------------------------
# submit_task_file
# ---------------------------------------------------------------------------


def test_submit_task_file_enqueues(tmp_path: Path) -> None:
    path = _write_tasks(tmp_path, [_sample_task(tmp_path, name="jobA")])
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    job_ids, max_conc = submit_task_file(mgr, path)
    assert len(job_ids) == 1
    assert uuid.UUID(job_ids[0])
    assert max_conc == 1
    data = mgr.get_job(job_ids[0])
    assert data["display_name"] == "jobA"
    assert data["status"] == "pending"
    assert data["calc_type"] == "opt"
    assert data["device"] == "cuda:0"
    # cmd must carry the interpreter + mliport.cli invocation
    assert data["cmd"][:3] == [sys.executable, "-m", "mliport.cli"]
    assert data["cmd"][data["cmd"].index("--name") + 1] == data["run_id"]


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
    assert len(job_ids) == 2
    assert max_conc == 2
    assert [mgr.get_job(job_id)["display_name"] for job_id in job_ids] == [
        "opt1",
        "md1",
    ]
    md_cmd = mgr.get_job(job_ids[1])["cmd"]
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


# ---------------------------------------------------------------------------
# R07/R08: worker exit codes and the PID-identity startup handshake
# ---------------------------------------------------------------------------


def _enqueue_raw(mgr: JobManager, name: str, code: str) -> str:
    job_id = _job_id(name)
    mgr.enqueue(
        job_id=job_id,
        display_name=name,
        calc_type="sp",
        structure="s.cif",
        formula="X",
        natoms=1,
        device="cpu",
        cmd=[sys.executable, "-c", code],
    )
    return job_id


def test_worker_maps_exit_codes_to_terminal_job_status(tmp_path: Path) -> None:
    """0 -> done, 2 -> not_converged, 130 -> cancelled, other -> failed."""
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    cases = {
        "ok": ("import sys; sys.exit(0)", "done"),
        "unconverged": ("import sys; sys.exit(2)", "not_converged"),
        "cancelled": ("import sys; sys.exit(130)", "cancelled"),
        "broken": ("import sys; sys.exit(3)", "failed"),
    }
    for name, (code, _expected) in cases.items():
        _enqueue_raw(mgr, name, code)
    scheduler = QueueScheduler(jobs_dir=mgr.jobs_dir, max_concurrent=4)
    statuses: dict[str, str] = {}
    deadline = time.time() + 30
    while time.time() < deadline:
        scheduler.run_once()
        statuses = {name: str(mgr.get_job(_job_id(name))["status"]) for name in cases}
        if all(
            status in {"done", "not_converged", "cancelled", "failed"}
            for status in statuses.values()
        ):
            break
        time.sleep(0.2)
    for name, (_code, expected) in cases.items():
        assert statuses.get(name) == expected, (
            name,
            statuses,
            mgr.get_job(_job_id(name)),
        )


def test_mark_running_refuses_unpinnable_pid(tmp_path: Path) -> None:
    """Fail closed: a process whose identity cannot be pinned is not promoted."""
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    job_id = _enqueue(mgr, "orphan")
    token = mgr.new_job_id()
    assert mgr.claim_job(
        job_id, claim_token=token, owner_pid=os.getpid(), device_uuid=None
    )
    # A PID that does not exist has no /proc identity: promotion must fail and
    # the record must stay CLAIMED (the caller owns killing its child).
    assert not mgr.mark_running(job_id, 999_999_999, claim_token=token)
    assert mgr.get_job(job_id)["status"] == "claimed"


def test_mark_running_does_not_touch_unrelated_records(tmp_path: Path) -> None:
    """A wrong claim token must never probe/signal the PID."""
    mgr = JobManager(jobs_dir=tmp_path / "jobs")
    job_id = _enqueue(mgr, "token-check")
    token = mgr.new_job_id()
    assert mgr.claim_job(
        job_id, claim_token=token, owner_pid=os.getpid(), device_uuid=None
    )
    assert not mgr.mark_running(job_id, os.getpid(), claim_token=mgr.new_job_id())
    assert mgr.get_job(job_id)["status"] == "claimed"


# ---------------------------------------------------------------------------
# QUEUE-01: no implicit UMA backend in task files
# ---------------------------------------------------------------------------


def test_queue_task_without_model_type_fails(tmp_path: Path) -> None:
    """QUEUE-01/QN-01: a new task must select its backend explicitly."""
    task = _sample_task(tmp_path)
    task.pop("model_type")
    with pytest.raises(ValueError, match="model_type|backend"):
        parse_task_file(_write_tasks(tmp_path, [task]))


def test_queue_fairchem_alias_canonicalizes_to_uma(tmp_path: Path) -> None:
    """QUEUE-01/QN-04: the fairchem alias is recorded as the UMA runtime."""
    task = _sample_task(tmp_path, model_type="fairchem")
    parsed = parse_task_file(_write_tasks(tmp_path, [task]))
    assert parsed["tasks"][0]["model_type"] == "uma"


def test_queue_resume_conflicting_backend_fails(tmp_path: Path) -> None:
    """QUEUE-01/QN-06: an explicit task backend must match the checkpoint."""
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    run = tmp_path / "run"
    images = [
        Atoms("H", positions=[[x, 0, 0]], cell=[4, 4, 4], pbc=True)
        for x in (0.5, 1.0, 1.5)
    ]
    NEBCheckpointStore(run).write(
        images,
        run_id=str(uuid.uuid4()),
        attempt_id=str(uuid.uuid4()),
        stage="neb",
        stage_step=1,
        resume_fingerprint={},
        resume_fingerprint_sha256="test",
        resolved_config={
            "model_type": "mace",
            "model_path": str(model),
            "task": "bulk",
            "device": "cpu",
            "inference_mode": "default",
            "calculator_options": {},
            "run_options": {},
            "settings": {},
        },
    )
    path = _write_tasks(
        tmp_path,
        [{"calc_type": "neb", "resume": str(run), "model_type": "dpa"}],
    )
    with pytest.raises(ValueError, match="conflict|does not match|checkpoint"):
        parse_task_file(path)


def test_build_command_requires_model_type(tmp_path: Path) -> None:
    """QUEUE-01/QN-01: the command builder has no backend default."""
    with pytest.raises(TypeError):
        build_mliport_command("sp", "s.cif", "m.pt")  # type: ignore[call-arg]
