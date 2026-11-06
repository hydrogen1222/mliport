"""Unit tests for the GPUMD subprocess boundary (no real gpumd required)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import pytest
from ase import Atoms

from mliport.gpumd.process import (
    gpumd_identity,
    resolve_gpumd_command,
    run_gpumd,
)
from mliport.gpumd.provenance import build_gpumd_provenance, write_provenance
from mliport.gpumd.runner import NativeGPUMDRunner, NEPGPUMDIdentity


def _script(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _workdir(tmp_path: Path, name: str = "work") -> Path:
    path = tmp_path / name
    path.mkdir(exist_ok=True)
    return path


def _assert_dead(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_exit_zero_reports_exited_and_creates_logs(tmp_path):
    script = _script(
        tmp_path,
        "fake_gpumd.sh",
        "echo hello-stdout\n"
        "echo hello-stderr >&2\n"
        "printf 'raw\\n' > thermo.out\n"
        "printf 'raw\\n' > traj.xyz\n"
        "exit 0",
    )
    workdir = _workdir(tmp_path)
    outcome = run_gpumd([str(script)], workdir)
    assert outcome.status == "exited"
    assert outcome.returncode == 0
    assert outcome.terminated is False
    assert Path(outcome.stdout_path).read_text(encoding="utf-8") == "hello-stdout\n"
    assert Path(outcome.stderr_path).read_text(encoding="utf-8") == "hello-stderr\n"
    assert (workdir / "thermo.out").is_file()
    assert (workdir / "traj.xyz").is_file()
    assert outcome.command == [str(script.resolve())]
    assert outcome.cwd == str(workdir.resolve())
    assert outcome.wall_time_s >= 0.0


def test_nonzero_exit_is_reported_not_raised(tmp_path):
    script = _script(tmp_path, "failing.sh", "echo boom >&2\nexit 3")
    outcome = run_gpumd([str(script)], _workdir(tmp_path))
    assert outcome.status == "exited"
    assert outcome.returncode == 3
    assert outcome.terminated is False
    assert "boom" in Path(outcome.stderr_path).read_text(encoding="utf-8")


def test_missing_binary_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        run_gpumd([str(tmp_path / "does-not-exist")], _workdir(tmp_path))
    with pytest.raises(FileNotFoundError, match="not found"):
        run_gpumd(["definitely-not-a-real-gpumd-binary"], _workdir(tmp_path))


def test_non_executable_file_raises_permission_error(tmp_path):
    script = _script(tmp_path, "noexec.sh", "exit 0")
    script.chmod(0o644)
    with pytest.raises(PermissionError, match="not executable"):
        run_gpumd([str(script)], _workdir(tmp_path))


def test_shell_string_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="argv list"):
        run_gpumd("/bin/echo hi; rm -rf /tmp/nope", _workdir(tmp_path))
    with pytest.raises(ValueError, match="argv list"):
        run_gpumd("/bin/echo hi", _workdir(tmp_path))
    with pytest.raises(ValueError, match="argv list"):
        run_gpumd([], _workdir(tmp_path))


def test_missing_workdir_raises_file_not_found(tmp_path):
    script = _script(tmp_path, "ok.sh", "exit 0")
    with pytest.raises(FileNotFoundError, match="work directory"):
        run_gpumd([str(script)], tmp_path / "missing")


def test_resolve_gpumd_command_returns_absolute_argv(tmp_path):
    script = _script(tmp_path, "ok.sh", "exit 0")
    argv = resolve_gpumd_command(script)
    assert argv == [str(script.resolve())]


def test_timeout_escalates_and_kills_process(tmp_path):
    script = _script(
        tmp_path,
        "slow.sh",
        "echo $$ > pid.txt\nexec sleep 30",
    )
    workdir = _workdir(tmp_path)
    started = time.monotonic()
    outcome = run_gpumd([str(script)], workdir, timeout_s=0.3, grace_s=1.0)
    elapsed = time.monotonic() - started
    assert outcome.status == "timeout"
    assert outcome.terminated is True
    assert elapsed < 10.0
    pid = int((workdir / "pid.txt").read_text(encoding="utf-8"))
    time.sleep(0.1)
    _assert_dead(pid)


def test_cancellation_sets_cancelled_and_kills_process(tmp_path):
    script = _script(
        tmp_path,
        "cancel.sh",
        "echo $$ > pid.txt\nexec sleep 30",
    )
    workdir = _workdir(tmp_path)
    cancel_event = threading.Event()

    def cancel_soon() -> None:
        time.sleep(0.4)
        cancel_event.set()

    thread = threading.Thread(target=cancel_soon)
    thread.start()
    try:
        outcome = run_gpumd(
            [str(script)],
            workdir,
            cancel_event=cancel_event,
            grace_s=1.0,
        )
    finally:
        thread.join(timeout=5.0)
    assert outcome.status == "cancelled"
    assert outcome.terminated is True
    pid = int((workdir / "pid.txt").read_text(encoding="utf-8"))
    time.sleep(0.1)
    _assert_dead(pid)


def test_environment_override_is_visible_to_child(tmp_path):
    script = _script(
        tmp_path,
        "env.sh",
        "printf '%s' \"$MLIPORT_GPUMD_TEST_ENV\"",
    )
    outcome = run_gpumd(
        [str(script)],
        _workdir(tmp_path),
        env={"MLIPORT_GPUMD_TEST_ENV": "visible-value"},
    )
    assert outcome.returncode == 0
    assert Path(outcome.stdout_path).read_text(encoding="utf-8") == "visible-value"


def test_gpumd_identity_parses_banner_and_hashes_binary(tmp_path):
    script = _script(
        tmp_path,
        "banner.sh",
        "echo '***************************************************************'\n"
        "echo '*                 Welcome to use GPUMD                        *'\n"
        "echo '*                   version 5.9.1                             *'\n"
        "exit 0",
    )
    identity = gpumd_identity(script)
    assert identity["version"] == "5.9.1"
    assert identity["version_error"] is None
    assert identity["real_path"] == str(script.resolve())
    expected = hashlib.sha256(script.read_bytes()).hexdigest()
    assert identity["sha256"] == expected


def test_gpumd_identity_records_version_error_without_guessing(tmp_path):
    script = _script(tmp_path, "bad.sh", "echo 'no banner here'\nexit 4")
    identity = gpumd_identity(script)
    assert identity["version"] is None
    assert identity["version_error"]
    assert "returncode=4" in identity["version_error"]


def _fake_nep(tmp_path: Path) -> Path:
    model = tmp_path / "nep.txt"
    model.write_text("nep4 1 H\ncutoff 8 4\n", encoding="utf-8")
    return model


def test_nep_gpumd_identity_is_not_an_ase_calculator():
    identity = NEPGPUMDIdentity(
        task="bulk", model_path="/tmp/nep.txt", gpumd_version="5.9.1"
    )
    assert identity.task == "bulk"
    assert identity.implemented_properties == ("energy", "forces", "stress")
    info = identity.info()
    assert info["model_type"] == "nep"
    assert info["backend_version"] == "5.9.1"
    assert info["is_ase_calculator"] is False
    with pytest.raises(RuntimeError, match="external process, not an ASE calculator"):
        identity.get_calculator()


def test_provenance_is_deterministic_and_records_identity(tmp_path):
    script = _script(tmp_path, "fake.sh", "exit 0")
    nep = _fake_nep(tmp_path)
    model = tmp_path / "model.xyz"
    model.write_text('1\nLattice="1 1 1"\nH 0 0 0\n', encoding="utf-8")
    identity = {
        "path": str(script),
        "real_path": str(script.resolve()),
        "sha256": "deadbeef",
        "version": "5.9.1",
        "version_error": None,
    }
    payload = build_gpumd_provenance(
        gpumd_path=script,
        nep_path=nep,
        run_in_text="potential nep.txt\nrun 1\n",
        model_xyz_path=model,
        device_requested="cpu",
        gpumd_identity_record=identity,
    )
    assert payload["numeric_precision"] == "float32"
    assert "float32" in payload["numeric_precision_note"]
    assert payload["gpumd"]["version"] == "5.9.1"
    assert payload["nep"]["element_types"] == ["H"]
    assert (
        payload["run_in"]["sha256"]
        == hashlib.sha256(b"potential nep.txt\nrun 1\n").hexdigest()
    )
    assert payload["actual_device_uuid"] is None
    assert (
        payload["model_xyz"]["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    )

    first = write_provenance(tmp_path / "provenance.json", payload)
    first_text = first.read_text(encoding="utf-8")
    second = write_provenance(tmp_path / "provenance.json", payload)
    assert second.read_text(encoding="utf-8") == first_text
    assert first_text.endswith("\n")
    assert json.loads(first_text)["schema"] == "mliport.gpumd-provenance/1"


def test_runner_exit_zero_missing_output_returns_failed(tmp_path):
    script = _script(tmp_path, "empty.sh", "exit 0")
    runner = NativeGPUMDRunner(
        {
            "gpumd_command": str(script),
            "nep_path": str(_fake_nep(tmp_path)),
            "task": "bulk",
        },
        calc_type="sp",
        output_dir=tmp_path / "run",
        verbose=False,
    )
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True)
    result = runner.execute(atoms)
    assert result["status"] == "failed"
    assert result["parse_error"]
    assert result["energy"] is None
    manifest = json.loads(
        (tmp_path / "run" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "GPUMDParseError"
    assert (tmp_path / "run" / "native" / "run.in").is_file()
    assert (tmp_path / "run" / "native" / "stdout.log").is_file()


def test_runner_nonzero_exit_raises_but_keeps_raw_files(tmp_path):
    script = _script(tmp_path, "crash.sh", "echo crash >&2\nexit 1")
    runner = NativeGPUMDRunner(
        {
            "gpumd_command": str(script),
            "nep_path": str(_fake_nep(tmp_path)),
            "task": "bulk",
        },
        calc_type="sp",
        output_dir=tmp_path / "run",
        verbose=False,
    )
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True)
    with pytest.raises(RuntimeError, match="return code 1"):
        runner.execute(atoms)
    native = tmp_path / "run" / "native"
    assert (native / "stderr.log").is_file()
    assert "crash" in (native / "stderr.log").read_text(encoding="utf-8")
    manifest = json.loads(
        (tmp_path / "run" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"


def test_runner_cancellation_returns_cancelled_and_kills_process(tmp_path):
    script = _script(
        tmp_path,
        "cancel_runner.sh",
        "if [ -f run.in ]; then\n"
        "  echo $$ > pid.txt\n"
        "  exec sleep 30\n"
        "else\n"
        "  echo 'version 5.9.1'\n"
        "  exit 0\n"
        "fi",
    )
    runner = NativeGPUMDRunner(
        {
            "gpumd_command": str(script),
            "nep_path": str(_fake_nep(tmp_path)),
            "task": "bulk",
        },
        calc_type="sp",
        output_dir=tmp_path / "run",
        verbose=False,
        cancel_event=threading.Event(),
    )
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True)

    def cancel_soon() -> None:
        time.sleep(0.5)
        runner._cancel_event.set()

    thread = threading.Thread(target=cancel_soon)
    thread.start()
    try:
        result = runner.execute(atoms)
    finally:
        thread.join(timeout=5.0)
    assert result["status"] == "cancelled"
    native = tmp_path / "run" / "native"
    pid = int((native / "pid.txt").read_text(encoding="utf-8"))
    time.sleep(0.1)
    _assert_dead(pid)
    manifest = json.loads(
        (tmp_path / "run" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "cancelled"
    assert manifest["process"]["status"] == "cancelled"


# ---------------------------------------------------------------------------
# NEP-G06: the child process must be pinned to the leased GPU
# ---------------------------------------------------------------------------


def test_leased_gpu_uuid_reaches_the_child_environment(tmp_path):
    """Recording the UUID is not enough: the child must actually see it.

    The runner resolves a GPU UUID (from the queue lease or ``--device``) and
    records it in provenance.  If it were not exported as
    ``CUDA_VISIBLE_DEVICES`` the spawned ``gpumd`` would ignore the lease and
    run on device 0.
    """
    from types import SimpleNamespace

    # A fake executable that reports the environment it actually received.
    script = tmp_path / "fake_gpumd.sh"
    script.write_text(
        "#!/bin/sh\n"
        'printf "%s" "${CUDA_VISIBLE_DEVICES-unset}" > cvd.txt\n'
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)

    model = tmp_path / "nep.txt"
    model.write_text("nep4 1 O\ncutoff 8 4 232 26\n", encoding="utf-8")
    atoms = Atoms("O2", positions=[[0, 0, 0], [1.2, 0, 0]], cell=[6, 6, 6], pbc=True)

    engine = SimpleNamespace(
        gpumd_command=str(script),
        nep_path=str(model),
        task="bulk",
    )
    runner = NativeGPUMDRunner(
        engine,
        calc_type="sp",
        output_dir=tmp_path / "run",
        verbose=False,
        device="cuda:0",
        cuda_visible_devices="GPU-1234abcd-0000-1111-2222-333344445555",
    )
    # The fake binary produces no GPUMD output, so the run itself is expected to
    # fail closed; what matters here is the environment the child received.
    with contextlib.suppress(Exception):
        runner.execute(atoms)

    seen = (tmp_path / "run" / "native" / "cvd.txt").read_text(encoding="utf-8")
    assert seen == "GPU-1234abcd-0000-1111-2222-333344445555"
    manifest = json.loads(
        (tmp_path / "run" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] != "completed"
    assert (
        manifest["provenance"]["cuda_visible_devices"]
        == "GPU-1234abcd-0000-1111-2222-333344445555"
    )
