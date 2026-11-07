"""Real (non-mock) installation tests for the NEP backend (tracking #61).

These tests execute ``scripts/install_mliport.sh`` for real: they create a
virtual environment, download wheels, install an editable mliport and run
``mliport doctor`` from the freshly created ``.venv-nep``.  They need network
access, ``uv`` and minutes of time, so they only run when
``MLIPORT_REAL_INSTALL=1`` is set (the dedicated ``installer-real-nep`` CI job
does this).  Plan-level behaviour is covered, without network, by
``test_install_plan.py``.

Every installation happens in a throw-away copy of the repository so that the
real checkout never gets a ``.venv-nep`` and "no other environment was created"
can be asserted exactly.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ENABLED = os.environ.get("MLIPORT_REAL_INSTALL") == "1"
LOG_DIR = Path(os.environ.get("MLIPORT_REAL_INSTALL_LOGDIR", "")).expanduser()
INSTALL_TIMEOUT = 1800

pytestmark = [
    pytest.mark.skipif(
        not ENABLED, reason="real installs run only with MLIPORT_REAL_INSTALL=1"
    ),
    pytest.mark.skipif(
        sys.platform == "win32"
        or shutil.which("uv") is None
        or shutil.which("bash") is None,
        reason="needs POSIX bash and uv",
    ),
]

TUNA_PROBE = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/calorine/"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The user's shell, minus anything that could mask an installer defect.

    ``PYTHONPATH`` in particular is what let the old NEP install reach the
    ``inventory`` stage without mliport in ``.venv-nep``.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("PIP_", "UV_", "MLIPORT_", "NEP_", "PYTHON"))
        and k not in {"VIRTUAL_ENV", "CONDA_PREFIX"}
    }
    env.update(extra or {})
    return env


def _copy_workspace(destination: Path) -> Path:
    """Copy what the installer needs (``mliport/`` and ``scripts/``)."""
    destination.mkdir(parents=True)
    ignore = shutil.ignore_patterns(
        "__pycache__",
        "*.egg-info",
        "*.pyc",
        ".venv*",
        "build",
        "dist",
        ".pytest_cache",
        ".ruff_cache",
    )
    for name in ("mliport", "scripts"):
        shutil.copytree(REPO / name, destination / name, ignore=ignore)
    return destination


def _run(
    argv: list[str],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
    log_name: str | None = None,
    timeout: int = INSTALL_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=env if env is not None else _clean_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if log_name and str(LOG_DIR) not in {"", "."}:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / f"{log_name}.log").write_text(
            f"$ {' '.join(argv)}\n(cwd={cwd})\nexit={result.returncode}\n\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}\n",
            encoding="utf-8",
        )
    return result


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return f"{result.stdout}\n{result.stderr}"


def _install(
    workspace: Path,
    *extra: str,
    env: dict[str, str] | None = None,
    log_name: str,
) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "bash",
            "scripts/install_mliport.sh",
            "--engines",
            "nep",
            "--non-interactive",
            *extra,
        ],
        workspace,
        env=env,
        log_name=log_name,
    )


def _is_executable_file(path: Path) -> bool:
    return (
        path.is_file()
        and bool(path.stat().st_mode & stat.S_IXUSR)
        and os.access(path, os.X_OK)
    )


def _doctor(
    workspace: Path,
    device: str,
    env: dict[str, str] | None = None,
    log_name: str | None = None,
):
    return _run(
        [
            str(workspace / ".venv-nep" / "bin" / "mliport"),
            "doctor",
            "--engine",
            "nep",
            "--device",
            device,
        ],
        workspace,
        env=env,
        log_name=log_name,
        timeout=300,
    )


def _no_gpumd_env() -> dict[str, str]:
    env = _clean_env()
    if shutil.which("gpumd", path=env.get("PATH")):
        pytest.skip(
            "a gpumd executable is on PATH; the missing-binary case cannot be shown"
        )
    return env


# ---------------------------------------------------------------------------
# first install, official sources (shared by the tests below)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> Path:
    path = _copy_workspace(tmp_path_factory.mktemp("nep-real") / "repo")
    result = _install(
        path,
        "--device",
        "cpu",
        "--source",
        "official",
        log_name="01-install-official-cpu",
    )
    assert result.returncode == 0, _output(result)
    return path


def test_install_creates_executable_mliport_in_venv_nep(workspace: Path) -> None:
    launcher = workspace / ".venv-nep" / "bin" / "mliport"
    assert _is_executable_file(launcher), "the verify stage launches this file"
    help_run = _run([str(launcher), "--help"], workspace, timeout=120)
    assert help_run.returncode == 0, _output(help_run)
    assert "doctor" in _output(help_run)


def test_install_writes_the_nep_launcher(workspace: Path) -> None:
    assert _is_executable_file(workspace / "bin" / "mliport-nep")


def test_installed_mliport_is_the_editable_workspace_copy(workspace: Path) -> None:
    python = workspace / ".venv-nep" / "bin" / "python"
    probe = _run(
        [
            str(python),
            "-I",
            "-c",
            "import mliport, calorine, ase; from importlib.metadata import version; "
            "print(mliport.__file__); print(version('calorine')); print(ase.__version__)",
        ],
        workspace,
        timeout=120,
    )
    assert probe.returncode == 0, _output(probe)
    lines = probe.stdout.strip().splitlines()
    assert Path(lines[0]).resolve().is_relative_to((workspace / "mliport").resolve())
    assert lines[1] == "4.1"


def test_doctor_cpu_passes_without_gpumd(workspace: Path) -> None:
    result = _doctor(
        workspace, "cpu", env=_no_gpumd_env(), log_name="02-doctor-cpu-no-gpumd"
    )
    assert result.returncode == 0, _output(result)


def test_doctor_cuda_names_the_missing_gpumd_binary(workspace: Path) -> None:
    result = _doctor(
        workspace, "cuda", env=_no_gpumd_env(), log_name="03-doctor-cuda-no-gpumd"
    )
    assert result.returncode != 0, "must fail closed, not pretend the GPU path works"
    assert "gpumd" in _output(result).lower()


def test_doctor_cuda_accepts_gpumd_given_through_mliport_gpumd_command(
    workspace: Path, tmp_path: Path
) -> None:
    fake = tmp_path / "gpumd"
    fake.write_text("#!/bin/sh\necho 'fake gpumd for doctor probe'\n", encoding="utf-8")
    fake.chmod(0o755)
    env = _no_gpumd_env()
    env["MLIPORT_GPUMD_COMMAND"] = str(fake)
    result = _doctor(workspace, "cuda", env=env, log_name="04-doctor-cuda-fake-gpumd")
    assert result.returncode == 0, _output(result)

    # doctor does not read NEP_GPUMD_COMMAND; documented, and pinned here.
    env = _no_gpumd_env()
    env["NEP_GPUMD_COMMAND"] = str(fake)
    other = _doctor(workspace, "cuda", env=env)
    assert other.returncode != 0


def test_nep_install_is_isolated_from_other_backends(workspace: Path) -> None:
    created = sorted(p.name for p in workspace.iterdir() if p.name.startswith(".venv"))
    assert created == [".venv-nep"]
    python = workspace / ".venv-nep" / "bin" / "python"
    listing = _run(
        ["uv", "pip", "list", "--python", str(python)], workspace, timeout=120
    )
    assert listing.returncode == 0, _output(listing)
    installed = {
        line.split()[0].lower()
        for line in listing.stdout.splitlines()[2:]
        if line.strip()
    }
    for forbidden in (
        "torch",
        "mace-torch",
        "fairchem-core",
        "deepmd-kit",
        "tensorflow",
        "tensorpotential",
    ):
        assert forbidden not in installed, forbidden
    assert {"calorine", "ase", "numpy", "mliport"} <= installed
    for other in ("mace", "dpa", "grace", "uma"):
        assert not (workspace / "bin" / f"mliport-{other}").exists()


def test_repeat_install_keeps_working_and_keeps_the_environment(
    workspace: Path,
) -> None:
    marker = workspace / ".venv-nep" / "reinstall-marker"
    marker.write_text("kept", encoding="utf-8")
    result = _install(
        workspace, "--device", "cpu", "--source", "official", log_name="05-reinstall"
    )
    assert result.returncode == 0, _output(result)
    assert marker.read_text(encoding="utf-8") == "kept", (
        "plain re-run must not delete the environment"
    )
    assert _is_executable_file(workspace / ".venv-nep" / "bin" / "mliport")
    assert _doctor(workspace, "cpu", env=_no_gpumd_env()).returncode == 0


def test_clean_reinstall_recreates_only_venv_nep(workspace: Path) -> None:
    marker = workspace / ".venv-nep" / "clean-marker"
    marker.write_text("gone", encoding="utf-8")
    bystander = workspace / ".venv-bystander"
    bystander.mkdir()
    (bystander / "keep").write_text("kept", encoding="utf-8")
    try:
        result = _install(
            workspace,
            "--device",
            "cpu",
            "--source",
            "official",
            "--clean",
            log_name="06-clean-reinstall",
        )
        assert result.returncode == 0, _output(result)
        assert not marker.exists()
        assert (bystander / "keep").read_text(encoding="utf-8") == "kept"
        assert _is_executable_file(workspace / ".venv-nep" / "bin" / "mliport")
    finally:
        shutil.rmtree(bystander, ignore_errors=True)


def test_offline_reinstall_from_the_warm_cache(workspace: Path) -> None:
    """The earlier installs filled uv's cache; ``--source offline`` must then
    rebuild the environment without touching the network."""
    result = _install(
        workspace,
        "--device",
        "cpu",
        "--source",
        "offline",
        "--clean",
        log_name="07-offline-reinstall",
    )
    assert result.returncode == 0, _output(result)
    assert _is_executable_file(workspace / ".venv-nep" / "bin" / "mliport")
    assert _doctor(workspace, "cpu", env=_no_gpumd_env()).returncode == 0


def test_gpu_host_install_without_gpumd_stops_at_doctor_and_names_gpumd(
    tmp_path: Path,
) -> None:
    """A GPU host (here: a V100 reported by an ``nvidia-smi`` shim, since CI has
    no GPU) with ``--device auto`` asks for the CUDA path.  Without gpumd the
    closing doctor must fail naming it, the environment itself stays installed,
    and the launcher is not advertised as installed."""
    shim_dir = tmp_path / "gpu-shim"
    shim_dir.mkdir()
    smi = shim_dir / "nvidia-smi"
    smi.write_text(
        "#!/bin/sh\necho 'Tesla V100-SXM2-16GB, 7.0, 580.173.02, 16384, 0, "
        "GPU-00000000-0000-0000-0000-000000000000'\n",
        encoding="utf-8",
    )
    smi.chmod(0o755)
    env = _no_gpumd_env()
    env["PATH"] = f"{shim_dir}{os.pathsep}{env['PATH']}"

    path = _copy_workspace(tmp_path / "repo")
    result = _install(
        path,
        "--device",
        "auto",
        "--source",
        "official",
        env=env,
        log_name="08-install-gpu-host-no-gpumd",
    )
    assert result.returncode != 0
    assert "gpumd" in _output(result).lower()
    assert _is_executable_file(path / ".venv-nep" / "bin" / "mliport"), (
        "environment itself is installed"
    )
    assert not (path / "bin" / "mliport-nep").exists()


# ---------------------------------------------------------------------------
# China mirror
# ---------------------------------------------------------------------------


def _tuna_reachable() -> bool:
    try:
        with urllib.request.urlopen(TUNA_PROBE, timeout=20) as response:  # noqa: S310
            return response.status == 200
    except OSError:
        return False


def test_china_source_really_uses_the_tuna_mirror(tmp_path: Path) -> None:
    """A ``uv`` shim records every invocation, so the mirror's presence on the
    NEP install step is observed in a real run, not read off the plan."""
    real_uv = shutil.which("uv")
    assert real_uv
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    uv_log = tmp_path / "uv-calls.log"
    shim = shim_dir / "uv"
    shim.write_text(
        f'#!/bin/sh\nprintf \'%s\\n\' "$*" >> "{uv_log}"\nexec "{real_uv}" "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    env = _clean_env()
    env["PATH"] = f"{shim_dir}{os.pathsep}{env['PATH']}"

    path = _copy_workspace(tmp_path / "repo")
    reachable = _tuna_reachable()
    result = _install(
        path,
        "--device",
        "cpu",
        "--source",
        "china",
        env=env,
        log_name="09-install-china-cpu",
    )

    calls = uv_log.read_text(encoding="utf-8").splitlines()
    nep_install = [
        c
        for c in calls
        if c.startswith("pip install") and "-e ./mliport[analysis-all]" in c
    ]
    assert nep_install, calls
    assert all(
        "--index-url https://mirrors.tuna.tsinghua.edu.cn/" in c for c in nep_install
    ), nep_install
    assert all("pypi.org" not in c for c in nep_install)
    if reachable:
        assert result.returncode == 0, _output(result)
        assert _is_executable_file(path / ".venv-nep" / "bin" / "mliport")
        assert _doctor(path, "cpu", env=_no_gpumd_env()).returncode == 0
    else:
        # Fail closed: an unreachable mirror is an error, never a silent switch to PyPI.
        assert result.returncode != 0
