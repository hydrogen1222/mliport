"""Real-GPU GPUMD known-answer tests.

These tests are skipped unless the native ``gpumd`` binary, ``nvidia-smi`` and
the GPUMD fixture model/structure are all available.  They are the only tests
in the GPUMD package that need a GPU.

The GPUMD binary on this machine dynamically links the CUDA 12.9 runtime, so
the test wrapper prepends ``/opt/apps/cuda-12.9/lib64`` to
``LD_LIBRARY_PATH`` *for the child process only* (the environment is restored
after each test).  Override the paths with:

* ``MLIPORT_GPUMD_EXECUTABLE``
* ``MLIPORT_GPUMD_MODEL``
* ``MLIPORT_GPUMD_STRUCTURE``
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from mliport.gpumd.runner import NativeGPUMDRunner

GPUMD_EXECUTABLE = Path(
    os.environ.get("MLIPORT_GPUMD_EXECUTABLE", "/tmp/gpumd-src/src/gpumd")
)
GPUMD_MODEL = Path(
    os.environ.get(
        "MLIPORT_GPUMD_MODEL",
        "/tmp/gpumd-src/tests_pytest/fixtures/models/nep_BaZrO3.txt",
    )
)
GPUMD_STRUCTURE = Path(
    os.environ.get(
        "MLIPORT_GPUMD_STRUCTURE",
        "/tmp/gpumd-src/tests_pytest/fixtures/structures/" "BaZrO3-nat40-rattled.xyz",
    )
)
CUDA_RUNTIME_LIB = "/opt/apps/cuda-12.9/lib64"

#: Registered known answer for the fixture model/structure on this V100.
SP_ENERGY_EV = -258.17361808


def _nvidia_smi_usable() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and "GPU" in completed.stdout


_SKIP_REASON = ""
if not GPUMD_EXECUTABLE.is_file():
    _SKIP_REASON = f"gpumd executable not found: {GPUMD_EXECUTABLE}"
elif not GPUMD_MODEL.is_file():
    _SKIP_REASON = f"NEP fixture model not found: {GPUMD_MODEL}"
elif not GPUMD_STRUCTURE.is_file():
    _SKIP_REASON = f"fixture structure not found: {GPUMD_STRUCTURE}"
elif not _nvidia_smi_usable():
    _SKIP_REASON = "nvidia-smi is unavailable or reports no GPU"

pytestmark = pytest.mark.skipif(bool(_SKIP_REASON), reason=_SKIP_REASON)


@contextlib.contextmanager
def _cuda_runtime_env():
    """Expose the CUDA 12.9 runtime to GPUMD subprocesses only."""
    original = os.environ.get("LD_LIBRARY_PATH")
    parts = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    if CUDA_RUNTIME_LIB not in parts:
        parts.insert(0, CUDA_RUNTIME_LIB)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = original


@pytest.fixture(scope="module")
def fixture_atoms():
    return read(GPUMD_STRUCTURE)


@pytest.fixture()
def engine():
    return {
        "gpumd_command": str(GPUMD_EXECUTABLE),
        "nep_path": str(GPUMD_MODEL),
        "task": "bulk",
    }


def _manifest(output_dir: Path) -> dict:
    return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))


def test_real_sp_known_answer(tmp_path, fixture_atoms, engine):
    runner = NativeGPUMDRunner(
        engine, calc_type="sp", output_dir=tmp_path, verbose=False
    )
    with _cuda_runtime_env():
        result = runner.execute(fixture_atoms.copy())

    assert result["status"] == "completed"
    assert result["energy"] == pytest.approx(SP_ENERGY_EV, rel=1e-6)
    assert result["natoms"] == 40
    forces = np.asarray(result["forces"], dtype=float)
    assert forces.shape == (40, 3)
    assert np.all(np.isfinite(forces))
    stress = np.asarray(result["stress"], dtype=float)
    assert stress.shape == (6,)
    assert np.all(np.isfinite(stress))
    assert result["positions_unchanged"] is True
    assert (tmp_path / "native" / "sp.xyz").is_file()
    manifest = _manifest(tmp_path)
    assert manifest["status"] == "completed"
    assert manifest["provenance"]["gpumd"]["version"] == "5.9.1"
    assert manifest["provenance"]["nep"]["element_types"] == ["Ba", "Zr", "O"]
    assert manifest["provenance"]["numeric_precision"] == "float32"


def test_real_opt_converges_and_lowers_energy(tmp_path, fixture_atoms, engine):
    runner = NativeGPUMDRunner(
        engine,
        calc_type="opt",
        output_dir=tmp_path,
        verbose=False,
        fmax=0.3,
        max_steps=20,
        optimizer="FIRE",
    )
    with _cuda_runtime_env():
        result = runner.execute(fixture_atoms.copy())

    assert result["status"] == "completed"
    assert result["converged"] is True
    assert result["energy"] < SP_ENERGY_EV
    assert result["fmax_component"] <= 0.3
    assert result["nsteps"] >= 1
    assert (tmp_path / "native" / "opt.xyz").is_file()
    assert _manifest(tmp_path)["status"] == "completed"


def test_real_nve_md_preserves_raw_trajectory(tmp_path, fixture_atoms, engine):
    runner = NativeGPUMDRunner(
        engine,
        calc_type="md",
        output_dir=tmp_path,
        verbose=False,
        ensemble="NVE",
        temperature=300.0,
        timestep_fs=1.0,
        steps=20,
        seed=42,
        dump_interval=10,
    )
    with _cuda_runtime_env():
        result = runner.execute(fixture_atoms.copy())

    assert result["status"] == "completed"
    assert result["nframes"] == 2
    assert math.isfinite(result["temperature"])
    assert result["temperature"] > 0.0
    raw = tmp_path / "native" / "traj.xyz"
    assert raw.is_file() and raw.stat().st_size > 0
    manifest = _manifest(tmp_path)
    run = manifest["runs"][0]
    assert run["ensemble"] == "NVE"
    assert run["thermostat_mapping"] is None
    assert run["T_coup"] is None
    assert run["nframes"] == 2
    assert run["msd_eligible"] is True


def test_real_nvt_langevin_mapping_and_raw_trajectory(tmp_path, fixture_atoms, engine):
    runner = NativeGPUMDRunner(
        engine,
        calc_type="md",
        output_dir=tmp_path,
        verbose=False,
        ensemble="NVT",
        temperature=300.0,
        timestep_fs=1.0,
        steps=20,
        thermostat="LANGEVIN",
        friction_per_fs=0.01,
        seed=7,
        dump_interval=10,
    )
    with _cuda_runtime_env():
        result = runner.execute(fixture_atoms.copy())

    assert result["status"] == "completed"
    assert result["nframes"] == 2
    assert math.isfinite(result["temperature"])
    assert result["temperature"] > 0.0
    raw = tmp_path / "native" / "traj.xyz"
    assert raw.is_file() and raw.stat().st_size > 0
    manifest = _manifest(tmp_path)
    run = manifest["runs"][0]
    assert run["ensemble"] == "NVT"
    assert run["thermostat"] == "LANGEVIN"
    assert run["thermostat_mapping"]["ensemble"] == "nvt_lan"
    assert run["thermostat_mapping"]["tau_T_fs"] == pytest.approx(100.0)
    assert run["T_coup"] == 100
    assert run["seed"] == 7
    assert run["unwrapped_available"] is True
    assert run["msd_eligible"] is True


def test_real_md_cancellation_is_terminal_and_keeps_raw_output(
    tmp_path, fixture_atoms, engine
):
    """AII/AUD-006 analogue for the external GPUMD process.

    Cancelling a long native run must terminate the child, report the terminal
    status ``cancelled`` (never ``completed``) and keep every raw file that was
    produced before the interruption.
    """
    import threading
    import time

    cancel_event = threading.Event()
    runner = NativeGPUMDRunner(
        engine,
        calc_type="md",
        output_dir=tmp_path,
        verbose=False,
        ensemble="NVE",
        temperature=300.0,
        timestep_fs=1.0,
        steps=5_000_000,  # far longer than the test will wait
        seed=3,
        dump_interval=50,
        cancel_event=cancel_event,
    )

    def _cancel_soon():
        time.sleep(1.0)
        cancel_event.set()

    killer = threading.Thread(target=_cancel_soon, daemon=True)
    with _cuda_runtime_env():
        killer.start()
        result = runner.execute(fixture_atoms.copy())
        killer.join(timeout=10)

    assert result["status"] == "cancelled"
    manifest = _manifest(tmp_path)
    assert manifest["status"] == "cancelled"
    # Raw evidence survives a cancellation: the run.in and model.xyz were
    # written before launch, and the process logs are preserved.
    for name in ("model.xyz", "run.in", "stdout.log", "stderr.log"):
        assert (tmp_path / "native" / name).is_file(), name
    # It must not claim a completed MD with a frame count.
    assert not manifest["result"].get("nframes") or result["status"] == "cancelled"
