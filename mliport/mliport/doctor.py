# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified for the mliport project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
"""Target-aware environment diagnostics for mliport.

The default command is a side-effect-free package and hardware inventory.
Runtime imports happen only when an engine is selected explicitly. This keeps
optional TensorFlow/DeepMD backends from initialising merely because ``doctor``
was asked to list the current environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement

from mliport.gpu_compat import arch_supports_device
from mliport.gpu_setup import TorchRecommendation, recommend_torch
from mliport.install.compatibility import BACKENDS, PROJECT_REQUIRES_PYTHON
from mliport.install.hardware import (
    MIN_VRAM_MIB_WARN,
    GpuInfo,
    cc_arch_name,
    detect_gpus,
)

_PROBE_SENTINEL = "__MLIPORT_DOCTOR_JSON__="

_ENGINE_SPECS: dict[str, dict[str, str]] = {
    "uma": {
        "label": "UMA",
        "distribution": "fairchem-core",
        "module": "fairchem.core",
        "framework": "torch",
        "install": "Run the mliport installer: ./scripts/install_mliport.sh (or `mliport setup` for the machine-specific command).",
    },
    "mace": {
        "label": "MACE",
        "distribution": "mace-torch",
        "module": "mace.calculators",
        "framework": "torch",
        "install": "Install MACE in the dedicated .venv-mace environment.",
    },
    "dpa": {
        "label": "DPA",
        "distribution": "deepmd-kit",
        "module": "deepmd.calculator",
        "framework": "torch",
        "install": "Install deepmd-kit[torch] in the dedicated .venv-dpa environment.",
    },
    "grace": {
        "label": "GRACE",
        "distribution": "tensorpotential",
        "module": "tensorpotential.calculator",
        "framework": "tensorflow",
        "install": "Install tensorpotential in the dedicated .venv-grace environment.",
    },
}


def _installed_dependency_conflicts(
    distribution_names: tuple[str, ...],
) -> list[str]:
    """Return installed requirements that are not satisfied.

    Import-only checks are insufficient for plugin-style engines: MACE can be
    imported with a newer e3nn and then fail only while unpickling a model.
    Inspecting distribution metadata catches that invalid environment before a
    calculation is submitted.
    """
    conflicts: list[str] = []
    for distribution_name in distribution_names:
        try:
            requirements = metadata.requires(distribution_name) or []
        except metadata.PackageNotFoundError:
            continue
        for raw_requirement in requirements:
            requirement = Requirement(raw_requirement)
            # e3nn is the shared, mutually exclusive UMA/MACE dependency.
            # Other matrix pins (notably the legacy-GPU torch channel) are
            # diagnosed by their dedicated checks.
            if requirement.name.lower() != "e3nn":
                continue
            if requirement.marker and not requirement.marker.evaluate():
                continue
            try:
                installed = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                continue
            if requirement.specifier and installed not in requirement.specifier:
                conflicts.append(
                    f"{distribution_name} requires {requirement.name}"
                    f"{requirement.specifier}, but {installed} is installed"
                )
    return conflicts


def _distribution_version(distribution_name: str) -> str | None:
    """Return an installed distribution version without importing its modules."""
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None


def _normalize_device(device: str) -> str:
    """Validate and normalize a doctor device target."""
    value = str(device).strip().lower()
    if value == "gpu":
        return "cuda"
    if value in {"auto", "cpu", "cuda"}:
        return value
    if value.startswith("cuda:") and value[5:].isdigit():
        return value
    raise ValueError("device must be auto, cpu, cuda, gpu, or cuda:N")


def _probe_output_tail(stdout: str, stderr: str, *, limit: int = 8) -> str:
    """Return a compact diagnostic tail from a failed isolated probe."""
    lines = [
        line.strip()
        for line in (stderr.splitlines() + stdout.splitlines())
        if line.strip() and not line.startswith(_PROBE_SENTINEL)
    ]
    return "\n".join(lines[-limit:])


def _probe_environment(device: str) -> dict[str, str]:
    """Use the same immutable identity/isolation contract as queued jobs."""
    from mliport.job_worker import leased_process_environment
    from mliport.queue import resolve_device_uuid

    _, environment = leased_process_environment(
        {"device": device, "device_uuid": resolve_device_uuid(device)}, [], os.environ
    )
    return environment


def _runtime_probe(
    engine: str,
    device: str,
    *,
    framework: str | None = None,
) -> dict[str, Any]:
    """Import one selected backend and execute a tiny framework operation.

    TensorFlow, tensorpotential and DeepMD may configure global runtime state
    during import. Capturing that import in a child process keeps ``doctor``
    itself quiet and prevents an inventory check from mutating the caller.
    """
    spec = _ENGINE_SPECS[engine]
    selected_framework = framework or spec["framework"]
    compute_target = "cpu" if device == "cpu" else "cuda"
    script = f"""
import importlib
import json

payload = {{"ok": False}}
try:
    importlib.import_module({spec['module']!r})
    payload["framework"] = {selected_framework!r}
    if payload["framework"] == "torch":
        import torch
        payload.update(
            torch_version=str(torch.__version__),
            cuda_runtime=str(torch.version.cuda) if torch.version.cuda else None,
            cuda_available=bool(torch.cuda.is_available()),
            device_count=int(torch.cuda.device_count()),
            arch_list=list(torch.cuda.get_arch_list()),
            gpus=[],
        )
        if payload["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                major, minor = torch.cuda.get_device_capability(index)
                payload["gpus"].append({{
                    "name": str(torch.cuda.get_device_name(index)),
                    "major": int(major),
                    "minor": int(minor),
                    "total_memory": int(props.total_memory),
                }})
        target = {compute_target!r}
        if target == "cuda" and not payload["cuda_available"]:
            raise RuntimeError("PyTorch cannot see the requested CUDA device")
        tensor_device = torch.device("cuda:0" if target == "cuda" else "cpu")
        x = torch.tensor([1.0, 2.0, 3.0], device=tensor_device)
        probe_value = float(torch.dot(x, x).item())
        if target == "cuda":
            torch.cuda.synchronize(tensor_device)
        if probe_value != 14.0:
            raise RuntimeError(f"unexpected PyTorch probe value {{probe_value}}")
        payload["compute_probe"] = f"tensor dot product on {{tensor_device}}"
    else:
        import tensorflow as tf
        physical = list(tf.config.list_physical_devices("GPU"))
        payload.update(
            tensorflow_version=str(tf.__version__),
            cuda_available=bool(physical),
            device_count=len(physical),
            gpus=[{{"name": str(item.name)}} for item in physical],
        )
        target = {compute_target!r}
        if target == "cuda" and not physical:
            raise RuntimeError("TensorFlow cannot see the requested CUDA device")
        tf.config.set_soft_device_placement(False)
        tensor_device = "/GPU:0" if target == "cuda" else "/CPU:0"
        with tf.device(tensor_device):
            x = tf.constant([1.0, 2.0, 3.0])
            probe_value = float(tf.reduce_sum(x * x).numpy())
        if probe_value != 14.0:
            raise RuntimeError(f"unexpected TensorFlow probe value {{probe_value}}")
        payload["compute_probe"] = f"tensor dot product on {{tensor_device}}"
    payload["ok"] = True
except BaseException as exc:
    payload.update(
        error_type=type(exc).__name__,
        error=str(exc),
    )

print({_PROBE_SENTINEL!r} + json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["ok"] else 1)
"""
    env = _probe_environment(device)

    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error_type": "TimeoutExpired",
            "error": "Backend import did not finish within 90 seconds",
        }
    except OSError as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    payload_line = next(
        (
            line[len(_PROBE_SENTINEL) :]
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(_PROBE_SENTINEL)
        ),
        None,
    )
    if payload_line is None:
        return {
            "ok": False,
            "error_type": "ProbeProtocolError",
            "error": _probe_output_tail(completed.stdout, completed.stderr)
            or f"Backend probe exited with status {completed.returncode}",
        }
    try:
        payload = json.loads(payload_line)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": f"Invalid backend probe response: {exc}",
        }
    if not payload.get("ok"):
        tail = _probe_output_tail(completed.stdout, completed.stderr)
        if tail:
            payload["detail"] = tail
    return payload


def _model_probe(
    engine: str,
    device: str,
    *,
    model_path: Path,
    task: str,
    head: str | None = None,
    structure_path: Path | None = None,
    default_dtype: str = "float64",
) -> dict[str, Any]:
    """Load a selected model and optionally evaluate one structure in a child."""
    probe_device = "cuda" if device.startswith("cuda") else device
    request = {
        "engine": engine,
        "device": probe_device,
        "model_path": str(model_path.resolve()),
        "task": task,
        "head": head,
        "structure_path": (
            str(structure_path.resolve()) if structure_path is not None else None
        ),
        "default_dtype": default_dtype,
    }
    script = f"""
import json
import tempfile

import numpy as np

request = {request!r}
payload = {{"ok": False}}
try:
    from ase.io import read
    from mliport.calculators.factory import CalculatorFactory
    from mliport.runners.singlepoint import SinglePointRunner

    options = {{}}
    if request["head"] is not None:
        options["head"] = request["head"]
    if request["engine"] == "mace":
        options["default_dtype"] = request["default_dtype"]
    wrapper = CalculatorFactory.create(
        request["engine"],
        request["model_path"],
        device=request["device"],
        task=request["task"],
        strict=True,
        **options,
    )
    calculator = wrapper.get_calculator()
    info = wrapper.info()
    payload.update(
        model_type=str(info.get("model_type", request["engine"])),
        task=str(wrapper.task),
        requested_head=info.get("requested_head", info.get("head")),
        active_head=info.get("active_head"),
        available_heads=info.get("available_heads", []),
        model_precision=info.get("model_precision", info.get("default_dtype")),
    )
    if request["structure_path"] is not None:
        atoms = read(request["structure_path"], index=-1)
        with tempfile.TemporaryDirectory(prefix="mliport-doctor-") as directory:
            runner = SinglePointRunner(
                wrapper,
                output_dir=directory,
                write_outcar=False,
                write_json=False,
                write_contcar=False,
                verbose=False,
            )
            atoms = runner._prepare_atoms(atoms)
        atoms.calc = calculator
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
            raise RuntimeError("model returned a non-finite energy or force")
        evaluation = {{
            "atoms": int(len(atoms)),
            "energy_eV": energy,
            "max_force_eV_A": float(np.max(np.linalg.norm(forces, axis=1))),
        }}
        if atoms.pbc.all() and wrapper.has_stress:
            stress = np.asarray(atoms.get_stress(), dtype=float)
            if not np.all(np.isfinite(stress)):
                raise RuntimeError("model returned non-finite stress")
            evaluation["stress_checked"] = True
        else:
            evaluation["stress_checked"] = False
        payload["evaluation"] = evaluation
    payload["ok"] = True
except BaseException as exc:
    payload.update(error_type=type(exc).__name__, error=str(exc))

print({_PROBE_SENTINEL!r} + json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["ok"] else 1)
"""
    env = _probe_environment(device)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error_type": "TimeoutExpired",
            "error": "Model probe did not finish within 300 seconds",
        }
    except OSError as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}

    payload_line = next(
        (
            line[len(_PROBE_SENTINEL) :]
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(_PROBE_SENTINEL)
        ),
        None,
    )
    if payload_line is None:
        return {
            "ok": False,
            "error_type": "ProbeProtocolError",
            "error": _probe_output_tail(completed.stdout, completed.stderr)
            or f"Model probe exited with status {completed.returncode}",
        }
    try:
        payload = json.loads(payload_line)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": f"Invalid model probe response: {exc}",
        }
    if not payload.get("ok"):
        tail = _probe_output_tail(completed.stdout, completed.stderr)
        if tail:
            payload["detail"] = tail
    return payload


def _should_warn_torch_mismatch(
    *,
    uma_installed: bool,
    installed_torch: str | None,
    recommended_torch: str,
) -> bool:
    """Apply the matrix Torch recommendation only to an UMA environment.

    MACE and DPA intentionally use their own Torch builds. Recommending UMA's
    matrix torch inside a DPA environment would break DeepMD's Torch 2.10 ABI
    requirement even when the installed CUDA kernel already supports the GPU.
    """
    return bool(
        uma_installed and installed_torch and recommended_torch not in installed_torch
    )


def _recommendation_detail(
    rec: TorchRecommendation, *, installed_torch: str | None = None
) -> str:
    """Build a detail string for a GPU from a TorchRecommendation."""
    lines = [rec.rationale]
    if not rec.supported:
        lines.append("")
        lines.append("  Use --device cpu, or upgrade to a Maxwell (GTX 900) GPU.")
        return "\n  ".join(lines)

    lines.append("")
    if installed_torch is not None and installed_torch != rec.version:
        lines.append(
            f"  Installed torch ({installed_torch}) does NOT match the "
            f"recommended ({rec.version}) for this GPU and engine."
        )
        lines.append("  Switch to the recommended build:")
    else:
        lines.append("  Install / pin the recommended build:")
    for cmd in rec.install_commands:
        lines.append(f"    {cmd}")
    lines.append("")
    lines.append("  Or run:  uv run mliport setup")
    lines.append("  After fixing, repeat the same doctor command.")
    return "\n  ".join(lines)


def _runtime_gpu_vram_mib(
    gpu: dict[str, Any], hardware_gpus: list[GpuInfo]
) -> int | None:
    """Return runtime GPU memory, with a validated nvidia-smi fallback.

    Some backend imports leave ``torch.cuda.get_device_properties()`` with a
    zero ``total_memory`` even though CUDA computation succeeds.  A zero is
    unknown, not a real zero-capacity GPU.  Fall back only when nvidia-smi has
    a device with the same name and compute capability; otherwise preserve
    the uncertainty instead of emitting a false low-VRAM warning.
    """
    try:
        runtime_bytes = int(gpu.get("total_memory") or 0)
    except (TypeError, ValueError):
        runtime_bytes = 0
    if runtime_bytes > 0:
        return runtime_bytes // (1024**2)

    try:
        major = int(gpu["major"])
        minor = int(gpu["minor"])
    except (KeyError, TypeError, ValueError):
        return None
    name = str(gpu.get("name") or "")
    matches = [
        item
        for item in hardware_gpus
        if item.name == name
        and item.cc_major == major
        and item.cc_minor == minor
        and item.vram_mib > 0
    ]
    sizes = {item.vram_mib for item in matches}
    if len(sizes) == 1:
        return sizes.pop()
    return None


def run_diagnostics(
    model_path: str | Path | None = None,
    *,
    engine: str = "auto",
    device: str = "auto",
    task: str | None = None,
    head: str | None = None,
    structure_path: str | Path | None = None,
    default_dtype: str = "float64",
) -> tuple[list[dict[str, Any]], int]:
    """Run inventory checks and, when selected, one engine runtime probe.

    Args:
        model_path: Optional model to load in an isolated subprocess.
        engine: ``auto`` for package inventory, or one explicit MLIP engine.
        device: ``auto``, ``cpu``, ``cuda``, ``gpu``, or ``cuda:N``.
        task: Explicit model task/PBC semantic, required with ``model_path``.
        head: Optional MACE head or DPA branch.
        structure_path: Optional structure for a real energy/force smoke test.
        default_dtype: MACE dtype used for the model probe.

    Returns:
        Tuple of (results list, number of failures).
    """
    checks: list[dict[str, Any]] = []
    from mliport.install.inventory import verify_inventory

    try:
        inventory_present = verify_inventory()
        checks.append(
            {
                "name": "Installer inventory",
                "value": "matched" if inventory_present else "not recorded",
                "status": "ok" if inventory_present else "warn",
                "detail": "Exact installed package/RECORD identity checked."
                if inventory_present
                else "Legacy/manual environment: no installer inventory certification.",
            }
        )
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        checks.append(
            {
                "name": "Installer inventory",
                "value": "mismatch",
                "status": "fail",
                "detail": str(exc),
            }
        )
    target_engine = str(engine).strip().lower()
    if target_engine == "fairchem":
        target_engine = "uma"
    if target_engine != "auto" and target_engine not in _ENGINE_SPECS:
        raise ValueError("engine must be auto, uma, mace, dpa, or grace")
    requested_device = _normalize_device(device)
    if default_dtype not in {"float32", "float64"}:
        raise ValueError("default_dtype must be float32 or float64")
    hw_gpus = list(detect_gpus() or [])

    if target_engine == "auto":
        checks.append(
            {
                "name": "Target engine",
                "value": "inventory only",
                "status": "skip",
                "detail": (
                    "No backend is imported in inventory mode. Choose --engine "
                    "uma, mace, dpa, or grace for a runtime check."
                ),
            }
        )
    else:
        checks.append(
            {
                "name": "Target engine",
                "value": _ENGINE_SPECS[target_engine]["label"],
                "status": "ok",
            }
        )

    if requested_device == "auto" and target_engine == "auto":
        resolved_device: str | None = None
        checks.append(
            {
                "name": "Target device",
                "value": "not selected",
                "status": "skip",
                "detail": "Select --engine to resolve auto to CUDA or CPU.",
            }
        )
    else:
        resolved_device = requested_device
        auto_detail = None
        if requested_device == "auto":
            resolved_device = "cuda" if hw_gpus else "cpu"
            auto_detail = (
                "auto selected CUDA because nvidia-smi found an NVIDIA GPU"
                if hw_gpus
                else "auto selected CPU because no NVIDIA GPU was detected"
            )
        checks.append(
            {
                "name": "Target device",
                "value": str(resolved_device).upper(),
                "status": "ok",
                **({"detail": auto_detail} if auto_detail else {}),
            }
        )

    # Python support must match the package metadata, including its upper bound.
    py_ver = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    if (3, 10) <= sys.version_info < (3, 13):
        checks.append({"name": "Python", "value": py_ver, "status": "ok"})
    else:
        checks.append(
            {
                "name": "Python",
                "value": py_ver,
                "status": "fail",
                "detail": f"mliport requires Python >=3.10,<3.13; found {py_ver}",
            }
        )

    mliport_version = _distribution_version("mliport")
    if mliport_version is None:
        checks.append(
            {
                "name": "mliport",
                "value": "not installed",
                "status": "fail",
                "detail": "Install with: cd mliport && python -m pip install -e .",
            }
        )
    else:
        checks.append(
            {"name": "mliport", "value": f"v{mliport_version}", "status": "ok"}
        )

    # nvidia-smi is optional for inventory and CPU targets, required for CUDA.
    cuda_requested = bool(resolved_device and resolved_device.startswith("cuda"))
    if hw_gpus:
        checks.append(
            {
                "name": "NVIDIA driver",
                "value": hw_gpus[0].driver_version,
                "status": "ok",
            }
        )
    elif cuda_requested:
        checks.append(
            {
                "name": "NVIDIA driver",
                "value": "not found",
                "status": "fail",
                "detail": (
                    "nvidia-smi found no usable NVIDIA driver or GPU. "
                    "Install a driver, or request --device cpu."
                ),
            }
        )
    else:
        reason = "CPU target" if resolved_device == "cpu" else "inventory mode"
        checks.append(
            {
                "name": "NVIDIA driver",
                "value": f"not required ({reason})",
                "status": "skip",
            }
        )

    cuda_selection_valid = bool(hw_gpus) if cuda_requested else True
    if cuda_requested and resolved_device and resolved_device.startswith("cuda:"):
        requested_index = int(resolved_device.split(":", 1)[1])
        if hw_gpus and requested_index >= len(hw_gpus):
            cuda_selection_valid = False
            checks.append(
                {
                    "name": "CUDA selection",
                    "value": resolved_device,
                    "status": "fail",
                    "detail": (
                        f"nvidia-smi reports {len(hw_gpus)} GPU(s); index "
                        f"{requested_index} is unavailable."
                    ),
                }
            )

    # Inventory uses distribution metadata only; it never imports an engine.
    installed_versions: dict[str, str | None] = {}
    for engine_name, spec in _ENGINE_SPECS.items():
        version = _distribution_version(spec["distribution"])
        installed_versions[engine_name] = version
        selected = engine_name == target_engine
        if selected:
            backend = BACKENDS[engine_name]
            compatible = (
                py_ver in PROJECT_REQUIRES_PYTHON and py_ver in backend.requires_python
            )
            checks.append(
                {
                    "name": f"{spec['label']} installer Python constraint",
                    "value": py_ver,
                    "status": "ok" if compatible else "fail",
                    "detail": (
                        f"mliport requires-python {PROJECT_REQUIRES_PYTHON}; "
                        f"installer pin {backend.requirement} requires-python "
                        f"{backend.requires_python} (fixed-version PyPI metadata). "
                        "This describes the installer pin, not other installed versions."
                    ),
                }
            )
        if version is not None:
            checks.append(
                {
                    "name": f"{spec['label']} package",
                    "value": f"{spec['distribution']} {version}",
                    "status": "ok",
                }
            )
        elif selected:
            checks.append(
                {
                    "name": f"{spec['label']} package",
                    "value": "not installed",
                    "status": "fail",
                    "detail": spec["install"],
                }
            )
        else:
            checks.append(
                {
                    "name": f"{spec['label']} package",
                    "value": "not installed (optional)",
                    "status": "skip",
                    "detail": spec["install"],
                }
            )

    installed_labels = [
        _ENGINE_SPECS[name]["label"]
        for name, version in installed_versions.items()
        if version is not None
    ]
    if len(installed_labels) > 1:
        checks.append(
            {
                "name": "Environment isolation",
                "value": f"multiple engines ({', '.join(installed_labels)})",
                "status": "warn",
                "detail": (
                    "Dedicated environments are recommended because backend "
                    "Torch/CUDA/TensorFlow requirements may conflict."
                ),
            }
        )
    else:
        checks.append(
            {
                "name": "Environment isolation",
                "value": "one or no engine installed",
                "status": "ok",
            }
        )
    # Detect the known mutually incompatible UMA/MACE e3nn combination before
    # importing either backend. It is fatal only when that environment is the
    # object of the check (or when inventory itself claims to be healthy).
    dependency_conflicts = _installed_dependency_conflicts(
        ("mace-torch", "fairchem-core")
    )
    if dependency_conflicts:
        conflict_status = "fail" if target_engine in {"auto", "uma", "mace"} else "warn"
        checks.append(
            {
                "name": "UMA/MACE dependencies",
                "value": "incompatible",
                "status": conflict_status,
                "detail": (
                    "\n".join(dependency_conflicts)
                    + "\n  Keep UMA and MACE in separate virtual environments."
                ),
            }
        )
    else:
        checks.append(
            {
                "name": "UMA/MACE dependencies",
                "value": "compatible",
                "status": "ok",
            }
        )

    torch_ver = _distribution_version("torch")
    if installed_versions["uma"] is not None and torch_ver is not None:
        fairchem_torch_requirement = None
        try:
            fairchem_requirements = metadata.requires("fairchem-core") or []
        except metadata.PackageNotFoundError:
            fairchem_requirements = []
        for raw_requirement in fairchem_requirements:
            requirement = Requirement(raw_requirement)
            if requirement.name.lower() == "torch" and (
                not requirement.marker or requirement.marker.evaluate()
            ):
                fairchem_torch_requirement = requirement
                break
        if (
            fairchem_torch_requirement is not None
            and fairchem_torch_requirement.specifier
            and torch_ver not in fairchem_torch_requirement.specifier
        ):
            checks.append(
                {
                    "name": "UMA dependency profile",
                    "value": "matrix mismatch",
                    "status": "warn",
                    "detail": (
                        f"Installed torch {torch_ver} does not satisfy FairChem's "
                        f"declared requirement {fairchem_torch_requirement}. "
                        "The mliport compatibility matrix selects torch 2.8.x for "
                        "fairchem-core 2.21.0; a patch-version deviation is a "
                        "warning, not a failure. Runtime versions are recorded "
                        "in MD artifacts.json."
                    ),
                }
            )

    # Runtime imports are explicit and isolated. Package presence alone never
    # earns a green runtime check.
    runtime: dict[str, Any] | None = None
    if target_engine == "auto":
        checks.append(
            {
                "name": "Engine runtime",
                "value": "not imported",
                "status": "skip",
            }
        )
    elif installed_versions[target_engine] is None:
        checks.append(
            {
                "name": "Engine runtime",
                "value": "not checked",
                "status": "skip",
                "detail": "Install the selected engine package first.",
            }
        )
    else:
        assert resolved_device is not None
        probe_framework = _ENGINE_SPECS[target_engine]["framework"]
        if (
            target_engine == "dpa"
            and model_path is not None
            and Path(model_path).suffix.lower() == ".pb"
        ):
            probe_framework = "tensorflow"
        runtime = _runtime_probe(
            target_engine,
            resolved_device,
            framework=probe_framework,
        )
        if runtime.get("ok"):
            checks.append(
                {
                    "name": "Engine runtime",
                    "value": f"{_ENGINE_SPECS[target_engine]['module']} compute succeeded",
                    "status": "ok",
                    "detail": runtime.get("compute_probe"),
                }
            )
            if runtime.get("framework") == "torch":
                framework_value = str(runtime.get("torch_version", "unknown"))
                if runtime.get("cuda_runtime"):
                    framework_value += f" (CUDA {runtime['cuda_runtime']})"
                checks.append(
                    {
                        "name": "PyTorch runtime",
                        "value": framework_value,
                        "status": "ok",
                    }
                )
            else:
                checks.append(
                    {
                        "name": "TensorFlow runtime",
                        "value": str(runtime.get("tensorflow_version", "unknown")),
                        "status": "ok",
                    }
                )
        else:
            error = f"{runtime.get('error_type', 'Error')}: {runtime.get('error', '')}".strip()
            detail = runtime.get("detail")
            if detail:
                error += f"\n{detail}"
            checks.append(
                {
                    "name": "Engine runtime",
                    "value": "compute probe failed",
                    "status": "fail",
                    "detail": error,
                }
            )

    if target_engine != "auto" and resolved_device == "cpu" and runtime:
        if runtime.get("ok"):
            checks.append({"name": "Compute device", "value": "CPU", "status": "ok"})
    elif target_engine != "auto" and cuda_requested:
        if not cuda_selection_valid:
            checks.append(
                {
                    "name": "CUDA runtime",
                    "value": "not checked",
                    "status": "skip",
                    "detail": "Resolve the NVIDIA driver/device selection first.",
                }
            )
        elif runtime and runtime.get("ok"):
            if runtime.get("cuda_available"):
                checks.append(
                    {
                        "name": "CUDA runtime",
                        "value": f"available ({runtime.get('device_count', 0)} visible GPU(s))",
                        "status": "ok",
                    }
                )
            else:
                framework = runtime.get("framework", "selected backend")
                checks.append(
                    {
                        "name": "CUDA runtime",
                        "value": "unavailable",
                        "status": "fail",
                        "detail": (
                            f"{framework} cannot see a CUDA GPU. Check the installed "
                            "framework build and CUDA_VISIBLE_DEVICES, or use --device cpu."
                        ),
                    }
                )

    # PyTorch architecture checks apply only to the explicitly selected
    # PyTorch engine. TensorFlow does not expose an equivalent wheel arch list.
    if (
        target_engine != "auto"
        and cuda_requested
        and runtime
        and runtime.get("ok")
        and runtime.get("framework") == "torch"
        and runtime.get("cuda_available")
    ):
        arch_list = list(runtime.get("arch_list") or [])
        for index, gpu in enumerate(runtime.get("gpus") or []):
            major = int(gpu["major"])
            minor = int(gpu["minor"])
            gpu_cc = f"sm_{major}{minor}"
            vram_mib = _runtime_gpu_vram_mib(gpu, hw_gpus)
            vram_label = (
                f"{vram_mib / 1024:.1f} GB" if vram_mib is not None else "VRAM unknown"
            )
            value = (
                f"{gpu['name']} ({vram_label}, CC {major}.{minor}, "
                f"{cc_arch_name(major, minor)})"
            )
            rec = recommend_torch(major, minor, engine=target_engine)
            if arch_supports_device(gpu_cc, arch_list):
                if rec.supported and _should_warn_torch_mismatch(
                    uma_installed=target_engine == "uma",
                    installed_torch=str(runtime.get("torch_version") or ""),
                    recommended_torch=rec.version,
                ):
                    checks.append(
                        {
                            "name": f"Runtime GPU {index}",
                            "value": value,
                            "status": "warn",
                            "detail": (
                                f"Kernel supports {gpu_cc}, but this UMA environment "
                                f"uses torch {runtime.get('torch_version')} instead of "
                                f"the matrix recommendation {rec.version}."
                            ),
                        }
                    )
                else:
                    checks.append(
                        {"name": f"Runtime GPU {index}", "value": value, "status": "ok"}
                    )
            else:
                checks.append(
                    {
                        "name": f"Runtime GPU {index}",
                        "value": value,
                        "status": "fail",
                        "detail": (
                            f"PyTorch wheel architectures {arch_list!r} do not support "
                            f"{gpu_cc}.\n  "
                            + _recommendation_detail(
                                rec,
                                installed_torch=str(runtime.get("torch_version") or ""),
                            )
                        ),
                    }
                )
            if vram_mib is None:
                checks.append(
                    {
                        "name": f"Runtime GPU {index} VRAM",
                        "value": "unknown",
                        "status": "warn",
                        "detail": (
                            "The framework did not report GPU memory and its device "
                            "could not be matched unambiguously to nvidia-smi."
                        ),
                    }
                )
            elif vram_mib < MIN_VRAM_MIB_WARN:
                checks.append(
                    {
                        "name": f"Runtime GPU {index} VRAM",
                        "value": vram_label,
                        "status": "warn",
                        "detail": (
                            f"Below {MIN_VRAM_MIB_WARN // 1024} GB; use small systems "
                            "and conservative memory settings."
                        ),
                    }
                )

    # Explicit model checks load the checkpoint.  A structure adds one real
    # energy/force evaluation; neither mode writes scientific output files.
    model_ready = False
    mp = Path(model_path).expanduser() if model_path else None
    if mp is None:
        checks.append(
            {
                "name": "Model file",
                "value": "not requested",
                "status": "skip",
                "detail": (
                    "Pass --model PATH --task TASK to load a model; add "
                    "--structure FILE for a real single-point smoke test."
                ),
            }
        )
        if head is not None or structure_path is not None or task is not None:
            checks.append(
                {
                    "name": "Model request",
                    "value": "incomplete",
                    "status": "fail",
                    "detail": "--task, --head, and --structure require --model.",
                }
            )
    elif not mp.exists():
        checks.append(
            {
                "name": "Model file",
                "value": f"{mp} — not found",
                "status": "fail",
                "detail": "Check the model path and selected engine.",
            }
        )
    else:
        wrong_kind = (target_engine == "grace" and not mp.is_dir()) or (
            target_engine in {"uma", "mace", "dpa"} and not mp.is_file()
        )
        if wrong_kind:
            expected = (
                "SavedModel directory" if target_engine == "grace" else "model file"
            )
            checks.append(
                {
                    "name": "Model file",
                    "value": str(mp),
                    "status": "fail",
                    "detail": f"{_ENGINE_SPECS[target_engine]['label']} expects a {expected}.",
                }
            )
        else:
            value = (
                f"{mp.name} (directory)"
                if mp.is_dir()
                else f"{mp.name} ({mp.stat().st_size / (1024**3):.2f} GB)"
            )
            checks.append({"name": "Model file", "value": value, "status": "ok"})
            model_ready = True

    structure_ready = structure_path is None
    structure = Path(structure_path).expanduser() if structure_path else None
    if structure is not None:
        if structure.is_file():
            checks.append(
                {
                    "name": "Smoke structure",
                    "value": str(structure),
                    "status": "ok",
                }
            )
            structure_ready = True
        else:
            checks.append(
                {
                    "name": "Smoke structure",
                    "value": f"{structure} — not found",
                    "status": "fail",
                }
            )

    request_ready = model_ready and structure_ready
    if model_ready and target_engine == "auto":
        request_ready = False
        checks.append(
            {
                "name": "Model load",
                "value": "not attempted",
                "status": "fail",
                "detail": "Loading a model requires an explicit --engine.",
            }
        )
    elif model_ready and task is None:
        request_ready = False
        checks.append(
            {
                "name": "Model load",
                "value": "not attempted",
                "status": "fail",
                "detail": (
                    "Loading a model requires an explicit --task so doctor "
                    "cannot guess PBC or model-task semantics."
                ),
            }
        )
    elif model_ready and target_engine != "auto":
        if (
            installed_versions[target_engine] is None
            or not runtime
            or not runtime.get("ok")
        ):
            checks.append(
                {
                    "name": "Model load",
                    "value": "not attempted",
                    "status": "skip",
                    "detail": "Resolve the selected engine runtime failure first.",
                }
            )
        elif request_ready:
            assert resolved_device is not None
            probe = _model_probe(
                target_engine,
                resolved_device,
                model_path=mp,
                task=str(task),
                head=head,
                structure_path=structure,
                default_dtype=default_dtype,
            )
            if probe.get("ok"):
                details: list[str] = [f"task={probe.get('task', task)}"]
                if probe.get("active_head") is not None:
                    details.append(f"active_head={probe['active_head']}")
                if probe.get("model_precision"):
                    details.append(f"precision={probe['model_precision']}")
                checks.append(
                    {
                        "name": "Model load",
                        "value": "checkpoint loaded",
                        "status": "ok",
                        "detail": ", ".join(details),
                    }
                )
                evaluation = probe.get("evaluation")
                if isinstance(evaluation, dict):
                    checks.append(
                        {
                            "name": "Single-point smoke",
                            "value": (
                                f"{evaluation['atoms']} atoms, "
                                f"E={evaluation['energy_eV']:.8g} eV"
                            ),
                            "status": "ok",
                            "detail": (
                                f"max|F|={evaluation['max_force_eV_A']:.8g} eV/Å; "
                                f"stress_checked={evaluation['stress_checked']}"
                            ),
                        }
                    )
            else:
                error = (
                    f"{probe.get('error_type', 'Error')}: " f"{probe.get('error', '')}"
                ).strip()
                if probe.get("detail"):
                    error += f"\n{probe['detail']}"
                checks.append(
                    {
                        "name": "Model load",
                        "value": "failed",
                        "status": "fail",
                        "detail": error,
                    }
                )
        else:
            checks.append(
                {
                    "name": "Model load",
                    "value": "not attempted",
                    "status": "skip",
                    "detail": "Resolve the smoke-structure path first.",
                }
            )

    failures = sum(check["status"] == "fail" for check in checks)
    return checks, failures


def format_diagnostics(checks: list[dict[str, Any]]) -> str:
    """Format diagnostic results as a readable text table.

    Args:
        checks: List of check result dicts from run_diagnostics().

    Returns:
        Formatted string ready for display.
    """
    icons = {"ok": "✓", "fail": "✗", "warn": "!", "skip": "-"}

    lines = []
    lines.append("")
    lines.append("=" * 68)
    lines.append(" mliport Environment Diagnostic")
    lines.append("=" * 68)
    lines.append("")

    max_name = max(len(c["name"]) for c in checks)

    for c in checks:
        icon = icons.get(c["status"], "?")
        name_padded = c["name"].ljust(max_name)
        value = c.get("value", "")

        if c["status"] == "fail":
            lines.append(f"  {name_padded}  {value:30s}  {icon} FAIL")
        elif c["status"] == "warn":
            lines.append(f"  {name_padded}  {value:30s}  {icon} WARN")
        elif c["status"] == "skip":
            lines.append(f"  {name_padded}  {value:30s}  {icon} SKIP")
        else:
            lines.append(f"  {name_padded}  {value:30s}  {icon}")

        if c.get("detail"):
            for detail_line in c["detail"].split("\n"):
                lines.append(f"    {detail_line}")

    lines.append("")
    lines.append("=" * 68)

    total_fails = sum(1 for c in checks if c["status"] == "fail")
    total_warns = sum(1 for c in checks if c["status"] == "warn")
    target = next(c for c in checks if c["name"] == "Target engine")
    target_device = next(c for c in checks if c["name"] == "Target device")
    inventory_only = target["value"] == "inventory only"
    warning_label = "warning" if total_warns == 1 else "warnings"

    if inventory_only and total_fails == 0:
        warning_text = f"; {total_warns} {warning_label}" if total_warns else ""
        lines.append(f" Inventory complete{warning_text}; no backend was imported.")
        lines.append(
            " For a runtime check, rerun with --engine ENGINE --device DEVICE "
            "(see --help)."
        )
    elif total_fails == 0:
        warning_text = f" ({total_warns} {warning_label})" if total_warns else ""
        lines.append(
            f" Environment checks passed for {target['value']} on "
            f"{target_device['value']}{warning_text}."
        )
    else:
        failure_label = "check" if total_fails == 1 else "checks"
        scope = (
            "the environment inventory"
            if inventory_only
            else f"{target['value']} on {target_device['value']}"
        )
        lines.append(f" {total_fails} required {failure_label} failed for {scope}.")
        lines.append(" Fix the failures above, then repeat the same doctor command.")

    lines.append("=" * 68)

    return "\n".join(lines)
