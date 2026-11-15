"""Opt-in bounded real-model SP / shared-calculator / finite-difference probe.

Run in a pre-existing isolated backend environment. External timeout is
mandatory for unattended invocation. No downloads/install. The report contains
no paths, raw UUIDs, hostnames or full exception messages.

The probe is device-neutral: ``--device cpu`` proves that the adapter really
runs on the CPU (no GPU UUID is needed) and ``--device cuda:N`` keeps the
runtime UUID confirmation. ``--structure`` replaces the built-in Cu cell, and
``--task`` names the model task family instead of assuming UMA's ``omat``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import time
from pathlib import Path

from ase.build import bulk
from ase.io import read

from mliport.calculators.factory import CalculatorFactory
from mliport.capabilities import model_identity
from mliport.config import resolve_config
from mliport.model_record import (
    build_model_record,
    distribution_version,
    hash_model_path,
)
from mliport.neb.workflow import run_neb_workflow
from mliport.runners.md import MDRunner
from mliport.validation import displacement_checks, evaluate

#: The distribution that provides each backend, for the version block.
BACKEND_DISTRIBUTIONS = {
    "uma": "fairchem-core",
    "mace": "mace-torch",
    "dpa": "deepmd-kit",
    "grace": "tensorpotential",
}

#: Built-in fixture: a rattled four-atom fcc Cu cell (2-32 atoms, no constraints).
CU_CELL_A = 3.65
CU_DISPLACEMENT_A = (0.07, -0.03, 0.02)


def normalise_device(value: str) -> str:
    """Accept exactly ``cpu`` or ``cuda:N``; anything else fails closed."""
    text = str(value).strip().lower()
    if text == "cpu":
        return "cpu"
    if text.startswith("cuda:") and text[5:].isdigit():
        return text
    raise ValueError(f"Unsupported --device {value!r}: use 'cpu' or 'cuda:N'")


def device_type(device: str) -> str:
    """``cpu`` or ``cuda`` -- the device type an evidence id names."""
    return "cpu" if device == "cpu" else "cuda"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=sorted(BACKEND_DISTRIBUTIONS))
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--device",
        required=True,
        help="Execution device: 'cpu' or 'cuda:N'. Required; never inferred.",
    )
    parser.add_argument(
        "--structure",
        type=Path,
        help="ASE-readable structure file (2-32 atoms, no constraints). "
        "Default: the built-in rattled Cu fcc cell.",
    )
    parser.add_argument(
        "--task",
        help="Model task family (UMA: 'omat', 'omol', …). Required for uma; "
        "defaults to 'bulk', the PBC hint for the other backends.",
    )
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--head")
    parser.add_argument(
        "--dpa-runtime",
        dest="dpa_runtime",
        choices=["current", "legacy"],
        help="DPA environment: 'current' (default) or 'legacy' (.venv-dpa-legacy)",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--full", action="store_true", help="Also run 10 MD steps and a five-image NEB"
    )
    parser.add_argument("--raw-dir", type=Path)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate arguments without touching a calculator."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.device = normalise_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    if args.backend == "uma" and not args.task:
        parser.error(
            "uma needs an explicit --task (for example --task omat); "
            "mliport never guesses a UMA task family"
        )
    if args.task is None:
        args.task = "bulk"
    if args.full and args.raw_dir is None:
        parser.error("--full requires --raw-dir (raw scientific outputs are retained)")
    return args


def load_fixture(structure: Path | None) -> tuple[object, dict]:
    """Return the single-point fixture and the record of what it is.

    The report never contains a local path: a user-supplied structure is
    identified by its SHA-256 and its shape, not by its name.
    """
    if structure is None:
        atoms = bulk("Cu", "fcc", a=CU_CELL_A, cubic=True)
        atoms.positions[0] += CU_DISPLACEMENT_A
        return atoms, {
            "fixture": "built-in fcc Cu",
            "element": "Cu",
            "atoms": len(atoms),
            "cell_A": [CU_CELL_A] * 3,
            "pbc": [True] * 3,
            "initial_displacement_A": list(CU_DISPLACEMENT_A),
        }
    atoms = read(str(structure))
    return atoms, {
        "fixture": "user structure",
        "structure_sha256": hash_model_path(structure, error_class=ValueError),
        "symbols": sorted(set(atoms.get_chemical_symbols())),
        "atoms": len(atoms),
        "cell_A": [round(float(value), 6) for value in atoms.cell.array.ravel()],
        "pbc": [bool(value) for value in atoms.pbc],
        "cell_rank": int(atoms.cell.rank),
    }


def device_identity(requested: str, info: dict) -> dict:
    """Confirm the runtime really used the requested device.

    A CPU run must not need (or claim) a GPU UUID; a CUDA run keeps today's
    UUID confirmation and additionally rejects a model that landed on another
    ordinal than the one requested.
    """
    actual_type = info.get("actual_device_type")
    report = {
        "requested_device": requested,
        "actual_device_type": actual_type,
        "actual_device_logical_index": info.get("actual_device_logical_index"),
        "model_precision": info.get("model_precision"),
        "active_head": info.get("active_head"),
        "inference_mode": info.get("inference_mode"),
        "dpa_runtime": info.get("dpa_runtime"),
        "deepmd_version": info.get("deepmd_version"),
    }
    if requested == "cpu":
        if actual_type != "cpu":
            raise RuntimeError(
                f"Requested cpu but the runtime reported {actual_type!r}"
            )
        return report
    expected_index = int(requested.split(":", 1)[1])
    if actual_type != "cuda":
        raise RuntimeError(
            f"Requested {requested} but the runtime reported {actual_type!r}"
        )
    uuid = info.get("actual_device_uuid")
    if not uuid:
        raise RuntimeError(f"Missing runtime UUID confirmation for {requested}")
    report["gpu_uuid_sha256"] = hashlib.sha256(uuid.encode()).hexdigest()
    actual_index = info.get("actual_device_logical_index")
    if actual_index is not None and int(actual_index) != expected_index:
        raise RuntimeError(
            f"Requested {requested} but the model tensors are on "
            f"cuda:{actual_index}"
        )
    return report


def evidence_id(
    device: str, backend: str, model_sha256: str, dtype: str, cache: bool
) -> str:
    """Name the device type, not a host: ``cpu-…`` or ``cuda-…``."""
    return f"{device_type(device)}-{backend}-{model_sha256[:12]}-{dtype}-cache{cache}"


def gpu_record() -> dict | None:
    """Hardware identity for a CUDA run, or None when it cannot be read."""
    try:
        from mliport.install.hardware import detect_gpus
    except Exception:
        return None
    try:
        gpus = detect_gpus() or []
    except Exception:
        return None
    if not gpus:
        return None
    gpu = gpus[0]
    return {
        "name": gpu.name,
        "compute_capability": [gpu.cc_major, gpu.cc_minor],
        "driver": gpu.driver_version,
    }


def peak_vram_MiB(framework: str) -> float | None:
    """Peak GPU memory of this process, or None on a CPU-only runtime."""
    try:
        if framework == "torch":
            import torch

            if not torch.cuda.is_available():
                return None
            return float(torch.cuda.max_memory_allocated() / 2**20)
        import tensorflow as tf

        return float(tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20)
    except Exception:
        return None


#: Backends whose adapter refuses to run in a process that never isolated its
#: visible devices (they cannot change a parent process's GPU visibility).
ISOLATED_PROCESS_BACKENDS = frozenset({"dpa", "grace"})


def check_process_isolation(backend: str, environ) -> None:
    """Refuse to start a backend that needs process-level device isolation.

    The CLI restarts itself with ``CUDA_VISIBLE_DEVICES`` set; a standalone
    probe has to do that itself before the framework is imported. Failing here
    with the exact command is friendlier than the adapter's later refusal.
    """
    if backend not in ISOLATED_PROCESS_BACKENDS or "CUDA_VISIBLE_DEVICES" in environ:
        return
    raise SystemExit(
        f"{backend} requires process-level device isolation before the framework "
        "is imported. Start this probe as:"
        f"\n  CUDA_VISIBLE_DEVICES='' <python> scripts/validate_runtime.py "
        f"{backend} ... --device cpu"
        f"\n  CUDA_VISIBLE_DEVICES=<single GPU UUID> <python> "
        f"scripts/validate_runtime.py {backend} ... --device cuda:0"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    check_process_isolation(args.backend, os.environ)

    def timed_out(*_):
        raise TimeoutError("Bounded validation time limit reached")

    signal.signal(signal.SIGTERM, timed_out)
    started = time.monotonic()
    atoms, fixture = load_fixture(args.structure)
    report = {
        "schema": "mliport.runtime-validation/2",
        "backend": args.backend,
        "python": platform.python_version(),
        "os": platform.system(),
        "model_sha256": hash_model_path(args.model, error_class=ValueError),
        "device": args.device,
        "task": args.task,
        "requested_head": args.head,
        "fixture": fixture,
        "workloads": {},
    }
    kwargs = {
        "uma": {"inference_mode": "default"},
        "mace": {"default_dtype": args.dtype, "head": args.head},
        "dpa": {
            "head": args.head,
            **({"dpa_runtime": args.dpa_runtime} if args.dpa_runtime else {}),
        },
        "grace": {
            "neighbor_cache": not args.no_cache,
            # A GPU memory cap is refused on a CPU device by the GRACE adapter.
            "gpu_memory_limit_mb": 6144 if device_type(args.device) == "cuda" else None,
        },
    }[args.backend]
    framework = "tensorflow" if args.backend == "grace" else "torch"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = (
            subprocess.run(
                ["git", "diff", "--quiet"], check=False, timeout=5
            ).returncode
            != 0
        )
        report.update(source_commit=commit, source_dirty=dirty)
        wrapper = CalculatorFactory.create(
            args.backend,
            args.model,
            device=args.device,
            task=args.task,
            **kwargs,
        )
        atoms.calc = wrapper.get_calculator()
        evaluate(atoms)  # Some upstream models move tensors only on first inference.
        report["runtime"] = device_identity(args.device, wrapper.info())
        if device_type(args.device) == "cuda":
            gpu = gpu_record()
            if gpu is not None:
                report["gpu"] = gpu
        report["versions"] = {
            name: found
            for name, found in (
                (framework, distribution_version(framework)),
                ("mliport", distribution_version("mliport")),
                (
                    args.backend,
                    distribution_version(BACKEND_DISTRIBUTIONS[args.backend]),
                ),
            )
            if found is not None
        }
        layer = {
            "model_type": args.backend,
            "model_path": str(args.model.resolve()),
            "task": args.task,
            "device": args.device,
            **kwargs,
        }
        if args.backend == "grace":
            layer["neighbor_cache"] = not args.no_cache
        resolved = resolve_config(calc_type="neb", cli=layer)
        report["model_identity"] = model_identity(build_model_record(wrapper, resolved))
        report["evidence_id"] = evidence_id(
            args.device,
            args.backend,
            report["model_sha256"],
            args.dtype if args.backend == "mace" else "native",
            not args.no_cache,
        )
        tolerances = (
            (5e-3, 2e-5)
            if "float32" in str(report["runtime"].get("model_precision") or [])
            else (1e-4, 1e-8)
        )
        report["workloads"] = displacement_checks(
            atoms, force_atol=tolerances[0], repeat_atol=tolerances[1]
        )
        if args.full and all(
            value["status"] == "passed" for value in report["workloads"].values()
        ):
            md = MDRunner(
                wrapper,
                ensemble="NVE",
                temperature=100,
                timestep=0.5,
                steps=10,
                seed=42,
                save_interval=1,
                pre_relax=False,
                output_dir=args.raw_dir / "md",
                verbose=False,
            ).run(atoms.copy())
            report["workloads"]["short_md"] = {
                "status": "passed" if md["md_steps"] == 10 else "failed",
                "steps": md["md_steps"],
                "frames": md["trajectory_frame_count"],
            }
            # The five-image NEB reuses the fixture: the last atom is moved
            # 0.25 A along x, so no second structure or model is needed.
            initial = atoms.copy()
            initial.calc = None
            final = initial.copy()
            final.positions[-1] += [0.25, 0.0, 0.0]
            neb_config = resolve_config(
                calc_type="neb",
                cli={
                    **layer,
                    "n_intermediate_images": 3,
                    "climb": True,
                    "neb_pre_max_steps": 30,
                    "max_steps": 50,
                    "endpoint_policy": "relax",
                    "path_convention": "unwrapped",
                    "checkpoint_interval": 5,
                },
            )
            result = run_neb_workflow(
                wrapper,
                neb_config,
                output_dir=args.raw_dir / "neb",
                initial=initial,
                final=final,
                verbose=False,
            )
            report["workloads"]["neb"] = {
                "status": "passed" if result["converged"] else result["status"],
                "total_images": 5,
                "converged": result["converged"],
                "saddle_validation": "not_performed",
            }
        if device_type(args.device) == "cuda":
            peak = peak_vram_MiB(framework)
            if peak is not None:
                report["peak_vram_MiB"] = peak
        report["status"] = (
            "passed"
            if all(v["status"] == "passed" for v in report["workloads"].values())
            else "failed"
        )
    except Exception as exc:
        report.update(
            status="timed_out" if isinstance(exc, TimeoutError) else "failed",
            error_type=type(exc).__name__,
        )
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["workloads"].setdefault("short_md", {"status": "not_run"})
        report["workloads"].setdefault("neb", {"status": "not_run"})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
