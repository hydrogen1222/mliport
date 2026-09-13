"""Opt-in bounded real-model SP / shared-calculator / finite-difference probe.

Run in a pre-existing isolated backend environment with one visible GPU UUID.
External timeout is mandatory for unattended invocation. No downloads/install.
The report contains no paths, raw UUIDs, hostnames or full exception messages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import signal
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

from ase.build import bulk

from mliport.calculators.factory import CalculatorFactory
from mliport.capabilities import model_identity
from mliport.config import resolve_config
from mliport.neb.workflow import _hash_model, _model_record, run_neb_workflow
from mliport.runners.md import MDRunner
from mliport.validation import displacement_checks, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["uma", "mace", "dpa", "grace"])
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--head")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--full", action="store_true", help="Also run 10 MD steps and a five-image NEB"
    )
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args()
    if args.full and args.raw_dir is None:
        parser.error("--full requires --raw-dir (raw scientific outputs are retained)")

    def timed_out(*_):
        raise TimeoutError("Bounded validation time limit reached")

    signal.signal(signal.SIGTERM, timed_out)
    started = time.monotonic()
    report = {
        "schema": "mliport.runtime-validation/1",
        "backend": args.backend,
        "python": platform.python_version(),
        "os": platform.system(),
        "model_sha256": _hash_model(args.model),
        "task": "omat" if args.backend == "uma" else "bulk",
        "requested_head": args.head,
        "workloads": {},
    }
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
        kwargs = {
            "uma": {"inference_mode": "default"},
            "mace": {"default_dtype": args.dtype, "head": args.head},
            "dpa": {"head": args.head},
            "grace": {"neighbor_cache": not args.no_cache, "gpu_memory_limit_mb": 6144},
        }[args.backend]
        wrapper = CalculatorFactory.create(
            args.backend, args.model, device="cuda:0", task=report["task"], **kwargs
        )
        atoms = bulk("Cu", "fcc", a=3.65, cubic=True)
        atoms.positions[0] += [0.07, -0.03, 0.02]
        atoms.calc = wrapper.get_calculator()
        evaluate(atoms)  # Some upstream models move tensors only on first inference.
        info = wrapper.info()
        if info.get("actual_device_type") != "cuda" or not info.get(
            "actual_device_uuid"
        ):
            raise RuntimeError("Missing runtime UUID confirmation")
        report["runtime"] = {
            key: info.get(key)
            for key in (
                "actual_device_type",
                "actual_device_logical_index",
                "model_precision",
                "active_head",
                "inference_mode",
            )
        }
        report["runtime"]["gpu_uuid_sha256"] = hashlib.sha256(
            info["actual_device_uuid"].encode()
        ).hexdigest()
        from mliport.install.hardware import detect_gpus

        gpu = (detect_gpus() or [])[0]
        report["gpu"] = {
            "name": gpu.name,
            "compute_capability": [gpu.cc_major, gpu.cc_minor],
            "driver": gpu.driver_version,
        }
        backend_dist = {
            "uma": "fairchem-core",
            "mace": "mace-torch",
            "dpa": "deepmd-kit",
            "grace": "tensorpotential",
        }[args.backend]
        framework = "tensorflow" if args.backend == "grace" else "torch"
        report["versions"] = {
            framework: version(framework),
            "mliport": version("mliport"),
            args.backend: version(backend_dist),
        }
        layer = {
            "model_type": args.backend,
            "model_path": str(args.model.resolve()),
            "task": report["task"],
            "device": "cuda:0",
            **kwargs,
        }
        if args.backend == "grace":
            layer["neighbor_cache"] = not args.no_cache
        resolved = resolve_config(calc_type="neb", cli=layer)
        report["model_identity"] = model_identity(_model_record(wrapper, resolved))
        report["evidence_id"] = (
            f"v100-{args.backend}-{report['model_sha256'][:12]}-{args.dtype if args.backend == 'mace' else 'native'}-cache{not args.no_cache}"
        )
        report["fixture"] = {
            "element": "Cu",
            "atoms": 4,
            "cell_A": [3.65] * 3,
            "pbc": [True] * 3,
            "initial_displacement_A": [0.07, -0.03, 0.02],
        }
        tolerances = (
            (5e-3, 2e-5)
            if "float32" in info.get("model_precision", [])
            else (1e-4, 1e-8)
        )
        report["workloads"] = displacement_checks(
            atoms, force_atol=tolerances[0], repeat_atol=tolerances[1]
        )
        if args.full and all(
            v["status"] == "passed" for v in report["workloads"].values()
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
            initial = bulk("Cu", "fcc", a=3.65, cubic=True)
            del initial[0]
            final = initial.copy()
            final.positions[-1] = [0, 0, 0]
            neb_config = resolve_config(
                calc_type="neb",
                cli={
                    **layer,
                    "n_intermediate_images": 3,
                    "climb": True,
                    "neb_pre_max_steps": 30,
                    "max_steps": 50,
                    "endpoint_policy": "validate",
                    "path_convention": "unwrapped",
                    "checkpoint_interval": 5,
                    "allow_unvalidated_neb": True,
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
        if framework == "torch":
            import torch

            report["peak_vram_MiB"] = torch.cuda.max_memory_allocated() / 2**20
        else:
            import tensorflow as tf

            report["peak_vram_MiB"] = (
                tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20
            )
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
