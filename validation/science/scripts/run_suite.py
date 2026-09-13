"""Sequential orchestrator for the beta validation suite.

Runs the per-tier scripts inside each backend's dedicated virtualenv, one
engine at a time (serial GPU discipline, taskbook section 2.1).  The
orchestrator itself never imports a backend -- it only launches the
backend venv's python against the scripts in this directory and relays
exit codes.  A missing venv produces an explicit ``blocked`` record, not
a silent skip.

Usage::

    python run_suite.py --tiers t1,t2 \
        --engine mace --profile-id mace_omat --model models/mace/... \
        --out .validation-work

Engine venv resolution order: ``--venv`` > ``MLIPORT_SCIENCE_VENV_<ENGINE>``
> ``<repo>/.venv-<engine>``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import common  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent

TIER_COMMANDS: dict[str, list[str]] = {
    "t1": ["inference.py"],
    "t2": ["invariance.py"],
    "t2fd": ["finite_difference.py"],
    "t3": ["omat_eval.py"],
    "t4": ["static_suite.py"],
    "t5": ["neb_suite.py"],
    "t6": ["md_suite.py"],
    "t7": ["analysis_suite.py"],
    "t8": ["performance_suite.py"],
}

#: backend venvs live in the repository root, next to .venv-mace etc.
DEFAULT_VENV_ROOT = str(Path(__file__).resolve().parents[3])


def venv_python(engine: str, override: str | None) -> str | None:
    if override:
        return override
    env_key = f"MLIPORT_SCIENCE_VENV_{engine.upper()}"
    if os.environ.get(env_key):
        return os.environ[env_key]
    default = Path(DEFAULT_VENV_ROOT) / f".venv-{engine}" / "bin" / "python"
    return str(default) if default.exists() else None


def write_blocked(engine: str, reason: str, out_dir: Path) -> None:
    rec = common.result_record(
        case_id="orchestrator",
        test_id="engine_provisioning",
        status="blocked",
        engine=engine,
        model_identity=engine,
        model_sha256="n/a",
        task=None,
        head=None,
        dtype="n/a",
        device={
            "requested": "cuda:0",
            "actual": None,
            "gpu_name": None,
            "gpu_uuid_hash": None,
        },
        input_structure_id=None,
        parameters={"reason": reason},
        metrics={},
    )
    common.write_result(rec, out_dir)


def is_cuda(device: str) -> bool:
    d = str(device).lower()
    return d in {"cuda", "gpu"} or d.startswith("cuda:")


def resolve_gpu_uuid(device: str) -> str | None:
    """Resolve the physical GPU UUID for process-level isolation.

    Backends whose ASE adapter has no per-calculator device (DPA) require
    CUDA_VISIBLE_DEVICES to be fixed to a single GPU before framework
    imports (mliport.devices.require_isolated_visibility).  The orchestrator
    applies that isolation to every engine uniformly.  Raw UUIDs are used
    only for the child process environment; records store their hash.
    """
    if not is_cuda(device):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    uuids = [u.strip() for u in out.stdout.splitlines() if u.strip()]
    if not uuids:
        return None
    if device.startswith("cuda:"):
        suffix = device.split(":", 1)[1]
        idx = int(suffix) if suffix.isdigit() else 0
        return uuids[idx] if idx < len(uuids) else None
    return uuids[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers", default="t1")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--head", default=None)
    parser.add_argument("--dtype", default=None, help="MACE only")
    parser.add_argument("--venv", default=None, help="python executable override")
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--cases",
        default=None,
        help="tier-script case selection forwarded verbatim (t7: md / "
        "transport / gemdat)",
    )
    args = parser.parse_args()

    python = venv_python(args.engine, args.venv)
    out_dir = Path(args.out)
    if python is None or not Path(python).exists():
        write_blocked(
            args.engine,
            f"no backend venv provisioned for engine {args.engine!r} "
            f"(looked at --venv, MLIPORT_SCIENCE_VENV_{args.engine.upper()}, "
            f"{DEFAULT_VENV_ROOT}/.venv-{args.engine})",
            out_dir,
        )
        print(f"[run_suite] {args.engine}: BLOCKED (no venv)", file=sys.stderr)
        return 2

    manifest_profile = json.loads(Path(args.manifest).read_text(encoding="utf-8"))[
        "profiles"
    ][args.profile_id]
    head = args.head or manifest_profile.get("head")
    dtype = args.dtype or manifest_profile.get("dtype")
    dtype_arg = (
        dtype if args.engine == "mace" and dtype in ("float32", "float64") else None
    )
    child_env = os.environ.copy()
    gpu_uuid = resolve_gpu_uuid(args.device)
    if gpu_uuid:
        # process-level device isolation before any framework import
        child_env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    elif is_cuda(args.device) and args.engine == "dpa":
        write_blocked(
            args.engine,
            "cannot resolve a single GPU UUID via nvidia-smi; DPA requires "
            "process-level CUDA_VISIBLE_DEVICES isolation",
            out_dir,
        )
        return 2
    exit_code = 0
    for tier in [t for t in args.tiers.split(",") if t]:
        scripts = TIER_COMMANDS.get(tier)
        if not scripts:
            print(f"[run_suite] unknown tier {tier!r}", file=sys.stderr)
            exit_code = 1
            continue
        for script in scripts:
            cmd = [
                python,
                str(SCRIPTS / script),
                "--engine",
                args.engine,
                "--profile-id",
                args.profile_id,
                "--model",
                args.model,
                "--manifest",
                args.manifest,
                "--out",
                str(out_dir / tier),
                "--device",
                args.device,
            ]
            if head:
                cmd += ["--head", head]
            if dtype_arg:
                cmd += ["--dtype", dtype_arg]
            if args.tag:
                cmd += ["--tag", args.tag]
            if args.cases:
                cmd += ["--cases", args.cases]
            print(f"[run_suite] {args.engine}/{tier}: {Path(script).name}")
            proc = subprocess.run(cmd, check=False, env=child_env)  # noqa: S603
            if proc.returncode != 0:
                exit_code = proc.returncode
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
