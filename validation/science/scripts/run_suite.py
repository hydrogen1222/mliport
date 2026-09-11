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

Engine venv resolution order: ``--venv`` > ``MLIPX_SCIENCE_VENV_<ENGINE>``
> ``/mnt/hdd500/mlipx/.venv-<engine>``.
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
}

DEFAULT_VENV_ROOT = "/mnt/hdd500/mlipx"


def venv_python(engine: str, override: str | None) -> str | None:
    if override:
        return override
    env_key = f"MLIPX_SCIENCE_VENV_{engine.upper()}"
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
        device={"requested": "cuda:0", "actual": None, "gpu_name": None, "gpu_uuid_hash": None},
        input_structure_id=None,
        parameters={"reason": reason},
        metrics={},
    )
    common.write_result(rec, out_dir)


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
    args = parser.parse_args()

    python = venv_python(args.engine, args.venv)
    out_dir = Path(args.out)
    if python is None or not Path(python).exists():
        write_blocked(
            args.engine,
            f"no backend venv provisioned for engine {args.engine!r} "
            f"(looked at --venv, MLIPX_SCIENCE_VENV_{args.engine.upper()}, "
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
    dtype_arg = dtype if args.engine == "mace" and dtype in ("float32", "float64") else None

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
            print(f"[run_suite] {args.engine}/{tier}: {Path(script).name}")
            proc = subprocess.run(cmd, check=False)  # noqa: S603
            if proc.returncode != 0:
                exit_code = proc.returncode
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
