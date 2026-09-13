"""Exact-release four-backend bridge smoke (recovery plan sections 11-15, 46-49).

The full scientific campaign targets ``5f8d91d``; the final release candidate
contains later productization/configuration changes.  This bridge verifies, at
one explicit release commit, that:

- each backend loads its pinned model and runs a finite single point on the
  same known structure;
- an explicit backend resolves to the correct calculator and device;
- omitting the backend fails closed (no implicit UMA/default fallback);
- a misspelled configuration key is fatal under strict config;
- the ``fairchem`` alias still resolves to UMA.

It does not regenerate long trajectories or any other scientific tier.
Records are small JSON documents under ``--out`` plus a campaign summary.

Usage::

    python validation/science/scripts/release_bridge.py \
        --release-commit <40-hex> \
        --campaign 20260913-release-bridge-<short> \
        --out validation/science/bridge/<campaign> \
        [--manifest validation/science/model_manifest.json] \
        [--models-root .] [--device cuda:0] [--engines mace,dpa,grace,uma]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "mliport"))

from mliport.install.compatibility import BACKENDS  # noqa: E402

SCIENTIFIC_CAMPAIGN_TARGET = "5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c"
BRIDGE_RECORD_SCHEMA = "mliport.release-bridge/1"
BRIDGE_SUMMARY_SCHEMA = "mliport.release-bridge-campaign/1"
NO_BACKEND_MESSAGE = "No MLIP backend was selected"
STRICT_CONFIG_MESSAGES = ("Strict config validation failed", "Unknown key")

ENGINE_CASES: dict[str, dict[str, Any]] = {
    "mace": {
        "profile_id": "mace_omat",
        "model": "models/mace/mace-omat-0-medium.model",
    },
    "dpa": {
        "profile_id": "dpa_omat",
        "model": "models/dpa/DPA-3.1-3M.pt",
        "extra_cli": ("--head", "Omat24"),
    },
    "grace": {
        "profile_id": "grace_omat",
        "model": "models/grace-omat-base",
    },
    "uma": {
        "profile_id": "uma_omat",
        "model": "models/uma/uma-s-1p2.pt",
        "extra_cli": ("--task", "omat"),
    },
}

#: Deterministic 4-atom fcc Cu cell (a = 3.6 A): a small periodic structure
#: every pinned model family contains in its training distribution.
KNOWN_STRUCTURE = """Cu4 fcc bridge
3.6000000000
0.0000000000 1.8000000000 1.8000000000
1.8000000000 0.0000000000 1.8000000000
1.8000000000 1.8000000000 0.0000000000
Cu
4
Direct
0.0000000000 0.0000000000 0.0000000000
0.5000000000 0.5000000000 0.0000000000
0.5000000000 0.0000000000 0.5000000000
0.0000000000 0.5000000000 0.5000000000
"""


# --------------------------------------------------------------- pure helpers
def bridge_status(checks: dict[str, dict[str, Any]]) -> str:
    """pass only when every check reports pass (missing checks fail)."""
    if not checks:
        return "fail"
    return "pass" if all(c.get("status") == "pass" for c in checks.values()) else "fail"


def evaluate_negative(
    *,
    exit_code: int,
    output: str,
    expected_phrases: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    """A negative test passes only when it fails *as expected*."""
    matched = [phrase for phrase in expected_phrases if phrase in output]
    status = "pass" if exit_code != 0 and matched else "fail"
    return {
        "status": status,
        "label": label,
        "expected": "non-zero exit and one of " + repr(expected_phrases),
        "observed_exit_code": exit_code,
        "observed_message": next(
            (
                line.strip()
                for line in output.splitlines()
                if any(p in line for p in expected_phrases)
            ),
            "",
        ),
    }


def check_single_point(
    result: dict[str, Any] | None,
    resolved: dict[str, Any] | None,
    *,
    engine: str,
    model_path: str,
) -> dict[str, Any]:
    """Finite energy/forces/stress plus explicit-backend attribution."""
    if not isinstance(result, dict) or not isinstance(resolved, dict):
        return {"status": "fail", "reason": "missing result/resolved JSON"}
    metadata = result.get("metadata", {}) or {}
    calculation = result.get("calculation", {}) or {}
    results = calculation.get("results", {}) or {}
    energy = results.get("energy")
    forces = results.get("forces")
    stress = results.get("stress")
    has_stress = bool(metadata.get("has_stress"))
    finite = (
        isinstance(energy, int | float)
        and energy == energy  # NaN guard
        and abs(energy) != float("inf")
    )
    forces_ok = isinstance(forces, list) and bool(forces)
    if forces_ok:
        for vector in forces:
            if not isinstance(vector, list) or any(
                not isinstance(value, int | float)
                or value != value
                or abs(value) == float("inf")
                for value in vector
            ):
                forces_ok = False
                break
    stress_ok = True
    if has_stress:
        stress_ok = (
            isinstance(stress, list)
            and bool(stress)
            and all(
                isinstance(value, int | float)
                and value == value
                and abs(value) != float("inf")
                for value in stress
            )
        )
    requested_type = str(resolved.get("model_type"))
    metadata_type = str(metadata.get("model_type"))
    resolved_path = Path(str(resolved.get("model_path") or "")).resolve()
    ok = bool(
        finite
        and forces_ok
        and stress_ok
        and requested_type == engine
        and metadata_type == engine
        and resolved_path == Path(model_path).resolve()
    )
    failure_reason = ""
    if not ok:
        failure_reason = (
            f"finite={finite} forces_ok={forces_ok} stress_ok={stress_ok} "
            f"resolved_type={requested_type!r} metadata_type={metadata_type!r} "
            f"resolved_path={str(resolved_path)!r}"
        )
    return {
        "status": "pass" if ok else "fail",
        "reason": failure_reason,
        "energy_eV": energy,
        "fmax_eV_A": (results.get("force_statistics") or {}).get("fmax"),
        "natoms": (calculation.get("system") or {}).get("natoms"),
        "stress_supported": has_stress,
        "requested_model_type": requested_type,
        "resolved_model_type": metadata_type,
        "requested_device": resolved.get("device"),
        "actual_device_type": metadata.get("actual_device_type"),
        "actual_device_uuid": metadata.get("actual_device_uuid"),
        "implemented_properties": metadata.get("implemented_properties"),
        "model_precision": metadata.get("model_precision"),
        "mliport_version": result.get("mliport_version"),
    }


def check_device_attribution(
    single_point: dict[str, Any], requested_device: str
) -> dict[str, Any]:
    """A requested CUDA run must report a CUDA actual device."""
    if "cuda" not in str(requested_device):
        return {"status": "pass", "detail": "cpu requested"}
    actual = str(single_point.get("actual_device_type") or "unknown")
    return {
        "status": "pass" if actual.startswith("cuda") else "fail",
        "requested": requested_device,
        "actual": actual,
        "detail": "no silent CPU fallback for a CUDA request",
    }


def check_alias_resolution(
    resolved: dict[str, Any] | None, result: dict[str, Any] | None
) -> dict[str, Any]:
    """``fairchem`` must resolve through the UMA runtime."""
    if not isinstance(resolved, dict) or not isinstance(result, dict):
        return {"status": "fail", "reason": "missing alias result"}
    resolved_type = str(resolved.get("model_type"))
    metadata_type = str((result.get("metadata") or {}).get("model_type"))
    ok = resolved_type == "fairchem" and metadata_type == "uma"
    return {
        "status": "pass" if ok else "fail",
        "resolved_model_type": resolved_type,
        "runtime_model_type": metadata_type,
        "detail": "fairchem is a legitimate alias for the UMA runtime",
    }


# ------------------------------------------------------------- I/O helpers
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(child.read_bytes())
    return digest.hexdigest()


def model_digest(path: Path) -> tuple[str, str]:
    if path.is_dir():
        return "tree", tree_sha256(path)
    return "file", sha256_file(path)


def run_command(
    argv: list[str], *, cwd: Path, env: dict[str, str], timeout: int = 600
) -> dict[str, Any]:
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output": result.stdout + result.stderr,
    }


def child_env(device: str, campaign: str, release_commit: str) -> dict[str, str]:
    """Child-process environment with documented device isolation.

    DPA/GRACE fail closed unless ``CUDA_VISIBLE_DEVICES`` was set before
    framework import.  The bridge sets it for its own child processes only
    (single GPU, no comma), exactly like the queue worker; it never changes
    the caller's environment.
    """
    env = dict(os.environ)
    env["MLIPORT_SOFTWARE_COMMIT"] = release_commit
    env["MLIPORT_VALIDATION_CAMPAIGN"] = campaign
    if "cuda" in str(device):
        visibility = env.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visibility or "," in visibility:
            index = str(device).split(":", 1)[1] if ":" in str(device) else "0"
            env["CUDA_VISIBLE_DEVICES"] = index
        env.pop("DEVICE", None)
        env["LOCAL_RANK"] = "0"
    return env


def venv_tool(engine: str) -> Path:
    """Console script of the compatibility-registry environment for engine."""
    return REPO / BACKENDS[engine].venv_name / "bin" / "mliport"


def venv_python(engine: str) -> Path:
    return REPO / BACKENDS[engine].venv_name / "bin" / "python"


def environment_identity(engine: str) -> dict[str, Any]:
    python = venv_python(engine)
    distribution = BACKENDS[engine].distribution
    code = (
        "import sys, importlib.metadata as m\n"
        "print(sys.version.split()[0])\n"
        "print(m.version('mliport'))\n"
        f"print(m.version({distribution!r}))\n"
    )
    result = subprocess.run(
        [str(python), "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    lines = result.stdout.splitlines()
    if result.returncode != 0 or len(lines) < 3:
        return {
            "venv": BACKENDS[engine].venv_name,
            "error": (result.stdout + result.stderr).strip(),
        }
    return {
        "venv": BACKENDS[engine].venv_name,
        "python": lines[0],
        "mliport_version": lines[1],
        "backend_distribution": distribution,
        "backend_version": lines[2],
    }


def resolve_profile(manifest: dict[str, Any], profile_id: str) -> dict[str, Any]:
    profiles = manifest.get("profiles", {})
    if profile_id not in profiles:
        raise KeyError(f"profile {profile_id!r} not in model manifest")
    return dict(profiles[profile_id])


def cleanup_work_dirs(out_dir: Path) -> None:
    """Drop raw child-run outputs, keeping only bridge evidence records."""
    import shutil  # noqa: PLC0415

    for engine_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        for name in (
            "sp",
            "scratch",
            "negative-no-backend",
            "negative-strict-config",
        ):
            shutil.rmtree(engine_dir / name, ignore_errors=True)
        (engine_dir / "strict-typo.incar").unlink(missing_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )


# ------------------------------------------------------------- orchestration
def _extra_cli(profile: dict[str, Any], case: dict[str, Any]) -> list[str]:
    args = list(case.get("extra_cli", ()))
    dtype = str(profile.get("dtype") or "")
    if "upstream" not in dtype and dtype and "--dtype" not in args:
        args += ["--dtype", dtype]
    return args


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_engine_bridge(
    engine: str,
    *,
    case: dict[str, Any],
    profile: dict[str, Any],
    model_path: Path,
    structure: Path,
    out_dir: Path,
    campaign: str,
    release_commit: str,
    device: str,
) -> dict[str, Any]:
    engine_dir = out_dir / engine
    engine_dir.mkdir(parents=True, exist_ok=True)
    scratch = engine_dir / "scratch"
    scratch.mkdir(exist_ok=True)
    tool = venv_tool(engine)
    env = child_env(device, campaign, release_commit)
    extra = _extra_cli(profile, case)

    sp_output = engine_dir / "sp"
    positive = run_command(
        [
            str(tool),
            "sp",
            str(structure),
            "--model",
            str(model_path),
            "--model-type",
            engine,
            *extra,
            "--device",
            device,
            "--output",
            str(sp_output),
        ],
        cwd=scratch,
        env=env,
    )
    resolved = _read_json(sp_output / "resolved_config.json")
    result = _read_json(sp_output / "mliport_results.json")
    model_load = {
        "status": "pass"
        if positive["exit_code"] == 0 and result is not None
        else "fail",
        "exit_code": positive["exit_code"],
        "detail": (positive["output"].strip().splitlines() or [""])[-1],
    }
    single_point = check_single_point(
        result, resolved, engine=engine, model_path=str(model_path)
    )
    device_check = check_device_attribution(single_point, device)

    negative_dir = engine_dir / "negative-no-backend"
    negative_dir.mkdir(parents=True, exist_ok=True)
    negative_run = run_command(
        [
            str(tool),
            "sp",
            str(structure),
            "--model",
            str(model_path),
            "--output",
            str(negative_dir / "run"),
        ],
        cwd=negative_dir,
        env=env,
    )
    no_backend = evaluate_negative(
        exit_code=negative_run["exit_code"],
        output=negative_run["output"],
        expected_phrases=(NO_BACKEND_MESSAGE,),
        label="no_implicit_fallback",
    )

    incar = engine_dir / "strict-typo.incar"
    incar.write_text(
        "CALC_TYPE = SP\n"
        f"MODEL_TYPE = {engine.upper()}\n"
        f"MODEL_TPYE = {engine.upper()}\n"
        f"MODEL_PATH = {model_path}\n",
        encoding="utf-8",
    )
    strict_dir = engine_dir / "negative-strict-config"
    strict_dir.mkdir(parents=True, exist_ok=True)
    strict_run = run_command(
        [
            str(tool),
            "run",
            "--incar",
            str(incar),
            "--structure",
            str(structure),
            "--output",
            str(strict_dir / "run"),
        ],
        cwd=strict_dir,
        env=env,
    )
    strict_config = evaluate_negative(
        exit_code=strict_run["exit_code"],
        output=strict_run["output"],
        expected_phrases=STRICT_CONFIG_MESSAGES,
        label="strict_config",
    )

    checks = {
        "model_load": model_load,
        "single_point": single_point,
        "device_attribution": device_check,
        "no_implicit_fallback": no_backend,
        "strict_config": strict_config,
    }
    digest_kind, digest = model_digest(model_path)
    record = {
        "schema": BRIDGE_RECORD_SCHEMA,
        "campaign_id": campaign,
        "release_candidate_commit": release_commit,
        "scientific_campaign_target": SCIENTIFIC_CAMPAIGN_TARGET,
        "engine": engine,
        "profile": {
            "profile_id": case["profile_id"],
            "identity": profile.get("identity"),
            "dtype": profile.get("dtype"),
            "task": profile.get("task"),
            "head": profile.get("head"),
            "manifest_model_sha256": profile.get("model_sha256"),
        },
        "model": {
            "path": str(model_path),
            "digest_kind": digest_kind,
            "computed_sha256": digest,
        },
        "environment": environment_identity(engine),
        "structure": {"path": str(structure), "sha256": sha256_file(structure)},
        "checks": checks,
        "status": bridge_status(checks),
    }
    write_json(engine_dir / f"{engine}.json", record)
    return record


def run_alias_bridge(
    *,
    out_dir: Path,
    structure: Path,
    model_path: Path,
    campaign: str,
    release_commit: str,
    device: str,
) -> dict[str, Any]:
    """Resolve ``fairchem`` through the real UMA runtime (no CLI choice bump).

    The CLI ``--model-type`` choices mirror the documented engine names; the
    alias contract lives in the backend-selection layer, so it is verified
    through the factory/resolver in the UMA environment instead.
    """
    alias_dir = out_dir / "fairchem-alias"
    alias_dir.mkdir(parents=True, exist_ok=True)
    env = child_env(device, campaign, release_commit)
    program = (
        "import json\n"
        "from mliport.calculators.factory import CalculatorFactory\n"
        "from mliport.config.resolver import resolve_config\n"
        f"model = {str(model_path)!r}\n"
        "wrapper = CalculatorFactory.create('fairchem', model, task='omat', device='cpu')\n"
        "resolved = resolve_config(calc_type='sp', cli={\n"
        "    'model_type': 'fairchem', 'model_path': model, 'task': 'omat'})\n"
        "print(json.dumps({\n"
        "    'wrapper_module': type(wrapper).__module__,\n"
        "    'wrapper_class': type(wrapper).__name__,\n"
        "    'resolved_model_type': resolved.model_type,\n"
        "}))\n"
    )
    run = run_command(
        [str(venv_python("uma")), "-c", program],
        cwd=alias_dir,
        env=env,
        timeout=300,
    )
    payload: dict[str, Any] = {}
    if run["exit_code"] == 0:
        try:
            payload = json.loads(run["stdout"].strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            payload = {}
    runtime_type = (
        "uma" if str(payload.get("wrapper_class")) == "UMACalculator" else "not-uma"
    )
    resolved = {"model_type": payload.get("resolved_model_type")}
    result = {"metadata": {"model_type": runtime_type}}
    check = check_alias_resolution(resolved, result)
    check["wrapper_module"] = payload.get("wrapper_module")
    check["wrapper_class"] = payload.get("wrapper_class")
    check["exit_code"] = run["exit_code"]
    if run["exit_code"] != 0:
        check["status"] = "fail"
        check["reason"] = (run["output"].strip().splitlines() or [""])[-1]
    write_json(alias_dir / "alias.json", {"checks": {"fairchem_alias": check}})
    return check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-commit", required=True)
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--manifest", default="validation/science/model_manifest.json")
    parser.add_argument("--models-root", default=".")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--engines", default="mace,dpa,grace,uma")
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="keep each engine's raw SP/config-run outputs (default: records only)",
    )
    args = parser.parse_args()

    release_commit = args.release_commit.strip().lower()
    if len(release_commit) != 40 or any(
        ch not in "0123456789abcdef" for ch in release_commit
    ):
        print(
            "[bridge] --release-commit must be a 40-character hex sha", file=sys.stderr
        )
        return 2
    campaign = args.campaign or f"20260913-release-bridge-{release_commit[:8]}"
    # Every path handed to a child process must be absolute: the children run
    # with an isolated cwd, so a relative structure/output path would resolve
    # inside that scratch directory (this bit the first b3 bridge run).
    out_dir = Path(args.out or f"validation/science/bridge/{campaign}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    structure = (out_dir / "structure.vasp").resolve()
    structure.write_text(KNOWN_STRUCTURE, encoding="utf-8")
    manifest_path = (REPO / args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    unknown = [e for e in engines if e not in ENGINE_CASES]
    if unknown:
        print(f"[bridge] unknown engines: {unknown}", file=sys.stderr)
        return 2

    records: dict[str, dict[str, Any]] = {}
    for engine in engines:
        case = ENGINE_CASES[engine]
        profile = resolve_profile(manifest, case["profile_id"])
        model_path = (REPO / args.models_root / case["model"]).resolve()
        if not model_path.exists():
            records[engine] = {
                "schema": BRIDGE_RECORD_SCHEMA,
                "campaign_id": campaign,
                "release_candidate_commit": release_commit,
                "engine": engine,
                "checks": {
                    "model_present": {
                        "status": "fail",
                        "reason": f"missing model {model_path}",
                    }
                },
                "status": "fail",
            }
            write_json(out_dir / engine / f"{engine}.json", records[engine])
            continue
        records[engine] = run_engine_bridge(
            engine,
            case=case,
            profile=profile,
            model_path=model_path,
            structure=structure,
            out_dir=out_dir,
            campaign=campaign,
            release_commit=release_commit,
            device=args.device,
        )

    alias_profile = resolve_profile(manifest, ENGINE_CASES["uma"]["profile_id"])
    alias_model = (REPO / args.models_root / ENGINE_CASES["uma"]["model"]).resolve()
    if alias_model.exists():
        alias = run_alias_bridge(
            out_dir=out_dir,
            structure=structure,
            model_path=alias_model,
            campaign=campaign,
            release_commit=release_commit,
            device=args.device,
        )
    else:
        alias = {
            "status": "fail",
            "reason": f"missing UMA model for the fairchem alias check: {alias_model}",
        }

    engine_status = {engine: record["status"] for engine, record in records.items()}
    negative_status = {
        f"{engine}:{name}": check["status"]
        for engine, record in records.items()
        for name, check in record.get("checks", {}).items()
        if name in {"no_implicit_fallback", "strict_config"}
    }
    overall = bridge_status(
        {
            **{f"engine:{e}": {"status": s} for e, s in engine_status.items()},
            **{
                f"negative:{name}": {"status": s} for name, s in negative_status.items()
            },
            "fairchem_alias": alias,
        }
    )
    summary = {
        "schema": BRIDGE_SUMMARY_SCHEMA,
        "campaign_id": campaign,
        "release_candidate_commit": release_commit,
        "scientific_campaign_target": SCIENTIFIC_CAMPAIGN_TARGET,
        "scientific_target_note": (
            "The full scientific campaign was produced at the scientific "
            "target; this bridge re-verifies selection/load/inference/config "
            "paths at the release candidate without regenerating long "
            "trajectories."
        ),
        "status": overall,
        "engines": engine_status,
        "negative_checks": negative_status,
        "fairchem_alias": alias,
        "records": [f"{engine}/{engine}.json" for engine in records],
        "environment": {engine: environment_identity(engine) for engine in records},
        "model_manifest": {
            "path": str(manifest_path.relative_to(REPO)),
            "sha256": sha256_file(manifest_path),
        },
        "structure": {"path": "structure.vasp", "sha256": sha256_file(structure)},
        "device": args.device,
    }
    write_json(out_dir / "summary.json", summary)
    if not args.keep_work:
        cleanup_work_dirs(out_dir)
    print(
        f"[bridge] campaign={campaign} release={release_commit[:8]} "
        f"status={overall} engines={engine_status} alias={alias['status']}"
    )
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
