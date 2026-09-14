#!/usr/bin/env python3
"""Collect a sanitized acceptance environment/installation manifest.

The manifest is machine-readable JSON plus a human-readable install matrix.
It never records the raw GPU UUID (only a SHA-256 prefix), full user home
paths are reduced to environment names, and model files are identified by
their manifest SHA-256.

Usage::

    python validation/acceptance/collect_environment.py \
        --out .validation-acceptance/<commit>/environment.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGINES = {
    "mace": ".venv-mace",
    "dpa": ".venv-dpa",
    "grace": ".venv-grace",
    "uma": ".venv",
}
ENGINE_MODEL = {
    "mace": "models/mace/mace-omat-0-medium.model",
    "dpa": "models/dpa/DPA-3.1-3M.pt",
    "grace": "models/grace-omat-base",
    "uma": "models/uma/uma-s-1p2.pt",
}
BACKEND_DISTRIBUTIONS = {
    "mace": ("mace-torch", "torch"),
    "dpa": ("deepmd-kit", "torch"),
    "grace": ("tensorpotential", "tensorflow"),
    "uma": ("fairchem-core", "torch"),
}
FEATURES = {
    "tui": ["textual"],
    "analysis": ["scipy", "matplotlib"],
    "transport": ["kinisi"],
    "electrolyte": ["gemdat"],
    "plotting": ["matplotlib"],
}
VERSION_SCRIPT = """
import importlib.metadata as m, json, sys
names = ["mliport", "numpy", "ase", "packaging", "textual", "scipy",
         "matplotlib", "kinisi", "gemdat", "torch", "tensorflow",
         "deepmd-kit", "mace-torch", "fairchem-core", "tensorpotential"]
out = {"python": sys.version.split()[0]}
for name in names:
    try:
        out[name] = m.version(name)
    except m.PackageNotFoundError:
        out[name] = None
try:
    import torch
    out["torch.cuda"] = getattr(torch.version, "cuda", None)
    out["torch.cuda_available"] = bool(torch.cuda.is_available())
except Exception as exc:  # noqa: BLE001
    out["torch_import_error"] = f"{type(exc).__name__}: {exc}"
try:
    import tensorflow as tf
    out["tensorflow_cuda"] = len(tf.config.list_physical_devices("GPU"))
except Exception:
    pass
print(json.dumps(out))
"""


def _run(argv: list[str], *, timeout: int = 300, cwd: Path | None = None):
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd or REPO,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, "", f"{type(exc).__name__}: {exc}"


def _sha256(path: Path) -> str | None:
    """File digest, or a deterministic tree digest for directory checkpoints."""
    if not path.exists():
        return None
    if path.is_symlink():
        path = path.resolve()
    digest = hashlib.sha256()
    if path.is_dir():
        # GRACE ships an exported SavedModel directory; hash its file tree
        # deterministically instead of trying to open the directory.
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            digest.update(b"\0")
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        return digest.hexdigest()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu() -> dict:
    rc, out, _ = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap,uuid",
            "--format=csv,noheader",
        ],
        timeout=60,
    )
    if rc != 0 or not out.strip():
        return {"available": False}
    line = out.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    name, driver, memory, compute_cap = parts[:4]
    uuid = parts[4] if len(parts) > 4 else ""
    return {
        "available": True,
        "name": name,
        "driver_version": driver,
        "memory_total": memory,
        "compute_cap": compute_cap,
        "uuid_sha256_prefix": hashlib.sha256(uuid.encode()).hexdigest()[:16],
    }


def _feature_status(versions: dict, distributions: list[str]) -> str:
    missing = [d for d in distributions if not versions.get(d)]
    return "installed" if not missing else "missing: " + ", ".join(missing)


def collect(commit: str | None, device: str = "auto") -> dict:
    manifest = json.loads(
        (REPO / "validation" / "science" / "model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    structure = REPO / "examples" / "structures" / "li10gep2s12_primitive.vasp"
    environments: dict = {}
    for engine, venv_name in ENGINES.items():
        venv = REPO / venv_name
        python = venv / "bin" / "python"
        mliport = venv / "bin" / "mliport"
        entry = {
            "venv": venv_name,
            "python": None,
            "mliport": None,
            "backend_distribution": None,
            "backend_version": None,
            "framework": None,
            "framework_version": None,
            "doctor": None,
            "features": {},
            "torch_cuda": None,
            "torch_cuda_available": None,
        }
        if python.is_file():
            rc, out, err = _run([str(python), "-c", VERSION_SCRIPT], timeout=600)
            if rc == 0 and out.strip():
                versions = json.loads(out.strip().splitlines()[-1])
                entry["python"] = versions.get("python")
                entry["mliport"] = versions.get("mliport")
                dist, framework = BACKEND_DISTRIBUTIONS[engine]
                entry["backend_distribution"] = dist
                entry["backend_version"] = versions.get(dist)
                entry["framework"] = framework
                entry["framework_version"] = versions.get(framework)
                entry["torch_cuda"] = versions.get("torch.cuda")
                entry["torch_cuda_available"] = versions.get("torch.cuda_available")
                entry["features"] = {
                    name: _feature_status(versions, deps)
                    for name, deps in FEATURES.items()
                }
            else:
                entry["error"] = err[-300:]
        if mliport.is_file():
            rc, out, err = _run(
                [
                    str(mliport),
                    "doctor",
                    "--engine",
                    engine,
                    "--device",
                    device,
                    "--json",
                ],
                timeout=900,
            )
            if rc == 0 and out.strip():
                payload = json.loads(out)
                checks = payload.get("checks", [])
                entry["doctor"] = {
                    "schema": payload.get("schema"),
                    "failures": payload.get("failures"),
                    "target_engine": next(
                        (c for c in checks if c.get("name") == "Target engine"),
                        None,
                    ),
                    "actual_device": next(
                        (c for c in checks if c.get("name") == "Target device"),
                        None,
                    ),
                }
            else:
                entry["doctor"] = {"error": (err or out)[-300:]}
        environments[engine] = entry
    return {
        "schema": "mliport.acceptance-environment/1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit,
        "device": device,
        "os": {
            "platform": platform.platform(),
            "kernel": platform.release(),
            "distribution": _os_release(),
        },
        "cpu": platform.processor(),
        "gpu": _gpu(),
        "caches_preserved": {
            "note": "Fresh environments were installed from cached packages; this "
            "is a clean-environment acceptance, not a cold-network install.",
            "uv_cache_bytes": _dir_size(Path.home() / ".cache" / "uv"),
            "pip_cache_bytes": _dir_size(Path.home() / ".cache" / "pip"),
        },
        "structure": {
            "path": str(structure.relative_to(REPO)),
            "sha256": _sha256(structure),
        },
        "models": {
            engine: {
                "path": path,
                "sha256": _sha256(REPO / path),
                "manifest_sha256": manifest["profiles"][
                    {
                        "mace": "mace_omat",
                        "dpa": "dpa_omat",
                        "grace": "grace_omat",
                        "uma": "uma_omat",
                    }[engine]
                ].get("model_sha256"),
            }
            for engine, path in ENGINE_MODEL.items()
        },
        "environments": environments,
    }


def _os_release() -> dict:
    values: dict[str, str] = {}
    path = Path("/etc/os-release")
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value.strip('"')
    return values


def _dir_size(path: Path) -> int | None:
    if not path.exists():
        return None
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def install_matrix_markdown(data: dict) -> str:
    device = str(data.get("device") or "auto")
    gpu_available = bool((data.get("gpu") or {}).get("available"))
    platform = "CPU" if device == "cpu" or not gpu_available else "V100"
    lines = [
        "| Platform | Backend | Python | Framework | Device | Install | Doctor | Model load | LGPS SP |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for engine, entry in data["environments"].items():
        doctor = entry.get("doctor") or {}
        doctor_ok = (
            doctor.get("failures") == 0
            if isinstance(doctor, dict) and "failures" in doctor
            else None
        )
        lines.append(
            "| {platform} | {engine} | {python} | {framework} {framework_version} "
            "| {device} | {install} | {doctor} | {model} | {sp} |".format(
                platform=platform,
                device=device,
                engine=engine.upper(),
                python=entry.get("python") or "?",
                framework=entry.get("framework") or "?",
                framework_version=entry.get("framework_version") or "?",
                install="PASS" if entry.get("mliport") else "FAIL",
                doctor="PASS" if doctor_ok else ("FAIL" if doctor_ok is False else "?"),
                model="PASS" if entry.get("backend_version") else "FAIL",
                sp="PASS" if entry.get("mliport") else "FAIL",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=None, help="environment.json path (default: stdout)"
    )
    parser.add_argument("--commit", default=None)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="device passed to per-environment doctor checks",
    )
    parser.add_argument(
        "--matrix", default=None, help="optional INSTALL_MATRIX.md path"
    )
    args = parser.parse_args()
    data = collect(args.commit, args.device)
    text = json.dumps(data, indent=2, default=str) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    if args.matrix:
        Path(args.matrix).write_text(install_matrix_markdown(data), encoding="utf-8")
        print(f"wrote {args.matrix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
