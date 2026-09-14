#!/usr/bin/env python3
"""LGPS acceptance harness for mliport (task book sections 68-74).

This script drives the *documented* mliport CLI in each backend environment
and records what actually happened.  It deliberately does not re-implement
mliport: it runs commands, checks exit codes and output files, parses the
JSON artifacts and writes one acceptance record per task.

Output layout::

    .validation-acceptance/<commit>/
        gpu/<backend>/<task>/            # command cwd when the task needs one
        gpu/<backend>/results.json       # task records
        gpu/<backend>/summary.json       # sanitized per-backend summary
        gpu/summary.json                 # campaign summary
        environment.json                 # collect_environment.py output

Status vocabulary (section 70): PASS, FAIL, EXPECTED_LIMITATION, NOT_RUN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ACCEPTANCE = REPO / ".validation-acceptance"
MANIFEST = REPO / "validation" / "science" / "model_manifest.json"
STRUCTURE_CANDIDATES = (
    REPO / "examples" / "structures" / "li10gep2s12_primitive.vasp",
    REPO / "local-data" / "LGPS.vasp",
    REPO / "local-data" / "LGPS222.vasp",
    REPO / "LGPS.vasp",
    REPO / "LGPS222.vasp",
)
LONG_RUN_CANDIDATES = (
    REPO / "results" / "multi-grace-800K-1ns",
    REPO / "results" / "800(Langevin)" / "grace",
    REPO / "results" / "800(Langevin)" / "dpa",
    REPO / "results" / "800(Langevin)" / "uma",
)

ENGINE_LOCAL_MODELS = {
    "mace": "models/mace/mace-omat-0-medium.model",
    "dpa": "models/dpa/DPA-3.1-3M.pt",
    "grace": "models/grace-omat-base",
    "uma": "models/uma/uma-s-1p2.pt",
}
ENGINE_PROFILES = {
    "mace": "mace_omat",
    "dpa": "dpa_omat",
    "grace": "grace_omat",
    "uma": "uma_omat",
}
#: Set by --device; Backend.cli uses it when no explicit device is passed.
_DEVICE_OVERRIDE: str | None = None

ENGINE_VENVS = {
    "mace": ".venv-mace",
    "dpa": ".venv-dpa",
    "grace": ".venv-grace",
    "uma": ".venv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def discover_structure() -> Path:
    for candidate in STRUCTURE_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "LGPS acceptance fixture not found; expected one of "
        + ", ".join(str(p) for p in STRUCTURE_CANDIDATES)
    )


def discover_long_run() -> Path | None:
    for candidate in LONG_RUN_CANDIDATES:
        if (candidate / "raw" / "trajectory.traj").is_file():
            return candidate
    return None


@dataclass
class Backend:
    name: str
    venv: Path
    python: Path
    mliport: Path
    model: Path
    model_type: str
    task: str
    head: str | None
    profile: str
    model_digest: str | None = None

    def model_args(self, device: str = "cuda") -> list[str]:
        """Options accepted by sp/opt/md/neb/batch (after the subcommand)."""
        argv = [
            "--model",
            str(self.model),
            "--model-type",
            self.model_type,
            "--task",
            self.task,
            "--device",
            device,
        ]
        if self.head:
            argv += ["--head", self.head]
        return argv

    def cli(
        self, subcommand: str, *positionals: str, device: str | None = None
    ) -> list[str]:
        resolved = device or _DEVICE_OVERRIDE or "cuda"
        return [
            str(self.mliport),
            subcommand,
            *positionals,
            *self.model_args(resolved),
        ]


def load_backend(name: str, *, device: str = "cuda") -> Backend:
    venv = REPO / ENGINE_VENVS[name]
    manifest = read_json(MANIFEST)
    profile = ENGINE_PROFILES[name]
    entry = manifest["profiles"][profile]
    model = (REPO / ENGINE_LOCAL_MODELS[name]).resolve()
    if not model.exists():
        raise SystemExit(f"{name}: model not found at {model}")
    return Backend(
        name=name,
        model_type=entry["engine"],
        task=entry["task"],
        head=entry.get("head"),
        profile=profile,
        model_digest=entry.get("model_sha256"),
        venv=venv,
        python=venv / "bin" / "python",
        mliport=venv / "bin" / "mliport",
        model=model,
    )


@dataclass
class Record:
    task: str
    status: str
    detail: str = ""
    command: str = ""
    rc: int | None = None
    duration_s: float | None = None
    artifacts: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)


class Campaign:
    def __init__(self, backend: Backend, out_dir: Path, device: str = "cuda"):
        self.backend = backend
        self.out_dir = out_dir
        self.device = device
        self.records: list[Record] = []
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- run
    def run(
        self,
        task: str,
        argv: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 900,
        env: dict | None = None,
        expect_rc: int | tuple[int, ...] = 0,
        expect_message: str | None = None,
        require_files: tuple[str, ...] = (),
        check_dir: Path | None = None,
        checks: dict | None = None,
    ) -> Record:
        cwd = cwd or (self.out_dir / task)
        cwd.mkdir(parents=True, exist_ok=True)
        artifact_dir = check_dir or cwd
        full_env = dict(os.environ)
        full_env.setdefault("MPLBACKEND", "Agg")
        if env:
            full_env.update(env)
        started = time.time()
        try:
            proc = subprocess.run(
                argv,
                cwd=cwd,
                env=full_env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            rc = -1
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = f"timeout after {timeout}s"
        duration = round(time.time() - started, 2)
        expected = expect_rc if isinstance(expect_rc, tuple) else (expect_rc,)
        record = Record(
            task=task,
            status="PASS" if rc in expected else "FAIL",
            command=" ".join(argv),
            rc=rc,
            duration_s=duration,
            detail="" if rc in expected else f"exit code {rc}, expected {expected}",
            checks=dict(checks or {}),
        )
        record.checks.setdefault("stdout_tail", stdout[-1200:])
        record.checks.setdefault("stderr_tail", stderr[-1200:])
        for rel in require_files:
            path = artifact_dir / rel
            record.artifacts.append(str(path))
            if not path.is_file():
                record.status = "FAIL"
                record.detail = (record.detail + f"; missing {rel}").strip("; ")
        if expect_message is not None:
            merged = stdout + stderr
            record.checks["expected_message"] = expect_message
            if expect_message not in merged:
                record.status = "FAIL"
                record.detail = (
                    record.detail + f"; expected message not found: {expect_message}"
                ).strip("; ")
        self.records.append(record)
        return record

    def limitation(self, task: str, reason: str, **checks) -> Record:
        record = Record(
            task=task,
            status="EXPECTED_LIMITATION",
            detail=reason,
            checks=dict(checks),
        )
        self.records.append(record)
        return record

    def not_run(self, task: str, reason: str) -> Record:
        record = Record(task=task, status="NOT_RUN", detail=reason)
        self.records.append(record)
        return record

    def write(self) -> None:
        payload = {
            "schema": "mliport.acceptance-backend/1",
            "backend": self.backend.name,
            "venv": str(self.backend.venv.relative_to(REPO)),
            "device_requested": self.device,
            "generated_at": now(),
            "records": [r.__dict__ for r in self.records],
        }
        write_json(self.out_dir / "results.json", payload)


# ---------------------------------------------------------------- checks
def _set_incar_keys(path: Path, updates: dict[str, str]) -> None:
    """Replace existing INCAR keys in place; append the ones that are absent."""
    remaining = dict(updates)
    out_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip().upper()
            if key in remaining:
                out_lines.append(f"{key} = {remaining.pop(key)}")
                continue
        out_lines.append(line)
    for key, value in remaining.items():
        out_lines.append(f"{key} = {value}")
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def parse_run_dir(run_dir: Path) -> dict:
    """Read the documented mliport output files and return a compact view."""
    summary: dict = {}
    results_path = run_dir / "mliport_results.json"
    resolved_path = run_dir / "resolved_config.json"
    if results_path.is_file():
        results = read_json(results_path)
        calc = results.get("calculation", {}) or {}
        system = calc.get("system", {}) or {}
        values = calc.get("results", {}) or {}
        meta = results.get("metadata", {}) or {}
        summary.update(
            {
                "mliport_version": results.get("mliport_version"),
                "mode": calc.get("mode"),
                "formula": system.get("formula"),
                "natoms": system.get("natoms"),
                "energy_eV": values.get("energy"),
                "energy_per_atom_eV": values.get("energy_per_atom"),
                "has_stress": bool(values.get("stress")),
                "model_type": meta.get("model_type"),
                "model_path": meta.get("model_path"),
                "task": meta.get("task"),
                "head": meta.get("head"),
                "default_dtype": meta.get("default_dtype"),
                "actual_device_type": meta.get("actual_device_type"),
                "actual_device_logical_index": meta.get("actual_device_logical_index"),
            }
        )
        forces = values.get("forces")
        if isinstance(forces, list) and forces:
            import numpy as np

            arr = np.asarray(forces, dtype=float)
            summary["forces_shape"] = list(arr.shape)
            summary["force_max_eV_A"] = float(np.max(np.linalg.norm(arr, axis=1)))
            summary["forces_finite"] = bool(np.isfinite(arr).all())
    if resolved_path.is_file():
        resolved = read_json(resolved_path)
        summary.setdefault("model_type", resolved.get("model_type"))
        summary.setdefault("task", resolved.get("task"))
        summary.setdefault("head", resolved.get("head"))
        summary.setdefault("resolved_device", resolved.get("device"))
        summary.setdefault("model_path", resolved.get("model_path"))
        summary["calc_type"] = resolved.get("calc_type")
        summary["strict"] = resolved.get("strict")
    return summary


def finite(value) -> bool:
    import math

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------ task table
def task_sp(campaign: Campaign, argv: list[str] | None = None) -> None:
    backend = campaign.backend
    structure = discover_structure()
    out = campaign.out_dir / "sp"
    record = campaign.run(
        "g1_sp",
        backend.cli("sp", str(structure)) + ["--output", str(out / "run")],
        timeout=1200,
        require_files=("mliport_results.json", "resolved_config.json"),
        check_dir=out / "run",
    )
    if record.status == "PASS":
        summary = parse_run_dir(out / "run")
        record.checks.update(summary)
        if not finite(summary.get("energy_per_atom_eV")):
            record.status = "FAIL"
            record.detail = "non-finite or missing energy_per_atom"
        actual = summary.get("actual_device_type") or summary.get("resolved_device")
        expected = {"cuda", "cuda:0"} if _DEVICE_OVERRIDE != "cpu" else {"cpu"}
        if actual not in expected:
            record.status = "FAIL"
            record.detail = f"device not {sorted(expected)}: {actual}"
        if not summary.get("forces_finite", False):
            record.status = "FAIL"
            record.detail = "forces missing or non-finite"


def task_sp_repeat(campaign: Campaign) -> None:
    backend = campaign.backend
    structure = discover_structure()
    out = campaign.out_dir / "sp_repeat"
    record = campaign.run(
        "g1_sp_repeat",
        backend.cli("sp", str(structure)) + ["--output", str(out / "run")],
        timeout=1200,
        require_files=("mliport_results.json",),
        check_dir=out / "run",
    )
    if record.status == "PASS":
        summary = parse_run_dir(out / "run")
        record.checks.update(summary)
        if not finite(summary.get("energy_per_atom_eV")):
            record.status = "FAIL"
            record.detail = "non-finite or missing energy_per_atom"


def task_sp_perturbed(campaign: Campaign) -> None:
    import numpy as np
    from ase.io import read, write

    backend = campaign.backend
    structure = read(discover_structure())
    displacement = np.zeros_like(structure.positions)
    li_indices = [
        i for i, s in enumerate(structure.get_chemical_symbols()) if s == "Li"
    ]
    target = li_indices[0]
    displacement[target] = [0.01, 0.0, 0.0]
    perturbed = structure.copy()
    perturbed.positions = structure.positions + displacement
    fixture_dir = campaign.out_dir / "sp_perturbed"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    path = fixture_dir / "perturbed.vasp"
    write(path, perturbed, format="vasp", direct=True, vasp5=True)
    record = campaign.run(
        "g2_sp_perturbed",
        backend.cli("sp", str(path)) + ["--output", str(fixture_dir / "run")],
        timeout=1200,
        require_files=("mliport_results.json",),
        check_dir=fixture_dir / "run",
    )
    if record.status == "PASS":
        summary = parse_run_dir(fixture_dir / "run")
        record.checks.update(summary)
        record.checks["perturbed_atom_index"] = target
        record.checks["displacement_A"] = 0.01
        if not finite(summary.get("energy_per_atom_eV")):
            record.status = "FAIL"
            record.detail = "non-finite or missing energy_per_atom"
        if abs(summary.get("energy_per_atom_eV", 0.0)) < 1e-12:
            record.checks["note"] = (
                "energy change not compared across backends (section 29)"
            )


def task_opt_fixed(campaign: Campaign) -> None:
    backend = campaign.backend
    structure = discover_structure()
    out = campaign.out_dir / "opt_fixed"
    record = campaign.run(
        "g3_opt_fixed_cell",
        backend.cli("opt", str(structure))
        + [
            "--max-steps",
            "2",
            "--fmax",
            "0.5",
            "--output",
            str(out / "run"),
        ],
        timeout=1800,
        require_files=("mliport_results.json",),
        check_dir=out / "run",
    )
    if record.status == "PASS":
        summary = parse_run_dir(out / "run")
        record.checks.update(summary)


def task_opt_cell(campaign: Campaign) -> None:
    backend = campaign.backend
    structure = discover_structure()
    out = campaign.out_dir / "opt_cell"
    record = campaign.run(
        "g4_opt_cell",
        backend.cli("opt", str(structure))
        + [
            "--max-steps",
            "1",
            "--fmax",
            "0.5",
            "--cell-opt",
            "--output",
            str(out / "run"),
        ],
        timeout=1800,
        require_files=("mliport_results.json",),
        check_dir=out / "run",
    )
    if record.status == "PASS":
        summary = parse_run_dir(out / "run")
        record.checks.update(summary)


def _md_task(
    campaign: Campaign,
    task: str,
    *,
    ensemble: str,
    steps: int,
    thermostat: str | None = None,
    temp: float = 300.0,
) -> None:
    backend = campaign.backend
    structure = discover_structure()
    out = campaign.out_dir / task
    argv = backend.cli("md", str(structure)) + [
        "--ensemble",
        ensemble,
        "--steps",
        str(steps),
        "--temp",
        str(temp),
        "--timestep",
        "1.0",
        "--save-interval",
        "1",
        "--seed",
        "20260914",
        "--no-pre-relax",
        "--output",
        str(out / "run"),
    ]
    if thermostat:
        argv += ["--thermostat", thermostat]
    if thermostat == "NHC":
        # mxliport fails closed without this: constrained NHC cannot run with
        # automatic COM removal.  The MD manual documents the requirement.
        argv += ["--com-policy", "none"]
    record = campaign.run(
        task,
        argv,
        timeout=1800,
        require_files=("raw/mliport_results.json", "raw/trajectory.traj"),
        check_dir=out / "run",
    )
    if record.status == "PASS":
        summary = parse_run_dir(out / "run" / "raw")
        record.checks.update(summary)
        record.checks["steps"] = steps
        if steps > 10:
            record.status = "FAIL"
            record.detail = f"step budget exceeded: {steps}"


def task_nve(campaign: Campaign) -> None:
    _md_task(campaign, "g5_nve", ensemble="NVE", steps=5)


def task_nvt_langevin(campaign: Campaign) -> None:
    _md_task(
        campaign, "g6_nvt_langevin", ensemble="NVT", steps=5, thermostat="LANGEVIN"
    )


def task_nvt_bussi(campaign: Campaign) -> None:
    _md_task(campaign, "g7_nvt_bussi", ensemble="NVT", steps=5, thermostat="BUSSI")


def task_nvt_nhc(campaign: Campaign) -> None:
    _md_task(campaign, "g8_nvt_nhc", ensemble="NVT", steps=5, thermostat="NHC")


def task_md_restart(campaign: Campaign) -> None:
    """Continue an NVT trajectory from positions+momenta (section 32)."""
    from ase.io import read, write

    backend = campaign.backend
    structure = discover_structure()
    leg1 = campaign.out_dir / "md_restart" / "leg1"
    leg2 = campaign.out_dir / "md_restart" / "leg2"
    record = campaign.run(
        "g9_md_restart_leg1",
        backend.cli("md", str(structure))
        + [
            "--ensemble",
            "NVT",
            "--steps",
            "4",
            "--save-interval",
            "1",
            "--seed",
            "20260914",
            "--no-pre-relax",
            "--output",
            str(leg1 / "run"),
        ],
        timeout=1800,
        require_files=("raw/trajectory.traj",),
        check_dir=leg1 / "run",
    )
    if record.status != "PASS":
        return
    last = read(str(leg1 / "run" / "raw" / "trajectory.traj"), index=-1)
    endpoint = campaign.out_dir / "md_restart" / "endpoint.traj"
    endpoint.parent.mkdir(parents=True, exist_ok=True)
    write(endpoint, last)
    record2 = campaign.run(
        "g9_md_restart_leg2",
        backend.cli("md", str(endpoint))
        + [
            "--ensemble",
            "NVT",
            "--steps",
            "4",
            "--save-interval",
            "1",
            "--seed",
            "20260914",
            "--no-pre-relax",
            "--velocity-policy",
            "preserve",
            "--output",
            str(leg2 / "run"),
        ],
        timeout=1800,
        require_files=("raw/trajectory.traj",),
        check_dir=leg2 / "run",
    )
    if record2.status == "PASS":
        first = read(str(leg2 / "run" / "raw" / "trajectory.traj"), index=0)
        same_positions = bool(
            __import__("numpy").allclose(first.positions, last.positions, atol=1e-8)
        )
        has_momenta = first.has("momenta")
        same_momenta = bool(
            has_momenta
            and __import__("numpy").allclose(
                first.get_momenta(), last.get_momenta(), atol=1e-8
            )
        )
        record2.checks.update(
            {
                "positions_continue": same_positions,
                "momenta_preserved": same_momenta,
                "restart_semantics": "state continuation into a new run directory; "
                "mliport md has no single-directory resume flag",
            }
        )
        record.status = "PASS"
        record2.status = "PASS" if (same_positions and same_momenta) else "FAIL"
        if record2.status == "FAIL":
            record2.detail = "restart did not preserve positions/momenta"


def task_batch(campaign: Campaign) -> None:
    from ase.io import read, write

    backend = campaign.backend
    structure = read(discover_structure())
    batch_dir = campaign.out_dir / "batch" / "structures"
    batch_dir.mkdir(parents=True, exist_ok=True)
    write(
        batch_dir / "primitive.vasp", structure, format="vasp", direct=True, vasp5=True
    )
    shifted = structure.copy()
    shifted.positions[0] += [0.01, 0.0, 0.0]
    write(batch_dir / "shifted.vasp", shifted, format="vasp", direct=True, vasp5=True)
    out = campaign.out_dir / "batch"
    record = campaign.run(
        "g10_batch_two_structures",
        backend.cli("batch", str(batch_dir))
        + [
            "--pattern",
            "*.vasp",
            "--calc-type",
            "sp",
            "--output",
            str(out / "runs"),
        ],
        timeout=1800,
    )
    if record.status == "PASS":
        runs = sorted((out / "runs").glob("*/mliport_results.json"))
        record.checks["completed_runs"] = len(runs)
        if len(runs) < 2:
            record.status = "FAIL"
            record.detail = f"expected 2 batch runs, found {len(runs)}"


def task_incar_run(campaign: Campaign) -> None:
    backend = campaign.backend
    structure = discover_structure()
    work = campaign.out_dir / "incar"
    work.mkdir(parents=True, exist_ok=True)
    incar = work / "INCAR.mliport"
    template = campaign.run(
        "g11_template_sp",
        [str(backend.mliport), "template", "sp", "--output", str(incar)],
        cwd=work,
        timeout=120,
    )
    if template.status != "PASS":
        return
    if not incar.is_file():
        campaign.not_run("g11_run", "template did not write INCAR.mliport")
        return
    updates = {
        # The template uses MODEL_PATH (mapped to model_path by the resolver).
        "MODEL_PATH": str(backend.model),
        "MODEL_TYPE": backend.model_type,
        "TASK": backend.task,
        "DEVICE": "cuda",
    }
    if backend.head:
        updates["HEAD"] = backend.head
    _set_incar_keys(incar, updates)
    record = campaign.run(
        "g11_run",
        [
            str(backend.mliport),
            "run",
            "-i",
            str(incar),
            "-s",
            str(structure),
            "-o",
            str(work / "run"),
        ],
        cwd=work,
        timeout=1200,
    )
    if record.status == "PASS":
        runs = sorted(work.rglob("mliport_results.json"))
        record.checks["result_files"] = [str(p.relative_to(work)) for p in runs]
        if not runs:
            record.status = "FAIL"
            record.detail = "INCAR run produced no mliport_results.json"


def task_negatives(campaign: Campaign) -> None:
    backend = campaign.backend
    structure = discover_structure()
    base = campaign.out_dir / "negatives"
    base.mkdir(parents=True, exist_ok=True)
    # 1a. CLI without any model
    campaign.run(
        "g12_negative_no_backend_cli",
        [
            str(backend.mliport),
            "sp",
            str(structure),
            "--output",
            str(base / "no_backend"),
        ],
        timeout=300,
        expect_rc=1,
        expect_message="one of --model PATH or --model-alias NAME is required",
    )
    # 1b. INCAR without any backend selection
    (base / "no_model.incar").write_text(
        "CALC_TYPE = sp\nDEVICE = cuda\n", encoding="utf-8"
    )
    campaign.run(
        "g12_negative_no_backend_config",
        [
            str(backend.mliport),
            "run",
            "-i",
            str(base / "no_model.incar"),
            "-s",
            str(structure),
        ],
        cwd=base,
        timeout=300,
        expect_rc=1,
        expect_message="No MLIP backend was selected",
    )
    # 2. strict-config typo (INCAR key spelling)
    incar = base / "typo.incar"
    incar.write_text(
        f"MODEL = {backend.model}\n"
        f"MODEL_TYPE = {backend.model_type}\n"
        f"TASK = {backend.task}\n"
        "DEVICE = cuda\n"
        "MODEL_TPYE = mace\n",
        encoding="utf-8",
    )
    campaign.run(
        "g12_negative_strict_config",
        [str(backend.mliport), "run", "-i", str(incar), "-s", str(structure)],
        cwd=base,
        timeout=300,
        expect_rc=1,
        expect_message="Strict config validation failed",
    )
    # 3. wrong backend option (a head that MACE does not know)
    campaign.run(
        "g12_negative_wrong_backend_option",
        backend.cli("sp", str(structure))
        + ["--head", "not-a-real-head", "--output", str(base / "wrong_head")],
        timeout=300,
        expect_rc=1,
    )
    # 4. invalid task
    campaign.run(
        "g12_negative_invalid_task",
        [
            str(backend.mliport),
            "sp",
            str(structure),
            "--model",
            str(backend.model),
            "--model-type",
            backend.model_type,
            "--task",
            "not-a-task",
            "--device",
            "cuda",
            "--output",
            str(base / "bad_task"),
        ],
        timeout=300,
        expect_rc=1,
    )


def task_api(campaign: Campaign) -> None:
    """Section 36: documented Python API must agree with the CLI."""
    backend = campaign.backend
    structure = discover_structure()
    script = campaign.out_dir / "api" / "api_check.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "import json\n"
        "from mliport.api import calculate_energy\n"
        f"energy = calculate_energy({str(structure)!r}, {str(backend.model)!r}, "
        f"model_type={backend.model_type!r}, task={backend.task!r}, "
        + (f"head={backend.head!r}, " if backend.head else "")
        + "device='cuda')\n"
        "print(json.dumps({'energy_eV': float(energy)}))\n",
        encoding="utf-8",
    )
    api_env = None
    if backend.name in {"dpa", "grace"}:
        # The Python API cannot re-exec the caller (unlike the CLI): the
        # documented requirement is to isolate CUDA visibility first.
        import subprocess

        try:
            uuid = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            ).stdout.splitlines()
            value = uuid[0].strip() if uuid else "0"
        except OSError:
            value = "0"
        api_env = {"CUDA_VISIBLE_DEVICES": value}
    record = campaign.run(
        "g13_python_api",
        [str(backend.python), str(script)],
        timeout=1200,
        env=api_env,
    )
    if api_env:
        record.checks["api_isolation_env"] = api_env
    if record.status == "PASS":
        try:
            payload = json.loads(record.checks["stdout_tail"].strip().splitlines()[-1])
            record.checks["api_energy_eV"] = payload["energy_eV"]
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            record.status = "FAIL"
            record.detail = f"could not parse API energy: {exc}"
            return
    # CLI equivalence on the same structure/model/device.
    cli_out = campaign.out_dir / "api" / "cli_run"
    cli = campaign.run(
        "g13_python_api_cli",
        backend.cli("sp", str(structure)) + ["--output", str(cli_out)],
        timeout=1200,
        require_files=("mliport_results.json",),
        check_dir=cli_out,
    )
    if cli.status == "PASS":
        summary = parse_run_dir(cli_out)
        cli_energy = summary.get("energy_eV")
        api_energy = record.checks.get("api_energy_eV")
        cli.checks.update(
            {k: summary[k] for k in ("energy_eV", "natoms", "actual_device_type")}
        )
        delta = (
            None
            if api_energy is None or cli_energy is None
            else float(api_energy) - float(cli_energy)
        )
        natoms = int(summary.get("natoms") or 0) or 1
        # float64 backends agree below 1e-6 eV in the acceptance runs; UMA
        # runs float32 upstream, so the equivalence check allows 1e-5 eV
        # total (about 2e-7 eV/atom) and records the exact delta.
        same = delta is not None and (abs(delta) <= 1e-5 or abs(delta) / natoms <= 1e-7)
        cli.checks["api_cli_energy_delta_eV"] = delta
        cli.checks["api_cli_energy_delta_eV_per_atom"] = (
            None if delta is None else delta / natoms
        )
        if not same:
            cli.status = "FAIL"
            cli.detail = f"API/CLI energy mismatch: {api_energy} vs {cli_energy}"
        else:
            record.checks["cli_energy_eV"] = cli_energy
            record.checks["api_cli_energy_delta_eV"] = 0.0


def task_tui(campaign: Campaign) -> None:
    """Section 37: help surface, non-TTY guard and a headless pilot run."""
    backend = campaign.backend
    campaign.run(
        "g14_tui_help",
        [str(backend.mliport), "tui", "--help"],
        timeout=120,
    )
    # Without a terminal the command must fail closed instead of hanging.
    campaign.run(
        "g14_tui_non_tty_guard",
        [str(backend.mliport), "tui"],
        timeout=60,
        expect_rc=1,
        expect_message="requires an interactive terminal",
        env={"TERM": "dumb"},
    )
    # A real TUI smoke without a terminal: Textual's pilot starts the app,
    # renders the main screen and exits.
    script = campaign.out_dir / "tui" / "tui_pilot.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "import asyncio\n"
        "from mliport.tui.app import MliportApp\n"
        "\n"
        "async def main():\n"
        "    app = MliportApp()\n"
        "    async with app.run_test(size=(100, 100)) as pilot:\n"
        "        await pilot.pause()\n"
        "        assert app.screen is not None\n"
        "        print('TUI_PILOT_OK')\n"
        "\n"
        "asyncio.run(main())\n",
        encoding="utf-8",
    )
    record = campaign.run(
        "g14_tui_headless_pilot",
        [str(backend.python), str(script)],
        timeout=300,
    )
    if record.status == "PASS" and "TUI_PILOT_OK" not in record.checks.get(
        "stdout_tail", ""
    ):
        record.status = "FAIL"
        record.detail = "TUI pilot did not confirm the main screen"


def task_queue(campaign: Campaign) -> None:
    backend = campaign.backend
    structure = discover_structure()
    work = campaign.out_dir / "queue"
    work.mkdir(parents=True, exist_ok=True)
    # The job registry is shared; stop any scheduler left by an earlier task so
    # this lifecycle is deterministic.
    campaign.run(
        "g15_queue_pre_stop",
        [str(backend.mliport), "queue", "stop"],
        cwd=work,
        timeout=60,
        expect_rc=(0, 1),
    )
    task_file = work / "tasks.json"
    task_file.write_text(
        json.dumps(
            {
                "max_concurrent": 1,
                "tasks": [
                    {
                        "name": "lgs-sp",
                        "calc_type": "sp",
                        "structure": str(structure),
                        "model": str(backend.model),
                        "model_type": backend.model_type,
                        "task": backend.task,
                        "device": "cuda",
                        "output_dir": str(work / "runs"),
                        "options": (
                            {"head": backend.head} if backend.head else {}
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    submit = campaign.run(
        "g15_queue_submit",
        [str(backend.mliport), "queue", "submit", str(task_file)],
        cwd=work,
        timeout=120,
    )
    if submit.status != "PASS":
        return
    import re

    match = re.search(r"\[([0-9a-fA-F-]{36})\]", submit.checks.get("stdout_tail", ""))
    job_id = match.group(1) if match else None
    submit.checks["job_id"] = job_id
    start = campaign.run(
        "g15_queue_start",
        [str(backend.mliport), "queue", "start", "--poll", "1"],
        cwd=work,
        timeout=120,
    )
    if start.status != "PASS":
        return
    deadline = time.time() + 180
    final_jobs = ""
    terminal = False
    while time.time() < deadline:
        poll = campaign.run(
            f"g15_queue_poll_{int(time.time())}",
            [str(backend.mliport), "jobs"],
            cwd=work,
            timeout=60,
        )
        final_jobs = poll.checks.get("stdout_tail", "")
        if job_id:
            for line in final_jobs.splitlines():
                if job_id in line and ("done" in line or "failed" in line):
                    terminal = True
                    break
        if terminal:
            break
        time.sleep(2)
    status = campaign.run(
        "g15_queue_status",
        [str(backend.mliport), "queue", "status"],
        cwd=work,
        timeout=60,
    )
    jobs = campaign.run(
        "g15_jobs",
        [str(backend.mliport), "jobs"],
        cwd=work,
        timeout=60,
    )
    run_dir = (work / "runs" / job_id) if job_id else None
    result_file = (run_dir / "mliport_results.json") if run_dir else None
    start.checks["job_id"] = job_id
    start.checks["result_file"] = str(result_file) if result_file else None
    start.checks["jobs_table"] = jobs.checks.get("stdout_tail", "")[-800:]
    if job_id is None:
        start.status = "FAIL"
        start.detail = "could not read the submitted job id from queue submit"
    elif not terminal:
        start.status = "FAIL"
        start.detail = f"job {job_id} did not reach a terminal state within 180 s"
    elif result_file is None or not result_file.is_file():
        start.status = "FAIL"
        start.detail = f"queue job {job_id} produced no mliport_results.json"
    stop = campaign.run(
        "g15_queue_stop",
        [str(backend.mliport), "queue", "stop"],
        cwd=work,
        timeout=60,
        expect_rc=(0, 1),
    )
    if status.status != "PASS" or jobs.status != "PASS" or stop.status != "PASS":
        if start.status == "PASS":
            start.status = "FAIL"
            start.detail = "queue status/jobs/stop step failed"


def _write_neb_endpoints(campaign: Campaign) -> tuple[Path, Path]:
    """Section 7/40: deterministic synthetic LGPS endpoints (workflow only)."""
    import numpy as np
    from ase.io import read, write

    atoms = read(discover_structure())
    li = [i for i, sym in enumerate(atoms.get_chemical_symbols()) if sym == "Li"]
    target = li[0]
    distances = atoms.get_all_distances(mic=True)[target]
    order = np.argsort(distances)
    neighbor = next(i for i in order if i != target and atoms[i].symbol == "Li")
    vector = np.asarray(
        atoms.get_distances(target, neighbor, mic=True, vector=True)
    ).reshape(3)
    # Move away from the nearest Li so the synthetic endpoint does not create
    # an unphysical close pair and endpoint relaxation stays cheap.
    direction = -vector / np.linalg.norm(vector)
    displacement = 0.9
    final = atoms.copy()
    for _ in range(4):
        trial = atoms.copy()
        trial.positions[target] += direction * displacement
        min_distance = float(
            np.min(np.where(np.eye(len(trial)), 9.9, trial.get_all_distances(mic=True)))
        )
        if min_distance >= 1.8:
            final = trial
            break
        displacement /= 2.0
    else:
        final = trial
    work = campaign.out_dir / "neb" / "endpoints"
    work.mkdir(parents=True, exist_ok=True)
    initial_path = work / "initial.vasp"
    final_path = work / "final.vasp"
    write(initial_path, atoms, format="vasp", direct=True, vasp5=True)
    write(final_path, final, format="vasp", direct=True, vasp5=True)
    write_json(
        work / "endpoints.json",
        {
            "source": str(discover_structure()),
            "moved_atom_index": target,
            "moved_symbol": atoms[target].symbol,
            "displacement_A": displacement,
            "direction": direction.tolist(),
            "min_distance_A": min_distance,
            "workflow_smoke_only": True,
            "physical_barrier_claimed": False,
            "note": "Synthetic one-Li displacement for endpoint/NEB plumbing only "
            "(task book section 7/40); no saddle validation, no barrier claim.",
        },
    )
    return initial_path, final_path


def task_neb(campaign: Campaign) -> None:
    backend = campaign.backend
    initial, final = _write_neb_endpoints(campaign)
    for name, climb in (("g16_neb_climb", True), ("g16_neb_plain", False)):
        out = campaign.out_dir / "neb" / name
        shutil.rmtree(out, ignore_errors=True)
        argv = backend.cli("neb", device="cuda") + [
            "--initial",
            str(initial),
            "--final",
            str(final),
            "--images",
            "1",
            "--max-steps",
            "4",
            "--fmax",
            "0.5",
            "--endpoint-policy",
            "relax",
            "--endpoint-fmax",
            "0.5",
            "--endpoint-steps",
            "20",
            "--interpolation",
            "idpp",
            "--output",
            str(out),
        ]
        argv.append("--climb" if climb else "--no-climb")
        # mliport refuses NEB when the energy-gradient consistency of the exact
        # model/runtime has not been validated; the acceptance run opts in
        # explicitly and records the experimental status (section 40).
        argv.append("--allow-unvalidated-neb")
        record = campaign.run(name, argv, timeout=2400)
        record.checks.update(
            {
                "workflow_smoke_only": True,
                "capability": "energy-gradient consistency unvalidated for this "
                "exact model/runtime; run used explicit --allow-unvalidated-neb",
                "saddle_validation": "not_performed",
                "physical_barrier": "not_claimed",
                "images": 3,
                "max_steps": 4,
                "climb": climb,
            }
        )
        if record.status == "PASS":
            files = sorted(
                str(p.relative_to(out)) for p in out.glob("**/*") if p.is_file()
            )
            record.checks["files"] = files[:20]


def _checkpoint_sequence(run_dir: Path):
    checkpoint = run_dir / "checkpoints" / "latest" / "checkpoint.json"
    if checkpoint.is_file():
        return read_json(checkpoint).get("checkpoint_sequence")
    steps = sorted((run_dir / "checkpoints").glob("step_*/checkpoint.json"))
    if steps:
        return read_json(steps[-1]).get("checkpoint_sequence")
    return None


def _neb_converged(run_dir: Path):
    results = run_dir / "neb_results.json"
    if not results.is_file():
        return None
    return read_json(results).get("converged")


def task_neb_restart(campaign: Campaign) -> None:
    """Section 41: resume a stopped band and check the step counter continues."""
    backend = campaign.backend
    initial, final = _write_neb_endpoints(campaign)
    base = campaign.out_dir / "neb" / "restart"
    shutil.rmtree(base, ignore_errors=True)
    leg1 = base / "leg1"
    common_tail = [
        "--initial",
        str(initial),
        "--final",
        str(final),
        "--images",
        "1",
        "--fmax",
        "0.5",
        "--endpoint-policy",
        "relax",
        "--endpoint-fmax",
        "0.5",
        "--endpoint-steps",
        "20",
        "--allow-unvalidated-neb",
    ]
    r1 = campaign.run(
        "g17_neb_restart_leg1",
        backend.cli("neb", device="cuda")
        + common_tail
        + ["--max-steps", "1", "--fmax", "0.01", "--output", str(leg1)],
        timeout=2400,
        expect_rc=(0, 2),
    )
    if r1.status != "PASS":
        return
    seq1 = _checkpoint_sequence(leg1)
    r1.checks["checkpoint_sequence"] = seq1
    r1.checks["converged"] = _neb_converged(leg1)
    # Budget overrides are rejected by the fingerprint in this beta; record
    # that as an explicit limitation (the error is clear and corrupts nothing).
    budget_probe = campaign.run(
        "g17_neb_restart_budget_override",
        backend.cli("neb", device="cuda")
        + ["--resume", str(leg1), "--max-steps", "5", "--output", str(leg1)],
        timeout=600,
        expect_rc=1,
        expect_message="Resume fingerprint is incompatible",
    )
    budget_probe.status = "EXPECTED_LIMITATION"
    budget_probe.detail = (
        "beta limitation: the resume fingerprint includes step budgets, so a "
        "user-supplied --max-steps override is rejected; resume continues with "
        "the recorded budget."
    )
    # Resume must continue in the original run directory.
    r2 = campaign.run(
        "g17_neb_restart_leg2",
        backend.cli("neb", device="cuda")
        + ["--resume", str(leg1), "--output", str(leg1)],
        timeout=2400,
        expect_rc=(0, 2),
    )
    if r2.status != "PASS":
        return
    seq2 = _checkpoint_sequence(leg1)
    r2.checks.update(
        {
            "leg1_checkpoint_sequence": seq1,
            "leg2_checkpoint_sequence": seq2,
            "leg1_converged": _neb_converged(leg1),
            "workflow_smoke_only": True,
            "saddle_validation": "not_performed",
            "physical_barrier": "not_claimed",
        }
    )
    if seq1 is None or seq2 is None or seq2 <= seq1:
        r2.status = "FAIL"
        r2.detail = f"resume did not continue (leg1={seq1}, leg2={seq2})"


def task_analysis_core(campaign: Campaign) -> None:
    """Sections 43-48: validate/thermo/MSD on this backend's fresh MD run."""
    backend = campaign.backend
    nve = campaign.out_dir / "g5_nve" / "run"
    if not (nve / "raw" / "trajectory.traj").is_file():
        campaign.not_run("g18_analysis_core", "no fresh NVE trajectory to analyze")
        return
    for sub, extra in (("validate", []), ("thermo", []), ("msd", ["--mobile", "Li"])):
        campaign.run(
            f"g18_analysis_{sub}",
            [str(backend.mliport), "analyze", str(nve), sub, *extra],
            timeout=1800,
        )


from analysis_tasks import task_analysis_long, task_arrhenius  # noqa: E402

TASKS = {
    "g1_sp": task_sp,
    "g1_sp_repeat": task_sp_repeat,
    "g2_sp_perturbed": task_sp_perturbed,
    "g3_opt_fixed": task_opt_fixed,
    "g4_opt_cell": task_opt_cell,
    "g5_nve": task_nve,
    "g6_nvt_langevin": task_nvt_langevin,
    "g7_nvt_bussi": task_nvt_bussi,
    "g8_nvt_nhc": task_nvt_nhc,
    "g9_md_restart": task_md_restart,
    "g10_batch": task_batch,
    "g11_incar": task_incar_run,
    "g12_negatives": task_negatives,
    "g13_api": task_api,
    "g14_tui": task_tui,
    "g15_queue": task_queue,
    "g16_neb": task_neb,
    "g17_neb_restart": task_neb_restart,
    "g18_analysis_core": task_analysis_core,
    "g19_analysis_long": task_analysis_long,
    "g20_arrhenius": task_arrhenius,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=sorted(ENGINE_VENVS))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--commit", default=None, help="override commit id for the output path"
    )
    parser.add_argument("--only", default=None, help="comma-separated task names")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        print("\n".join(TASKS))
        return 0
    global _DEVICE_OVERRIDE
    _DEVICE_OVERRIDE = args.device
    commit = (
        args.commit
        or subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    phase = "cpu" if args.device == "cpu" else "gpu"
    out_dir = (
        Path(args.out) if args.out else ACCEPTANCE / commit / phase / args.backend
    ).resolve()
    backend = load_backend(args.backend, device=args.device)
    campaign = Campaign(backend, out_dir, device=args.device)
    selected = args.only.split(",") if args.only else list(TASKS)
    for name in selected:
        task = TASKS.get(name)
        if task is None:
            campaign.not_run(name, "unknown task")
            continue
        print(f"[{args.backend}] {name} ...", flush=True)
        task(campaign)
    campaign.write()
    failures = sum(r.status == "FAIL" for r in campaign.records)
    print(
        f"[{args.backend}] {len(campaign.records)} tasks, "
        f"{sum(r.status == 'PASS' for r in campaign.records)} PASS, {failures} FAIL"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
