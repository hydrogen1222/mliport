"""Tiered entry point for the mliport literature benchmark suite.

One command imports the pinned literature references, evaluates a model
profile on them (tier ``t9``), optionally runs the existing scientific harness
tiers, aggregates the evidence and writes a Markdown/JSON report::

    python validation/science/scripts/benchmark.py \
        --tier core --datasets lips25,lips20,omat24 \
        --engine mace --profile-id mace_omat \
        --model /path/to/mace-omat-0-medium.model \
        --manifest validation/science/model_manifest.json \
        --out .validation-work/literature

Tiers (see ``validation/science/literature/README.md`` for the full contract):

``core``
    Light, CI-friendly: a deterministic handful of structures per dataset and
    only the literature evaluation (t9). No MD, no phonons, no downloads of
    anything larger than the pinned reference files.
``standard``
    The core selection plus the standard selection of every dataset, plus
    inference (t1), static properties (t4: relax/cellrelax/EOS) and a short
    NVE drift check (t6: nve_dt_sweep).
``research``
    The research selection plus the full existing harness (t1, t2, t2fd, t4,
    t5, t6, t7, t8).  The per-tier budgets are the tier scripts' own; this
    orchestrator never extends them.

The orchestrator never imports a backend: it launches each engine venv through
``run_suite.py``, so model loading, device isolation and calculator identity
checks are exactly the ones the existing suite already enforces.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
_SCIENCE_ROOT = SCRIPTS.parent
_REPO_ROOT = _SCIENCE_ROOT.parents[1]
for _path in (str(SCRIPTS), str(_SCIENCE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import common  # noqa: E402
from literature import (  # noqa: E402
    datasets as literature_datasets,
    importers as literature_importers,
)

#: Existing harness tiers executed for each literature tier, with optional
#: workflow restrictions for the scripts that support ``--workflows``.
TIER_PLAN: dict[str, dict[str, Any]] = {
    "core": {"tiers": [], "workflows": {}},
    "standard": {
        "tiers": ["t1", "t4", "t6"],
        "workflows": {"t4": "relax,cellrelax,eos", "t6": "nve_dt_sweep"},
    },
    "research": {
        "tiers": ["t1", "t2", "t2fd", "t4", "t5", "t6", "t7", "t8"],
        "workflows": {},
    },
}

#: Splits that are geometry-only inputs and must be relaxed before scoring.
RELAX_SPLITS = {"task1_unrelaxed"}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _venv_python(engine: str, override: str | None) -> str | None:
    if override:
        return override
    env_key = f"MLIPORT_SCIENCE_VENV_{engine.upper()}"
    if os.environ.get(env_key):
        return os.environ[env_key]
    candidate = Path(common.__file__).resolve().parents[3] / f".venv-{engine}/bin/python"
    return str(candidate) if candidate.exists() else None


def _write_blocked(
    out_dir: Path,
    dataset: str,
    split: str,
    profile_id: str,
    profile: dict[str, Any],
    reason: str,
    device: str,
) -> None:
    record = common.result_record(
        case_id=f"t9_{dataset}",
        test_id=f"t9_{dataset}_{split}",
        status="blocked",
        engine=profile.get("engine", "unknown"),
        model_identity=profile.get("identity", profile.get("upstream_model_id", "unknown")),
        model_sha256=profile.get("model_sha256", "0" * 64),
        task=profile.get("task"),
        head=profile.get("head"),
        dtype=profile.get("dtype", "unknown"),
        device=common.device_record(device),
        input_structure_id=None,
        parameters={"dataset": dataset, "split": split},
        metrics={},
        diagnostics={"reason": reason, "profile_id": profile_id},
        exception=reason,
        inference_mode="literature",
    )
    common.write_result(record, out_dir)


def _run(cmd: list[str], env: dict[str, str] | None = None) -> int:
    print("[benchmark] $ " + " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(cmd, check=False, env=env).returncode  # noqa: S603


def main() -> int:  # noqa: PLR0915 - a single, explicit orchestration path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="core", choices=sorted(TIER_PLAN))
    parser.add_argument(
        "--datasets",
        default="lips25,lips20,omat24",
        help="comma list from the literature registry",
    )
    parser.add_argument("--engine", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--head", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--venv", default=None)
    parser.add_argument(
        "--cache", default=None, help="dataset cache directory (raw upstream files)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never download: reuse the verified cache or fail explicitly",
    )
    parser.add_argument(
        "--reference-manifest",
        default=None,
        help="local manifest for datasets that cannot be downloaded (ms25)",
    )
    parser.add_argument(
        "--import-only",
        action="store_true",
        help="only fetch/import the reference data and write resolved.json",
    )
    parser.add_argument(
        "--skip-existing-tiers",
        action="store_true",
        help="run only t9, skip the tier-plan additions (t1/t4/t6/...)",
    )
    parser.add_argument(
        "--skip-report", action="store_true", help="do not write the Markdown report"
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="cap the structures per split (quick smoke runs; recorded)",
    )
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    registry = literature_datasets.load_registry()
    dataset_ids = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in dataset_ids if d not in registry["datasets"]]
    if unknown:
        parser.error(
            f"unknown dataset(s) {unknown}; known: {sorted(registry['datasets'])}"
        )
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.profile_id not in manifest["profiles"]:
        parser.error(
            f"profile {args.profile_id!r} is not in {args.manifest} "
            f"(known: {sorted(manifest['profiles'])})"
        )
    profile = manifest["profiles"][args.profile_id]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tier_dir = out_dir / "t9"
    python = _venv_python(args.engine, args.venv)

    resolved: dict[str, Any] = {
        "schema": "mliport.literature-benchmark-config/1",
        "generated_utc": _now(),
        "tier": args.tier,
        "tier_plan": TIER_PLAN[args.tier],
        "datasets": dataset_ids,
        "engine": args.engine,
        "profile_id": args.profile_id,
        "profile": profile,
        "model_path": str(Path(args.model).expanduser().resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "device": args.device,
        "dtype": args.dtype or profile.get("dtype"),
        "head": args.head or profile.get("head"),
        "cache": str(literature_datasets.cache_root(args.cache)),
        "offline": bool(args.offline),
        "backend_python": python,
        "selections": {},
        "imports": [],
        "commands": [],
    }

    exit_code = 0
    imported: list[tuple[str, dict[str, Any]]] = []
    for dataset in dataset_ids:
        selection_dir = out_dir / "data" / dataset
        try:
            document = literature_importers.import_dataset(
                dataset,
                tier=args.tier,
                cache=args.cache,
                out_dir=selection_dir,
                registry=registry,
                fetch=True,
                offline=args.offline,
                reference_manifest=args.reference_manifest,
            )
        except literature_datasets.RegistryError as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"[benchmark] {dataset}: BLOCKED ({reason})", file=sys.stderr)
            resolved["imports"].append({"dataset": dataset, "status": "blocked", "reason": reason})
            _write_blocked(
                tier_dir, dataset, "import", args.profile_id, profile, reason, args.device
            )
            # a dataset that cannot be acquired is recorded, never faked
            exit_code = max(exit_code, 2)
            continue
        resolved["selections"][dataset] = {
            "directory": str(selection_dir),
            "splits": {
                split: {
                    "output": meta.get("output"),
                    "selected_records": meta.get("selected_records"),
                    "available_records": meta.get("available_records"),
                    "rule": meta.get("rule"),
                }
                for split, meta in document.get("splits", {}).items()
            },
        }
        resolved["imports"].append(
            {"dataset": dataset, "status": "imported", "splits": list(document.get("splits", {}))}
        )
        imported.append((dataset, document))

    (out_dir / "resolved.json").write_text(
        json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
    )
    if args.import_only:
        print(f"[benchmark] imported {len(imported)} dataset(s); resolved config at {out_dir / 'resolved.json'}")
        return exit_code

    if python is None:
        reason = (
            f"no backend venv for engine {args.engine!r} (pass --venv or set "
            f"MLIPORT_SCIENCE_VENV_{args.engine.upper()})"
        )
        print(f"[benchmark] BLOCKED: {reason}", file=sys.stderr)
        resolved["imports"].append({"dataset": None, "status": "blocked", "reason": reason})
        (out_dir / "resolved.json").write_text(
            json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
        )
        for dataset, document in imported:
            for split in document.get("splits", {}):
                _write_blocked(
                    tier_dir, dataset, split, args.profile_id, profile, reason, args.device
                )
        return 2

    common_args = [
        "--engine",
        args.engine,
        "--profile-id",
        args.profile_id,
        "--model",
        args.model,
        "--manifest",
        args.manifest,
        "--out",
        str(out_dir),
        "--device",
        args.device,
    ]
    if args.head or profile.get("head"):
        common_args += ["--head", args.head or profile["head"]]
    if args.dtype or profile.get("dtype") in ("float32", "float64"):
        common_args += ["--dtype", args.dtype or profile["dtype"]]
    if args.venv:
        common_args += ["--venv", args.venv]
    if args.tag:
        common_args += ["--tag", args.tag]

    run_suite = SCRIPTS / "run_suite.py"
    for dataset, document in imported:
        for split, meta in document.get("splits", {}).items():
            output = meta.get("output")
            if not output:
                continue
            subset = out_dir / "data" / dataset / output
            if not subset.exists():
                reason = f"imported split file is missing: {subset}"
                print(f"[benchmark] {dataset}/{split}: BLOCKED ({reason})", file=sys.stderr)
                _write_blocked(
                    tier_dir, dataset, split, args.profile_id, profile, reason, args.device
                )
                exit_code = max(exit_code, 2)
                continue
            command = [
                sys.executable,
                str(run_suite),
                "--tiers",
                "t9",
                "--dataset",
                dataset,
                "--split",
                split,
                "--subset",
                str(subset),
                *common_args,
            ]
            if split in RELAX_SPLITS:
                command += ["--relax"]
            resolved["commands"].append(
                {"stage": f"t9/{dataset}/{split}", "argv": command}
            )
            code = _run(command)
            exit_code = max(exit_code, 0 if code == 0 else 1)

    plan = TIER_PLAN[args.tier]
    if plan["tiers"] and not args.skip_existing_tiers:
        # one invocation per tier so a tier-specific workflow budget can be
        # forwarded without inventing a per-tier syntax in run_suite.py
        for tier in plan["tiers"]:
            command = [
                sys.executable,
                str(run_suite),
                "--tiers",
                tier,
                *common_args,
            ]
            workflow = plan["workflows"].get(tier)
            if workflow:
                command += ["--workflows", workflow]
            resolved["commands"].append(
                {"stage": f"tiers/{tier}", "argv": command}
            )
            code = _run(command)
            exit_code = max(exit_code, 0 if code == 0 else 1)
    (out_dir / "resolved.json").write_text(
        json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
    )

    aggregate = SCRIPTS / "aggregate.py"
    _run(
        [
            sys.executable,
            str(aggregate),
            "--results",
            str(out_dir),
            "--manifest",
            args.manifest,
            "--out",
            str(out_dir / "summary.json"),
        ]
    )
    if not args.skip_report:
        report = SCRIPTS / "literature_report.py"
        _run(
            [
                sys.executable,
                str(report),
                "--results",
                str(out_dir),
                "--registry",
                str(literature_datasets.registry_path()),
                "--out",
                str(out_dir),
            ]
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
