#!/usr/bin/env python
"""T6 confsearch workflow validation (spec section 13.3).

Owner-run, GPU-time validation of the configuration-search *workflow*. It is
**not** a CI suite and it makes no accuracy claim against DFT: it compares

* (i)  N random placements relaxed without MD (the multi-start baseline, run
       through the product with ``--cycles 0``), and
* (ii) confsearch with the same N starts (same seed, so the same placement
       stream),

on one small public system -- CH3OH on a slab cut from the repository's LGPS
structure (``examples/structures/li10gep2s12_primitive.vasp``) -- with one real
backend, and reports best energies, unique counts and the discovery curve.

The only claim is workflow behaviour: the search must never report a worse
"lowest found" than its own baseline, the candidate counts must be consistent
with the registry, and the discovery curve must be monotone in the number of
unique minima. Absolute energies from a foundation model on a sulfide surface
are not DFT-quality and must be re-ranked before any chemistry is read off
them.

Usage::

    .venv-mace/bin/python validation/science/scripts/confsearch_suite.py \
        --engine mace --model /path/model.model --device cuda:0 \
        --n-starts 8 --cycles 3 --output validation/science/reports/confsearch-mace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engines  # noqa: E402  (validation helper module in this directory)

REPO = Path(__file__).resolve().parents[3]
LGPS_PATH = REPO / "examples" / "structures" / "li10gep2s12_primitive.vasp"

#: Slab thickness kept from the bottom of the LGPS cell, in Angstrom.
SLAB_THICKNESS_A = 6.0
#: Vacuum added on +z, in Angstrom (the surface normal must be +z).
VACUUM_A = 15.0
#: Adsorbate height above the slab top used by the placement, in Angstrom.
PLACEMENT_HEIGHT_A = 2.4


def build_slab(thickness_A: float = SLAB_THICKNESS_A, vacuum_A: float = VACUUM_A):
    """Cut a fixed slab from the bottom of the repository's LGPS cell.

    The LGPS primitive cell already has ``a_z = b_z = 0``, so the surface normal
    is +z; the cell is extended along c to open the vacuum and the atoms above
    ``z_min + thickness_A`` are dropped, which leaves one contiguous slab with
    its vacuum across the c boundary (the convention confsearch requires).
    """
    from ase import Atoms
    from ase.constraints import FixAtoms
    from ase.io import read

    if not LGPS_PATH.is_file():
        raise FileNotFoundError(f"LGPS structure not found: {LGPS_PATH}")
    cell = read(LGPS_PATH)
    z = np.asarray(cell.get_positions(), dtype=float)[:, 2]
    keep = z <= float(np.min(z)) + float(thickness_A)
    slab = Atoms(
        numbers=np.asarray(cell.numbers, dtype=int)[keep],
        positions=np.asarray(cell.get_positions(), dtype=float)[keep],
        cell=np.asarray(cell.cell, dtype=float),
        pbc=True,
    )
    matrix = np.asarray(slab.cell, dtype=float).copy()
    matrix[2, 2] += float(vacuum_A)
    slab.cell = matrix
    slab.set_constraint(FixAtoms(indices=list(range(len(slab)))))
    return slab


def build_molecule():
    """CH3OH in Cartesian coordinates, PBC ignored (mode B contract)."""
    from ase.build import molecule

    adsorbate = molecule("CH3OH")
    adsorbate.pbc = False
    return adsorbate


def _summary(results: dict[str, Any]) -> dict[str, Any]:
    ranking = results.get("ranking") or {}
    diagnostics = results.get("diagnostics") or {}
    entries = (ranking.get("refined") or []) + (ranking.get("unrefined") or [])
    return {
        "status": results.get("status"),
        "best_energy_eV": ranking.get("best_energy_eV"),
        "n_rankable": ranking.get("n_rankable"),
        "n_quenches": diagnostics.get("n_quenches"),
        "cumulative_md_ps": diagnostics.get("cumulative_md_ps"),
        "hit_counts": sorted((entry["hit_count"] for entry in entries), reverse=True),
        "baseline": results.get("baseline"),
        "warnings": results.get("warnings") or [],
    }


def _discovery_rows(run_dir: Path) -> list[dict[str, str]]:
    import csv

    path = run_dir / "diagnostics" / "discovery.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _monotone_unique_counts(rows: list[dict[str, str]]) -> bool:
    previous = 0
    for row in rows:
        value = int(float(row.get("unique_valid") or 0))
        if value < previous:
            return False
        previous = value
    return True


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    """Run the baseline and the search with the same starts and report both."""
    from ase.io import write

    from mliport.api import run_confsearch

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    slab = build_slab()
    molecule = build_molecule()
    write(output / "slab.vasp", slab, format="vasp")
    write(output / "molecule.xyz", molecule, format="xyz")

    common_kwargs: dict[str, Any] = {
        "slab": str(output / "slab.vasp"),
        "molecule": str(output / "molecule.xyz"),
        "n_starts": int(args.n_starts),
        "model_path": args.model,
        "model_type": args.engine,
        "task": "bulk",
        "device": args.device,
        "seed": int(args.seed),
        "verbose": True,
        "refine": False,
        "quench_fmax": args.quench_fmax,
        "quench_steps": args.quench_steps,
    }
    print("baseline: N placements relaxed without MD (cycles=0)")
    baseline = run_confsearch(cycles=0, output_dir=output / "baseline", **common_kwargs)
    print("search: the same N starts with MD cycles")
    search = run_confsearch(
        cycles=int(args.cycles),
        output_dir=output / "search",
        temperature_high=args.temperature_high,
        temperature_low=args.temperature_low,
        **common_kwargs,
    )

    baseline_summary = _summary(baseline)
    search_summary = _summary(search)
    rows = _discovery_rows(output / "search")
    best_baseline = baseline_summary["best_energy_eV"]
    best_search = search_summary["best_energy_eV"]
    report = {
        "schema": "mliport.confsearch-validation/1",
        "system": "CH3OH on an LGPS-derived slab (examples/structures)",
        "engine": args.engine,
        "model": str(args.model),
        "device": args.device,
        "n_starts": int(args.n_starts),
        "cycles": int(args.cycles),
        "seed": int(args.seed),
        "baseline": baseline_summary,
        "search": search_summary,
        "comparison": {
            "best_energy_difference_eV": (
                None
                if best_baseline is None or best_search is None
                else float(best_search) - float(best_baseline)
            ),
            "search_improved_on_baseline": (
                best_baseline is not None
                and best_search is not None
                and float(best_search) < float(best_baseline) - 1e-9
            ),
            "unique_minima_baseline": baseline_summary["n_rankable"],
            "unique_minima_search": search_summary["n_rankable"],
            "discovery_rows": len(rows),
            "discovery_unique_counts_monotone": _monotone_unique_counts(rows),
        },
        "claims": (
            "Workflow behaviour only. No accuracy claim against DFT; the MLIP "
            "ranking is a screening result and every exported candidate must be "
            "re-ranked with DFT."
        ),
    }
    (output / "confsearch_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "CONFSEARCH_VALIDATION.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    return report


def _markdown(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    baseline = report["baseline"]
    search = report["search"]
    return "\n".join(
        [
            "# confsearch workflow validation",
            "",
            f"System: {report['system']}",
            f"Engine: `{report['engine']}` on `{report['device']}`",
            f"Starts: {report['n_starts']} (same placement stream for both runs)",
            f"Cycles: {report['cycles']} (baseline: 0)",
            "",
            "| | baseline (no MD) | confsearch |",
            "|---|---|---|",
            f"| status | {baseline['status']} | {search['status']} |",
            f"| lowest found (eV) | {baseline['best_energy_eV']} | "
            f"{search['best_energy_eV']} |",
            f"| unique minima | {baseline['n_rankable']} | {search['n_rankable']} |",
            f"| quenches | {baseline['n_quenches']} | {search['n_quenches']} |",
            "",
            f"Search improved on the baseline: "
            f"{comparison['search_improved_on_baseline']}",
            f"Discovery rows: {comparison['discovery_rows']} "
            f"(unique counts monotone: "
            f"{comparison['discovery_unique_counts_monotone']})",
            "",
            "This report checks workflow behaviour only. It is not an accuracy",
            "claim against DFT: the MLIP ranking is a screening result and every",
            "exported candidate must be re-ranked with DFT.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=sorted(engines._ENGINES))
    parser.add_argument("--model", required=True, help="model checkpoint path")
    parser.add_argument("--profile", default=None, help="model manifest profile id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-starts", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--temperature-high", type=float, default=500.0)
    parser.add_argument("--temperature-low", type=float, default=300.0)
    parser.add_argument("--quench-fmax", type=float, default=0.05)
    parser.add_argument("--quench-steps", type=int, default=2000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.n_starts < 1:
        parser.error("--n-starts must be >= 1")
    report = run_campaign(args)
    print(json.dumps(report["comparison"], indent=2, sort_keys=True))
    print(f"Report written to {Path(args.output) / 'CONFSEARCH_VALIDATION.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
