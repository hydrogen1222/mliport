"""Small, repeatable real-backend MD smoke test.

Run this script with the backend's isolated interpreter. It deliberately uses
four Ar atoms, five steps by default, and CPU unless ``--device`` is explicitly
overridden, so it does not compete with production GPU jobs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms

from mliport.calculators.factory import CalculatorFactory
from mliport.runners.md import MDRunner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", required=True, choices=["uma", "mace", "dpa", "grace"]
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--head", default=None)
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("md-backend-smoke"))
    parser.add_argument("--system", choices=["bulk", "molecule"], default="bulk")
    return parser


def _atoms(system: str) -> Atoms:
    atoms = Atoms(
        "Ar4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [0.0, 0.0, 2.5],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=system == "bulk",
    )
    return atoms


def main() -> int:
    args = _parser().parse_args()
    if args.steps < 1 or args.steps > 20:
        raise ValueError("--steps must be between 1 and 20 for this smoke test")

    task = args.task or ("omat" if args.backend == "uma" else args.system)
    calculator_options: dict[str, object] = {}
    if args.backend == "mace":
        calculator_options["default_dtype"] = args.dtype
        if args.head:
            calculator_options["head"] = args.head
    elif args.backend == "dpa" and args.head:
        calculator_options["head"] = args.head

    wrapper = CalculatorFactory.create(
        args.backend,
        args.model,
        device=args.device,
        task=task,
        **calculator_options,
    )
    methods = [
        ("nve", "NVE", "LANGEVIN"),
        ("langevin", "NVT", "LANGEVIN"),
        ("bussi", "NVT", "BUSSI"),
        ("nhc", "NVT", "NHC"),
    ]
    for name, ensemble, thermostat in methods:
        output = args.output / args.backend / args.system / name
        runner = MDRunner(
            wrapper,
            ensemble=ensemble,
            thermostat=thermostat,
            temperature=300.0,
            timestep=0.25,
            steps=args.steps,
            save_interval=1,
            output_dir=output,
            pre_relax=False,
            seed=7,
            verbose=False,
        )
        results = runner.run(_atoms(args.system))
        if not np.isfinite(results["energy"]):
            raise RuntimeError(f"{name}: non-finite energy")
        if not np.isfinite(results["temperature"]):
            raise RuntimeError(f"{name}: non-finite temperature")
        if not (output / "raw" / "trajectory.traj").is_file():
            raise RuntimeError(f"{name}: trajectory was not written")
        print(
            f"PASS {args.backend:5s} {args.system:8s} {name:9s} "
            f"E={results['energy']:.6f} eV T={results['temperature']:.2f} K"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
