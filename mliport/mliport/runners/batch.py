# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Batch calculation runner.

Processes multiple structures sequentially while reusing one loaded model.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.io import read
from tqdm import tqdm

from mliport.runners.optimization import OptimizationRunner
from mliport.runners.singlepoint import SinglePointRunner
from mliport.writers.json_writer import JsonWriter

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms


class BatchRunner:
    """Run calculations on multiple structures in batch mode.

    Processes a directory of structure files sequentially. A single calculator
    instance is deliberately not shared across worker threads: ASE/backend
    calculators mutate cached results during evaluation and are not generally
    thread-safe.

    Example:
        >>> runner = BatchRunner(
        ...     calculator,
        ...     calc_type="sp",
        ...     output_dir="batch_results"
        ... )
        >>> results = runner.run_from_directory("structures/")
        >>> print(f"Processed: {results['success']}/{results['total']}")
    """

    def __init__(
        self,
        calculator,
        calc_type: str = "sp",
        output_dir: Path | str = ".",
        parallel: bool = False,
        max_workers: int = 1,
        verbose: bool = True,
        job_name: str | None = None,
        **calc_kwargs,
    ):
        """Initialize batch runner.

        Args:
            calculator: UMA calculator wrapper
            calc_type: Type of calculation (sp, opt, md)
            output_dir: Directory for output files
            parallel: Reserved; parallel model execution is not yet supported.
            max_workers: Reserved worker count (must currently be one).
            verbose: Whether to print progress messages
            job_name: Optional job name for organizing results
            **calc_kwargs: Additional arguments for calculation runner
        """
        self.calculator = calculator
        self.calc_type = calc_type.lower()
        self.job_name = job_name
        self.parallel = parallel
        self.max_workers = max_workers
        self.verbose = verbose
        self.calc_kwargs = calc_kwargs

        # Build output directory path
        base_dir = Path(output_dir)
        if job_name:
            self.output_dir = base_dir / job_name
        else:
            self.output_dir = base_dir

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate calc_type
        valid_types = {"sp", "opt", "md"}
        if self.calc_type not in valid_types:
            raise ValueError(
                f"Unknown calculation type: {calc_type}. "
                f"Use one of: {', '.join(valid_types)}"
            )
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.parallel and self.max_workers > 1:
            raise NotImplementedError(
                "Parallel BatchRunner execution is not safe with a shared ASE "
                "calculator. Use serial mode (parallel=False, max_workers=1)."
            )

    def run_from_directory(
        self,
        input_dir: Path | str,
        pattern: str | None = None,
    ) -> dict[str, Any]:
        """Run calculations on all matching files in directory.

        Args:
            input_dir: Input directory containing structure files
            pattern: Explicit glob pattern. If omitted, discover CIF, XYZ,
                VASP and POSCAR files.

        Returns:
            Dictionary with batch results summary
        """
        input_dir = Path(input_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Batch input is not a directory: {input_dir}")

        if pattern is not None:
            files = list(input_dir.glob(pattern))
        else:
            files = []
            for supported_pattern in ("*.cif", "*.xyz", "*.vasp", "POSCAR*"):
                files.extend(input_dir.glob(supported_pattern))

        # Remove duplicates and sort
        files = sorted(set(files))

        if not files:
            print(f"No structure files found in {input_dir}")
            return {"total": 0, "success": 0, "failed": 0}

        print(f"Found {len(files)} structure files")

        return self.run_from_files(files)

    def run_from_files(self, files: list[Path | str]) -> dict[str, Any]:
        """Run calculations on a list of files.

        Args:
            files: List of structure file paths

        Returns:
            Dictionary with batch results summary
        """
        paths = [Path(filepath) for filepath in files]
        stem_counts: dict[str, int] = {}
        for filepath in paths:
            stem_counts[filepath.stem] = stem_counts.get(filepath.stem, 0) + 1

        results_list = []
        success_count = 0
        failed_count = 0

        iterator = tqdm(paths, desc="Processing structures") if self.verbose else paths
        for filepath in iterator:
            try:
                atoms = read(filepath)
                output_name = (
                    filepath.stem
                    if stem_counts[filepath.stem] == 1
                    else filepath.name.replace(".", "_")
                )
                sub_dir = self.output_dir / output_name
                sub_dir.mkdir(exist_ok=True)
                result = self._run_single(atoms, sub_dir, filepath.stem)
                results_list.append(
                    {
                        "filename": filepath.name,
                        "formula": atoms.get_chemical_formula(),
                        "natoms": len(atoms),
                        "success": True,
                        **result,
                    }
                )
                success_count += 1
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e!s}"
                results_list.append(
                    {
                        "filename": filepath.name,
                        "success": False,
                        "error": error_msg,
                        "traceback": traceback.format_exc(),
                    }
                )
                failed_count += 1
                if self.verbose:
                    print(f"\nError processing {filepath.name}: {error_msg}")

        # Write summary
        summary = {
            "total": len(paths),
            "success": success_count,
            "failed": failed_count,
            "results": results_list,
        }

        self._write_summary(summary)

        return summary

    def _run_single(
        self,
        atoms: Atoms,
        sub_dir: Path,
        name: str,
    ) -> dict[str, Any]:
        """Run single calculation.

        Args:
            atoms: ASE Atoms object
            sub_dir: Sub-directory for output
            name: Structure name

        Returns:
            Dictionary with results
        """
        # Create appropriate runner
        if self.calc_type == "sp":
            runner = SinglePointRunner(
                self.calculator,
                output_dir=sub_dir,
                verbose=False,
                **self.calc_kwargs,
            )
        elif self.calc_type == "opt":
            runner = OptimizationRunner(
                self.calculator,
                output_dir=sub_dir,
                verbose=False,
                **self.calc_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Batch mode for {self.calc_type} not yet implemented"
            )

        # Run calculation
        results = runner.execute(atoms)

        # Extract key results
        return {
            "energy": results.get("energy"),
            "fmax": (
                np.max(np.linalg.norm(results.get("forces"), axis=1))
                if results.get("forces") is not None
                else None
            ),
            "time": results.get("time"),
            "timing": results.get("timing"),
            "output_dir": str(sub_dir),
        }

    def _write_summary(self, summary: dict[str, Any]) -> None:
        """Write batch summary to file.

        Args:
            summary: Summary dictionary
        """
        if not self.calc_kwargs.get("write_json", True):
            return

        # Write JSON summary
        json_path = self.output_dir / "batch_summary.json"
        writer = JsonWriter()
        metadata = {
            "calc_type": self.calc_type,
            "total": summary["total"],
            "success": summary["success"],
            "failed": summary["failed"],
        }
        if self.job_name:
            metadata["job_name"] = self.job_name
        writer.write_batch(
            summary["results"],
            json_path,
            metadata=metadata,
        )

        if self.verbose:
            print(f"\nBatch summary written to: {json_path}")
            print(
                f"Total: {summary['total']}, Success: {summary['success']}, Failed: {summary['failed']}"
            )
