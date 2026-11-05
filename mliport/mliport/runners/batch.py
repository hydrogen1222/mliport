# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Batch calculation runner.

Processes multiple structures sequentially while reusing one loaded model.
"""

from __future__ import annotations

import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.io import read
from tqdm import tqdm

from mliport.protocols import CancellationRequested
from mliport.run_status import RunOutcome, classify_result
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

    Output layout: each structure gets its own sub-directory that mirrors its
    path relative to the input root, so ``A/rank_1/1.cif`` and
    ``B/rank_1/1.cif`` land in ``OUT/A/rank_1/1`` and ``OUT/B/rank_1/1``.
    Every finished structure writes ``batch_status.json`` into its directory
    immediately, and the batch root keeps ``batch_manifest.json`` (the
    settings that define the batch) plus an append-only
    ``batch_progress.jsonl``. With ``resume=True`` a structure whose status
    file records a finished outcome (completed or not converged) under the
    same manifest, for the same input content and with all of its recorded
    output files intact, is reused instead of recomputed; everything else
    (failed, unfinished, changed input, missing outputs) runs again.
    """

    MANIFEST_NAME = "batch_manifest.json"
    STATUS_NAME = "batch_status.json"
    PROGRESS_NAME = "batch_progress.jsonl"
    #: Outcomes that are final for a given manifest and are reused on resume.
    #: Re-running a deterministic optimization that missed max_steps with the
    #: same settings cannot change its outcome; a failure may be transient.
    REUSABLE_OUTCOMES = frozenset(
        {RunOutcome.COMPLETED.value, RunOutcome.NOT_CONVERGED.value}
    )

    def __init__(
        self,
        calculator,
        calc_type: str = "sp",
        output_dir: Path | str = ".",
        parallel: bool = False,
        max_workers: int = 1,
        verbose: bool = True,
        job_name: str | None = None,
        resume: bool = False,
        run_identity: dict[str, Any] | None = None,
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
            resume: Reuse finished structures recorded by an earlier run of
                the same batch (identical manifest) instead of recomputing.
            run_identity: Extra settings that define the batch scientifically
                (model path/type/task, calculator options). They are stored in
                the manifest and must match on resume.
            **calc_kwargs: Additional arguments for calculation runner
        """
        self.calculator = calculator
        self.calc_type = calc_type.lower()
        self.job_name = job_name
        self.parallel = parallel
        self.max_workers = max_workers
        self.verbose = verbose
        self.calc_kwargs = calc_kwargs
        self.resume = resume
        self.run_identity = dict(run_identity or {})

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
            return {"total": 0, "success": 0, "failed": 0, "not_converged": 0}

        print(f"Found {len(files)} structure files")

        return self.run_from_files(files, input_root=input_dir)

    @staticmethod
    def output_names(
        files: list[Path], input_root: Path | str | None = None
    ) -> dict[Path, Path]:
        """Map each input file to a unique output sub-directory (relative).

        The sub-directory mirrors the file's directory relative to
        ``input_root`` (default: the deepest common parent of all inputs), so
        identically named files from different directories never share an
        output directory. Within one directory the file stem is used, or the
        full name with dots replaced when two formats share a stem
        (``sample.xyz`` / ``sample.cif``). Any remaining collision is an error:
        overwriting one structure's results with another's is never allowed.
        """
        resolved = [Path(f).expanduser().resolve() for f in files]
        if len(set(resolved)) != len(resolved):
            raise ValueError("Batch input contains the same structure file twice.")
        if not resolved:
            return {}
        if input_root is None:
            root = Path(os.path.commonpath([str(p.parent) for p in resolved]))
        else:
            root = Path(input_root).expanduser().resolve()

        stem_counts: dict[tuple[Path, str], int] = {}
        for path in resolved:
            key = (path.parent, path.stem)
            stem_counts[key] = stem_counts.get(key, 0) + 1

        names: dict[Path, Path] = {}
        owners: dict[Path, Path] = {}
        for original, path in zip(files, resolved):
            try:
                rel_parent = path.parent.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"Structure {path} is outside the batch input root {root}."
                ) from exc
            leaf = (
                path.stem
                if stem_counts[(path.parent, path.stem)] == 1
                else path.name.replace(".", "_")
            )
            rel = rel_parent / leaf
            if rel in owners:
                raise ValueError(
                    f"Structures {owners[rel]} and {path} would share the output "
                    f"directory {rel}; rename one of them."
                )
            owners[rel] = path
            names[Path(original)] = rel
        return names

    def _manifest(self) -> dict[str, Any]:
        """Settings that define this batch; resume requires an exact match."""
        params: dict[str, Any] = {"calc_type": self.calc_type}
        if self.calc_type == "opt":
            from mliport.config.defaults import BUILTIN_DEFAULTS  # noqa: PLC0415

            for key in ("optimizer", "fmax", "max_steps", "cell_opt", "fix_symmetry"):
                value = self.calc_kwargs.get(key, BUILTIN_DEFAULTS["opt"][key])
                params[key] = value.upper() if key == "optimizer" else value
        for key in ("charge", "spin"):
            params[key] = self.calc_kwargs.get(key)
        params.update(self.run_identity)
        # Normalise through JSON so a re-read manifest compares equal.
        return json.loads(json.dumps(params, sort_keys=True, default=str))

    def _prepare_manifest(self) -> dict[str, Any]:
        manifest = self._manifest()
        path = self.output_dir / self.MANIFEST_NAME
        if self.resume:
            if not path.exists():
                raise FileNotFoundError(
                    f"Cannot resume: {path} not found. Resume only works in the "
                    "output directory of an earlier batch run."
                )
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous != manifest:
                changed = sorted(
                    key
                    for key in set(previous) | set(manifest)
                    if previous.get(key) != manifest.get(key)
                )
                details = ", ".join(
                    f"{key}: {previous.get(key)!r} -> {manifest.get(key)!r}"
                    for key in changed
                )
                raise ValueError(
                    "Cannot resume: batch settings differ from the earlier run "
                    f"({details}). Results computed with different settings must "
                    "not be mixed; use a new --output directory."
                )
        else:
            _atomic_write_json(path, manifest)
        return manifest

    def _reusable_record(
        self, filepath: Path, sub_dir: Path, manifest: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Return the recorded result only if it is still trustworthy.

        Reuse requires: a finished outcome under the identical manifest, the
        same input file content (sha256) as when it was computed, and every
        output file the run produced still present with its recorded size.
        Anything else is recomputed.
        """
        status_path = sub_dir / self.STATUS_NAME
        if not status_path.exists():
            return None
        try:
            record = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if record.get("manifest") != manifest:
            return None
        if record.get("status") not in self.REUSABLE_OUTCOMES:
            return None
        try:
            if record.get("input_sha256") != _file_sha256(filepath):
                return None
        except OSError:
            return None
        outputs = record.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            return None
        if self.calc_type == "opt" and "CONTCAR" not in outputs:
            return None
        for name, size in outputs.items():
            path = sub_dir / name
            if not path.is_file() or path.stat().st_size != size:
                return None
        return record

    def run_from_files(
        self,
        files: list[Path | str],
        input_root: Path | str | None = None,
    ) -> dict[str, Any]:
        """Run calculations on a list of files.

        Args:
            files: List of structure file paths
            input_root: Directory that output sub-directories are made
                relative to (default: deepest common parent of ``files``).

        Returns:
            Dictionary with batch results summary
        """
        paths = [Path(filepath) for filepath in files]
        names = self.output_names(paths, input_root)
        manifest = self._prepare_manifest()
        progress_path = self.output_dir / self.PROGRESS_NAME

        results_list = []
        success_count = 0
        failed_count = 0
        not_converged_count = 0
        reused_count = 0

        iterator = tqdm(paths, desc="Processing structures") if self.verbose else paths
        for filepath in iterator:
            rel = names[filepath]
            sub_dir = self.output_dir / rel
            record: dict[str, Any] | None = None
            reused = False
            if self.resume:
                record = self._reusable_record(filepath, sub_dir, manifest)
                reused = record is not None
            if record is None:
                record = self._process_one(filepath, rel, sub_dir, manifest)
            else:
                record = {**record, "reused": True}
                reused_count += 1

            outcome = RunOutcome(record["status"])
            if outcome is RunOutcome.COMPLETED:
                success_count += 1
            else:
                failed_count += 1
                if outcome is RunOutcome.NOT_CONVERGED:
                    not_converged_count += 1
            for internal in ("manifest", "outputs"):
                record.pop(internal, None)
            results_list.append(record)
            if not reused:
                _append_jsonl(progress_path, record)

        # Write summary
        summary = {
            "total": len(paths),
            "success": success_count,
            "failed": failed_count,
            "not_converged": not_converged_count,
            "reused": reused_count,
            "results": results_list,
        }

        self._write_summary(summary)

        return summary

    def _process_one(
        self,
        filepath: Path,
        rel: Path,
        sub_dir: Path,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        """Run one structure and persist its outcome before moving on."""
        status_path = sub_dir / self.STATUS_NAME
        sub_dir.mkdir(parents=True, exist_ok=True)
        # A stale status from an earlier attempt must never survive a rerun
        # that is interrupted before it records its own outcome.
        status_path.unlink(missing_ok=True)
        base = {
            "filename": filepath.name,
            "source": str(filepath),
            "output_name": rel.as_posix(),
        }
        try:
            # Hash first: the reused record must describe exactly this input.
            base["input_sha256"] = _file_sha256(filepath)
            atoms = read(filepath)
            result = self._run_single(atoms, sub_dir, filepath.stem)
            outcome = RunOutcome(result["status"])
            record = {
                **base,
                "formula": atoms.get_chemical_formula(),
                "natoms": len(atoms),
                "success": outcome is RunOutcome.COMPLETED,
                **result,
            }
        except CancellationRequested:
            # Leave no status: an interrupted structure runs again on resume.
            raise
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e!s}"
            record = {
                **base,
                "success": False,
                "status": RunOutcome.FAILED.value,
                "error": error_msg,
                "traceback": traceback.format_exc(),
            }
            if self.verbose:
                print(f"\nError processing {filepath}: {error_msg}")
        outputs = {
            child.name: child.stat().st_size
            for child in sorted(sub_dir.iterdir())
            if child.is_file()
            and child.name != self.STATUS_NAME
            and not child.name.startswith(".")
        }
        _atomic_write_json(
            status_path, {**record, "outputs": outputs, "manifest": manifest}
        )
        return record

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

        # A returned optimizer result is not automatically a successful run:
        # failed optimizers may keep partial results, and max_steps can be
        # reached without convergence. Preserve the runner's outcome.
        outcome = classify_result(results, calc_type=self.calc_type)

        return {
            "status": outcome.value,
            "converged": results.get("converged") if self.calc_type == "opt" else None,
            "nsteps": results.get("nsteps") if self.calc_type == "opt" else None,
            "error": results.get("error") if outcome is RunOutcome.FAILED else None,
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
            "not_converged": summary["not_converged"],
            "reused": summary.get("reused", 0),
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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON so a crash never leaves a truncated status file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one record and flush it to disk immediately."""
    line = json.dumps(
        {k: v for k, v in payload.items() if k != "traceback"},
        default=_json_default,
    )
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
