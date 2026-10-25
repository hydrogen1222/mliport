"""Confsearch workflow: model identity, gates, run context and the engine call.

This is the layer the CLI, the Python API and the queue use.  It owns everything
that needs a loaded backend:

* the shared model-identity record (including ``dispersion``);
* the direct-force gate and the energy-force gradient pre-flight (section 6);
* the resume fingerprint and its key-level refusal;
* ``run_context.json`` and ``run.log``.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mliport.confsearch.checkpoint import (
    CheckpointData,
    ConfsearchCheckpointStore,
    build_resume_fingerprint,
    compare_fingerprints,
)
from mliport.confsearch.engine import run_confsearch
from mliport.confsearch.errors import (
    ConfsearchGradientConsistencyError,
    ConfsearchPreparationError,
)
from mliport.confsearch.options import ConfsearchOptions
from mliport.confsearch.prepare import input_hashes, prepare_starts
from mliport.confsearch.revisions import (
    CS_CHECKPOINT_SCHEMA_REVISION,
    CS_SCIENTIFIC_REVISION,
    CS_SPEC_REVISION,
)
from mliport.logger import LiveRunLogger
from mliport.model_record import build_model_record
from mliport.neb.preflight import (
    PREFLIGHT_SCHEMA,
    PreflightNaming,
    run_gradient_preflight,
)
from mliport.neb.io import atomic_write_json

if TYPE_CHECKING:
    from collections.abc import Callable

    from mliport.config.resolver import ResolvedConfig

RUN_CONTEXT_SCHEMA = "mliport.confsearch-run-context/1"

#: confsearch wording for the generalized NEB gradient pre-flight.
CONFSEARCH_PREFLIGHT_NAMING = PreflightNaming(
    context="confsearch",
    item_label="Structure",
    item_label_lower="structure",
    subject="structure",
    schema="mliport.confsearch-gradient-preflight/1",
    sampled_key="sampled_structures",
    error_class=ConfsearchGradientConsistencyError,
    scope=(
        "spot check on the first prepared start and sampled directions; not a "
        "proof of consistency over the whole potential energy surface"
    ),
    consequence=(
        "confsearch ranks force-converged minima by energy, so the ranking "
        "would be meaningless. Use a conservative (autograd-force) model or "
        "precision."
    ),
)


@dataclass
class ConfsearchWorkflowResult:
    """Everything the CLI/API needs to report one confsearch invocation."""

    results: dict
    run_context: dict
    dry_run: bool = False

    @property
    def status(self) -> str:
        return str(self.results.get("status", "completed"))


def _options_from_resolved(resolved: ResolvedConfig) -> ConfsearchOptions:
    known = {item.name for item in fields(ConfsearchOptions)}
    values: dict[str, Any] = {}
    for key, value in dict(resolved.run_options).items():
        if key in known:
            values[key] = value
    for key, value in dict(resolved.settings).items():
        if key in known and key not in values:
            values[key] = value
    if values.get("seed") is None:
        values["seed"] = int(np.random.SeedSequence().entropy % (2**31 - 1))
    return ConfsearchOptions(**values)


def _rng_streams(seed: int, n_replicas: int):
    """Spec 7.5: placement stream first, then one (md, mc) pair per replica."""
    sequence = np.random.SeedSequence(int(seed))
    placement_ss, *replica_sequences = sequence.spawn(1 + max(1, n_replicas))
    return placement_ss, replica_sequences


def archive_for_extension(run_dir: str | Path, *, index: int) -> Path:
    """Archive the previous top-level results before an extension (spec 11.3)."""
    run_dir = Path(run_dir)
    history = run_dir / "history" / f"extension_{int(index)}"
    history.mkdir(parents=True, exist_ok=True)
    for name in (
        "search_results.json",
        "artifacts.json",
        "top",
        "diagnostics",
        "minima",
    ):
        source = run_dir / name
        if source.exists():
            source.rename(history / name)
    return history


def record_extension(
    run_dir: str | Path, *, added_cycles: int, total_cycles: int
) -> dict:
    """Append an extension record to ``run_context.json`` (spec 11.3)."""
    import json as _json  # noqa: PLC0415

    path = Path(run_dir) / "run_context.json"
    context = _json.loads(path.read_text(encoding="utf-8"))
    extensions = list(context.get("extensions") or [])
    record = {
        "index": len(extensions) + 1,
        "added_cycles": int(added_cycles),
        "total_cycles": int(total_cycles),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mliport_version": _mliport_version(),
    }
    extensions.append(record)
    context["extensions"] = extensions
    # The recorded budget must follow the extension, otherwise a second extend
    # would compare against the original cycle count and refuse.
    options_record = dict(context.get("options") or {})
    options_record["cs_cycles"] = int(total_cycles)
    context["options"] = options_record
    path.write_text(
        _json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def _mliport_version() -> str:
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    try:
        return version("mliport")
    except PackageNotFoundError:  # pragma: no cover - always installed
        return "unknown"


def run_confsearch_workflow(
    calculator,
    resolved: ResolvedConfig,
    *,
    output_dir: str | Path | None = None,
    resume: bool = False,
    dry_run: bool = False,
    allow_cycle_change: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> ConfsearchWorkflowResult:
    """Run (or resume) one confsearch job from a resolved configuration."""
    options = _options_from_resolved(resolved)
    output = Path(output_dir or resolved.output_dir)
    placement_ss, _replica_sequences = _rng_streams(
        int(options.seed or 0), int(options.cs_n_starts or len(options.cs_starts or ()))
    )
    placement_rng = np.random.default_rng(placement_ss)
    starts = prepare_starts(options, placement_rng=placement_rng)
    hashes = input_hashes(options)
    regions_identity = {
        "adsorbate": starts[0].regions.adsorbate.tolist(),
        "slab": starts[0].regions.slab.tolist(),
        "fixed": starts[0].regions.fixed.tolist(),
        "relax": starts[0].regions.relax.tolist(),
    }

    if dry_run:
        return ConfsearchWorkflowResult(
            results={
                "schema": "mliport.confsearch-results/1",
                "status": "dry_run",
                "n_starts": len(starts),
                "budget": {
                    "cycles": int(options.cs_cycles),
                    "md_steps_per_cycle": int(options.total_md_steps_per_cycle),
                    "md_steps_total": int(
                        len(starts)
                        * options.total_md_steps_per_cycle
                        * options.cs_cycles
                    ),
                    "quench_upper_bound": int(
                        len(starts) * options.total_quench_upper_bound
                    ),
                },
                "regions": regions_identity,
                "warnings": sorted(
                    {warning for start in starts for warning in start.regions.warnings}
                ),
            },
            run_context={"schema": RUN_CONTEXT_SCHEMA, "dry_run": True},
            dry_run=True,
        )

    # A dry run validates the inputs and returns; it must not create the output
    # directory or a run.log, otherwise the real run with the same --output
    # would be refused by the "output directory already contains results" check.
    output.mkdir(parents=True, exist_ok=True)
    # Spec 11.1 revision 5: run.log is the live, append-only LiveRunLogger, so
    # it exists from the first line, survives a crash and grows on resume or
    # extend instead of being rewritten at the end of the run.
    run_logger = LiveRunLogger(output / "run.log", callback=log_fn, append=True)

    def log(message: str) -> None:
        run_logger(str(message))


    try:
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        try:
            mliport_version = version("mliport")
        except PackageNotFoundError:  # pragma: no cover - always installed
            mliport_version = "unknown"

        model = build_model_record(
            calculator, resolved, error_class=ConfsearchPreparationError
        )
        warnings: list[str] = []
        preflight: dict[str, Any] = {"status": "not_run"}
        info = dict(calculator.info())
        if info.get("direct_forces") is True:
            if not options.cs_allow_direct_forces:
                raise ConfsearchGradientConsistencyError(
                    "confsearch: the model reports direct_forces=True, so its forces "
                    "are not verifiably the gradient of its energy and the energy "
                    "ranking would be meaningless. Set cs_allow_direct_forces=true "
                    "(--allow-direct-forces) to accept this explicitly."
                )
            model["energy_gradient_consistency"] = "not_guaranteed"
            warnings.append(
                "cs_allow_direct_forces is set and the model reports "
                "direct_forces=True: the forces are not guaranteed to be -grad(E), "
                "so the ranking is not meaningful."
            )
            preflight = {"status": "skipped", "reason": "direct_forces allowed"}
        else:
            ase_calculator = calculator.get_calculator()
            preflight = run_gradient_preflight(
                ase_calculator,
                [starts[0].atoms],
                dtype=model.get("dtype_effective"),
                minimum_structures=1,
                naming=CONFSEARCH_PREFLIGHT_NAMING,
            )
            model["energy_gradient_consistency"] = "measured"
            log(
                "gradient pre-flight passed: "
                f"{preflight['n_checks']} checks, max |FD - (-F.u)| = "
                f"{preflight['max_abs_error_eV_A']:.2e} eV/A"
            )

        comparison_options = options
        if allow_cycle_change:
            # `extend` is the one supported budget change (spec 11.3): everything
            # else in the fingerprint must still match, so compare with the
            # checkpointed cycle count substituted back in.
            recorded_context = output / "run_context.json"
            if recorded_context.is_file():
                recorded = json.loads(recorded_context.read_text(encoding="utf-8"))
                recorded_cycles = (recorded.get("options") or {}).get("cs_cycles")
                if recorded_cycles is not None:
                    comparison_options = replace(options, cs_cycles=int(recorded_cycles))
        fingerprint, fingerprint_sha = build_resume_fingerprint(
            options=options,
            model=model,
            input_hashes=hashes,
            regions=regions_identity,
            seed=options.seed,
            mliport_version=mliport_version,
        )
        store = ConfsearchCheckpointStore(output)
        checkpoint: CheckpointData | None = None
        if resume:
            checkpoint, recorded = store.load()
            comparison_fingerprint = (
                build_resume_fingerprint(
                    options=comparison_options,
                    model=model,
                    input_hashes=hashes,
                    regions=regions_identity,
                    seed=options.seed,
                    mliport_version=mliport_version,
                )[0]
                if allow_cycle_change
                else fingerprint
            )
            differences = compare_fingerprints(recorded, comparison_fingerprint)
            if differences:
                raise ConfsearchPreparationError(
                    "Cannot resume: the checkpoint fingerprint does not match this "
                    "configuration/input:\n  - " + "\n  - ".join(differences)
                )
            log(f"resuming from checkpoint unit {checkpoint.unit_index}")
            if allow_cycle_change:
                # Spec 11.3 revision 5: the fingerprint has been validated (only
                # cs_cycles exempt), so this is the point where the previous results
                # may be archived and the extension recorded.  Doing it here -- and
                # not in the CLI before the call -- keeps a refused extension from
                # moving the previous results away.
                added = int(options.cs_cycles) - int(comparison_options.cs_cycles)
                if added > 0:
                    index = (
                        len(
                            json.loads(
                                (output / "run_context.json").read_text(encoding="utf-8")
                            ).get("extensions")
                            or []
                        )
                        + 1
                    )
                    history = archive_for_extension(output, index=index)
                    record_extension(
                        output,
                        added_cycles=added,
                        total_cycles=int(options.cs_cycles),
                    )
                    log(
                        f"extending by {added} cycle(s); previous results archived "
                        f"under {history.relative_to(output)}"
                    )

        run_context = {
            "schema": RUN_CONTEXT_SCHEMA,
            "spec_revision": CS_SPEC_REVISION,
            "scientific_revision": CS_SCIENTIFIC_REVISION,
            "checkpoint_schema_revision": CS_CHECKPOINT_SCHEMA_REVISION,
            "mliport_version": mliport_version,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": options.seed,
            "device": resolved.device,
            # Spec 11.4 revision 5: index sets alone are not enough (two different
            # slabs can share them), so the merge key needs the atomic numbers in
            # order, the cell, the PBC and the electronic state.
            "system": {
                "numbers": [int(z) for z in starts[0].atoms.numbers],
                "cell_A": np.asarray(starts[0].atoms.cell, dtype=float).tolist(),
                "pbc": [bool(value) for value in starts[0].atoms.pbc],
            },
            "electronic_state": {
                "charge": resolved.run_options.get("charge"),
                "spin": resolved.run_options.get("spin"),
            },
            "resume": bool(resume),
            "model": model,
            "input_hashes": hashes,
            "regions": regions_identity,
            "budget": {
                "replicas": len(starts),
                "cycles": int(options.cs_cycles),
                "md_steps_per_cycle": int(options.total_md_steps_per_cycle),
                "md_steps_total": int(
                    len(starts) * options.total_md_steps_per_cycle * options.cs_cycles
                ),
                "quench_upper_bound": int(len(starts) * options.total_quench_upper_bound),
            },
            "gradient_preflight": preflight,
            "resume_fingerprint_sha256": fingerprint_sha,
            "starts": [start.to_dict() for start in starts],
            "options": {
                item.name: _jsonable(getattr(options, item.name))
                for item in fields(ConfsearchOptions)
                if not item.name.startswith("_")
            },
        }
        if not resume:
            _write_resolved_config(resolved, output)
        resume_entry = (
            {
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "unit_index": None if checkpoint is None else int(checkpoint.unit_index),
                "seed": options.seed,
                "mliport_version": mliport_version,
            }
            if resume
            else None
        )
        run_context = _update_run_context(
            output / "run_context.json", base=run_context, resume_entry=resume_entry
        )

        results = run_confsearch(
            calculator.get_calculator(),
            options,
            starts=starts,
            output_dir=output,
            seed=int(options.seed or 0),
            model_record=model,
            gradient_preflight=preflight,
            resume=checkpoint,
            resume_fingerprint=fingerprint,
            log=log,
        )
        results["warnings"] = list(
            dict.fromkeys(list(results.get("warnings", [])) + warnings)
        )
        (output / "search_results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, default=_jsonable) + "\n",
            encoding="utf-8",
        )
        log(f"confsearch finished with status {results.get('status')}")
        run_context = _update_run_context(output / "run_context.json")
        return ConfsearchWorkflowResult(results=results, run_context=run_context)
    except Exception as exc:
        # A failed run's log must end with the reason (spec 11.1).
        log(f"confsearch failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        run_logger.close()


def _write_resolved_config(resolved: ResolvedConfig, output_dir: Path) -> None:
    """Write ``resolved_config.json`` for a new run (spec 11.1 revision 5).

    Mirrors the CLI's emitter (atomic replace, honours ``write_resolved_config``)
    so both entry points produce the same file.
    """
    import os  # noqa: PLC0415

    if not resolved.settings.get("write_resolved_config", True):
        return
    target = Path(output_dir) / "resolved_config.json"
    payload = json.dumps(resolved.as_dict(), indent=2, default=str) + "\n"
    tmp = target.parent / f".resolved_config.json.tmp-{os.getpid()}"
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, target)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def _update_run_context(
    path: Path,
    *,
    base: dict[str, Any] | None = None,
    resume_entry: dict[str, Any] | None = None,
    extension: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create run_context.json once, then update it in place (spec 11.1 rev 5).

    A resume appends to ``resumes[]``; an extension (recorded by the CLI) appends
    to ``extensions[]``.  The recorded identity, regions, budget, starts and
    options are never rebuilt from scratch, so the file stays the run's history.
    """
    if path.is_file():
        context = json.loads(path.read_text(encoding="utf-8"))
    elif base is not None:
        context = dict(base)
    else:
        raise ConfsearchPreparationError(
            f"Cannot update {path}: it does not exist and no base record was given"
        )
    context.setdefault("resumes", [])
    context.setdefault("extensions", [])
    if resume_entry is not None:
        context["resumes"].append(resume_entry)
    if extension is not None:
        context["extensions"].append(extension)
    context["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(path, context)
    return context


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "CONFSEARCH_PREFLIGHT_NAMING",
    "PREFLIGHT_SCHEMA",
    "RUN_CONTEXT_SCHEMA",
    "ConfsearchWorkflowResult",
    "run_confsearch_workflow",
]
