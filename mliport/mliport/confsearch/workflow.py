"""Confsearch workflow: model identity, gates, run context and the engine call.

This is the layer the CLI, the Python API and the queue use.  It owns everything
that needs a loaded backend:

* the shared model-identity record (including ``dispersion``);
* the direct-force gate and the energy-force gradient pre-flight (section 6);
* the resume fingerprint and its key-level refusal;
* ``run_context.json`` and ``run.log``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, fields
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


def run_confsearch_workflow(
    calculator,
    resolved: ResolvedConfig,
    *,
    output_dir: str | Path | None = None,
    resume: bool = False,
    dry_run: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> ConfsearchWorkflowResult:
    """Run (or resume) one confsearch job from a resolved configuration."""
    options = _options_from_resolved(resolved)
    output = Path(output_dir or resolved.output_dir)
    lines: list[str] = []

    def log(message: str) -> None:
        lines.append(message)
        if log_fn is not None:
            log_fn(message)

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
        differences = compare_fingerprints(recorded, fingerprint)
        if differences:
            raise ConfsearchPreparationError(
                "Cannot resume: the checkpoint fingerprint does not match this "
                "configuration/input:\n  - " + "\n  - ".join(differences)
            )
        log(f"resuming from checkpoint unit {checkpoint.unit_index}")

    run_context = {
        "schema": RUN_CONTEXT_SCHEMA,
        "spec_revision": CS_SPEC_REVISION,
        "scientific_revision": CS_SCIENTIFIC_REVISION,
        "checkpoint_schema_revision": CS_CHECKPOINT_SCHEMA_REVISION,
        "mliport_version": mliport_version,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": options.seed,
        "device": resolved.device,
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
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "run_context.json", run_context)

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
    (output / "run.log").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )
    run_context["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(output / "run_context.json", run_context)
    return ConfsearchWorkflowResult(results=results, run_context=run_context)


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
