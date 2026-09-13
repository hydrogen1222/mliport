"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Modified for the mliport project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
Programmatic API for mliport.

Provides high-level functions for running calculations from Python scripts.
This module is designed for external scripts that need to integrate MLIP
calculations into complex workflows.  All functions route through
``resolve_config()`` so built-in defaults, model aliases and settings.ini
are honoured (Phase 1 plan).

Example:
    >>> from mliport.api import run_single_point, calculate_energy
    >>>
    >>> # Run a single point calculation
    >>> results = run_single_point(
    ...     structure="structure.cif",
    ...     model_path="uma-s-1.pt",
    ...     task="omat",
    ...     job_name="my_calculation"
    ... )
    >>> print(f"Energy: {results['energy']:.4f} eV")
    >>>
    >>> # Just get the energy
    >>> energy = calculate_energy("structure.cif", "uma-s-1.pt", task="omat")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from ase import Atoms
from ase.io import read

from mliport.engine import CalculationEngine, EngineConfig

if TYPE_CHECKING:
    from typing import Any


def _load_structure(structure: Atoms | str | Path) -> Atoms:
    """Load structure from various input types.

    Args:
        structure: ASE Atoms object or path to structure file

    Returns:
        ASE Atoms object

    Raises:
        ValueError: If structure cannot be loaded
    """
    if isinstance(structure, Atoms):
        return structure

    structure_path = Path(structure).expanduser().resolve()
    if not structure_path.exists():
        raise ValueError(f"Structure file not found: {structure_path}")

    try:
        return read(structure_path)
    except Exception as e:
        raise ValueError(f"Error reading structure: {e}") from e


def _console_log(message: str, level: str = "info") -> None:
    """Print a live engine message for verbose API calls."""
    print(message, flush=True)


def _build_api_cli(
    calc_type: str,
    model_type: str | None,
    task: str | None,
    device: str | None,
    inference_mode: str | None,
    default_dtype: str | None,
    head: str | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """Collect non-None API kwargs into a canonical option dict for the resolver."""
    cli: dict[str, Any] = {}
    for key, value in (
        ("model_type", model_type),
        ("task", task),
        ("device", device),
        ("inference_mode", inference_mode),
        ("default_dtype", default_dtype),
        ("head", head),
    ):
        if value is not None:
            cli[key] = value
    if extra:
        for k, v in extra.items():
            if v is not None:
                cli[k] = v
    return cli


def _api_resolve(
    calc_type: str,
    model_path: str,
    cli: dict[str, Any],
    output_dir: str,
    job_name: str | None,
    settings_path: str | None,
    model_alias: str | None,
    profile: str | None,
    strict_config: bool | None,
) -> EngineConfig:
    """Thin wrapper around resolve_config for the API layer."""
    config, _resolved = _api_resolve_full(
        calc_type,
        model_path,
        cli,
        output_dir,
        job_name,
        settings_path,
        model_alias,
        profile,
        strict_config,
    )
    return config


def _api_resolve_full(
    calc_type: str,
    model_path: str | None,
    cli: dict[str, Any],
    output_dir: str | Path,
    job_name: str | None,
    settings_path: str | None,
    model_alias: str | None,
    profile: str | None,
    strict_config: bool | None,
    *,
    incar_layer: dict[str, Any] | None = None,
) -> tuple[EngineConfig, Any]:
    """Resolve API inputs while retaining immutable provenance for workflows."""
    from mliport.config import load_settings  # noqa: PLC0415
    from mliport.config import resolve_config  # noqa: PLC0415

    settings = load_settings(explicit=settings_path)
    if model_path:
        cli.setdefault("model_path", model_path)
    if strict_config is not None:
        cli["strict_config"] = strict_config
    resolved = resolve_config(
        calc_type=calc_type,
        settings=settings,
        incar=incar_layer,
        cli=cli,
        model_alias_name=model_alias,
        profile_name=profile,
        cli_base_dir=Path.cwd(),
    )
    ec = EngineConfig.from_resolved(resolved)
    ec.output_dir = Path(output_dir).expanduser().resolve()
    if job_name:
        ec.job_name = job_name
    return ec, resolved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_single_point(
    structure: Atoms | str | Path,
    model_path: str,
    model_type: str | None = None,
    task: str | None = None,
    device: str | None = None,
    inference_mode: str | None = None,
    job_name: str | None = None,
    output_dir: str = "./results",
    verbose: bool = True,
    settings_path: str | None = None,
    model_alias: str | None = None,
    default_dtype: str | None = None,
    head: str | None = None,
    profile: str | None = None,
    strict_config: bool | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run single point calculation.

    Calculates energy, forces, and stress for a given structure.

    Args:
        structure: ASE Atoms object or path to structure file
        model_path: Path to model checkpoint
        model_type: MLIP engine (uma, mace, dpa, grace). None = resolver decides.
        task: Model head (omat, omol, oc20, oc25, odac, omc).
        device: Device for calculation (cpu or cuda).
        inference_mode: Inference-mode override.
        job_name: Optional job name for organizing results.
        output_dir: Base directory for output files.
        verbose: Whether to print progress messages.
        settings_path: Explicit path to settings.ini.
        model_alias: Named model alias from settings.ini [model:NAME].
        default_dtype: MACE dtype override (float32 or float64).
        head: MACE head or DeepMD/DPA multi-task branch name.
        profile: Reusable profile from settings.ini [profile:NAME].
        strict_config: Override strict-config behaviour.
        **kwargs: Additional calc/run options forwarded to the resolver.

    Returns:
        Dictionary with results (energy, forces, stress, time)

    Example:
        >>> results = run_single_point(
        ...     structure="structure.cif",
        ...     model_path="uma-s-1.pt",
        ...     task="omat",
        ...     job_name="sp_calc"
        ... )
        >>> print(f"Energy: {results['energy']:.4f} eV")
    """
    started_at = time.perf_counter()
    atoms = _load_structure(structure)
    if verbose:
        print(f"System: {atoms.get_chemical_formula()}")
        print(f"Atoms: {len(atoms)}")
        print(f"Loading model: {model_path}")

    cli = _build_api_cli(
        "sp", model_type, task, device, inference_mode, default_dtype, head, kwargs
    )
    config = _api_resolve(
        "sp",
        model_path,
        cli,
        output_dir,
        job_name,
        settings_path,
        model_alias,
        profile,
        strict_config,
    )
    engine = CalculationEngine.from_config(config)
    return engine.run(
        atoms,
        log_fn=_console_log if verbose else None,
        started_at=started_at,
    )


def run_optimization(
    structure: Atoms | str | Path,
    model_path: str,
    model_type: str | None = None,
    task: str | None = None,
    device: str | None = None,
    inference_mode: str | None = None,
    job_name: str | None = None,
    output_dir: str = "./results",
    fmax: float | None = None,
    max_steps: int | None = None,
    optimizer: str | None = None,
    cell_opt: bool | None = None,
    fix_symmetry: bool | None = None,
    verbose: bool = True,
    settings_path: str | None = None,
    model_alias: str | None = None,
    default_dtype: str | None = None,
    head: str | None = None,
    profile: str | None = None,
    strict_config: bool | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run geometry optimization.

    Optimizes atomic positions and optionally cell parameters
    until forces converge below threshold.  All defaults are drawn from
    ``resolve_config()``; only explicitly-passed values act as overrides.

    Args:
        structure: ASE Atoms object or path to structure file.
        model_path: Path to model checkpoint.
        model_type: MLIP engine (uma, mace, dpa, grace).
        task: Model head.
        device: Device for calculation (cpu or cuda).
        inference_mode: Inference-mode override.
        job_name: Optional job name for organizing results.
        output_dir: Base directory for output files.
        fmax: Force convergence threshold in eV/Å.
        max_steps: Maximum optimization steps.
        optimizer: Optimization algorithm (FIRE, BFGS, LBFGS).
        cell_opt: Whether to optimize cell parameters.
        fix_symmetry: Whether to preserve symmetry.
        verbose: Whether to print progress messages.
        settings_path: Explicit path to settings.ini.
        model_alias: Named model alias from settings.ini.
        default_dtype: MACE dtype override.
        head: MACE head or DeepMD/DPA multi-task branch name.
        profile: Reusable profile from settings.ini.
        strict_config: Override strict-config behaviour.
        **kwargs: Additional options forwarded to the resolver.

    Returns:
        Dictionary with results (energy, converged, nsteps, etc.)

    Example:
        >>> results = run_optimization(
        ...     structure="structure.cif",
        ...     model_path="uma-s-1.pt",
        ...     fmax=0.02,
        ...     cell_opt=True,
        ...     job_name="opt_calc"
        ... )
        >>> print(f"Converged: {results['converged']}")
    """
    started_at = time.perf_counter()
    atoms = _load_structure(structure)
    if verbose:
        print(f"System: {atoms.get_chemical_formula()}")
        print(f"Atoms: {len(atoms)}")
        print(f"Loading model: {model_path}")

    extra = dict(kwargs)
    for name, value in (
        ("fmax", fmax),
        ("max_steps", max_steps),
        ("optimizer", optimizer),
        ("cell_opt", cell_opt),
        ("fix_symmetry", fix_symmetry),
    ):
        if value is not None:
            extra[name] = value

    cli = _build_api_cli(
        "opt", model_type, task, device, inference_mode, default_dtype, head, extra
    )
    config = _api_resolve(
        "opt",
        model_path,
        cli,
        output_dir,
        job_name,
        settings_path,
        model_alias,
        profile,
        strict_config,
    )
    engine = CalculationEngine.from_config(config)
    return engine.run(
        atoms,
        log_fn=_console_log if verbose else None,
        started_at=started_at,
    )


def run_md(
    structure: Atoms | str | Path,
    model_path: str,
    model_type: str | None = None,
    task: str | None = None,
    device: str | None = None,
    inference_mode: str | None = None,
    job_name: str | None = None,
    output_dir: str = "./results",
    ensemble: str | None = None,
    temperature: float | None = None,
    timestep: float | None = None,
    steps: int | None = None,
    thermostat: str | None = None,
    friction: float | None = None,
    bussi_tau: float | None = None,
    nhc_tdamp: float | None = None,
    nhc_tchain: int | None = None,
    nhc_tloop: int | None = None,
    save_interval: int | None = None,
    pre_relax: bool | None = None,
    verbose: bool = True,
    settings_path: str | None = None,
    model_alias: str | None = None,
    default_dtype: str | None = None,
    head: str | None = None,
    profile: str | None = None,
    strict_config: bool | None = None,
    com_policy: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run molecular dynamics simulation.

    Runs MD simulation using NVT (Langevin, Bussi/CSVR, or Nose-Hoover chain)
    or NVE (Velocity Verlet).
    Optional pre-relaxation reduces large initial atomic forces by optimizing
    positions only; it does not relax the cell or generally eliminate stress.
    All defaults are drawn from ``resolve_config()``.

    Args:
        structure: ASE Atoms object or path to structure file.
        model_path: Path to model checkpoint.
        model_type: MLIP engine (uma, mace, dpa, grace).
        task: Model head.
        device: Device for calculation (cpu or cuda).
        inference_mode: Inference-mode override (md defaults to "turbo").
        job_name: Optional job name for organizing results.
        output_dir: Base directory for output files.
        ensemble: MD ensemble (NVT or NVE).
        temperature: Temperature in Kelvin.
        timestep: Time step in femtoseconds.
        steps: Number of MD steps.
        thermostat: NVT thermostat (LANGEVIN, BUSSI, or NHC).
        friction: Langevin friction coefficient (1/fs).
        bussi_tau: Bussi/CSVR coupling time in femtoseconds.
        nhc_tdamp: Nose-Hoover-chain damping time in femtoseconds.
        nhc_tchain: Nose-Hoover chain length.
        nhc_tloop: Nose-Hoover thermostat integration substeps.
        save_interval: Interval for saving trajectory frames.
        pre_relax: Whether to perform pre-relaxation before MD.
        verbose: Whether to print progress messages.
        settings_path: Explicit path to settings.ini.
        model_alias: Named model alias from settings.ini.
        default_dtype: MACE dtype override.
        head: MACE head or DeepMD/DPA multi-task branch name.
        profile: Reusable profile from settings.ini.
        strict_config: Override strict-config behaviour.
        com_policy: Center-of-mass policy (auto, none, initialize_only, constraint).
        **kwargs: Additional options forwarded to the resolver.

    Returns:
        Dictionary with results (temperature, energy, etc.)

    Example:
        >>> results = run_md(
        ...     structure="structure.cif",
        ...     model_path="uma-s-1.pt",
        ...     ensemble="NVT",
        ...     temperature=300,
        ...     steps=10000,
        ...     job_name="md_calc"
        ... )
        >>> print(f"Final temperature: {results['temperature']:.1f} K")
    """
    started_at = time.perf_counter()
    atoms = _load_structure(structure)
    if verbose:
        print(f"System: {atoms.get_chemical_formula()}")
        print(f"Atoms: {len(atoms)}")
        print(f"Loading model: {model_path}")

    extra = dict(kwargs)
    for name, value in (
        ("ensemble", ensemble),
        ("temperature", temperature),
        ("timestep", timestep),
        ("steps", steps),
        ("thermostat", thermostat),
        ("friction", friction),
        ("bussi_tau", bussi_tau),
        ("nhc_tdamp", nhc_tdamp),
        ("nhc_tchain", nhc_tchain),
        ("nhc_tloop", nhc_tloop),
        ("save_interval", save_interval),
        ("com_policy", com_policy),
        ("pre_relax", pre_relax),
    ):
        if value is not None:
            extra[name] = value

    cli = _build_api_cli(
        "md", model_type, task, device, inference_mode, default_dtype, head, extra
    )
    config = _api_resolve(
        "md",
        model_path,
        cli,
        output_dir,
        job_name,
        settings_path,
        model_alias,
        profile,
        strict_config,
    )
    engine = CalculationEngine.from_config(config)
    return engine.run(
        atoms,
        log_fn=_console_log if verbose else None,
        started_at=started_at,
    )


def run_neb(
    initial: Atoms | str | Path | None = None,
    final: Atoms | str | Path | None = None,
    model_path: str | None = None,
    *,
    resume: str | Path | None = None,
    model_type: str | None = None,
    task: str | None = None,
    device: str | None = None,
    inference_mode: str | None = None,
    output_dir: str | Path | None = None,
    job_name: str | None = None,
    n_intermediate_images: int | None = None,
    climb: bool | None = None,
    method: str | None = None,
    interpolation: str | None = None,
    path_convention: str | None = None,
    spring: float | None = None,
    fmax: float | None = None,
    max_steps: int | None = None,
    pre_fmax: float | None = None,
    pre_max_steps: int | None = None,
    maxstep: float | None = None,
    endpoint_policy: str | None = None,
    endpoint_fmax: float | None = None,
    endpoint_steps: int | None = None,
    idpp_fmax: float | None = None,
    idpp_steps: int | None = None,
    idpp_mic: bool | None = None,
    min_distance: float | None = None,
    checkpoint_interval: int | None = None,
    allow_unvalidated_neb: bool | None = None,
    atom_map=None,
    image_shifts=None,
    verbose: bool = True,
    settings_path: str | None = None,
    model_alias: str | None = None,
    default_dtype: str | None = None,
    head: str | None = None,
    profile: str | None = None,
    strict_config: bool | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run or geometry-resume a fixed-cell NEB/CI-NEB workflow.

    New runs require two endpoints, a model (path or alias), and an explicit
    output directory. Resume restores model and scientific options from the
    checkpoint, preserves the original run ID/directory, and creates a new
    attempt ID. Every interface uses the same typed resolver and output schema.
    """
    from mliport.neb.workflow import (  # noqa: PLC0415
        checkpoint_resolved_layer,
        checkpoint_run_directory,
    )

    extra = dict(kwargs)
    for name, value in (
        ("n_intermediate_images", n_intermediate_images),
        ("climb", climb),
        ("neb_method", method),
        ("neb_interpolation", interpolation),
        ("path_convention", path_convention),
        ("neb_spring", spring),
        ("fmax", fmax),
        ("max_steps", max_steps),
        ("neb_pre_fmax", pre_fmax),
        ("neb_pre_max_steps", pre_max_steps),
        ("neb_maxstep", maxstep),
        ("endpoint_policy", endpoint_policy),
        ("endpoint_fmax", endpoint_fmax),
        ("endpoint_steps", endpoint_steps),
        ("idpp_fmax", idpp_fmax),
        ("idpp_steps", idpp_steps),
        ("idpp_mic", idpp_mic),
        ("neb_min_distance", min_distance),
        ("checkpoint_interval", checkpoint_interval),
        ("allow_unvalidated_neb", allow_unvalidated_neb),
    ):
        if value is not None:
            extra[name] = value

    checkpoint_layer = None
    if resume is not None:
        if initial is not None or final is not None:
            raise ValueError("resume cannot be combined with new NEB endpoints")
        if atom_map is not None or image_shifts is not None:
            raise ValueError("resume restores atom mapping/winding from checkpoint")
        if job_name is not None:
            raise ValueError("job_name cannot change a resumed run directory")
        if model_alias is not None or profile is not None:
            raise ValueError(
                "resume restores original alias/profile values; use direct kwargs, "
                "which are checked by the fingerprint"
            )
        run_dir = checkpoint_run_directory(resume)
        if (
            output_dir is not None
            and Path(output_dir).expanduser().resolve() != run_dir
        ):
            raise ValueError(
                f"output_dir must remain the original run directory {run_dir}"
            )
        output_dir = run_dir
        checkpoint_layer = checkpoint_resolved_layer(resume)
        initial_atoms = final_atoms = None
    else:
        if initial is None or final is None:
            raise ValueError("A new NEB run requires initial and final endpoints")
        if output_dir is None:
            raise ValueError("A new NEB run requires an explicit output_dir")
        initial_atoms = _load_structure(initial)
        final_atoms = _load_structure(final)
        if not isinstance(initial, Atoms):
            extra["neb_initial"] = str(initial)
        if not isinstance(final, Atoms):
            extra["neb_final"] = str(final)

    cli = _build_api_cli(
        "neb",
        model_type,
        task,
        device,
        inference_mode,
        default_dtype,
        head,
        extra,
    )
    config, resolved = _api_resolve_full(
        "neb",
        model_path,
        cli,
        output_dir,
        job_name,
        settings_path,
        model_alias,
        profile,
        strict_config,
        incar_layer=checkpoint_layer,
    )
    if verbose:
        if initial_atoms is not None:
            print(
                f"NEB endpoints: {initial_atoms.get_chemical_formula()} "
                f"({len(initial_atoms)} atoms, {n_intermediate_images or 'resolved'} images)"
            )
        print(f"Loading model: {resolved.model_path}")
    engine = CalculationEngine.from_config(config)
    return engine.run_neb(
        resolved,
        initial=initial_atoms,
        final=final_atoms,
        resume=resume,
        atom_map=atom_map,
        image_shifts=image_shifts,
        log_fn=_console_log if verbose else None,
    )


def calculate_energy(
    structure: Atoms | str | Path,
    model_path: str,
    model_type: str | None = None,
    task: str | None = None,
    device: str | None = None,
    relax: bool = False,
    fmax: float | None = None,
    max_steps: int | None = None,
    verbose: bool = False,
    settings_path: str | None = None,
    model_alias: str | None = None,
    default_dtype: str | None = None,
    head: str | None = None,
    profile: str | None = None,
    strict_config: bool | None = None,
    **kwargs,
) -> float:
    """Calculate energy of a structure.

    Simple interface to get just the energy value. Optionally performs
    a quick geometry optimization before calculating energy.

    Args:
        structure: ASE Atoms object or path to structure file.
        model_path: Path to model checkpoint.
        model_type: MLIP engine (uma, mace, dpa, grace).
        task: Model head.
        device: Device for calculation (cpu or cuda).
        relax: Whether to pre-relax structure before energy calculation.
        fmax: Force convergence threshold for relaxation (if relax=True).
        max_steps: Maximum steps for relaxation (if relax=True).
        verbose: Whether to print progress messages.
        settings_path: Explicit path to settings.ini.
        model_alias: Named model alias from settings.ini.
        default_dtype: MACE dtype override.
        head: MACE head or DeepMD/DPA multi-task branch name.
        profile: Reusable profile from settings.ini.
        strict_config: Override strict-config behaviour.
        **kwargs: Additional options.

    Returns:
        Energy in eV

    Example:
        >>> energy = calculate_energy("structure.cif", "uma-s-1.pt")
        >>> print(f"Energy: {energy:.4f} eV")
        >>>
        >>> # With pre-relaxation
        >>> energy = calculate_energy(
        ...     "structure.cif", "uma-s-1.pt", relax=True
        ... )
    """
    if relax:
        results = run_optimization(
            structure=structure,
            model_path=model_path,
            model_type=model_type,
            task=task,
            device=device,
            fmax=fmax,
            max_steps=max_steps,
            verbose=verbose,
            settings_path=settings_path,
            model_alias=model_alias,
            default_dtype=default_dtype,
            head=head,
            profile=profile,
            strict_config=strict_config,
            **kwargs,
        )
    else:
        results = run_single_point(
            structure=structure,
            model_path=model_path,
            model_type=model_type,
            task=task,
            device=device,
            verbose=verbose,
            settings_path=settings_path,
            model_alias=model_alias,
            default_dtype=default_dtype,
            head=head,
            profile=profile,
            strict_config=strict_config,
            **kwargs,
        )

    return results["energy"]


__all__ = [
    "calculate_energy",
    "run_md",
    "run_neb",
    "run_optimization",
    "run_single_point",
]
