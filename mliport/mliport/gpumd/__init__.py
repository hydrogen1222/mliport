"""Native GPUMD execution for the NEP backend (PR #61 §C).

GPUMD is an external GPU program, not an ASE calculator: one process performs
a whole SP / optimisation / MD run.  This package builds the two input files,
runs the executable safely, parses its output into mliport's normalised form,
and records provenance.  It never re-implements an integrator.

The package intentionally imports nothing heavy at import time; public names
are resolved lazily so ``import mliport.gpumd`` stays cheap.
"""

from __future__ import annotations

__all__ = [
    "ASE_TIME_UNIT_FS",
    "Frame",
    "GPUMDParseError",
    "NEPGPUMDIdentity",
    "NativeGPUMDRunner",
    "OPTIMIZER_MAPPING",
    "ProcessOutcome",
    "THERMOSTAT_MAPPING",
    "build_gpumd_provenance",
    "build_model_xyz",
    "build_run_in",
    "gpumd_identity",
    "iter_extxyz_frames",
    "md_frames_summary",
    "momenta_to_velocities_ang_per_fs",
    "optimization_result",
    "parse_extxyz_frames",
    "parse_nep_model",
    "parse_thermo_out",
    "provenance",
    "resolve_gpumd_command",
    "run_gpumd",
    "run_in_provenance",
    "single_point_from_dump",
    "validate_elements",
    "voigt_stress",
    "write_normalized_trajectory",
    "write_provenance",
]

_IMPORTS = {
    "ASE_TIME_UNIT_FS": ".input",
    "OPTIMIZER_MAPPING": ".input",
    "THERMOSTAT_MAPPING": ".input",
    "build_model_xyz": ".input",
    "build_run_in": ".input",
    "momenta_to_velocities_ang_per_fs": ".input",
    "parse_nep_model": ".input",
    "run_in_provenance": ".input",
    "validate_elements": ".input",
    "ProcessOutcome": ".process",
    "gpumd_identity": ".process",
    "resolve_gpumd_command": ".process",
    "run_gpumd": ".process",
    "Frame": ".parse",
    "GPUMDParseError": ".parse",
    "iter_extxyz_frames": ".parse",
    "md_frames_summary": ".parse",
    "optimization_result": ".parse",
    "parse_extxyz_frames": ".parse",
    "parse_thermo_out": ".parse",
    "single_point_from_dump": ".parse",
    "voigt_stress": ".parse",
    "write_normalized_trajectory": ".parse",
    "build_gpumd_provenance": ".provenance",
    "provenance": ".provenance",
    "write_provenance": ".provenance",
    "NEPGPUMDIdentity": ".runner",
    "NativeGPUMDRunner": ".runner",
}


def __getattr__(name: str):
    """Lazily import a public name to keep the package import-light."""
    module_name = _IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib  # noqa: PLC0415

    module = importlib.import_module(module_name, __package__)
    value = getattr(module, name)
    globals()[name] = value
    return value
