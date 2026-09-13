"""Thermodynamic trajectory diagnostics without equilibration heuristics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any

    from mliport.analysis.dataset import TrajectoryDataset


def _statistics(values: np.ndarray) -> dict[str, float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "last": float(finite[-1]),
    }


def block_statistics(values: np.ndarray, *, block_size_frames: int) -> dict[str, Any]:
    """Return reproducible non-overlapping block means and their spread."""

    if block_size_frames < 1:
        raise ValueError("block_size_frames must be >= 1")
    values = np.asarray(values, dtype=float)
    n_blocks = len(values) // block_size_frames
    if n_blocks < 1:
        raise ValueError("block_size_frames is longer than the selected trajectory")
    trimmed = values[: n_blocks * block_size_frames]
    blocks = trimmed.reshape(n_blocks, block_size_frames)
    means = np.nanmean(blocks, axis=1)
    finite = means[np.isfinite(means)]
    return {
        "block_size_frames": int(block_size_frames),
        "number_of_complete_blocks": int(n_blocks),
        "discarded_tail_frames": int(len(values) - len(trimmed)),
        "block_means": means,
        "mean_of_block_means": float(np.mean(finite)) if len(finite) else np.nan,
        "std_of_block_means": (
            float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        ),
    }


def thermodynamic_diagnostics(
    dataset: TrajectoryDataset,
    *,
    include_equilibration: bool = False,
    start: int | None = None,
    stop: int | None = None,
    block_size_frames: int | None = None,
) -> dict[str, Any]:
    """Select thermodynamic columns and summarize only available observables."""

    view = dataset.analysis_view(
        include_equilibration=include_equilibration, start=start, stop=stop
    )
    natoms = view.natoms
    columns: dict[str, np.ndarray] = {}
    if view.times_fs is not None:
        columns["time_ps"] = (view.times_fs - view.times_fs[0]) / 1000.0
        columns["absolute_time_ps"] = view.times_fs / 1000.0
    if view.steps is not None:
        columns["step"] = view.steps
    if view.phases is not None:
        columns["phase"] = view.phases
    mappings = (
        ("temperature_K", view.temperature_K, 1.0),
        ("potential_energy_eV_atom", view.potential_energy_eV, 1.0 / natoms),
        ("kinetic_energy_eV_atom", view.kinetic_energy_eV, 1.0 / natoms),
        ("total_energy_eV_atom", view.total_energy_eV, 1.0 / natoms),
        ("pressure_GPa", view.pressure_GPa, 1.0),
        ("volume_A3", view.volumes_A3, 1.0),
    )
    for name, values, scale in mappings:
        if values is not None:
            columns[name] = np.asarray(values, dtype=float) * scale

    summary: dict[str, Any] = {
        "n_frames": view.nframes,
        "phase": "all" if include_equilibration else "production",
    }
    if view.times_fs is not None:
        summary.update(
            {
                "time_start_ps": float(view.times_fs[0] / 1000.0),
                "time_end_ps": float(view.times_fs[-1] / 1000.0),
                "duration_ps": float((view.times_fs[-1] - view.times_fs[0]) / 1000.0),
            }
        )
    summary_fields = {
        "temperature_K": "temperature",
        "potential_energy_eV_atom": "potential_energy",
        "kinetic_energy_eV_atom": "kinetic_energy",
        "total_energy_eV_atom": "total_energy",
        "pressure_GPa": "pressure",
        "volume_A3": "volume",
    }
    for column, prefix in summary_fields.items():
        if column not in columns:
            continue
        stats = _statistics(columns[column])
        if stats is None:
            continue
        unit = column.removeprefix(prefix + "_")
        for statistic, value in stats.items():
            summary[f"{prefix}_{statistic}_{unit}"] = value

    ensemble = str(
        dataset.metadata.get("resolved_config", {})
        .get("run_options", {})
        .get("ensemble", "")
    ).upper()
    if (
        ensemble == "NVE"
        and "total_energy_eV_atom" in columns
        and view.times_fs is not None
        and view.nframes >= 2
    ):
        energy = columns["total_energy_eV_atom"]
        time_ps = (view.times_fs - view.times_fs[0]) / 1000.0
        finite = np.isfinite(energy) & np.isfinite(time_ps)
        if np.count_nonzero(finite) >= 2 and np.ptp(time_ps[finite]) > 0:
            slope, intercept = np.polyfit(time_ps[finite], energy[finite], 1)
            fitted = slope * time_ps[finite] + intercept
            initial = float(fitted[0])
            total_drift = float(fitted[-1] - fitted[0])
            summary["nve_total_energy_drift_eV_atom_ps"] = float(slope)
            summary["nve_total_energy_drift_meV_atom_ps"] = float(slope * 1000.0)
            summary["nve_fitted_total_drift_eV_atom"] = total_drift
            summary["nve_relative_drift_over_production"] = (
                total_drift / abs(initial) if not np.isclose(initial, 0.0) else None
            )

    blocks: dict[str, Any] = {}
    if block_size_frames is not None:
        for name, values in columns.items():
            if name in {"time_ps", "absolute_time_ps", "step", "phase"}:
                continue
            blocks[name] = block_statistics(
                np.asarray(values, dtype=float), block_size_frames=block_size_frames
            )
    return {"columns": columns, "summary": summary, "blocks": blocks}
