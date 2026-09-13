"""Generate the beta validation figures from machine-readable evidence.

Taskbook section 38: every figure regenerates from records produced by
the canonical evidence loader (``evidence.load_evidence``) plus immutable
per-structure CSV files; nothing is hand-edited and no figure parses raw
result JSON itself.  The figure manifest lists every contributing record
path.  Engine colours are consistent across all figures.

Usage::

    python generate_figures.py [--evidence-root .validation-work] \
        [--out validation/science/reports/figures]

A figure is skipped (with an explicit reason printed and returned in the
manifest) when the underlying evidence cannot support it -- e.g. the
OMat24 stress-parity figure, because the official validation split
carries no reference stress labels.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Canonical evidence layer: figures never parse raw records themselves
# (task book PR-A).  Every number they draw points at a record listed in the
# figure manifest.
_SCIENCE_ROOT = Path(__file__).resolve().parent.parent
if str(_SCIENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCIENCE_ROOT))

from evidence import (  # noqa: E402
    STATUS_RANK,
    TIER_NAMES,
    CampaignManifestError,
    build_version_block,
    load_evidence,
)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

#: consistent engine styling across every figure (taskbook section 38)
ENGINE_STYLE = {
    "mace": {"color": "#1f77b4", "label": "MACE-OMAT-0 (float64)"},
    "dpa": {"color": "#d62728", "label": "DPA-3.1-3M (Omat24)"},
    "grace": {"color": "#2ca02c", "label": "GRACE-2L-OMAT"},
    "uma": {"color": "#9467bd", "label": "UMA-s-1p2 (OMat)"},
}
ENGINE_ORDER = ("mace", "dpa", "grace", "uma")

DPI = 150


def _representative(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic figure representative for one engine/case.

    The newest non-failing profile wins; if every profile failed, the newest
    failure is still plotted (and the manifest lists every record considered).
    Never status-blind "prefer pass": a failure with no passing alternative
    remains visible.
    """
    ranked = sorted(
        records,
        key=lambda rec: (
            STATUS_RANK.get(str(rec.get("status")), -1) != STATUS_RANK["fail"],
            str(rec.get("git_commit") or ""),
            str(rec.get("_path") or ""),
        ),
        reverse=True,
    )
    return ranked[0]


def _group_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """compound case key -> {engine: representative record}.

    ``records`` must come from :func:`evidence.load_evidence`; the figure
    manifest records every ``_path`` that contributed so a plotted number can
    always be traced back to its immutable evidence.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        case_id = record.get("case_id")
        test_id = record.get("test_id")
        if not case_id or not test_id:
            continue
        grouped[f"{case_id}__{test_id}"].append(record)
    out: dict[str, dict[str, Any]] = {}
    for key, entries in grouped.items():
        per_engine: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in entries:
            per_engine[str(record.get("engine", "?"))].append(record)
        out[key] = {
            engine: _representative(engine_records)
            for engine, engine_records in per_engine.items()
        }
    return out


def _record_label(record: dict[str, Any]) -> str:
    """`engine/dtype#digest (path)` for one contributing record."""
    engine = str(record.get("engine", "?"))
    dtype = str(record.get("dtype", "?"))
    digest = str(record.get("profile_id") or "").rsplit("-", 1)[-1][:6]
    return f"{engine}/{dtype}#{digest} ({record.get('_path')})"


def _all_profile_labels(records: list[dict[str, Any]]) -> list[str]:
    """`record_label` for every record a figure series considered."""
    return [
        _record_label(record)
        for record in sorted(records, key=lambda r: str(r.get("_path") or ""))
    ]


def _footer(fig: plt.Figure, sources: list[str]) -> None:
    fig.text(
        0.01,
        0.01,
        "sources: " + "; ".join(sources[:4]),
        fontsize=5.5,
        color="0.35",
    )


def _style(engine: str) -> dict[str, str]:
    return ENGINE_STYLE.get(engine, {"color": "0.5", "label": engine})


def fig_omat24_energy(t3_dir: Path, out: Path) -> str | None:
    """Parity + error distribution from the per-structure CSV export."""
    series = []
    for eng in ENGINE_ORDER:
        path = t3_dir / f"t3_errors_{eng}.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open()))
        de = np.array([float(r["de_per_atom_eV"]) for r in rows if r["de_per_atom_eV"]])
        e_ref = np.array([float(r["e_ref_eV"]) for r in rows if r["de_per_atom_eV"]])
        e_mod = np.array([float(r["e_model_eV"]) for r in rows if r["de_per_atom_eV"]])
        nat = np.array([float(r["natoms"]) for r in rows if r["de_per_atom_eV"]])
        series.append((eng, e_ref / nat, e_mod / nat, de, path.name))
    if not series:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    all_de = np.concatenate([s[3] for s in series])
    lim = max(np.abs(all_de).max(), 1e-6) * 1.05
    ax = axes[0]
    ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.8, alpha=0.6)
    for eng, e_ref, e_mod, _de, _n in series:
        st = _style(eng)
        ax.plot(
            e_ref, e_mod, ".", ms=3, color=st["color"], alpha=0.7, label=st["label"]
        )
    ax.set_xlabel("OMat24 reference energy (eV/atom)")
    ax.set_ylabel("model energy (eV/atom)")
    ax.set_title("energy parity")
    ax.legend(fontsize=7)
    ax = axes[1]
    bins = np.linspace(-lim, lim, 60)
    for eng, _r, _m, de, _n in series:
        st = _style(eng)
        ax.hist(
            de, bins=bins, histtype="step", lw=1.4, color=st["color"], label=st["label"]
        )
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"energy error, model $-$ DFT (eV/atom)")
    ax.set_ylabel("structures")
    ax.set_title("energy error distribution")
    ax.legend(fontsize=7)
    fig.suptitle(
        "T3: OMat24 held-out subset (256 structures, no offsets fitted)",
        fontsize=10,
    )
    _footer(fig, [s[4] for s in series])
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out / "fig01_omat24_energy.png", dpi=DPI)
    plt.close(fig)
    return "fig01_omat24_energy.png"


def fig_omat24_forces(t3_dir: Path, out: Path) -> str | None:
    """Component error vs reference force magnitude, binned medians."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    plotted = False
    sources = []
    for eng in ENGINE_ORDER:
        path = t3_dir / f"t3_errors_{eng}.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open()))
        frms = np.array([float(r["f_ref_rms_eV_A"]) for r in rows if r["ok"] == "1"])
        err = np.array(
            [float(r["f_component_mae_eV_A"]) for r in rows if r["ok"] == "1"]
        )
        edges = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, float("inf")]
        xs, ys = [], []
        for lo, hi in itertools.pairwise(edges):
            sel = (frms >= lo) & (frms < hi)
            if sel.sum() >= 3:
                xs.append(np.sqrt(lo * hi) if np.isfinite(hi) else lo * 1.3)
                ys.append(np.median(err[sel]))
        if xs:
            st = _style(eng)
            ax.plot(xs, ys, "o-", ms=4, lw=1.2, color=st["color"], label=st["label"])
            plotted = True
        sources.append(path.name)
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel(r"reference force RMS per structure (eV/$\mathrm{\AA}$)")
    ax.set_ylabel(r"median component MAE (eV/$\mathrm{\AA}$)")
    ax.set_title("T3: force error vs reference force magnitude")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    _footer(fig, sources)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig02_omat24_forces.png", dpi=DPI)
    plt.close(fig)
    return "fig02_omat24_forces.png"


def fig_eos(t4_records: dict, out: Path) -> str | None:
    grouped = t4_records
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    sources = []
    plotted = False
    for ax, system, title in (
        (axes[0], "t4_eos__t4c_cu_fcc", "Cu fcc"),
        (axes[1], "t4_eos__t4c_si_diamond", "Si diamond"),
    ):
        per_engine = grouped.get(system, {})
        for eng in ENGINE_ORDER:
            rec = per_engine.get(eng)
            if rec is None:
                continue
            v = np.array(rec["metrics"]["volumes_A3"], dtype=float)
            e = np.array(rec["metrics"]["energy_per_atom_eV"], dtype=float)
            e = e - e.min()
            st = _style(eng)
            ax.plot(v, e, "o-", ms=3.5, lw=1.2, color=st["color"], label=st["label"])
            sources.append(f"{system}__{eng}")
            plotted = True
        ax.set_title(title)
        ax.set_xlabel(r"volume ($\mathrm{\AA}^3$)")
        ax.set_ylabel(r"$E - E_{\min}$ (eV/atom)")
    if plotted:
        axes[0].legend(fontsize=7)
        fig.suptitle("T4: EOS curves (9-point volume scan)", fontsize=10)
        _footer(fig, sources or ["t4_eos__t4c_* records"])
        fig.tight_layout(rect=(0, 0.03, 1, 0.93))
        fig.savefig(out / "fig04_eos.png", dpi=DPI)
        plt.close(fig)
        return "fig04_eos.png"
    plt.close(fig)
    return None


def fig_elastic(t4_records: dict, out: Path) -> str | None:
    grouped = t4_records
    constants = ("C11_GPa", "C12_GPa", "C44_GPa")
    systems = (
        ("t4_elastic__t4d_cu_fcc_clamped", "Cu fcc (clamped-ion)"),
        ("t4_elastic__t4d_cu_fcc_relaxed", "Cu fcc (relaxed-ion)"),
        ("t4_elastic__t4d_si_diamond_clamped", "Si diamond (clamped)"),
        ("t4_elastic__t4d_si_diamond_relaxed", "Si diamond (relaxed)"),
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(constants))
    width = 0.8 / max(len(systems), 1)
    plotted = False
    for i, (case, _label) in enumerate(systems):
        per_engine = grouped.get(case, {})
        for j, eng in enumerate(ENGINE_ORDER):
            rec = per_engine.get(eng)
            if rec is None:
                continue
            vals = [rec["metrics"].get(c, np.nan) for c in constants]
            offset = (i - (len(systems) - 1) / 2) * width
            ax.bar(
                x + offset + (j - 1.5) * width / 4,
                vals,
                width / 4,
                color=_style(eng)["color"],
                alpha=0.55 + 0.15 * (i % 2),
                edgecolor="none",
            )
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xticks(x)
    ax.set_xticklabels(["C11", "C12", "C44"])
    ax.set_ylabel("elastic constant (GPa)")
    ax.set_title("T4: cubic elastic constants (finite strain)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_style(e)["color"]) for e in ENGINE_ORDER
    ]
    ax.legend(handles, [_style(e)["label"] for e in ENGINE_ORDER], fontsize=7)
    _footer(fig, [case for case, _ in systems])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig05_elastic.png", dpi=DPI)
    plt.close(fig)
    return "fig05_elastic.png"


def fig_phonon(t4_records: dict, out: Path) -> str | None:
    grouped = t4_records
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    plotted = False
    for ax, case, title in (
        (axes[0], "t4_phonon__t4e_cu_fcc_prim_3x3x3_d0.005", "Cu fcc"),
        (axes[1], "t4_phonon__t4e_si_diamond_prim_3x3x3_d0.005", "Si diamond"),
    ):
        per_engine = grouped.get(case, {})
        for eng in ENGINE_ORDER:
            rec = per_engine.get(eng)
            if rec is None:
                continue
            rec = per_engine.get(eng)
            if rec is None:
                continue
            omegas = rec["metrics"].get("dispersion_omegas_sampled_eV")
            if not omegas:
                continue
            natoms_prim = int(
                rec.get("parameters", {}).get("fixture", {}).get("natoms_prim", 0)
            )
            nbranch = 3 * natoms_prim
            if nbranch <= 0 or len(omegas) % nbranch:
                continue  # cannot reshape the recorded sample honestly
            n_q = len(omegas) // nbranch
            om = np.array(omegas, dtype=float).reshape(n_q, nbranch)
            st = _style(eng)
            for branch in om.T:
                ax.plot(
                    np.arange(n_q), branch, "-", lw=0.9, color=st["color"], alpha=0.85
                )
            plotted = True
        ax.set_title(title)
        ax.set_xlabel("sampled path index (recorded 96-value subset)")
        ax.set_ylabel(r"$\omega$ (eV)")
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    if plotted:
        handles = [
            plt.Line2D((0, 0), (0, 0), color=_style(e)["color"], lw=1.5)
            for e in ENGINE_ORDER
        ]
        axes[0].legend(handles, [_style(e)["label"] for e in ENGINE_ORDER], fontsize=7)
        fig.suptitle(
            "T4: harmonic phonon dispersions (finite-displacement, ASR)",
            fontsize=10,
        )
        _footer(fig, ["t4_phonon__t4e_* records"])
        fig.tight_layout(rect=(0, 0.03, 1, 0.93))
        fig.savefig(out / "fig06_phonon.png", dpi=DPI)
        plt.close(fig)
        return "fig06_phonon.png"
    plt.close(fig)
    return None


def fig_vacancy(t4_records: dict, out: Path) -> str | None:
    grouped = t4_records
    cases = (
        ("t4_vacancy__t4g_cu_2x2x2", "2x2x2 (32)"),
        ("t4_vacancy__t4g_cu_3x3x3", "3x3x3 (108)"),
        ("t4_vacancy__t4g_cu_4x4x4", "4x4x4 (256)"),
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    plotted = False
    for eng in ENGINE_ORDER:
        xs, ys = [], []
        for i, (case, label) in enumerate(cases):
            rec = grouped.get(case, {}).get(eng)
            if rec is None:
                continue
            xs.append(label)
            ys.append(rec["metrics"]["evac_relaxed_eV"])
        if xs:
            st = _style(eng)
            ax.plot(
                range(len(xs)),
                ys,
                "o-",
                ms=4,
                lw=1.2,
                color=st["color"],
                label=st["label"],
            )
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([c[1] for c in cases])
    ax.set_xlabel("Cu supercell size (atoms)")
    ax.set_ylabel(r"relaxed vacancy energy $E_v$ (eV)")
    ax.set_title("T4: Cu vacancy finite-size trend")
    ax.legend(fontsize=8)
    _footer(fig, [c[0] for c in cases])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig07_vacancy.png", dpi=DPI)
    plt.close(fig)
    return "fig07_vacancy.png"


def fig_surface(t4_records: dict, out: Path) -> str | None:
    grouped = t4_records
    layers = ("4L", "6L", "8L")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    plotted = False
    for ax, vac, title in (
        (axes[0], "10A", "vacuum 10 Å"),
        (axes[1], "15A", "vacuum 15 Å"),
    ):
        for eng in ENGINE_ORDER:
            xs, ys = [], []
            for i, nl in enumerate(layers):
                case = f"t4_surface__t4h_cu111_{nl}_{vac}"
                rec = grouped.get(case, {}).get(eng)
                if rec is None:
                    continue
                xs.append(i)
                ys.append(rec["metrics"]["gamma_relaxed_J_m2"])
            if xs:
                st = _style(eng)
                ax.plot(
                    xs, ys, "o-", ms=4, lw=1.2, color=st["color"], label=st["label"]
                )
                plotted = True
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers)
        ax.set_title(title)
        ax.set_xlabel("Cu(111) slab layers")
    if plotted:
        axes[0].set_ylabel(r"relaxed surface energy $\gamma$ (J/m$^2$)")
        axes[0].legend(fontsize=7)
        fig.suptitle("T4: Cu(111) surface-energy convergence", fontsize=10)
        _footer(fig, ["t4_surface__t4h_cu111_* records"])
        fig.tight_layout(rect=(0, 0.03, 1, 0.93))
        fig.savefig(out / "fig08_surface.png", dpi=DPI)
        plt.close(fig)
        return "fig08_surface.png"
    plt.close(fig)
    return None


def _neb_profiles(t5_records: dict, case: str) -> dict[str, np.ndarray]:
    grouped = t5_records
    profiles = {}
    for eng in ENGINE_ORDER:
        rec = grouped.get(case, {}).get(eng)
        if rec is not None:
            e = rec["metrics"].get("energies_eV")
            if e:
                profiles[eng] = np.array(e, dtype=float)
    return profiles


def fig_neb(t5_records: dict, out: Path) -> str | None:
    panels = (
        ("t5_neb__t5d_cu_vacancy_ci_neb5", "Cu vacancy hop (CI-NEB 5 images)"),
        ("t5_neb__t5i_na3ps4_na_hop_ci_neb5", r"α-Na3PS4 Na hop (CI-NEB 5 images)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    plotted = False
    for ax, (case, title) in zip(axes, panels, strict=False):
        profiles = _neb_profiles(t5_records, case)
        for eng, e in profiles.items():
            st = _style(eng)
            x = np.linspace(0, 1, len(e))
            ax.plot(
                x, e - e.min(), "o-", ms=4, lw=1.2, color=st["color"], label=st["label"]
            )
            plotted = True
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("reaction coordinate (image)")
        ax.set_ylabel(r"$E - E_{\min}$ (eV)")
    if plotted:
        axes[0].legend(fontsize=7)
        fig.suptitle("T5: NEB minimum-energy paths", fontsize=10)
        _footer(fig, [c[0] for c in panels])
        fig.tight_layout(rect=(0, 0.03, 1, 0.93))
        fig.savefig(out / "fig09_10_neb_profiles.png", dpi=DPI)
        plt.close(fig)
        return "fig09_10_neb_profiles.png"
    plt.close(fig)
    return None


def fig_saddle(t5_records: dict, out: Path) -> str | None:
    grouped = t5_records
    case = "t5_saddle__t5h_saddle_hessian"
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    plotted = False
    for eng in ENGINE_ORDER:
        rec = grouped.get(case, {}).get(eng)
        if rec is None:
            continue
        per_delta = rec["metrics"].get("per_delta") or []
        deltas = rec["metrics"].get("deltas_A") or []
        mins = []
        for entry in per_delta:
            m = entry.get("min_eigenvalue_eV_A2")
            if m is None:
                continue
            mins.append(m)
        if len(deltas) == len(mins) and deltas:
            st = _style(eng)
            ax.plot(
                deltas, mins, "o-", ms=4, lw=1.2, color=st["color"], label=st["label"]
            )
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.6)
    ax.set_xlabel("FD step δ (Å)")
    ax.set_ylabel(r"lowest Hessian eigenvalue (eV/$\mathrm{\AA}^2$)")
    ax.set_title("T5: saddle Hessian negative-mode persistence")
    ax.legend(fontsize=8)
    _footer(fig, [case])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig11_saddle_hessian.png", dpi=DPI)
    plt.close(fig)
    return "fig11_saddle_hessian.png"


def fig_nve_drift(t6_records: dict, out: Path) -> str | None:
    grouped = t6_records
    case = "t6_nve__t6a_cu32_nve_timestep_sweep"
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    plotted = False
    for eng in ENGINE_ORDER:
        rec = grouped.get(case, {}).get(eng)
        if rec is None:
            continue
        per = rec["metrics"].get("per_timestep") or {}
        dts, slopes = [], []
        for dt, entry in sorted(per.items(), key=lambda kv: float(kv[0])):
            dts.append(float(dt))
            slopes.append(abs(entry["drift_slope_eV_atom_ps"]))
        if dts:
            st = _style(eng)
            ax.plot(
                dts, slopes, "o-", ms=4, lw=1.2, color=st["color"], label=st["label"]
            )
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("NVE timestep (fs)")
    ax.set_ylabel(r"|total-energy drift slope| (eV/atom/ps)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("T6: NVE energy drift vs timestep (Cu32, 300 K)")
    ax.legend(fontsize=8)
    _footer(fig, [case])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig12_nve_drift.png", dpi=DPI)
    plt.close(fig)
    return "fig12_nve_drift.png"


def fig_nvt_temperature(t6_dir: Path, out: Path) -> str | None:
    """Temperature trace + histogram from one NVT run directory."""
    run_root = t6_dir / "t6" / "md_runs"
    run_dirs = sorted(p for p in run_root.rglob("*") if p.is_dir())
    bussi_dirs = [
        p for p in run_dirs if "t6c_bussi" in p.name and (p / "raw" / "md.csv").exists()
    ]
    if not bussi_dirs:
        return None
    run = bussi_dirs[0]
    rows = list(csv.DictReader((run / "raw" / "md.csv").open()))
    t = np.array([float(r["time_fs"]) / 1000.0 for r in rows])
    temp = np.array([float(r["temperature_K"]) for r in rows])
    prod = np.array([r["phase"] == "production" for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    ax = axes[0]
    ax.plot(t, temp, lw=0.6, color=_style("mace")["color"])
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("temperature (K)")
    ax.set_title(f"NVT trace: {run.name}", fontsize=9)
    ax = axes[1]
    ax.hist(temp[prod], bins=40, color=_style("mace")["color"], alpha=0.8)
    ax.axvline(
        float(temp[prod].mean()),
        color="k",
        lw=1.0,
        label=f"mean {temp[prod].mean():.1f} K",
    )
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("frames")
    ax.set_title("production-phase distribution", fontsize=9)
    ax.legend(fontsize=8)
    fig.suptitle("T6: NVT thermostat behaviour (raw frames, no smoothing)", fontsize=10)
    _footer(fig, [str(run.relative_to(t6_dir.parent))])
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(out / "fig13_nvt_temperature.png", dpi=DPI)
    plt.close(fig)
    return "fig13_nvt_temperature.png"


def fig_transport(t7_records: dict, out: Path) -> str | None:
    grouped = t7_records
    case = "t7_transport__t7a_na3ps4_T700K"
    per_engine = grouped.get(case, {})
    if not per_engine:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = np.arange(len(per_engine))
    plotted = False
    for i, eng in enumerate(ENGINE_ORDER):
        rec = per_engine.get(eng)
        if rec is None:
            continue
        kin = rec["metrics"].get("kinisi_transport") or {}
        post = kin.get("D_posterior_m2_s") or {}
        if not post:
            continue
        mean = post["mean"]
        ci = post.get("credible_interval_95") or [mean, mean]
        yerr = [[mean - ci[0]], [ci[1] - mean]]
        st = _style(eng)
        ax.errorbar(
            i,
            mean * 1e8,
            yerr=np.array(yerr) * 1e8,
            fmt="o",
            ms=5,
            capsize=4,
            color=st["color"],
        )
        native = (rec["metrics"].get("native_msd") or {}).get("D_diagnostic_m2_s")
        if native:
            ax.plot(i, native * 1e8, "^", ms=5, mfc="none", color=st["color"])
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_style(e)["label"] for e in ENGINE_ORDER if e in per_engine],
        fontsize=8,
    )
    ax.set_ylabel(r"tracer $D$ at 700 K ($10^{-8}\,\mathrm{m^2/s}$)")
    ax.set_title(
        "T7: α-Na3PS4 transport at 700 K\n"
        "(circles: kinisi posterior mean ±95% CI; triangles: native MSD fit)",
        fontsize=9,
    )
    _footer(fig, [case])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig14_na3ps4_transport.png", dpi=DPI)
    plt.close(fig)
    return "fig14_na3ps4_transport.png"


def fig_performance(t8_records: dict, out: Path) -> str | None:
    grouped = t8_records
    sizes = (32, 128, 512)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    plotted = False
    ax = axes[0]
    for eng in ENGINE_ORDER:
        xs, ys = [], []
        for n in sizes:
            rec = grouped.get(f"t8_sp_scaling_{n}__t8_performance", {}).get(eng)
            if rec is None:
                continue
            xs.append(n)
            ys.append(rec["metrics"]["warm_call_median_seconds"] * 1000.0)
        if xs:
            st = _style(eng)
            ax.plot(xs, ys, "o-", ms=4, lw=1.2, color=st["color"], label=st["label"])
            plotted = True
    ax.set_xlabel("atoms (α-Na3PS4 supercell)")
    ax.set_ylabel("warm SP median latency (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(list(sizes))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_title("single-point latency scaling (V100)")
    ax = axes[1]
    width = 0.2
    md_sizes: set[int] = set()
    for j, eng in enumerate(ENGINE_ORDER):
        xs, ys = [], []
        for n in sizes:
            rec = grouped.get(f"t8_md_throughput_{n}__t8_performance", {}).get(eng)
            if rec is None:
                continue
            xs.append(n)
            ys.append(rec["metrics"]["steps_per_second"])
            md_sizes.add(n)
        if xs:
            st = _style(eng)
            ax.bar(
                np.array(xs, dtype=float) + (j - 1.5) * width,
                ys,
                width,
                color=st["color"],
            )
            plotted = True
    ax.set_xlabel("atoms")
    ax.set_ylabel("NVE steps/s")
    ax.set_xticks(sorted(md_sizes))
    ax.set_title("MD throughput (VelocityVerlet)")
    if plotted:
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=_style(e)["color"]) for e in ENGINE_ORDER
        ]
        axes[0].legend(handles, [_style(e)["label"] for e in ENGINE_ORDER], fontsize=7)
        fig.suptitle("T8: GPU performance characterization", fontsize=10)
        _footer(fig, ["t8_sp_scaling_* / t8_md_throughput_* records"])
        fig.tight_layout(rect=(0, 0.03, 1, 0.93))
        fig.savefig(out / "fig15_performance.png", dpi=DPI)
        plt.close(fig)
        return "fig15_performance.png"
    plt.close(fig)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", default=".validation-work")
    parser.add_argument("--out", default="validation/science/reports/figures")
    parser.add_argument(
        "--campaign",
        default=None,
        help="restrict figures to records tagged with this campaign id",
    )
    args = parser.parse_args()
    root = Path(args.evidence_root)
    out = Path(args.out)

    bundle = load_evidence(root, campaign=args.campaign)
    for problem in bundle.problems:
        print(f"[figures] PROBLEM: {problem}", file=sys.stderr)
    if bundle.problems:
        print(
            f"[figures] refusing to render: {len(bundle.problems)} evidence "
            "problem(s)",
            file=sys.stderr,
        )
        return 2
    if not bundle.records:
        print(
            f"[figures] refusing to render: no result records under {root}",
            file=sys.stderr,
        )
        return 3
    out.mkdir(parents=True, exist_ok=True)

    tiers = {tier: bundle.tier(tier) for tier in TIER_NAMES}
    t3_dir = root / "t3"
    t4_records = _group_records(tiers["t4"])
    t5_records = _group_records(tiers["t5"])
    t6_records = _group_records(tiers["t6"])
    t7_records = _group_records(tiers["t7"])
    t8_records = _group_records(tiers["t8"])

    generated: dict[str, str] = {}
    skipped: dict[str, str] = {}

    jobs = (
        ("fig01", lambda: fig_omat24_energy(t3_dir, out)),
        ("fig02", lambda: fig_omat24_forces(t3_dir, out)),
        ("fig03", lambda: None),
        ("fig04", lambda: fig_eos(t4_records, out)),
        ("fig05", lambda: fig_elastic(t4_records, out)),
        ("fig06", lambda: fig_phonon(t4_records, out)),
        ("fig07", lambda: fig_vacancy(t4_records, out)),
        ("fig08", lambda: fig_surface(t4_records, out)),
        ("fig09+10", lambda: fig_neb(t5_records, out)),
        ("fig11", lambda: fig_saddle(t5_records, out)),
        ("fig12", lambda: fig_nve_drift(t6_records, out)),
        ("fig13", lambda: fig_nvt_temperature(root / "t6", out)),
        ("fig14", lambda: fig_transport(t7_records, out)),
        ("fig15", lambda: fig_performance(t8_records, out)),
    )
    for name, fn in jobs:
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - report and continue
            skipped[name] = f"generator error: {type(exc).__name__}: {exc}"
            continue
        if result:
            generated[name] = result
        else:
            skipped[name] = "evidence unavailable"

    # fig03 is structurally unavailable: the official OMat24 validation
    # split ships no reference stress labels
    skipped["fig03"] = (
        "OMat24 stress parity not computable: the official validation "
        "split carries no reference stress labels (recorded in the T3 "
        "record metrics.stress)"
    )

    try:
        versions = build_version_block(
            bundle.records, evidence_campaign=bundle.campaign
        )
    except CampaignManifestError as exc:  # pragma: no cover - no manifest here
        print(f"[figures] refusing to render: {exc}", file=sys.stderr)
        return 2
    manifest = {
        "schema": "mlipx.beta-figures/1",
        "evidence_root": str(root),
        "campaign": bundle.campaign,
        "versions": versions.as_dict(),
        "validation_logic_version": bundle.records[0]["_identity"][
            "validation_logic_version"
        ],
        "selection_rule": (
            "newest non-failing profile per engine/case key; when every "
            "profile failed the newest failure is still plotted"
        ),
        "record_sources": {
            tier: _all_profile_labels(tiers[tier]) for tier in TIER_NAMES
        },
        "generated": generated,
        "skipped": skipped,
    }
    (out / "figures_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for name, path in generated.items():
        print(f"[figures] {name} -> {path}")
    for name, why in skipped.items():
        print(f"[figures] {name} SKIPPED: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
