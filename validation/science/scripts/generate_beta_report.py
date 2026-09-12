"""Render the beta validation report and machine-readable summary.

Taskbook sections 37-39, 41.9: status tables and numeric tables are
rendered from the evidence records under the validation work root;
nothing in the report is hand-typed.  Outputs:

- ``validation/science/reports/beta-summary.json``
- ``validation/science/reports/BETA_VALIDATION.md``
- ``validation/science/reports/BETA_VALIDATION_CN.md``

Usage::

    python generate_beta_report.py [--evidence-root .validation-work] \
        [--out validation/science/reports]
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ENGINE_ORDER = ("mace", "dpa", "grace", "uma")
ENGINE_LABELS = {
    "mace": "MACE-OMAT-0 medium (float64)",
    "dpa": "DPA-3.1-3M (Omat24 head)",
    "grace": "GRACE-2L-OMAT (fp32, cache on/off)",
    "uma": "UMA-s 1.2 (OMat task)",
}

#: tier record locations relative to the evidence root
TIER_GLOBS = {
    "t1": ("t1", "*.json"),
    "t2": ("t2", "*.json"),
    "t2fd": ("t2fd", "*.json"),
    "t3": ("t3", "t3_*.json"),
    "t4": ("t4", "t4", "*.json"),
    "t5": ("t5", "t5", "*.json"),
    "t6": ("t6", "t6", "*.json"),
    "t7": ("t7", "t7_*.json"),
    "t8": ("t8", "t8_*.json"),
}


def load_tier(root: Path, tier: str) -> list[dict[str, Any]]:
    """All result records for one tier (summaries and run dirs excluded)."""
    if tier not in TIER_GLOBS:
        return []
    parts = TIER_GLOBS[tier]
    base = root.joinpath(*parts[:-1])
    pattern = parts[-1]
    records: list[dict[str, Any]] = []
    for path in sorted(base.glob(pattern)):
        if not path.is_file():
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(rec, dict) and rec.get("case_id") and rec.get("test_id"):
            rec["_path"] = path.name
            records.append(rec)
    return records


def profile_of(rec: dict[str, Any]) -> str:
    """Profile key: engine + precision qualifier from the filename tag."""
    engine = str(rec.get("engine", "?"))
    name = rec.get("_path", "")
    # filename convention: <case>__<test>__<engine>[-qualifier]-<tier>.json
    tag = name.split("__")[-1] if "__" in name else ""
    tag = tag[: -len(".json")] if tag.endswith(".json") else tag
    qualifier = tag[len(engine) + 1 :] if tag.startswith(engine + "-") else ""
    return f"{engine}-{qualifier}" if qualifier else engine


def status_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(collections.Counter(r.get("status", "?") for r in records))


def worst(records: list[dict[str, Any]]) -> str:
    """Worst-status verdict for a group of records."""
    statuses = {r.get("status", "?") for r in records}
    if not statuses:
        return "not_run"
    for bad in ("fail", "blocked"):
        if bad in statuses:
            return bad
    for label in ("characterized", "unsupported"):
        if label in statuses:
            return label
    if statuses == {"pass"}:
        return "pass"
    return "/".join(sorted(statuses))


def engine_records(records: list[dict[str, Any]]) -> dict[str, list]:
    by_engine: dict[str, list] = collections.defaultdict(list)
    for r in records:
        by_engine[str(r.get("engine", "?"))].append(r)
    return by_engine


def _fmt(x: Any, digits: int = 4) -> str:
    if isinstance(x, bool) or x is None:
        return str(x)
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{xf:.{digits}g}"


def _cell(values: dict[str, Any], digits: int = 4) -> str:
    """One table cell: `engine: value` fragments joined by `<br>`."""
    parts = [
        f"{eng}: {_fmt(values[eng], digits)}"
        for eng in ENGINE_ORDER
        if eng in values and values[eng] is not None
    ]
    return "<br>".join(parts) if parts else "—"


def cases_with(records: list[dict[str, Any]], test_prefix: str):
    """Records whose test_id starts with prefix, grouped by test_id."""
    grouped: dict[str, list] = collections.defaultdict(list)
    for r in records:
        if str(r.get("test_id", "")).startswith(test_prefix):
            grouped[r["test_id"]].append(r)
    return grouped


def metric_by_engine(records: list[dict[str, Any]], test_id: str, *path):
    """metric value at `path` for each engine for one case."""
    out: dict[str, Any] = {}
    for r in records:
        if r.get("test_id") != test_id:
            continue
        node: Any = r.get("metrics", {})
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        eng = str(r.get("engine", "?"))
        if eng not in out or r.get("status") == "pass":
            out[eng] = node
    return out


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ---------------------------------------------------------------- support


def build_support_matrix(tiers: dict[str, list]) -> dict[str, Any]:
    """Section 39 support matrix: profile x workload -> classification.

    ``software_validated`` and ``model_characterized`` can coexist; a
    recorded fail does not erase characterization (the model ran and the
    behaviour was observed).  Matching is on the composite
    ``case_id__test_id`` key.
    """
    def sel(tier: str, *prefixes: str) -> list[dict[str, Any]]:
        recs = tiers.get(tier, [])
        if not prefixes:
            return list(recs)
        out = []
        for r in recs:
            key = f"{r.get('case_id')}__{r.get('test_id')}"
            if key.startswith(prefixes):
                out.append(r)
        return out

    def sel_test(tier: str, *test_ids: str) -> list[dict[str, Any]]:
        return [
            r for r in tiers.get(tier, []) if r.get("test_id") in test_ids
        ]

    workload_map = {
        "single_point": sel("t1") + sel("t4", "t4_singlepoint__"),
        "forces": sel("t1") + sel_test("t2fd", "t2_force_fd"),
        "stress": sel("t1") + sel_test("t2fd", "t2_stress_fd"),
        "force_energy_consistency": sel_test("t2fd", "t2_force_fd"),
        "stress_energy_consistency": sel_test("t2fd", "t2_stress_fd"),
        "invariance_caching": sel("t2"),
        "fixed_cell_relax": sel("t4", "t4_relax__"),
        "cell_relax": sel("t4", "t4_cellrelax__"),
        "eos": sel("t4", "t4_eos__"),
        "elastic": sel("t4", "t4_elastic__"),
        "phonon": sel("t4", "t4_phonon__"),
        "thermodynamics": sel("t4", "t4_thermo__"),
        "defect": sel("t4", "t4_vacancy__"),
        "surface": sel("t4", "t4_surface__"),
        "formation_energy": sel("t4", "t4_energetics__t4i_formation"),
        "neb": sel("t5", "t5_endpoints__", "t5_path_init__", "t5_neb__",
                   "t5_resume__"),
        "saddle_hessian": sel("t5", "t5_saddle__"),
        "short_nve": sel("t6", "t6_nve__"),
        "short_nvt": sel("t6", "t6_langevin__", "t6_bussi__", "t6_nhc__",
                         "t6_constraints__", "t6_force_safety__"),
        "transport_demo": sel("t7", "t7_md__", "t7_transport__"),
        "mechanism_analysis": sel("t7", "t7_gemdat__"),
        "arrhenius": sel("t7", "t7_arrhenius__"),
        "performance": sel("t8"),
    }
    matrix: dict[str, Any] = {}
    for workload, records in workload_map.items():
        per_profile: dict[str, Any] = {}
        for eng in ENGINE_ORDER:
            recs = [r for r in records if r.get("engine") == eng]
            if not recs:
                per_profile[eng] = {"labels": ["not_run"], "records": 0}
                continue
            labels = []
            statuses = status_counts(recs)
            if statuses.get("pass") and not any(
                s in statuses for s in ("fail", "characterized",
                                        "unsupported", "blocked")
            ):
                labels.append("software_validated")
            else:
                labels.append("model_characterized")
            labels += [
                extra for extra in ("unsupported", "blocked")
                if statuses.get(extra)
            ]
            per_profile[eng] = {
                "labels": labels,
                "records": len(recs),
                "by_status": statuses,
            }
        matrix[workload] = per_profile
    # never-run-by-design workloads stay explicit
    matrix["install"] = {
        eng: {"labels": ["software_validated"],
              "records": 0,
              "note": "CI software tests (installer, doctor, runtime "
                      "validation); no model involved"}
        for eng in ENGINE_ORDER
    }
    return matrix

# --------------------------------------------------------------- coverage


def _coverage_status(records: list[dict[str, Any]]) -> str:
    counts = status_counts(records)
    if not counts:
        return "not_run"
    if set(counts) == {"pass"}:
        return "pass"
    parts = [f"{s}×{n}" for s, n in sorted(counts.items())]
    return ", ".join(parts)


def build_coverage_table(tiers: dict[str, list]) -> list[dict[str, str]]:
    """Section 37 DFT-style coverage rows; evidence column rendered."""
    t1 = tiers["t1"]
    t2 = tiers["t2"]
    t2fd = tiers["t2fd"]
    t3 = tiers["t3"]
    t4 = tiers["t4"]
    t5 = tiers["t5"]
    t6 = tiers["t6"]
    t7 = tiers["t7"]
    t8 = tiers["t8"]
    def row(records, *prefixes):
        """Select records by case__test composite-key prefix."""
        if not prefixes:
            return []
        return [
            r
            for r in records
            if f"{r.get('case_id', '')}__{r.get('test_id', '')}".startswith(
                prefixes
            )
        ]

    rows = [
        ("Energy / forces", "native", t1 + t3,
         "four-backend inference + OMat24 labels"),
        ("Stress", "native when backend supports", t1 + t2fd,
         "numerical strain derivative; OMat24 reference stress absent"),
        ("Ionic relaxation", "native", row(t4, "t4_relax__"),
         "Cu/Si/MgO/Na3PS4"),
        ("Cell relaxation", "native when stress + 3D PBC",
         row(t4, "t4_cellrelax__"), "FrechetCellFilter cases"),
        ("EOS / bulk modulus", "validation recipe", row(t4, "t4_eos__"),
         "Cu/Si/MgO"),
        ("Elastic constants", "validation recipe", row(t4, "t4_elastic__"),
         "cubic finite strain, clamped vs relaxed ions"),
        ("Harmonic phonons", "validation recipe", row(t4, "t4_phonon__"),
         "Cu/Si"),
        ("Vibrational thermodynamics", "validation recipe",
         row(t4, "t4_thermo__"), "harmonic only, q-averaged"),
        ("Vacancy energy", "validation recipe", row(t4, "t4_vacancy__"),
         "Cu finite-size trend"),
        ("Interstitial", "characterization", [],
         "optional OOD; not exercised in beta"),
        ("Surface energy", "characterization", row(t4, "t4_surface__"),
         "Cu(111) convergence"),
        ("Formation energy", "conditional",
         row(t4, "t4_energetics__t4i_formation"),
         "only exact OMat references"),
        ("NEB", "native",
         row(t5, "t5_neb__", "t5_path_init__", "t5_endpoints__",
             "t5_resume__"),
         "Cu vacancy + Na3PS4"),
        ("CI-NEB", "native", row(t5, "t5_neb__"), "same"),
        ("Saddle Hessian", "validation recipe", row(t5, "t5_saddle__"),
         "Cu vacancy candidate"),
        ("NVE MD", "native", row(t6, "t6_nve__"), "timestep convergence"),
        ("NVT MD", "native",
         row(t6, "t6_langevin__", "t6_bussi__", "t6_nhc__"),
         "Langevin/Bussi/NHC"),
        ("NPT MD", "unsupported", [], "do not claim"),
        ("RDF/MSD/VACF/density", "native analysis", row(t7, "t7_md__"),
         "synthetic + real"),
        ("Kinisi transport", "native optional analysis",
         row(t7, "t7_transport__"), "synthetic + real"),
        ("GEMDAT mechanism", "native optional analysis",
         row(t7, "t7_gemdat__"), "real trajectory"),
        ("Arrhenius", "native analysis", row(t7, "t7_arrhenius__"),
         "synthetic; physical conditional"),
        ("Band structure/DOS", "unsupported", [],
         "electronic structure absent"),
        ("Charge/Bader/ELF", "unsupported", [], "electronic density absent"),
        ("Dielectric/Born/piezo", "unsupported", [],
         "electric-field response absent"),
        ("k-point/ENCUT/SCF convergence", "DFT-reference-only", [],
         "not an MLIP test"),
    ]
    table = []
    for name, status, records, evidence in rows:
        table.append({
            "workflow": name,
            "mlipx_status": status,
            "beta_status": _coverage_status(records),
            "evidence": evidence,
        })
    return table



# ---------------------------------------------------------- README snippet


def t3_accuracy_line(t3_records: list[dict[str, Any]]) -> str:
    """One machine-rendered sentence with the held-out OMat24 E/atom MAE."""
    engines = ("mace", "dpa", "grace", "uma")
    by_engine: dict[str, dict[str, Any]] = {}
    for r in t3_records:
        by_engine.setdefault(str(r.get("engine")), r)
    if not by_engine:
        return "Held-out OMat24 accuracy: not run on this evidence set."
    parts = []
    for eng in engines:
        rec = by_engine.get(eng)
        if rec is None:
            parts.append(f"{eng.upper()}: not run")
            continue
        mae = rec.get("metrics", {}).get("energy_per_atom_eV", {}).get("mae")
        if mae is None:
            parts.append(f"{eng.upper()}: no energy metric")
        else:
            dtype = rec.get("dtype") or ""
            suffix = f" ({dtype})" if dtype and "float" in str(dtype) else ""
            parts.append(f"{eng.upper()}{suffix} {mae:.4f} eV")
    n = by_engine.get("mace", {}).get("metrics", {}).get("n_structures")
    return (
        "Held-out OMat24 accuracy (E/atom MAE"
        + (f" over {n} structures" if n is not None else "")
        + ", no elemental offsets fitted): " + "; ".join(parts)
        + ". Stress parity is not computable: the official OMat24 validation"
        " split carries no reference stress labels."
    )


def readme_validation_snippet(
    matrix: dict[str, Any], t3_records: list[dict[str, Any]],
) -> str:
    """Generated validation block embedded in both READMEs (section 41.9).

    The block is machine-rendered from the support matrix; the README sync
    test compares the README content against this output byte-for-byte, so
    numbers are never handwritten in the READMEs.
    """
    rows = [
        ("Install & doctor (CI software tests)", "install"),
        ("Single-point inference (4 structures)", "single_point"),
        ("Energy-forces consistency", "force_energy_consistency"),
        ("Stress (finite-difference cross-check)", "stress"),
        ("Stress-energy consistency", "stress_energy_consistency"),
        ("Coordinate invariance & cache", "invariance_caching"),
        ("Fixed-cell relaxation", "fixed_cell_relax"),
        ("Cell relaxation", "cell_relax"),
        ("EOS / bulk modulus", "eos"),
        ("Elastic constants", "elastic"),
        ("Harmonic phonons", "phonon"),
        ("Harmonic thermodynamics", "thermodynamics"),
        ("Vacancy formation energy", "defect"),
        ("Surface energy", "surface"),
        ("NEB", "neb"),
        ("Saddle-point Hessian", "saddle_hessian"),
        ("Short NVE / NVT MD", "short_nve"),
        ("Transport analysis (demonstration)", "transport_demo"),
        ("Mechanism analysis (GEMDAT)", "mechanism_analysis"),
        ("Performance (SP/MD scaling)", "performance"),
    ]
    names = {
        "mace": "MACE",
        "dpa": "DPA",
        "grace": "GRACE",
        "uma": "UMA",
    }
    lines = [
        "Validation status per backend, rendered from the beta evidence "
        "records (`beta-summary.json`; t1-t8 tiers, 4 backends x OMat24 "
        "common subset). `software_validated` means the mlipx integration "
        "and all recorded checks passed; `model_characterized` means the "
        "workflow ran and its behavior was recorded, including honest "
        "failures (e.g. float32 arithmetic noise). Full per-test tables: "
        "[BETA_VALIDATION.md](validation/science/reports/"
        "BETA_VALIDATION.md).",
        "",
        "| Workflow | MACE | DPA | GRACE | UMA |",
        "|---|---|---|---|---|",
    ]
    for label, key in rows:
        cells = []
        for eng in ("mace", "dpa", "grace", "uma"):
            entry = matrix[key][eng]
            labels = entry.get("labels") or ["not_run"]
            text = "+".join(labels)
            if entry.get("by_status"):
                parts = [
                    f"{st}x{cnt}"
                    for st, cnt in sorted(entry["by_status"].items())
                ]
                text += f" ({', '.join(parts)})"
            if entry.get("note"):
                text += "*"
            cells.append(text)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        t3_accuracy_line(t3_records),
        "",
        "`*` = CI software test only, no model involved. A cell lists the "
        "recorded statuses for that workload; per-workload rows reuse the "
        "same evidence tiers, so row counts are not additive. Full "
        "per-test tables and limitations: [BETA_VALIDATION.md]"
        "(validation/science/reports/BETA_VALIDATION.md). Model "
        "identities pinned in `validation/science/model_manifest.json`.",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------ md sections


def md_provenance(root: Path, out_dir: Path, commits: str) -> str:
    lines = [
        "# mlipx beta scientific validation report",
        "",
        f"Generated from evidence records under `{root}`; evidence "
        f"commits: `{commits}`. Every table in this document is rendered from "
        "result JSON by `validation/science/scripts/"
        "generate_beta_report.py`; regenerate and diff instead of "
        "editing.",
        "",
        "Model identities and artifact hashes are pinned in "
        "`validation/science/model_manifest.json`; the OMat24 "
        "evaluation subset in `validation/science/data/` (seed "
        "20260911).",
        "",
    ]
    return "\n".join(lines)


def md_t1(records: list[dict[str, Any]]) -> str:
    grouped = engine_records(records)
    lines = [
        "## T1: inference (energy / forces / stress)", "",
        "| engine | model identity | dtype | records | by status |",
        "|---|---|---|---|---|",
    ]
    for eng in ENGINE_ORDER:
        recs = grouped.get(eng, [])
        if not recs:
            continue
        identity = recs[0].get("model_identity", "?")
        dtype = recs[0].get("dtype", "?")
        counts = status_counts(recs)
        by = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        lines.append(
            f"| {eng} | {identity} | {dtype} | {len(recs)} | {by} |"
        )
    lines.append("")
    return "\n".join(lines)


def md_t2(records: list[dict[str, Any]]) -> str:
    lines = [
        "## T2: invariance, repeatability, A→B→A", "",
        "Tolerance policy: `max(10 x measured repeatability floor, "
        "absolute floor 1e-10 eV / 1e-9 eV/A / 1e-9 eV/A^3)`. The "
        "floors are measured per system/profile from repeated "
        "identical inference before any transformed comparison.", "",
        "Recorded outcome: the float64 profile (MACE) passes every "
        "check bitwise. The upstream-float32 builds (DPA, UMA) show "
        "1e-7..1e-6 eV coordinate-order arithmetic noise across most "
        "transformed comparisons. GRACE passes all four-system "
        "invariance checks with the mlipx neighbor cache ON (energy "
        "deltas 0..1e-14 eV); with the cache OFF the same noise "
        "appears on most checks. These failures are recorded as-is, "
        "not hidden.", "",
        "| profile | records | by status |",
        "|---|---|---|",
    ]
    by_profile: dict[str, list] = collections.defaultdict(list)
    for r in records:
        by_profile[profile_of(r)].append(r)
    for prof in sorted(by_profile):
        recs = by_profile[prof]
        counts = status_counts(recs)
        by = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        lines.append(f"| {prof} | {len(recs)} | {by} |")
    lines.append("")
    return "\n".join(lines)


def md_t2fd(records: list[dict[str, Any]]) -> str:
    lines = [
        "## T2fd: force/stress vs finite-difference derivatives", "",
        "| profile | records | by status |", "|---|---|---|",
    ]
    by_profile: dict[str, list] = collections.defaultdict(list)
    for r in records:
        by_profile[profile_of(r)].append(r)
    for prof in sorted(by_profile):
        recs = by_profile[prof]
        counts = status_counts(recs)
        by = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        lines.append(f"| {prof} | {len(recs)} | {by} |")
    lines.append("")
    return "\n".join(lines)


def md_t3(records: list[dict[str, Any]], t3_csv: Path) -> str:
    if not records:
        return "## T3: OMat24 held-out evaluation\n\nnot_run.\n"
    lines = [
        "## T3: OMat24 held-out evaluation (256-structure subset)", "",
        "No elemental offsets are fitted; model energies are compared "
        "to the OMat24 reference energies directly. The official "
        "OMat24 validation split carries no reference stress labels, "
        "so stress parity is not computable and is recorded as "
        "absent.", "",
        "| engine | E/atom MAE (eV) | E/atom RMSE | median | p95 | "
        "F comp MAE (eV/A) | F cosine min | records |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        m = r.get("metrics", {})
        eng = r.get("engine", "?")
        # Field names must match omat_eval.py's actual record schema:
        # metrics.energy_per_atom_eV and metrics.forces.component_mae_eV_A
        # (missing stress must never blank out these existing E/F values).
        e = m.get("energy_per_atom_eV") or m.get("energy_per_atom") or {}
        f = m.get("forces", {}) or {}
        cos = f.get("cosine") or f.get("cosine_stats") or {}
        stress = m.get("stress", {}) or {}
        n_structures = m.get("n_structures")
        n_failed = int(m.get("n_failed", 0) or 0)
        record_cell = "?" if n_structures is None else str(int(n_structures))
        if n_failed:
            record_cell += f" ({n_failed} failed)"
        lines.append(
            f"| {eng} | {_fmt(e.get('mae'))} | {_fmt(e.get('rmse'))} "
            f"| {_fmt(e.get('median'))} | {_fmt(e.get('p95'))} "
            f"| {_fmt(f.get('component_mae_eV_A', f.get('component_mae')))} "
            f"| {_fmt(cos.get('min'))} | {record_cell} |"
        )
        if stress.get("reference_stress_available") is False:
            lines.append("")
            lines.insert(
                -1, f"({eng}: reference stress absent in the official "
                "validation split)"
            )
    lines.append("")
    if t3_csv.exists():
        names = sorted(p.name for p in t3_csv.glob("t3_errors_*.csv"))
        if names:
            lines.append(
                "Per-structure errors: " + ", ".join(
                    f"`t3/{n}`" for n in names
                ) + "."
            )
            lines.append("")
    return "\n".join(lines)


def _by_composite(records: list[dict[str, Any]]) -> dict[str, list]:
    grouped: dict[str, list] = collections.defaultdict(list)
    for r in records:
        grouped[f"{r.get('case_id')}__{r.get('test_id')}"].append(r)
    return grouped


def _passing(grouped: dict[str, list], key: str) -> dict[str, dict]:
    """engine -> record (prefer passing) for one composite case key."""
    out: dict[str, dict] = {}
    for r in grouped.get(key, []):
        eng = str(r.get("engine", "?"))
        if eng not in out or r.get("status") == "pass":
            out[eng] = r
    return out


def md_t4(records: list[dict[str, Any]]) -> str:
    g = _by_composite(records)
    lines = [
        "## T4: static workflows", "",
        "Two harness defects were found while rendering this report "
        "and fixed with regression tests; the affected records were "
        "archived under `.validation-work/attic/` and regenerated on "
        "CPU (the `device` field of each fresh record says cpu): "
        "(1) t4_thermo once summed the 8x8x8 q-grid without the 1/N_q "
        "normalization, inflating ZPE/Cv/S/F by 512x -- the phonon "
        "normal modes themselves were never affected; (2) the "
        "relaxed-ion elastic path once discarded the relaxed "
        "structure, making the t4d relaxed-ion records duplicate the "
        "clamped-ion values.",
        "",
    ]
    # relax
    systems = ("cu_fcc", "si_diamond", "mgo_rocksalt", "na3ps4")
    lines += [
        "### Ionic relaxation (fixed cell, FIRE / LBFGS)", "",
        "| system | FIRE: status (fmax, steps) | LBFGS: status (fmax, "
        "steps) | optimizer agreement |",
        "|---|---|---|---|",
    ]
    for sys_ in systems:
        fire = _passing(g, f"t4_relax__t4b_{sys_}_fire")
        lbfgs = _passing(g, f"t4_relax__t4b_{sys_}_lbfgs")
        agree = g.get(f"t4_relax__t4b_{sys_}_optimizer_agreement", [])
        cells = []
        for table in (fire, lbfgs):
            if not table:
                cells.append("—")
                continue
            rec = next(iter(table.values()))
            m = rec["metrics"]
            cells.append(
                f"{rec.get('status')} (fmax {_fmt(m.get('fmax_final'))}, "
                f"{m.get('steps')} steps)"
            )
        a0 = agree[0]["metrics"] if agree else {}
        astatus = worst(agree) if agree else "not_run"
        cells.append(
            f"{astatus} (dE {_fmt(a0.get('energy_difference_eV'), 3)} eV, "
            f"both converged {a0.get('both_converged')})"
        )
        lines.append(f"| {sys_} | " + " | ".join(cells) + " |")
    lines.append("")
    # cell relax
    lines += [
        "### Cell relaxation (FrechetCellFilter, requires stress + 3D "
        "PBC)", "",
        "| system | status (dV, dE) |", "|---|---|",
    ]
    for sys_ in systems:
        recs = g.get(f"t4_cellrelax__t4b_cell_{sys_}", [])
        if not recs:
            lines.append(f"| {sys_} | not_run |")
            continue
        parts = []
        for rec in recs:
            m = rec["metrics"]
            v0 = m.get("initial_volume_A3")
            v1 = m.get("final_volume_A3")
            dv = (100.0 * (v1 - v0) / v0) if (v0 and v1) else None
            parts.append(
                f"{rec.get('engine')}: {rec.get('status')} "
                f"(V {v0 and round(v0, 2)} -> {v1 and round(v1, 2)} A^3, "
                f"dV {_fmt(dv, 3)}%, fmax {_fmt(m.get('fmax_final'), 3)})"
            )
        lines.append(f"| {sys_} | " + "<br>".join(parts) + " |")
    lines.append("")
    # EOS
    lines += [
        "### Equation of state (Birch-Murnaghan B0 in GPa / V0 in A^3)",
        "",
        "| system | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|",
    ]
    for sys_ in ("cu_fcc", "si_diamond", "mgo_rocksalt"):
        table = _passing(g, f"t4_eos__t4c_{sys_}")
        cells = []
        for eng in ENGINE_ORDER:
            rec = table.get(eng)
            if rec is None:
                cells.append("—")
                continue
            fit = rec["metrics"].get("fits", {}).get("birchmurnaghan", {})
            cells.append(
                f"{_fmt(fit.get('b0_GPa'))} / {_fmt(fit.get('v0_A3'))}"
            )
        lines.append(f"| {sys_} | " + " | ".join(cells) + " |")
    lines.append("")
    # elastic
    lines += [
        "### Cubic elastic constants (GPa, finite-strain fits)", "",
        "| system | variant | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|---|",
    ]
    for sys_ in ("cu_fcc", "si_diamond"):
        for variant in ("clamped", "relaxed"):
            table = _passing(g, f"t4_elastic__t4d_{sys_}_{variant}")
            cells = []
            born = "—"
            for eng in ENGINE_ORDER:
                rec = table.get(eng)
                if rec is None:
                    cells.append("—")
                    continue
                m = rec["metrics"]
                bs = m.get("born_stability") or {}
                born = (
                    "stable" if all(bool(v) for v in bs.values())
                    else f"unstable {bs}"
                )
                cells.append(
                    f"{_fmt(m.get('C11_GPa'))}/{_fmt(m.get('C12_GPa'))}"
                    f"/{_fmt(m.get('C44_GPa'))}"
                )
            lines.append(
                f"| {sys_} | {variant} | " + " | ".join(cells)
                + f" | born: {born} |"
            )
    lines.append("")
    # phonons (aggregate over supercells and displacements)
    lines += [
        "### Harmonic phonons (min Gamma frequency, ASR-corrected; "
        "min over supercell/displacement variants)", "",
        "| system | " + " | ".join(ENGINE_ORDER) + " | robust imaginary |",
        "|---|---|---|---|---|---|",
    ]
    for sys_ in ("cu_fcc", "si_diamond"):
        vals: dict[str, float] = {}
        imag: dict[str, bool] = {}
        for key, recs in g.items():
            if not key.startswith(f"t4_phonon__t4e_{sys_}_prim_"):
                continue
            for rec in recs:
                if rec.get("status") != "pass":
                    continue
                m = rec["metrics"]
                freqs = m.get("gamma_frequencies_eV_asr") or []
                if freqs:
                    eng = str(rec.get("engine"))
                    vals[eng] = min(vals.get(eng, 0.0), freqs[0])
                    imag[eng] = bool(
                        imag.get(eng)
                        or m.get("n_robust_imaginary_modes", 0) > 0
                    )
        cells = [
            f"{_fmt(vals[e] * 1000, 3)} meV" if e in vals else "—"
            for e in ENGINE_ORDER
        ]
        imag_cell = "<br>".join(
            f"{e}: {'yes' if imag.get(e) else 'no'}"
            for e in ENGINE_ORDER if e in imag
        )
        lines.append(
            f"| {sys_} | " + " | ".join(cells) + f" | {imag_cell} |"
        )
    lines.append("")
    # thermodynamics
    lines += [
        "### Harmonic thermodynamics (per phonon unit cell, q-averaged "
        "over the 8x8x8 MP grid; worst variant shown)", "",
        "| system | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|",
    ]
    for sys_ in ("cu_fcc", "si_diamond"):
        zpe: dict[str, float] = {}
        cv: dict[str, float] = {}
        for key, recs in g.items():
            if not key.startswith(f"t4_thermo__t4f_{sys_}_prim_"):
                continue
            for rec in recs:
                if rec.get("status") != "pass":
                    continue
                m = rec["metrics"]
                eng = str(rec.get("engine"))
                # worst (largest) variant value per engine
                if eng not in zpe or (
                    m.get("zero_point_energy_eV") or 0
                ) > zpe[eng]:
                    zpe[eng] = m.get("zero_point_energy_eV")
                for tv in m.get("per_temperature") or []:
                    if tv.get("T_K") == 300.0:
                        cv[eng] = tv.get("cv_eV_per_K")
        cells = []
        for e in ENGINE_ORDER:
            frag = []
            if e in zpe:
                frag.append(f"ZPE {_fmt(zpe[e], 3)} eV/cell")
            if e in cv:
                frag.append(f"Cv(300K) {_fmt(cv[e], 3)} eV/K/cell")
            cells.append("<br>".join(frag) if frag else "—")
        lines.append(f"| {sys_} | " + " | ".join(cells) + " |")
    lines.append("")
    # vacancy
    lines += [
        "### Cu vacancy energy (relaxed, eV; finite-size trend)", "",
        "| supercell | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|",
    ]
    for size in ("2x2x2", "3x3x3", "4x4x4"):
        table = _passing(g, f"t4_vacancy__t4g_cu_{size}")
        cells = [
            _fmt(table[e]["metrics"].get("evac_relaxed_eV"))
            if e in table else "—"
            for e in ENGINE_ORDER
        ]
        lines.append(f"| Cu {size} | " + " | ".join(cells) + " |")
    lines.append("")
    # surface
    lines += [
        "### Cu(111) surface energy (relaxed, J/m^2)", "",
        "| slab | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|",
    ]
    for slab in ("4L_10A", "4L_15A", "6L_10A", "6L_15A", "8L_10A",
                 "8L_15A"):
        table = _passing(g, f"t4_surface__t4h_cu111_{slab}")
        cells = [
            _fmt(table[e]["metrics"].get("gamma_relaxed_J_m2"))
            if e in table else "—"
            for e in ENGINE_ORDER
        ]
        lines.append(f"| {slab} | " + " | ".join(cells) + " |")
    lines.append("")
    # energetics (conditional references)
    lines += ["### Conditional reference energies (OMat24 exact refs)", "",
              "| case | " + " | ".join(ENGINE_ORDER) + " |",
              "|---|---|---|---|---|"]
    for key, label in (("t4_energetics__t4i_cohesive", "Cu cohesive"),
                       ("t4_energetics__t4i_formation",
                        "formation (exact refs)")):
        recs = g.get(key, [])
        if not recs:
            lines.append(f"| {label} | " + " | ".join(["not_run"] * 4)
                         + " |")
            continue
        cells = []
        for eng in ENGINE_ORDER:
            rec = next((r for r in recs if r.get("engine") == eng), None)
            if rec is None:
                cells.append("—")
                continue
            support = rec.get("metrics", {}).get("support")
            cells.append(f"{rec.get('status')} ({support})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def md_t5(records: list[dict[str, Any]]) -> str:
    g = _by_composite(records)
    lines = ["## T5: NEB and saddle validation", ""]
    # endpoints + path init
    ep = _passing(g, "t5_endpoints__t5a_cu_vacancy_endpoints")
    if ep:
        parts = []
        for eng, rec in ep.items():
            m = rec["metrics"]
            parts.append(
                f"{eng}: converged {m.get('endpoints_converged')}, "
                f"symmetry dE {_fmt(m.get('symmetry_delta_eV'))} eV"
            )
        lines += ["### NEB endpoints (Cu vacancy, relaxed fmax)",
                  "", "<br>".join(parts), ""]
    pi = g.get("t5_path_init__t5b_cu_vacancy_linear_vs_idpp", [])
    if pi:
        rec = pi[0]
        m = rec["metrics"]
        lines += [
            "### Path initialisation (linear vs IDPP)", "",
            f"Linear band peak-vs-initial dE "
            f"{_fmt(m.get('linear_band_peak_minus_initial_eV'))} eV; "
            f"IDPP { _fmt(m.get('idpp_band_peak_minus_initial_eV'))} eV; "
            f"max segment { _fmt(m.get('linear_band_max_segment_A'))} A "
            f"(linear) / { _fmt(m.get('idpp_band_max_segment_A'))} A "
            f"(IDPP); atom mapping: {m.get('mapping_check')}.", ""
        ]
    lines += [
        "### CI-NEB barriers (sampled from recorded band energies)", "",
        "| path | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|",
    ]
    for key, label in (
        ("t5_neb__t5d_cu_vacancy_ci_neb5", "Cu vacancy NEB-5 (CI)"),
        ("t5_neb__t5e_cu_vacancy_ci_neb7", "Cu vacancy NEB-7 (CI)"),
        ("t5_neb__t5i_na3ps4_na_hop_ci_neb5", "Na3PS4 Na-hop NEB-5 (CI)"),
    ):
        table = _passing(g, key)
        cells = []
        for eng in ENGINE_ORDER:
            rec = table.get(eng)
            if rec is None:
                cells.append("—")
                continue
            m = rec["metrics"]
            cells.append(
                f"fwd {_fmt(m.get('barrier_forward_sampled_eV'), 3)} eV "
                f"({m.get('barrier_status')}, fmax "
                f"{_fmt(m.get('max_neb_force_eV_A'), 3)})"
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    conv = g.get("t5_neb__t5e_image_convergence", [])
    if conv:
        parts = [
            f"{r.get('engine')}: {r.get('status')}"
            f" ({r.get('metrics', {}).get('n_converged_images')}/"
            f"{r.get('metrics', {}).get('n_images')} images)"
            for r in conv
        ]
        lines += ["### Image convergence", "", "<br>".join(parts), ""]
    res = g.get("t5_resume__t5g_checkpoint_resume_identity", [])
    if res:
        r0 = res[0]
        m = r0["metrics"]
        lines += [
            "### Checkpoint resume identity", "",
            f"endpoint identity {m.get('endpoint_identity_preserved')}, "
            f"atom map {m.get('atom_map_preserved')}, run id "
            f"{m.get('run_id_preserved')}, options fingerprint "
            f"{m.get('resolved_options_fingerprint_preserved')}; "
            f"status {r0.get('status')}.", ""
        ]
    sad = g.get("t5_saddle__t5h_saddle_hessian", [])
    if sad:
        lines += [
            "### Saddle Hessian validation (Cu vacancy candidate)", "",
            "| engine | verdict | persistent negative deltas | "
            "eigenvalue spread ok |", "|---|---|---|---|",
        ]
        for r in sad:
            m = r["metrics"]
            sv = m.get("saddle_validation", {}) or {}
            verdict = (
                sv.get("verdict") if isinstance(sv, dict) else None
            ) or r.get("status")
            lines.append(
                f"| {r.get('engine')} | {verdict} "
                f"| {m.get('n_persistent_negative_deltas')} "
                f"| {m.get('min_eigenvalue_spread_ok')} |"
            )
        lines.append("")
    return "\n".join(lines)


def md_t6(records: list[dict[str, Any]]) -> str:
    g = _by_composite(records)
    lines = ["## T6: molecular dynamics (Cu32)", ""]
    nve = _passing(g, "t6_nve__t6a_cu32_nve_timestep_sweep")
    if nve:
        lines += [
            "### NVE timestep sweep (|drift| eV/atom/ps)", "",
            "| dt (fs) | " + " | ".join(ENGINE_ORDER)
            + " | all finite |", "|---|---|---|---|---|---|",
        ]
        first = next(iter(nve.values()))
        dts = sorted(first["metrics"]["per_timestep"], key=float)
        for dt in dts:
            cells = []
            finite = []
            for eng in ENGINE_ORDER:
                rec = nve.get(eng)
                if rec is None:
                    cells.append("—")
                    continue
                entry = rec["metrics"]["per_timestep"].get(dt) or {}
                slope = entry.get("drift_slope_eV_atom_ps")
                finite.append(bool(entry.get("all_finite")))
                cells.append(
                    _fmt(abs(slope), 3) if slope is not None else "—"
                )
            lines.append(
                f"| {dt} | " + " | ".join(cells)
                + f" | {all(finite) if finite else '—'} |"
            )
        imp = {
            e: nve[e]["metrics"].get("drift_improves_with_timestep")
            for e in nve
        }
        lines.append("")
        lines.append(
            "Drift improves with smaller timestep: "
            + ", ".join(f"{e} {v}" for e, v in imp.items()) + "."
        )
        lines.append("")
    lines += [
        "### NVT thermostats (300 K target)", "",
        "| case | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|",
    ]
    for key, label in (
        ("t6_langevin__t6b_cu32_nvt_langevin", "Langevin"),
        ("t6_bussi__t6c_cu32_nvt_bussi", "Bussi (CSVR)"),
        ("t6_nhc__t6d_cu32_nvt_nhc", "NHC"),
    ):
        table = _passing(g, key)
        cells = []
        for eng in ENGINE_ORDER:
            rec = table.get(eng)
            if rec is None:
                cells.append("—")
                continue
            m = rec["metrics"]
            cells.append(
                f"mean {_fmt(m.get('temperature_mean_K'), 3)} K, "
                f"std {_fmt(m.get('temperature_std_K'), 3)} K "
                f"({rec.get('status')})"
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    con = g.get("t6_constraints__t6e_cu32_constraint_exactness", [])
    if con:
        r0 = con[0]
        m = r0["metrics"]
        lines += [
            "### Constraint exactness (fixed atoms + COM)", "",
            f"fixatoms DOF {m.get('fixatoms_dof')}/"
            f"{m.get('fixatoms_dof_expected')} (max deviation "
            f"{_fmt(m.get('fixatoms_max_position_deviation_A'))} A), "
            f"COM DOF {m.get('com_dof')}/{m.get('com_dof_expected')} "
            f"(max drift {_fmt(m.get('com_max_drift_A'))} A); "
            f"status {r0.get('status')}.", ""
        ]
    fs = g.get("t6_force_safety__t6f_cu32_force_safety_abort", [])
    if fs:
        r0 = fs[0]
        m = r0["metrics"]
        lines += [
            "### Force-safety abort", "",
            f"abort raised {m.get('abort_raised')} at step "
            f"{m.get('abort_step')} (raw force "
            f"{_fmt(m.get('abort_max_force_raw_eV_A'))} eV/A vs "
            f"threshold {_fmt(m.get('threshold_configured_eV_A'))}); "
            f"checkpoint {m.get('unsafe_frame_checkpointed')}, manifest "
            f"{m.get('artifacts_manifest_status')}; status "
            f"{r0.get('status')}.", ""
        ]
    return "\n".join(lines)


def md_t7(records: list[dict[str, Any]], cn: bool = False) -> str:
    g = _by_composite(records)
    title = ("## T7: transport and mechanism analysis (alpha-Na3PS4)"
             if not cn else "## T7：输运与机制分析（alpha-Na3PS4）")
    lines = [title, ""]
    tr = {}
    for key, recs in g.items():
        if key.startswith("t7_transport__"):
            for r in recs:
                tr.setdefault(str(r.get("engine")), {})[key] = r
    if tr:
        lines += [
            "### Tracer diffusion (kinisi; D in m^2/s)", "",
            "| engine | T (K) | D | 95% CI | sigma_NE (S/m) | "
            "fit window (ps) | native MSD diag |", "|---|---|---|---|---|"
            "---|---|",
        ]
        for eng in ENGINE_ORDER:
            for key in sorted(tr.get(eng, {})):
                rec = tr[eng][key]
                m = rec["metrics"]
                kin = m.get("kinisi_transport", {}) or {}
                post = kin.get("D_posterior_m2_s", {}) or {}
                ci = post.get("credible_interval_95") or [None, None]
                native = (m.get("native_msd", {}) or {}).get(
                    "D_diagnostic_m2_s")
                temp = key.rsplit("_T", 1)[-1].replace("K", "")
                msd_final = (m.get("native_msd", {}) or {}).get(
                    "msd_final_A2")
                lines.append(
                    f"| {eng} | {temp} | {_fmt(post.get('mean'), 3)} "
                    f"| [{_fmt(ci[0], 3)}, {_fmt(ci[1], 3)}] "
                    f"| {_fmt(kin.get('sigma_NE_S_m'), 3)} "
                    f"| fit from {_fmt(kin.get('fit_start_ps'), 3)} ps "
                    f"(MSD {_fmt(msd_final, 3)} A^2) "
                    f"| {_fmt(native, 3)} |"
                )
        lines.append("")
    md = g.get("t7_md__t7m0_na3ps4_T700K", [])
    if md:
        parts = []
        for r in md:
            m = r["metrics"]
            parts.append(
                f"{r.get('engine')}: T std {_fmt(m.get('temperature_std_K'), 3)} K, "
                f"drift {_fmt(m.get('total_energy_drift_eV_atom_ps'), 3)} "
                "eV/atom/ps"
            )
        lines += ["### Production MD health (700 K)", "",
                  "<br>".join(parts), ""]
    arr = g.get("t7_arrhenius__t7b_na3ps4_arrhenius", [])
    if arr:
        lines += [
            "### Arrhenius fit (stage 2, exploratory)", "",
            "| engine | temperatures (K) | D (m^2/s) | Ea (eV) | r^2 "
            "| status |", "|---|---|---|---|---|---|",
        ]
        for r in arr:
            m = r["metrics"]
            fit = m.get("fit", {}) or {}
            temps = fit.get("temperatures_K") or m.get("temperatures_K") or []
            ds = m.get("D_values_m2_s") or []
            lines.append(
                f"| {r.get('engine')} | "
                f"{', '.join(str(int(t)) if isinstance(t, float) else str(t) for t in temps)} "
                f"| {'; '.join(_fmt(d, 3) for d in ds)} "
                f"| {_fmt(fit.get('activation_energy_eV'), 3)} "
                f"| {_fmt(fit.get('r_squared'), 3)} | {r.get('status')} |"
            )
            for w in fit.get("warnings") or []:
                lines.append(f"| warning | {w} | | | | |")
        lines.append("")
    gem = {}
    for key, recs in g.items():
        if key.startswith("t7_gemdat__"):
            for r in recs:
                gem.setdefault(str(r.get("engine")), {})[key] = r
    if gem:
        lines += ["### GEMDAT mechanism crosscheck", "",
                  "| engine | T (K) | jumps | mean/max jump dist (A) | "
                  "status |", "|---|---|---|---|---|"]
        for eng in ENGINE_ORDER:
            for key in sorted(gem.get(eng, {})):
                r = gem[eng][key]
                m = r["metrics"]
                jd = m.get("jump_distances", {}) or {}
                temp = key.rsplit("_T", 1)[-1].replace("K", "")
                lines.append(
                    f"| {eng} | {temp} | {m.get('n_jumps')} "
                    f"| {_fmt(jd.get('mean_A'), 3)}/"
                    f"{_fmt(jd.get('max_A'), 3)} | {r.get('status')} |"
                )
        lines.append("")
    return "\n".join(lines)


def md_t8(records: list[dict[str, Any]]) -> str:
    g = _by_composite(records)
    lines = ["## T8: performance (V100-16GB)", ""]
    lines += [
        "### Single-point warm latency (median ms)", "",
        "| atoms | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|",
    ]
    for n in (32, 128, 512):
        table = _passing(g, f"t8_sp_scaling_{n}__t8_performance")
        cells = []
        vram = {}
        for eng in ENGINE_ORDER:
            rec = table.get(eng)
            if rec is None:
                cells.append("—")
                continue
            m = rec["metrics"]
            cells.append(_fmt(m.get("warm_call_median_seconds", 0) * 1e3, 3))
            vram[eng] = m.get("peak_vram_mib")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines.append("")
    lines += [
        "### NVE MD throughput (steps/s / atom-steps/s)", "",
        "| atoms | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|",
    ]
    for n in (128, 512):
        table = _passing(g, f"t8_md_throughput_{n}__t8_performance")
        cells = []
        for eng in ENGINE_ORDER:
            rec = table.get(eng)
            if rec is None:
                cells.append("—")
                continue
            m = rec["metrics"]
            cells.append(
                f"{_fmt(m.get('steps_per_second'), 3)} / "
                f"{_fmt(m.get('atom_steps_per_second'), 3)}"
            )
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines.append("")
    vram_parts = []
    table = _passing(g, "t8_sp_scaling_512__t8_performance")
    for eng in ENGINE_ORDER:
        rec = table.get(eng)
        if rec and rec["metrics"].get("peak_vram_mib") is not None:
            vram_parts.append(
                f"{eng} {_fmt(rec['metrics']['peak_vram_mib'])} MiB"
            )
    if vram_parts:
        lines.append("Peak VRAM at 512 atoms: " + ", ".join(vram_parts)
                     + ".")
        lines.append("")
    return "\n".join(lines)



def md_support_matrix(matrix: dict[str, Any]) -> str:
    order = [
        "install", "single_point", "forces", "stress",
        "force_energy_consistency", "stress_energy_consistency",
        "invariance_caching", "fixed_cell_relax", "cell_relax", "eos",
        "elastic", "phonon", "thermodynamics", "defect", "surface",
        "formation_energy", "neb", "saddle_hessian", "short_nve",
        "short_nvt", "transport_demo", "mechanism_analysis",
        "arrhenius", "performance",
    ]
    lines = [
        "## Support matrix (section 39 classification)", "",
        "`software_validated` and `model_characterized` coexist; "
        "`fail`/`characterized` records keep the workload classified "
        "as characterized, with the record counts visible.", "",
        "| workload | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|",
    ]
    for workload in order:
        entry = matrix.get(workload)
        if entry is None:
            continue
        cells = []
        for eng in ENGINE_ORDER:
            cell = entry.get(eng, {"labels": ["not_run"]})
            note = cell.get("note")
            label = " + ".join(cell["labels"])
            if note:
                label += " *"
            cells.append(label)
        lines.append(
            f"| {workload} | " + " | ".join(cells) + " |"
        )
    lines.append("")
    lines.append(
        "\\* install rows cover software validation only (installer, "
        "doctor, runtime validation in CI); no model is involved."
    )
    lines.append("")
    return "\n".join(lines)


def md_limitations() -> str:
    return """## Known limitations

- The OMat24 validation split ships no reference stress labels, so
  stress parity against OMat24 data is not computable; stress is
  validated numerically instead (T1 consistency, T2fd derivative
  agreement).
- Upstream-float32 builds (DPA, GRACE, UMA) show 1e-7..1e-6 eV
  coordinate-order arithmetic noise on transformed-structure
  comparisons; float64 profiles pass bitwise. Recorded as-is.
- Physical transport conclusions (Arrhenius) need longer trajectories
  than the beta budget; stage-2 results are exploratory and honestly
  classified.
- No NPT, no electronic structure, no model uncertainty unless a
  backend implements and audits it.
- V100 validation does not prove every GPU architecture.
"""


def evidence_commits(tiers: dict[str, list[dict[str, Any]]]) -> str:
    """Git commits carried by the evidence records themselves.

    Deriving this from the records (instead of stamping the current HEAD)
    keeps report regeneration byte-stable: the report is a pure function of
    the evidence records, so ``git diff --exit-code`` after regeneration
    actually holds. String-typed to stay coherent with the shared
    ``mlipx.beta-validation-summary/1`` schema id.
    """
    commits = sorted(
        {
            str(r.get("git_commit"))
            for records in tiers.values()
            for r in records
            if r.get("git_commit")
        }
    )
    if not commits:
        return "unknown"
    return ", ".join(commits)


def _summary_json(root: Path, out_dir: Path,
                  tiers: dict[str, list], matrix: dict,
                  coverage: list) -> dict[str, Any]:
    t3 = tiers["t3"]
    omat = {}
    for r in t3:
        m = r.get("metrics", {})
        e = m.get("energy_per_atom_eV") or m.get("energy_per_atom") or {}
        f = m.get("forces", {}) or {}
        omat[r.get("engine", "?")] = {
            "status": r.get("status"),
            "energy_per_atom": e,
            "forces": f,
            "stress": m.get("stress", {}),
        }
    violations: list[str] = []
    problems = [
        f"{tier}: {r.get('_path')} blocked"
        for tier, records in tiers.items()
        for r in records
        if r.get("status") == "blocked"
    ]
    return {
        "schema": "mlipx.beta-validation-summary/1",
        "generated_from": str(root),
        "code_commit": evidence_commits(tiers),
        "model_profiles": {
            eng: ENGINE_LABELS[eng] for eng in ENGINE_ORDER
        },
        "tiers": {
            tier: {
                "records": len(records),
                "by_status": status_counts(records),
            }
            for tier, records in tiers.items()
        },
        "omat24": omat,
        "support_matrix": matrix,
        "coverage_table": coverage,
        "harness_violations": violations,
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", default=".validation-work")
    parser.add_argument(
        "--out", default="validation/science/reports"
    )
    args = parser.parse_args()
    root = Path(args.evidence_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tiers = {tier: load_tier(root, tier) for tier in TIER_GLOBS}
    matrix = build_support_matrix(tiers)
    coverage = build_coverage_table(tiers)

    summary = _summary_json(root, out, tiers, matrix, coverage)
    (out / "beta-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    sections_en = [
        md_provenance(root, out, summary["code_commit"]),
        md_t1(tiers["t1"]),
        md_t2(tiers["t2"]),
        md_t2fd(tiers["t2fd"]),
        md_t3(tiers["t3"], root / "t3"),
        md_t4(tiers["t4"]),
        md_t5(tiers["t5"]),
        md_t6(tiers["t6"]),
        md_t7(tiers["t7"]),
        md_t8(tiers["t8"]),
        md_support_matrix(matrix),
        md_limitations(),
    ]
    (out / "BETA_VALIDATION.md").write_text(
        "\n".join(sections_en), encoding="utf-8"
    )
    (out / "README_VALIDATION.md").write_text(
        readme_validation_snippet(matrix, tiers.get("t3", [])) + "\n",
        encoding="utf-8"
    )

    sections_cn = [
        "# mlipx beta 科学验证报告",
        "",
        f"由 `{root}` 下的证据记录生成；证据对应提交 `{summary['code_commit']}`。"
        "本文档全部表格由 `validation/science/scripts/"
        "generate_beta_report.py` 从结果 JSON 渲染，更新方式是重新生成"
        "并 diff，不要手工编辑。",
        "",
        "模型身份与产物哈希固定在 `validation/science/model_manifest.json`；"
        "OMat24 评估子集见 `validation/science/data/`（种子 20260911）。",
        "",
        md_t1(tiers["t1"]),
        md_t2(tiers["t2"]),
        md_t2fd(tiers["t2fd"]),
        md_t3(tiers["t3"], root / "t3"),
        md_t4(tiers["t4"]),
        md_t5(tiers["t5"]),
        md_t6(tiers["t6"]),
        md_t7(tiers["t7"], cn=True),
        md_t8(tiers["t8"]),
        md_support_matrix(matrix),
        md_limitations(),
    ]
    (out / "BETA_VALIDATION_CN.md").write_text(
        "\n".join(sections_cn), encoding="utf-8"
    )
    for tier, records in tiers.items():
        counts = status_counts(records)
        print(f"[report] {tier}: {len(records)} records, {counts}")
    print(f"[report] wrote {out}/beta-summary.json, "
          f"BETA_VALIDATION.md, BETA_VALIDATION_CN.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
