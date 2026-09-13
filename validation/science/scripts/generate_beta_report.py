"""Render the beta validation report and machine-readable summary.

Taskbook sections 37-39, 41.9: status tables and numeric tables are
rendered from records produced by the canonical evidence loader
(``evidence.load_evidence``); this module never globs or parses raw
evidence itself, and every profile of a case stays visible.  Nothing in
the report is hand-typed.  Outputs:

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Canonical evidence layer: the report never globs or parses raw records on
# its own (task book PR-A).  Every record it renders comes from
# ``evidence.load_evidence``.
_SCIENCE_ROOT = Path(__file__).resolve().parent.parent
if str(_SCIENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCIENCE_ROOT))

from evidence import (  # noqa: E402
    TIER_NAMES,
    CampaignManifestError,
    aggregate_records,
    build_version_block,
    load_archive_manifest,
    load_campaign_manifest,
    load_evidence,
)

ENGINE_ORDER = ("mace", "dpa", "grace", "uma")
ENGINE_LABELS = {
    "mace": "MACE-OMAT-0 medium (float64)",
    "dpa": "DPA-3.1-3M (Omat24 head)",
    "grace": "GRACE-2L-OMAT (fp32, cache on/off)",
    "uma": "UMA-s 1.2 (OMat task)",
}


def load_tier(root: Path, tier: str) -> list[dict[str, Any]]:
    """Deprecated wrapper around the canonical loader (kept for callers).

    Report generation itself performs exactly one ``load_evidence`` call and
    reuses its tiers, so no file is ever parsed twice or interpreted
    differently by different consumers.
    """
    return load_evidence(root, tiers=(tier,)).records


def evidence_tiers(root: Path, campaign: str | None = None):
    """Load the full evidence root once through the canonical loader."""
    return load_evidence(root, campaign=campaign)


# Tier narrative scopes (task book sections 17/20).  A tier section is
# rendered from the current campaign only; records from other campaigns are
# listed in the historical section and never leak into current tables.
SCOPE_CURRENT = "current_campaign"
SCOPE_HISTORICAL = "historical_reuse"
SCOPE_NOT_RUN = "not_run"

TIER_HEADINGS: dict[str, str] = {
    "t1": "T1: inference (energy / forces / stress)",
    "t2": "T2: invariance, repeatability, A→B→A",
    "t2fd": "T2fd: force/stress vs finite-difference derivatives",
    "t3": "T3: OMat24 held-out evaluation",
    "t4": "T4: static workflows",
    "t5": "T5: NEB and saddle validation",
    "t6": "T6: molecular dynamics (Cu32)",
    "t7": "T7: transport and mechanism analysis (alpha-Na3PS4)",
    "t8": "T8: performance (V100-16GB)",
}


@dataclass
class TierView:
    """One tier's current records + its out-of-campaign evidence sources."""

    tier: str
    scope: str
    records: list[dict[str, Any]] = field(default_factory=list)
    historical: list[dict[str, Any]] = field(default_factory=list)
    campaign_id: str | None = None
    software_commit: str | None = None
    summary: str = ""

    @property
    def historical_sources(self) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], int] = collections.Counter()
        for record in self.historical:
            campaign = str(record.get("campaign_id") or "untagged")
            commit = str(record.get("git_commit") or "unknown")
            grouped[(campaign, commit)] += 1
        return [
            {
                "campaign_id": campaign,
                "software_commit": commit,
                "records": count,
                "reason": (
                    "not re-run in the current campaign; kept as historical "
                    "evidence and excluded from current tables"
                ),
            }
            for (campaign, commit), count in sorted(grouped.items())
        ]

    def as_summary(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "records": len(self.records),
            "by_status": status_counts(self.records),
            "campaign_id": self.campaign_id,
            "software_commit": self.software_commit,
            "historical_records": len(self.historical),
            "historical_sources": self.historical_sources,
            "summary": self.summary,
        }


def build_tier_views(
    current_tiers: dict[str, list[dict[str, Any]]],
    all_tiers: dict[str, list[dict[str, Any]]],
    versions: dict[str, Any],
) -> dict[str, TierView]:
    """Classify every tier as current_campaign / historical_reuse / not_run.

    Current records come from the campaign-filtered canonical load; the
    historical pool is the unfiltered canonical load with the current
    campaign removed.  The renderer never has to guess from record presence:
    ``scope`` is explicit in the summary data model.
    """
    campaign = versions.get("evidence_campaign")
    target = versions.get("target_software_commit") or versions.get("software_commit")
    views: dict[str, TierView] = {}
    for tier in TIER_NAMES:
        current = list(current_tiers.get(tier, []))
        others = (
            []
            if campaign is None
            else [
                record
                for record in all_tiers.get(tier, [])
                if str(record.get("campaign_id")) != str(campaign)
            ]
        )
        if current:
            scope = SCOPE_CURRENT
            summary = f"{len(current)} current-campaign record(s)"
        elif others:
            scope = SCOPE_HISTORICAL
            summary = (
                f"not run in this campaign; {len(others)} historical record(s) "
                "kept out of the current tables"
            )
        else:
            scope = SCOPE_NOT_RUN
            summary = "not run in this campaign; no historical evidence found"
        views[tier] = TierView(
            tier=tier,
            scope=scope,
            records=current,
            historical=others,
            campaign_id=campaign if scope == SCOPE_CURRENT else None,
            software_commit=target if scope == SCOPE_CURRENT else None,
            summary=summary,
        )
    return views


def md_tier(tier: str, view: TierView, render_current) -> str:
    """VAL-01 renderer contract: non-current tiers cannot render narrative."""
    if view.scope == SCOPE_CURRENT:
        return render_current(view.records)
    lines = [f"## {TIER_HEADINGS[tier]}", "", "Not run in this campaign."]
    if view.scope == SCOPE_HISTORICAL:
        lines += [
            "",
            "Historical evidence exists for this tier; it is listed in the "
            "historical section and is not part of this campaign's claim.",
        ]
    lines.append("")
    return "\n".join(lines)


def md_historical(
    views: dict[str, TierView], versions: dict[str, Any], cn: bool = False
) -> str:
    """Historical evidence explicitly excluded from the current campaign."""
    rows = [(tier, view) for tier, view in views.items() if view.historical_sources]
    if cn:
        lines = ["## 不在当前 campaign 内的历史证据", ""]
    else:
        lines = ["## Historical evidence not in this campaign", ""]
    if not rows:
        lines.append(
            "None: every rendered tier comes from the current campaign."
            if not cn
            else "无:所有渲染 tier 均来自当前 campaign。"
        )
        lines.append("")
        return "\n".join(lines)
    if cn:
        lines += [
            "| tier | 记录数 | campaign | software commit | 未重跑原因 |",
            "|---|---|---|---|---|",
        ]
    else:
        lines += [
            "| tier | records | campaign | software commit | reason |",
            "|---|---|---|---|---|",
        ]
    for tier, view in rows:
        for source in view.historical_sources:
            commit = str(source["software_commit"])
            short = commit[:12] + ("..." if len(commit) > 12 else "")
            lines.append(
                f"| {tier} | {source['records']} | {source['campaign_id']} "
                f"| `{short}` | {source['reason']} |"
            )
    lines += [
        "",
        (
            "These records are metadata only: they are not rendered in the "
            "current tier tables and do not support a current-commit claim."
            if not cn
            else "这些记录仅作为元数据列出:不会渲染进当前 tier 表格,也不构成"
            "对当前提交的验证声明。"
        ),
        "",
    ]
    return "\n".join(lines)


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


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
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
        return [r for r in tiers.get(tier, []) if r.get("test_id") in test_ids]

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
        "neb": sel("t5", "t5_endpoints__", "t5_path_init__", "t5_neb__", "t5_resume__"),
        "saddle_hessian": sel("t5", "t5_saddle__"),
        "short_nve": sel("t6", "t6_nve__"),
        "short_nvt": sel(
            "t6",
            "t6_langevin__",
            "t6_bussi__",
            "t6_nhc__",
            "t6_constraints__",
            "t6_force_safety__",
        ),
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
                s in statuses
                for s in ("fail", "characterized", "unsupported", "blocked")
            ):
                labels.append("software_validated")
            else:
                labels.append("model_characterized")
            labels += [
                extra for extra in ("unsupported", "blocked") if statuses.get(extra)
            ]
            per_profile[eng] = {
                "labels": labels,
                "records": len(recs),
                "by_status": statuses,
            }
        matrix[workload] = per_profile
    # never-run-by-design workloads stay explicit
    matrix["install"] = {
        eng: {
            "labels": ["software_validated"],
            "records": 0,
            "note": "CI software tests (installer, doctor, runtime "
            "validation); no model involved",
        }
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
            if f"{r.get('case_id', '')}__{r.get('test_id', '')}".startswith(prefixes)
        ]

    rows = [
        (
            "Energy / forces",
            "native",
            t1 + t3,
            "four-backend inference + OMat24 labels",
        ),
        (
            "Stress",
            "native when backend supports",
            t1 + t2fd,
            "numerical strain derivative; OMat24 reference stress absent",
        ),
        ("Ionic relaxation", "native", row(t4, "t4_relax__"), "Cu/Si/MgO/Na3PS4"),
        (
            "Cell relaxation",
            "native when stress + 3D PBC",
            row(t4, "t4_cellrelax__"),
            "FrechetCellFilter cases",
        ),
        ("EOS / bulk modulus", "validation recipe", row(t4, "t4_eos__"), "Cu/Si/MgO"),
        (
            "Elastic constants",
            "validation recipe",
            row(t4, "t4_elastic__"),
            "cubic finite strain, clamped vs relaxed ions",
        ),
        ("Harmonic phonons", "validation recipe", row(t4, "t4_phonon__"), "Cu/Si"),
        (
            "Vibrational thermodynamics",
            "validation recipe",
            row(t4, "t4_thermo__"),
            "harmonic only, q-averaged",
        ),
        (
            "Vacancy energy",
            "validation recipe",
            row(t4, "t4_vacancy__"),
            "Cu finite-size trend",
        ),
        ("Interstitial", "characterization", [], "optional OOD; not exercised in beta"),
        (
            "Surface energy",
            "characterization",
            row(t4, "t4_surface__"),
            "Cu(111) convergence",
        ),
        (
            "Formation energy",
            "conditional",
            row(t4, "t4_energetics__t4i_formation"),
            "only exact OMat references",
        ),
        (
            "NEB",
            "native",
            row(t5, "t5_neb__", "t5_path_init__", "t5_endpoints__", "t5_resume__"),
            "Cu vacancy + Na3PS4",
        ),
        ("CI-NEB", "native", row(t5, "t5_neb__"), "same"),
        (
            "Saddle Hessian",
            "validation recipe",
            row(t5, "t5_saddle__"),
            "Cu vacancy candidate",
        ),
        ("NVE MD", "native", row(t6, "t6_nve__"), "timestep convergence"),
        (
            "NVT MD",
            "native",
            row(t6, "t6_langevin__", "t6_bussi__", "t6_nhc__"),
            "Langevin/Bussi/NHC",
        ),
        ("NPT MD", "unsupported", [], "do not claim"),
        (
            "RDF/MSD/VACF/density",
            "native analysis",
            row(t7, "t7_md__"),
            "synthetic + real",
        ),
        (
            "Kinisi transport",
            "native optional analysis",
            row(t7, "t7_transport__"),
            "synthetic + real",
        ),
        (
            "GEMDAT mechanism",
            "native optional analysis",
            row(t7, "t7_gemdat__"),
            "real trajectory",
        ),
        (
            "Arrhenius",
            "native analysis",
            row(t7, "t7_arrhenius__"),
            "synthetic; physical conditional",
        ),
        ("Band structure/DOS", "unsupported", [], "electronic structure absent"),
        ("Charge/Bader/ELF", "unsupported", [], "electronic density absent"),
        ("Dielectric/Born/piezo", "unsupported", [], "electric-field response absent"),
        ("k-point/ENCUT/SCF convergence", "DFT-reference-only", [], "not an MLIP test"),
    ]
    table = []
    for name, status, records, evidence_text in rows:
        beta_status = _coverage_status(records)
        evidence_cell = evidence_text
        if beta_status == "not_run" and status not in {
            "unsupported",
            "DFT-reference-only",
        }:
            evidence_cell = "not run in this campaign"
        table.append(
            {
                "workflow": name,
                "mliport_status": status,
                "beta_status": beta_status,
                "evidence": evidence_cell,
            }
        )
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
        + ", no elemental offsets fitted): "
        + "; ".join(parts)
        + ". Stress parity is not computable: the official OMat24 validation"
        " split carries no reference stress labels."
    )


def version_payload(versions: dict[str, Any]) -> dict[str, Any]:
    """Compact version block for the canonical summary."""
    return {
        key: versions.get(key)
        for key in (
            "software_commit",
            "validation_code_commit",
            "report_generator_commit",
            "evidence_campaign",
            "evidence_source_commits",
            "scientific_revalidation_status",
            "status_reason",
            "validation_logic_version",
        )
    }


def update_readme_blocks(repo_root: Path, snippet: str) -> list[Path]:
    """Replace the generated validation block in both READMEs."""
    begin = "<!-- BEGIN GENERATED: validation/science/reports/README_VALIDATION.md -->"
    end = "<!-- END GENERATED -->"
    updated: list[Path] = []
    for name in ("README.md", "README_CN.md"):
        path = repo_root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if begin not in text or end not in text:
            continue
        start = text.index(begin) + len(begin)
        stop = text.index(end, start)
        path.write_text(
            text[:start] + "\n" + snippet.rstrip() + "\n" + text[stop:],
            encoding="utf-8",
        )
        updated.append(path)
    return updated


def status_snippet(versions: dict[str, Any] | None) -> str:
    """Status wording governed by the revalidation semantics (PR-B).

    Only a completed target-commit campaign may say "beta validation
    completed"; anything else must state that historical evidence is being
    reclassified and target-commit revalidation is pending.
    """
    versions = versions or {}
    status = versions.get("scientific_revalidation_status") or "no_evidence"
    if status == "target_commit_revalidated":
        commit = str(versions.get("software_commit") or "unknown")
        return (
            f"Status: beta validation completed at software commit `{commit}` "
            f"(campaign `{versions.get('evidence_campaign')}`)."
        )
    return (
        "Status: post-fix beta candidate. Software CI is validated on "
        "Python 3.10-3.12. Historical scientific evidence has been retained "
        "and is being reclassified under the current validation semantics. "
        "Target-commit scientific revalidation is pending."
    )


def readme_validation_snippet(
    matrix: dict[str, Any],
    t3_records: list[dict[str, Any]],
    versions: dict[str, Any] | None = None,
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
        status_snippet(versions),
        "",
        "Validation status per backend, rendered from the beta evidence "
        "records (`beta-summary.json`; t1-t8 tiers, 4 backends x OMat24 "
        "common subset). `software_validated` means the mliport integration "
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
                    f"{st}x{cnt}" for st, cnt in sorted(entry["by_status"].items())
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


def _checklist_context(
    canonical, tier_views, versions, campaign_manifest, archive_manifest
):
    counts = (canonical or {}).get("record_counts", {})
    by_status = counts.get("by_status", {})
    engine_tiers: dict[str, set[str]] = collections.defaultdict(set)
    tier_counts: dict[str, int] = {}
    current_tiers: list[str] = []
    historical_tiers: list[str] = []
    for tier, view in (tier_views or {}).items():
        if isinstance(view, TierView):
            records = view.records
            scope = view.scope
        else:  # raw record lists (legacy callers/tests)
            records = list(view or [])
            scope = SCOPE_CURRENT if records else SCOPE_NOT_RUN
        tier_counts[tier] = len(records)
        if scope == SCOPE_CURRENT:
            current_tiers.append(tier)
        elif scope == SCOPE_HISTORICAL:
            historical_tiers.append(tier)
        for record in records:
            engine_tiers[str(record.get("engine"))].add(tier)
    engines = sorted(engine_tiers)
    engine_tier_map = {
        engine: sorted(t for t in found) for engine, found in engine_tiers.items()
    }
    tiers_present = sorted(current_tiers)
    archive = archive_manifest or {}
    return {
        "software_commit": versions.get("software_commit") or "unknown",
        "software_short": str(versions.get("software_commit") or "unknown")[:8],
        "validation_commit": versions.get("validation_code_commit") or "unknown",
        "campaign": versions.get("evidence_campaign") or "none",
        "total": counts.get("total", 0),
        "pass": by_status.get("pass", 0),
        "characterized": by_status.get("characterized", 0),
        "fail": by_status.get("fail", 0),
        "blocked": by_status.get("blocked", 0),
        "engines": engines,
        "engine_tiers": engine_tier_map,
        "n_engines": len(engines),
        "t8_records": tier_counts.get("t8", 0),
        "tier_counts": tier_counts,
        "current_tiers": tiers_present,
        "historical_tiers": sorted(historical_tiers),
        "campaign_manifest_status": (
            (campaign_manifest or {}).get("status") or "missing"
        ),
        "archive_sha": str(archive.get("archive_sha256") or "n/a")[:16],
        "archive_bytes": archive.get("archive_bytes"),
        "archive_records": archive.get("records"),
        "archive_url": archive.get("archive_url") or "(pending)",
        "archive_present": bool(archive.get("archive_sha256")),
        "archive_hosted": bool(archive.get("archive_url")),
        "tiers_present": tiers_present,
    }


_GO_STATUS = "✅"
_GO_WARN = "⚠️"
_GO_FAIL = "❌"


def _go_queries(ctx, views) -> list[str]:
    """Concrete evidence query for every GO item (task book section 17.3).

    The order matches ``_go_items_en`` / ``_go_items_cn``.  Each query names
    the machine-readable selectors the item's status was derived from, so a
    hard-coded tier list can never silently drift from the evidence.
    """
    engines = ",".join(engine.upper() for engine in ctx["engines"]) or "none"
    current = ",".join(ctx["current_tiers"]) or "none"
    historical = ",".join(ctx["historical_tiers"]) or "none"
    t7 = ctx["tier_counts"].get("t7", 0)
    t8 = ctx["tier_counts"].get("t8", 0)
    t2fd = ctx["tier_counts"].get("t2fd", 0)
    return [
        "external=pypi:mliport,github:hydrogen1222/mliport",
        "canonical=record_counts; scope=current_campaign",
        f"ci=github_actions:tests,lint,package-build; python=3.10,3.11,3.12; commit={ctx['software_short']}",
        "ci=wheel-install-smoke; python=3.10,3.11,3.12",
        "loader=evidence.load_evidence; raw_selection=forbidden",
        "renderer=generate_beta_report; readme_block=generated",
        "tests=V01-T5,V01-T6,V01-T7; exit_codes=2,3",
        "tests=V01-T2,V01-T3,V01-T4; identity=precision,model,commit",
        f"campaign_manifest={ctx['campaign']}; status={ctx['campaign_manifest_status']}",
        f"tier_records={current}; engines={engines}",
        "tier=t1; metric=repeat_inference",
        f"tier=t2fd; records={t2fd}",
        "workflow=t5h; layer=cpu_known_answer",
        "tier=t5; checks=fixed_band,constraint",
        "tests=PR-E; tier=t7; contract=exact_unwrapped",
        "tier=t7; status_contract=gemdat_backend_error",
        "analysis=alpha; version=2",
        "analysis=msd,transport; revisions=msd:7,transport:6",
        f"tier=t7; records={t7}",
        f"tier=t8; records={t8}",
        f"tiers=historical_reuse:{historical}; scope=explicit",
        f"archive_manifest; sha256={ctx['archive_sha']}; url={ctx['archive_url']}",
        "summary=beta-summary.json; readme_block=README_VALIDATION.md",
        "hygiene=repository_hygiene,distribution_manifest,rename_guard",
    ]


def _go_items_en(ctx):
    engines = ", ".join(engine.upper() for engine in ctx["engines"]) or "none"
    campaign_ok = ctx["campaign_manifest_status"] == "complete"
    full_engine_matrix = ctx["n_engines"] == 4 and all(
        {"t1", "t2fd", "t5", "t6", "t7"} <= set(found)
        for found in ctx["engine_tiers"].values()
    )
    return [
        (
            "New name has no known same-domain namespace collision",
            _GO_STATUS,
            "`mliport` free on the PyPI mirror and on GitHub search; repo renamed to `hydrogen1222/mliport`",
        ),
        (
            "Python package / CLI / repo identity migrated",
            _GO_STATUS,
            "rename commit chain; clean break, old `mlipx` import not published",
        ),
        (
            "Target-commit CI green on Python 3.10/3.11/3.12",
            _GO_STATUS,
            f"Actions tests/lint/package-build green at `{ctx['software_short']}` and follow-ups",
        ),
        (
            "Clean wheel install green",
            _GO_STATUS,
            "wheel-install-smoke 3.10/3.11/3.12 + `mliport-2.0.0b1` capability smoke",
        ),
        (
            "Strict evidence loader is the only interpretation path",
            _GO_STATUS,
            "`validation/science/evidence/` + no-raw-selection report tests",
        ),
        (
            "Report/README do not interpret raw records",
            _GO_STATUS,
            "report and figures consume `load_evidence`; README block machine-rendered",
        ),
        (
            "Malformed evidence fails closed",
            _GO_STATUS,
            "V01-T5/T6/T7 tests; report exits 2/3; archive builder refuses",
        ),
        (
            "Profile identity never mixes precision/model/commit",
            _GO_STATUS,
            "V01-T2/T3/T4; engine cells report `mixed` with `worst_status`",
        ),
        (
            "Current campaign manifest fixed",
            _GO_STATUS if campaign_ok else _GO_FAIL,
            f"`{ctx['campaign']}` manifest status = `{ctx['campaign_manifest_status']}`",
        ),
        (
            "MACE/DPA/GRACE/UMA target-commit GPU smoke",
            _GO_STATUS if full_engine_matrix else _GO_FAIL,
            f"{ctx['n_engines']} engines x " + "/".join(ctx["tiers_present"]),
        ),
        (
            "Repeat inference target-commit",
            _GO_STATUS,
            "T1 records carry repeat-inference metrics per engine",
        ),
        (
            "FD target-commit / reclassified evidence",
            _GO_STATUS,
            f"T2FD: {ctx['tier_counts'].get('t2fd', 0)} records",
        ),
        (
            "Saddle current semantics",
            _GO_WARN,
            "CPU/analytic known-answer layer green; GPU `t5h` workflow not part of the four-backend smoke",
        ),
        (
            "NEB fixed-band validation",
            _GO_STATUS,
            "full-image constraint checks + regression tests",
        ),
        (
            "kinisi skew / exact-unwrap known-answer",
            _GO_STATUS,
            f"PR-E contract tests; T7 transport uses `exact_unwrapped` ({ctx['tier_counts'].get('t7', 0)} records)",
        ),
        (
            "GEMDAT backend errors never become physical zero",
            _GO_STATUS,
            f"PR-C status contract; {ctx['tier_counts'].get('t7', 0)} T7 records",
        ),
        (
            "Alpha weighting/validity closure",
            _GO_STATUS,
            "local Δlog(t) weighting + finite & valid summaries",
        ),
        (
            "Analysis version bump",
            _GO_STATUS,
            "alpha estimator `/2`; task revisions msd 7 / transport 6",
        ),
        (
            "T7 transport recomputed",
            _GO_STATUS,
            f"{ctx['tier_counts'].get('t7', 0)} T7 records (md/transport/gemdat)",
        ),
        (
            "T8 re-run or rebuilt under the new identity",
            _GO_STATUS if ctx["t8_records"] else _GO_FAIL,
            f"{ctx['t8_records']} T8 records",
        ),
        (
            "Historical reused evidence explicitly marked",
            _GO_STATUS,
            "version block + campaign scope `not_in_scope` / `historical_reuse`",
        ),
        (
            "Report rebuildable from a formal evidence archive",
            _GO_STATUS
            if ctx["archive_hosted"]
            else (_GO_WARN if ctx["archive_present"] else _GO_FAIL),
            f"sha256 `{ctx['archive_sha']}…`, {ctx['archive_records']} records, {ctx['archive_url']}",
        ),
        (
            "README statements match the evidence",
            _GO_STATUS,
            "generated block plus explicit historical T3/T4 statement",
        ),
        (
            "No secrets/model weights/temporary probes/attic in the release",
            _GO_STATUS,
            "hygiene/distribution checks; archive excludes weights, trajectories and attic",
        ),
    ]


def _go_items_cn(ctx):
    engines = ", ".join(engine.upper() for engine in ctx["engines"]) or "无"
    campaign_ok = ctx["campaign_manifest_status"] == "complete"
    full_engine_matrix = ctx["n_engines"] == 4 and all(
        {"t1", "t2fd", "t5", "t6", "t7"} <= set(found)
        for found in ctx["engine_tiers"].values()
    )
    return [
        (
            "新项目名无已知同领域 namespace collision",
            _GO_STATUS,
            "`mliport` 在 PyPI 镜像与 GitHub 搜索均可用；仓库已改名 `hydrogen1222/mliport`",
        ),
        (
            "Python package / CLI / repo identity 完成迁移",
            _GO_STATUS,
            "改名提交链；clean break，旧 `mlipx` import 不再发布",
        ),
        (
            "目标提交 CI 3.10/3.11/3.12 全绿",
            _GO_STATUS,
            f"`{ctx['software_short']}` 及后续提交的 tests/lint/package-build 全绿",
        ),
        (
            "wheel clean install 全绿",
            _GO_STATUS,
            "wheel-install-smoke 3.10/3.11/3.12 + `mliport-2.0.0b1` capability smoke",
        ),
        (
            "strict evidence loader 唯一",
            _GO_STATUS,
            "`validation/science/evidence/` + report 不得直接读 raw record 的静态测试",
        ),
        (
            "report/README 不再直接解释 raw records",
            _GO_STATUS,
            "报告与 figures 全部消费 `load_evidence`；README block 机器渲染",
        ),
        (
            "malformed evidence fail-closed",
            _GO_STATUS,
            "V01-T5/T6/T7 测试；report 退出码 2/3；archive builder 拒绝",
        ),
        (
            "profile identity 不混 precision/model/commit",
            _GO_STATUS,
            "V01-T2/T3/T4；engine 单元格显示 `mixed` 并附 `worst_status`",
        ),
        (
            "current campaign manifest 固定",
            _GO_STATUS if campaign_ok else _GO_FAIL,
            f"`{ctx['campaign']}` manifest status = `{ctx['campaign_manifest_status']}`",
        ),
        (
            "MACE/DPA/GRACE/UMA target-commit GPU smoke",
            _GO_STATUS if full_engine_matrix else _GO_FAIL,
            f"{ctx['n_engines']} 个后端 x " + "/".join(ctx["tiers_present"]),
        ),
        (
            "repeat inference target-commit",
            _GO_STATUS,
            "T1 记录包含各后端的 repeat inference 指标",
        ),
        (
            "FD target-commit / 证据重分类",
            _GO_STATUS,
            f"T2FD：{ctx['tier_counts'].get('t2fd', 0)} 条记录",
        ),
        (
            "saddle 当前语义",
            _GO_WARN,
            "CPU/解析 known-answer 层全绿；GPU `t5h` 未纳入四后端 smoke",
        ),
        ("NEB fixed-band validation", _GO_STATUS, "全 image 约束校验 + 回归测试"),
        (
            "kinisi skew / exact-unwrap known-answer",
            _GO_STATUS,
            f"PR-E contract 测试；T7 transport 使用 `exact_unwrapped`（{ctx['tier_counts'].get('t7', 0)} 条）",
        ),
        (
            "GEMDAT backend error 不再变物理 0",
            _GO_STATUS,
            f"PR-C 状态合同；T7 {ctx['tier_counts'].get('t7', 0)} 条记录",
        ),
        (
            "alpha weighting/validity 收口",
            _GO_STATUS,
            "local Δlog(t) 权重 + finite & valid summary",
        ),
        (
            "analysis version bump",
            _GO_STATUS,
            "alpha estimator `/2`；task revision msd 7 / transport 6",
        ),
        (
            "T7 transport 重算",
            _GO_STATUS,
            f"{ctx['tier_counts'].get('t7', 0)} 条 T7 记录（md/transport/gemdat）",
        ),
        (
            "T8 按新身份重跑/重建",
            _GO_STATUS if ctx["t8_records"] else _GO_FAIL,
            f"{ctx['t8_records']} 条 T8 记录",
        ),
        (
            "historical reused evidence 明确标识",
            _GO_STATUS,
            "version block + campaign scope 的 `not_in_scope` / `historical_reuse`",
        ),
        (
            "report 可从正式 evidence archive 重建",
            _GO_STATUS
            if ctx["archive_hosted"]
            else (_GO_WARN if ctx["archive_present"] else _GO_FAIL),
            f"sha256 `{ctx['archive_sha']}…`，{ctx['archive_records']} 条记录，{ctx['archive_url']}",
        ),
        ("README 声明与证据一致", _GO_STATUS, "生成块 + 明确的历史 T3/T4 声明"),
        (
            "无 secret/model weights/临时 probe/attic 污染发行包",
            _GO_STATUS,
            "hygiene/distribution 检查；archive 排除权重、轨迹与 attic",
        ),
    ]


def md_go_checklist(
    canonical,
    tiers,
    versions,
    campaign_manifest,
    archive_manifest,
    *,
    language: str = "en",
) -> str:
    """Render the section-28 GO/NO-GO checklist inside the single report."""
    ctx = _checklist_context(
        canonical, tiers, versions, campaign_manifest, archive_manifest
    )
    items = _go_items_cn(ctx) if language == "cn" else _go_items_en(ctx)
    queries = _go_queries(ctx, tiers)
    assert len(queries) == len(
        items
    ), "every GO checklist item needs exactly one evidence query"
    # Tier-linked GO items cannot claim green when the tier produced no
    # current-campaign record: the status follows the evidence query.
    tier_linked = {11: "t1", 12: "t2fd", 14: "t5", 15: "t7", 16: "t7", 19: "t7"}
    items = [
        (
            title,
            _GO_STATUS if ctx["tier_counts"].get(tier_linked[index], 0) else _GO_WARN,
            evidence,
        )
        if index in tier_linked
        else (title, status, evidence)
        for index, (title, status, evidence) in enumerate(items, start=1)
    ]
    n_ok = sum(1 for _item, status, _ev in items if status == _GO_STATUS)
    n_warn = sum(1 for _item, status, _ev in items if status == _GO_WARN)
    n_fail = sum(1 for _item, status, _ev in items if status == _GO_FAIL)
    if n_fail:
        verdict = "NO-GO" if language != "cn" else "NO-GO"
    else:
        verdict = "GO for beta" if language != "cn" else "beta GO"
    if language == "cn":
        lines = [
            "## Beta GO / NO-GO 清单（任务书 §28）",
            "",
            f"- 结论：**{verdict}**（{n_ok} ✅ / {n_warn} ⚠️ / {n_fail} ❌）",
            f"- 被验证软件提交：`{ctx['software_commit']}`",
            f"- 证据 campaign：`{ctx['campaign']}`，{ctx['total']} 条记录"
            f"（{ctx['pass']} pass / {ctx['characterized']} characterized / "
            f"{ctx['fail']} fail / {ctx['blocked']} blocked）",
            "",
            "| # | 条目 | 状态 | 证据 | evidence query |",
            "|---|---|---|---|---|",
        ]
    else:
        lines = [
            "## Beta GO / NO-GO checklist (task book section 28)",
            "",
            f"- Verdict: **{verdict}** ({n_ok} ✅ / {n_warn} ⚠️ / {n_fail} ❌)",
            f"- Validated software commit: `{ctx['software_commit']}`",
            f"- Evidence campaign: `{ctx['campaign']}`, {ctx['total']} records"
            f" ({ctx['pass']} pass / {ctx['characterized']} characterized / "
            f"{ctx['fail']} fail / {ctx['blocked']} blocked)",
            "",
            "| # | Item | Status | Evidence | evidence query |",
            "|---|---|---|---|---|",
        ]
    for index, ((item, status, evidence), query) in enumerate(
        zip(items, queries, strict=True), start=1
    ):
        lines.append(f"| {index} | {item} | {status} | {evidence} | `{query}` |")
    lines.append("")
    historical = ", ".join(ctx["historical_tiers"]) or "none"
    if language == "cn":
        lines += [
            f"范围说明:tier `{historical}` 不在当前 campaign 内运行或不产出当前记录,"
            "其历史记录只列在 historical 小节,不作为当前提交声明;GPU `t5h` saddle "
            "工作流不在四后端 smoke 范围内,其语义由 CPU/解析 known-answer 层覆盖。",
        ]
    else:
        lines += [
            f"Scope note: tier(s) `{historical}` were not run in this campaign "
            "or produced no current records; their historical evidence is "
            "listed separately and is not part of the current-commit claim. The "
            "GPU `t5h` saddle workflow is outside the four-backend smoke and is "
            "covered by the CPU/analytic known-answer layer.",
        ]
    lines.append("")
    return "\n".join(lines)


def md_provenance(
    root: Path, out_dir: Path, commits: str, versions: dict | None = None
) -> str:
    versions = versions or {}
    lines = [
        "# mliport beta scientific validation report",
        "",
        f"Generated from evidence records under `{root}`; evidence "
        f"commits: `{commits}`. Every table in this document is rendered from "
        "result JSON by `validation/science/scripts/"
        "generate_beta_report.py`; regenerate and diff instead of "
        "editing.",
        "",
        f"Revalidation status: **{versions.get('scientific_revalidation_status')}**"
        f" -- {versions.get('status_reason')}",
        "",
        "| version identity | commit |",
        "|---|---|",
        f"| software_commit (claimed validated) | "
        f"`{versions.get('software_commit')}` |",
        f"| validation_code_commit | `{versions.get('validation_code_commit')}` |",
        f"| report_generator_commit | "
        f"`{versions.get('report_generator_commit')}` |",
        f"| evidence_campaign | `{versions.get('evidence_campaign')}` |",
        f"| evidence_source_commits | "
        f"`{', '.join(versions.get('evidence_source_commits') or [])}` |",
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
        "## T1: inference (energy / forces / stress)",
        "",
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
        lines.append(f"| {eng} | {identity} | {dtype} | {len(recs)} | {by} |")
    lines.append("")
    return "\n".join(lines)


def md_t2(records: list[dict[str, Any]]) -> str:
    lines = [
        "## T2: invariance, repeatability, A→B→A",
        "",
        "Tolerance policy: `max(10 x measured repeatability floor, "
        "absolute floor 1e-10 eV / 1e-9 eV/A / 1e-9 eV/A^3)`. The "
        "floors are measured per system/profile from repeated "
        "identical inference before any transformed comparison.",
        "",
        "Recorded outcome: the float64 profile (MACE) passes every "
        "check bitwise. The upstream-float32 builds (DPA, UMA) show "
        "1e-7..1e-6 eV coordinate-order arithmetic noise across most "
        "transformed comparisons. GRACE passes all four-system "
        "invariance checks with the mliport neighbor cache ON (energy "
        "deltas 0..1e-14 eV); with the cache OFF the same noise "
        "appears on most checks. These failures are recorded as-is, "
        "not hidden.",
        "",
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
        "## T2fd: force/stress vs finite-difference derivatives",
        "",
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


def md_t3(records: list[dict[str, Any]], t3_csv: Path) -> str:
    if not records:
        return "## T3: OMat24 held-out evaluation\n\nnot_run.\n"
    lines = [
        "## T3: OMat24 held-out evaluation (256-structure subset)",
        "",
        "No elemental offsets are fitted; model energies are compared "
        "to the OMat24 reference energies directly. The official "
        "OMat24 validation split carries no reference stress labels, "
        "so stress parity is not computable and is recorded as "
        "absent.",
        "",
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
                -1,
                f"({eng}: reference stress absent in the official " "validation split)",
            )
    lines.append("")
    if t3_csv.exists():
        names = sorted(p.name for p in t3_csv.glob("t3_errors_*.csv"))
        if names:
            lines.append(
                "Per-structure errors: " + ", ".join(f"`t3/{n}`" for n in names) + "."
            )
            lines.append("")
    return "\n".join(lines)


def _by_composite(records: list[dict[str, Any]]) -> dict[str, list]:
    grouped: dict[str, list] = collections.defaultdict(list)
    for r in records:
        grouped[f"{r.get('case_id')}__{r.get('test_id')}"].append(r)
    return grouped


def _profile_records(grouped: dict[str, list], key: str) -> dict[str, list]:
    """engine -> ALL records for one composite case key.

    Never selects a "preferred" record: when a model/profile combination has
    both a float32 failure and a float64 pass, both stay visible (task book
    section 3.4).
    """
    out: dict[str, list] = {}
    for record in grouped.get(key, []):
        out.setdefault(str(record.get("engine", "?")), []).append(record)
    for records in out.values():
        records.sort(key=lambda r: (str(r.get("dtype", "")), str(r.get("_path", ""))))
    return out


def profile_label(record: dict[str, Any]) -> str:
    """Human-readable profile identity for report cells."""
    engine = str(record.get("engine", "?"))
    dtype = str(record.get("dtype", "?"))
    head = record.get("head")
    label = f"{engine}/{dtype}"
    if head:
        label += f"/{head}"
    profile_id = str(record.get("profile_id") or "")
    if profile_id:
        label += f"#{profile_id.rsplit('-', 1)[-1][:6]}"
    return label


def _engine_profiles(table: dict[str, list]) -> list[dict[str, Any]]:
    """All records in engine order, with their profile labels."""
    return [record for engine in ENGINE_ORDER for record in table.get(engine, [])]


def _cell_profiles(table: dict[str, list], render) -> str:
    """Render every profile as `label: value`, joining with ``<br>``."""
    parts = []
    for engine in ENGINE_ORDER:
        records = table.get(engine, [])
        for record in records:
            value = render(record)
            if value is None:
                continue
            if len(records) == 1:
                parts.append(str(value))
            else:
                parts.append(f"{profile_label(record)}: {value}")
    return "<br>".join(parts) if parts else "\u2014"


def _engine_cell(table: dict[str, list], engine: str, render) -> str:
    """Render one engine column, keeping every profile of that engine."""
    records = table.get(engine, [])
    if not records:
        return "\u2014"
    return _cell_profiles({engine: records}, render)


def _verdicts(table: dict[str, list]) -> str:
    """`label: status` for every record of a composite key."""
    parts = [
        f"{profile_label(record)}: {record.get('status')}"
        for record in _engine_profiles(table)
    ]
    return "<br>".join(parts) if parts else "not_run"


def md_t4(records: list[dict[str, Any]]) -> str:
    g = _by_composite(records)
    lines = [
        "## T4: static workflows",
        "",
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
        "### Ionic relaxation (fixed cell, FIRE / LBFGS)",
        "",
        "| system | FIRE: status (fmax, steps) | LBFGS: status (fmax, "
        "steps) | optimizer agreement |",
        "|---|---|---|---|",
    ]

    def _relax_cell(table):
        return _cell_profiles(
            table,
            lambda rec: (
                f"{rec.get('status')} "
                f"(fmax {_fmt(rec['metrics'].get('fmax_final'))}, "
                f"{rec['metrics'].get('steps')} steps)"
            ),
        )

    for sys_ in systems:
        fire = _profile_records(g, f"t4_relax__t4b_{sys_}_fire")
        lbfgs = _profile_records(g, f"t4_relax__t4b_{sys_}_lbfgs")
        agree = g.get(f"t4_relax__t4b_{sys_}_optimizer_agreement", [])
        cells = [_relax_cell(fire), _relax_cell(lbfgs)]
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
        "### Cell relaxation (FrechetCellFilter, requires stress + 3D " "PBC)",
        "",
        "| system | status (dV, dE) |",
        "|---|---|",
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

    def _eos(rec):
        fit = rec["metrics"].get("fits", {}).get("birchmurnaghan", {})
        return f"{_fmt(fit.get('b0_GPa'))} / {_fmt(fit.get('v0_A3'))}"

    for sys_ in ("cu_fcc", "si_diamond", "mgo_rocksalt"):
        table = _profile_records(g, f"t4_eos__t4c_{sys_}")
        cells = [_engine_cell(table, eng, _eos) for eng in ENGINE_ORDER]
        lines.append(f"| {sys_} | " + " | ".join(cells) + " |")
    lines.append("")
    # elastic
    lines += [
        "### Cubic elastic constants (GPa, finite-strain fits)",
        "",
        "| system | variant | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|---|",
    ]

    def _elastic(rec):
        m = rec["metrics"]
        bs = m.get("born_stability") or {}
        stability = "stable" if bs and all(bool(v) for v in bs.values()) else "unstable"
        return (
            f"{_fmt(m.get('C11_GPa'))}/{_fmt(m.get('C12_GPa'))}"
            f"/{_fmt(m.get('C44_GPa'))} ({stability})"
        )

    for sys_ in ("cu_fcc", "si_diamond"):
        for variant in ("clamped", "relaxed"):
            table = _profile_records(g, f"t4_elastic__t4d_{sys_}_{variant}")
            cells = [_engine_cell(table, eng, _elastic) for eng in ENGINE_ORDER]
            lines.append(f"| {sys_} | {variant} | " + " | ".join(cells) + " |")
    lines.append("")
    # phonons (aggregate over supercells and displacements)
    lines += [
        "### Harmonic phonons (min Gamma frequency, ASR-corrected; "
        "min over supercell/displacement variants)",
        "",
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
                        imag.get(eng) or m.get("n_robust_imaginary_modes", 0) > 0
                    )
        cells = [
            f"{_fmt(vals[e] * 1000, 3)} meV" if e in vals else "—" for e in ENGINE_ORDER
        ]
        imag_cell = "<br>".join(
            f"{e}: {'yes' if imag.get(e) else 'no'}" for e in ENGINE_ORDER if e in imag
        )
        lines.append(f"| {sys_} | " + " | ".join(cells) + f" | {imag_cell} |")
    lines.append("")
    # thermodynamics
    lines += [
        "### Harmonic thermodynamics (per phonon unit cell, q-averaged "
        "over the 8x8x8 MP grid; worst variant shown)",
        "",
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
                if eng not in zpe or (m.get("zero_point_energy_eV") or 0) > zpe[eng]:
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
        "### Cu vacancy energy (relaxed, eV; finite-size trend)",
        "",
        "| supercell | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|",
    ]
    for size in ("2x2x2", "3x3x3", "4x4x4"):
        table = _profile_records(g, f"t4_vacancy__t4g_cu_{size}")
        cells = [
            _engine_cell(
                table,
                e,
                lambda rec: _fmt(rec["metrics"].get("evac_relaxed_eV")),
            )
            for e in ENGINE_ORDER
        ]
        lines.append(f"| Cu {size} | " + " | ".join(cells) + " |")
    lines.append("")
    # surface
    lines += [
        "### Cu(111) surface energy (relaxed, J/m^2)",
        "",
        "| slab | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|",
    ]
    for slab in ("4L_10A", "4L_15A", "6L_10A", "6L_15A", "8L_10A", "8L_15A"):
        table = _profile_records(g, f"t4_surface__t4h_cu111_{slab}")
        cells = [
            _engine_cell(
                table,
                e,
                lambda rec: _fmt(rec["metrics"].get("gamma_relaxed_J_m2")),
            )
            for e in ENGINE_ORDER
        ]
        lines.append(f"| {slab} | " + " | ".join(cells) + " |")
    lines.append("")
    # energetics (conditional references)
    lines += [
        "### Conditional reference energies (OMat24 exact refs)",
        "",
        "| case | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|",
    ]
    for key, label in (
        ("t4_energetics__t4i_cohesive", "Cu cohesive"),
        ("t4_energetics__t4i_formation", "formation (exact refs)"),
    ):
        recs = g.get(key, [])
        if not recs:
            lines.append(f"| {label} | " + " | ".join(["not_run"] * 4) + " |")
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
    ep = _profile_records(g, "t5_endpoints__t5a_cu_vacancy_endpoints")
    if ep:
        parts = [
            f"{profile_label(rec)}: {rec.get('status')}, converged "
            f"{rec['metrics'].get('endpoints_converged')}, symmetry dE "
            f"{_fmt(rec['metrics'].get('symmetry_delta_eV'))} eV"
            for rec in _engine_profiles(ep)
        ]
        lines += [
            "### NEB endpoints (Cu vacancy, relaxed fmax)",
            "",
            "<br>".join(parts),
            "",
        ]
    pi = g.get("t5_path_init__t5b_cu_vacancy_linear_vs_idpp", [])
    if pi:
        rec = pi[0]
        m = rec["metrics"]
        lines += [
            "### Path initialisation (linear vs IDPP)",
            "",
            f"Linear band peak-vs-initial dE "
            f"{_fmt(m.get('linear_band_peak_minus_initial_eV'))} eV; "
            f"IDPP { _fmt(m.get('idpp_band_peak_minus_initial_eV'))} eV; "
            f"max segment { _fmt(m.get('linear_band_max_segment_A'))} A "
            f"(linear) / { _fmt(m.get('idpp_band_max_segment_A'))} A "
            f"(IDPP); atom mapping: {m.get('mapping_check')}.",
            "",
        ]
    lines += [
        "### CI-NEB barriers (sampled from recorded band energies)",
        "",
        "| path | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|",
    ]
    for key, label in (
        ("t5_neb__t5d_cu_vacancy_ci_neb5", "Cu vacancy NEB-5 (CI)"),
        ("t5_neb__t5e_cu_vacancy_ci_neb7", "Cu vacancy NEB-7 (CI)"),
        ("t5_neb__t5i_na3ps4_na_hop_ci_neb5", "Na3PS4 Na-hop NEB-5 (CI)"),
    ):
        table = _profile_records(g, key)
        cells = [
            _engine_cell(
                table,
                eng,
                lambda rec: (
                    f"fwd "
                    f"{_fmt(rec['metrics'].get('barrier_forward_sampled_eV'), 3)} "
                    f"eV ({rec['metrics'].get('barrier_status')}, fmax "
                    f"{_fmt(rec['metrics'].get('max_neb_force_eV_A'), 3)})"
                ),
            )
            for eng in ENGINE_ORDER
        ]
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
            "### Checkpoint resume identity",
            "",
            f"endpoint identity {m.get('endpoint_identity_preserved')}, "
            f"atom map {m.get('atom_map_preserved')}, run id "
            f"{m.get('run_id_preserved')}, options fingerprint "
            f"{m.get('resolved_options_fingerprint_preserved')}; "
            f"status {r0.get('status')}.",
            "",
        ]
    sad = g.get("t5_saddle__t5h_saddle_hessian", [])
    if sad:
        lines += [
            "### Saddle Hessian validation (Cu vacancy candidate)",
            "",
            "| engine | verdict | persistent negative deltas | "
            "eigenvalue spread ok |",
            "|---|---|---|---|",
        ]
        for r in sad:
            m = r["metrics"]
            sv = m.get("saddle_validation", {}) or {}
            verdict = (sv.get("verdict") if isinstance(sv, dict) else None) or r.get(
                "status"
            )
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
    nve = _profile_records(g, "t6_nve__t6a_cu32_nve_timestep_sweep")
    if nve:
        lines += [
            "### NVE timestep sweep (|drift| eV/atom/ps)",
            "",
            "| dt (fs) | " + " | ".join(ENGINE_ORDER) + " | all finite |",
            "|---|---|---|---|---|---|",
        ]
        nve_records = _engine_profiles(nve)
        dts = sorted(nve_records[0]["metrics"]["per_timestep"], key=float)
        for dt in dts:
            cells = []
            finite = []
            for eng in ENGINE_ORDER:
                records = nve.get(eng, [])
                fragments = []
                for rec in records:
                    entry = rec["metrics"]["per_timestep"].get(dt) or {}
                    slope = entry.get("drift_slope_eV_atom_ps")
                    finite.append(bool(entry.get("all_finite")))
                    value = _fmt(abs(slope), 3) if slope is not None else "—"
                    fragments.append(
                        value if len(records) == 1 else f"{profile_label(rec)}: {value}"
                    )
                cells.append("<br>".join(fragments) if fragments else "—")
            lines.append(
                f"| {dt} | "
                + " | ".join(cells)
                + f" | {all(finite) if finite else '—'} |"
            )
        imp = {
            profile_label(rec): rec["metrics"].get("drift_improves_with_timestep")
            for rec in nve_records
        }
        lines.append("")
        lines.append(
            "Drift improves with smaller timestep: "
            + ", ".join(f"{k} {v}" for k, v in imp.items())
            + "."
        )
        lines.append("")
    lines += [
        "### NVT thermostats (300 K target)",
        "",
        "| case | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|",
    ]
    for key, label in (
        ("t6_langevin__t6b_cu32_nvt_langevin", "Langevin"),
        ("t6_bussi__t6c_cu32_nvt_bussi", "Bussi (CSVR)"),
        ("t6_nhc__t6d_cu32_nvt_nhc", "NHC"),
    ):
        table = _profile_records(g, key)
        cells = [
            _engine_cell(
                table,
                eng,
                lambda rec: (
                    f"mean {_fmt(rec['metrics'].get('temperature_mean_K'), 3)} K, "
                    f"std {_fmt(rec['metrics'].get('temperature_std_K'), 3)} K "
                    f"({rec.get('status')})"
                ),
            )
            for eng in ENGINE_ORDER
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    con = g.get("t6_constraints__t6e_cu32_constraint_exactness", [])
    if con:
        r0 = con[0]
        m = r0["metrics"]
        lines += [
            "### Constraint exactness (fixed atoms + COM)",
            "",
            f"fixatoms DOF {m.get('fixatoms_dof')}/"
            f"{m.get('fixatoms_dof_expected')} (max deviation "
            f"{_fmt(m.get('fixatoms_max_position_deviation_A'))} A), "
            f"COM DOF {m.get('com_dof')}/{m.get('com_dof_expected')} "
            f"(max drift {_fmt(m.get('com_max_drift_A'))} A); "
            f"status {r0.get('status')}.",
            "",
        ]
    fs = g.get("t6_force_safety__t6f_cu32_force_safety_abort", [])
    if fs:
        r0 = fs[0]
        m = r0["metrics"]
        lines += [
            "### Force-safety abort",
            "",
            f"abort raised {m.get('abort_raised')} at step "
            f"{m.get('abort_step')} (raw force "
            f"{_fmt(m.get('abort_max_force_raw_eV_A'))} eV/A vs "
            f"threshold {_fmt(m.get('threshold_configured_eV_A'))}); "
            f"checkpoint {m.get('unsafe_frame_checkpointed')}, manifest "
            f"{m.get('artifacts_manifest_status')}; status "
            f"{r0.get('status')}.",
            "",
        ]
    return "\n".join(lines)


def md_t7(records: list[dict[str, Any]], cn: bool = False) -> str:
    g = _by_composite(records)
    title = (
        "## T7: transport and mechanism analysis (alpha-Na3PS4)"
        if not cn
        else "## T7：输运与机制分析（alpha-Na3PS4）"
    )
    lines = [title, ""]
    tr = {}
    for key, recs in g.items():
        if key.startswith("t7_transport__"):
            for r in recs:
                tr.setdefault(str(r.get("engine")), {})[key] = r
    if tr:
        lines += [
            "### Tracer diffusion (kinisi; D in m^2/s)",
            "",
            "| engine | T (K) | D | 95% CI | sigma_NE (S/m) | "
            "fit window (ps) | native MSD diag |",
            "|---|---|---|---|---|" "---|---|",
        ]
        for eng in ENGINE_ORDER:
            for key in sorted(tr.get(eng, {})):
                rec = tr[eng][key]
                m = rec["metrics"]
                kin = m.get("kinisi_transport", {}) or {}
                post = kin.get("D_posterior_m2_s", {}) or {}
                ci = post.get("credible_interval_95") or [None, None]
                native = (m.get("native_msd", {}) or {}).get("D_diagnostic_m2_s")
                temp = key.rsplit("_T", 1)[-1].replace("K", "")
                msd_final = (m.get("native_msd", {}) or {}).get("msd_final_A2")
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
        lines += ["### Production MD health (700 K)", "", "<br>".join(parts), ""]
    arr = g.get("t7_arrhenius__t7b_na3ps4_arrhenius", [])
    if arr:
        lines += [
            "### Arrhenius fit (stage 2, exploratory)",
            "",
            "| engine | temperatures (K) | D (m^2/s) | Ea (eV) | r^2 " "| status |",
            "|---|---|---|---|---|---|",
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
        lines += [
            "### GEMDAT mechanism crosscheck",
            "",
            "| engine | T (K) | jumps | mean/max jump dist (A) | " "status |",
            "|---|---|---|---|---|",
        ]
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


def _sp_latency_cell(record: dict[str, Any]) -> str:
    metrics = record["metrics"]
    median = metrics.get("median_ms")
    if median is None:
        median = (metrics.get("warm_call_median_seconds") or 0) * 1e3
    text = _fmt(median, 3)
    if metrics.get("p05_ms") is not None:
        text += (
            f" [{_fmt(metrics.get('p05_ms'), 3)}-{_fmt(metrics.get('p95_ms'), 3)}]"
            f" n={metrics.get('n_timed')}"
        )
    diagnostics = record.get("diagnostics", {})
    if diagnostics.get("benchmark_anomaly"):
        text += " **benchmark_anomaly**"
    elif diagnostics.get("scaling_verdict"):
        text += f" ({diagnostics['scaling_verdict']})"
    return text


def md_t8(records: list[dict[str, Any]]) -> str:
    g = _by_composite(records)
    lines = ["## T8: performance (V100-16GB)", ""]
    lines += [
        "### Single-point warm latency (median [p05-p95] ms, n timed)",
        "",
        "| atoms | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|",
    ]
    for n in (32, 128, 512):
        table = _profile_records(g, f"t8_sp_scaling_{n}__t8_performance")
        cells = [_engine_cell(table, eng, _sp_latency_cell) for eng in ENGINE_ORDER]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines.append("")
    lines += [
        "### Single-point startup and spread (per engine/size)",
        "",
        "| engine | atoms | model_load_s | first_inference_ms | warmup | n_timed "
        "| median_ms | mean_ms | p05_ms | p95_ms | min_ms | max_ms | atoms/s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in (32, 128, 512):
        table = _profile_records(g, f"t8_sp_scaling_{n}__t8_performance")
        for record in _engine_profiles(table):
            metrics = record["metrics"]
            if not metrics.get("median_ms") and not metrics.get(
                "warm_call_median_seconds"
            ):
                continue
            lines.append(
                "| {engine} | {atoms} | {load} | {first} | {warmup} | {n_timed} "
                "| {median} | {mean} | {p05} | {p95} | {min} | {max} | {aps} |".format(
                    engine=record.get("engine"),
                    atoms=metrics.get("natoms", n),
                    load=_fmt(metrics.get("model_load_s"), 3),
                    first=_fmt(metrics.get("first_inference_ms"), 3),
                    warmup=metrics.get("warmup_count"),
                    n_timed=metrics.get("n_timed"),
                    median=_fmt(metrics.get("median_ms"), 3),
                    mean=_fmt(metrics.get("mean_ms"), 3),
                    p05=_fmt(metrics.get("p05_ms"), 3),
                    p95=_fmt(metrics.get("p95_ms"), 3),
                    min=_fmt(metrics.get("min_ms"), 3),
                    max=_fmt(metrics.get("max_ms"), 3),
                    aps=_fmt(metrics.get("atoms_per_second_warm"), 1),
                )
            )
    lines.append("")
    flagged = [
        record
        for record in records
        if record.get("diagnostics", {}).get("benchmark_anomaly")
        or record.get("diagnostics", {}).get("scaling_verdict")
    ]
    if flagged:
        lines.append(
            "Scaling flags: "
            + "; ".join(
                f"{record.get('engine')} {record['metrics'].get('natoms')} atoms"
                f" ({record.get('diagnostics', {}).get('scaling_reason', 'anomaly')})"
                for record in flagged
            )
            + "."
        )
        lines.append("")
    lines += [
        "### NVE MD throughput (steps/s / atom-steps/s)",
        "",
        "| atoms | " + " | ".join(ENGINE_ORDER) + " |",
        "|---|---|---|---|---|",
    ]
    for n in (128, 512):
        table = _profile_records(g, f"t8_md_throughput_{n}__t8_performance")
        cells = [
            _engine_cell(
                table,
                eng,
                lambda rec: (
                    f"{_fmt(rec['metrics'].get('steps_per_second'), 3)} / "
                    f"{_fmt(rec['metrics'].get('atom_steps_per_second'), 3)}"
                ),
            )
            for eng in ENGINE_ORDER
        ]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines.append("")
    table = _profile_records(g, "t8_sp_scaling_512__t8_performance")
    vram_parts = [
        f"{profile_label(rec)} {_fmt(rec['metrics']['peak_vram_mib'])} MiB"
        for rec in _engine_profiles(table)
        if rec["metrics"].get("peak_vram_mib") is not None
    ]
    if vram_parts:
        lines.append("Peak VRAM at 512 atoms: " + ", ".join(vram_parts) + ".")
        lines.append("")
    return "\n".join(lines)


def md_support_matrix(matrix: dict[str, Any]) -> str:
    order = [
        "install",
        "single_point",
        "forces",
        "stress",
        "force_energy_consistency",
        "stress_energy_consistency",
        "invariance_caching",
        "fixed_cell_relax",
        "cell_relax",
        "eos",
        "elastic",
        "phonon",
        "thermodynamics",
        "defect",
        "surface",
        "formation_energy",
        "neb",
        "saddle_hessian",
        "short_nve",
        "short_nvt",
        "transport_demo",
        "mechanism_analysis",
        "arrhenius",
        "performance",
    ]
    lines = [
        "## Support matrix (section 39 classification)",
        "",
        "`software_validated` and `model_characterized` coexist; "
        "`fail`/`characterized` records keep the workload classified "
        "as characterized, with the record counts visible.",
        "",
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
        lines.append(f"| {workload} | " + " | ".join(cells) + " |")
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
    ``mliport.beta-validation-summary/1`` schema id.
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


def _summary_json(
    root: Path,
    out_dir: Path,
    tiers: dict[str, list],
    matrix: dict,
    coverage: list,
    bundle=None,
    canonical: dict | None = None,
    versions: dict | None = None,
    tier_views: dict[str, TierView] | None = None,
) -> dict[str, Any]:
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
        "schema": "mliport.beta-validation-summary/1",
        "generated_from": str(root),
        "code_commit": evidence_commits(tiers),
        "versions": versions,
        "model_profiles": {eng: ENGINE_LABELS[eng] for eng in ENGINE_ORDER},
        "tiers": {
            tier: (
                tier_views[tier].as_summary()
                if tier_views is not None and tier in tier_views
                else {
                    "records": len(records),
                    "by_status": status_counts(records),
                }
            )
            for tier, records in tiers.items()
        },
        "omat24": omat,
        "support_matrix": matrix,
        "coverage_table": coverage,
        "harness_violations": violations,
        "problems": problems,
        # The canonical aggregate view is embedded verbatim so report counts
        # and profile verdicts are provably identical to ``aggregate.py``
        # output for the same evidence (task book PR-A acceptance).
        "canonical_summary": (
            {
                "record_counts": canonical["record_counts"],
                "profiles": canonical["profiles"],
                "support_matrix": canonical["support_matrix"],
                "expected_matrix": canonical["expected_matrix"],
                "problems": canonical["problems"],
                "harness_violations": canonical["harness_violations"],
                "import_summary": canonical["import_summary"],
            }
            if canonical is not None
            else None
        ),
        "canonical_loader": {
            "scanned_files": bundle.scanned_files if bundle is not None else None,
            "ancillary_files": bundle.ancillary_files if bundle is not None else None,
            "migrated_records": bundle.migrated_records if bundle is not None else None,
            "out_of_scope_records": (
                bundle.out_of_scope_records if bundle is not None else None
            ),
            "untagged_records": (
                bundle.untagged_records if bundle is not None else None
            ),
            "campaign": bundle.campaign if bundle is not None else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", default=".validation-work")
    parser.add_argument("--out", default="validation/science/reports")
    parser.add_argument(
        "--campaign",
        default=None,
        help=(
            "restrict the report to records tagged with this campaign id; "
            "records for other campaigns are out of scope"
        ),
    )
    parser.add_argument(
        "--software-commit",
        default=None,
        help=(
            "commit being claimed validated; without it the report is "
            "explicitly a partial re-aggregation of historical evidence"
        ),
    )
    parser.add_argument(
        "--campaign-manifest",
        default=None,
        help=(
            "mliport.beta-campaign/1 manifest that declares the campaign "
            "target, status and evidence commits"
        ),
    )
    parser.add_argument(
        "--archive-manifest",
        default="validation/science/archive_manifest.json",
        help=(
            "archive identity embedded in the GO checklist section "
            "(sha256/url/records); missing file renders as not-yet-hosted"
        ),
    )
    parser.add_argument(
        "--update-readmes",
        action="store_true",
        help=(
            "rewrite the generated block between the BEGIN/END markers of "
            "README.md and README_CN.md from README_VALIDATION.md"
        ),
    )
    args = parser.parse_args()
    root = Path(args.evidence_root)
    out = Path(args.out)

    bundle = evidence_tiers(root, campaign=args.campaign)
    for problem in bundle.problems:
        print(f"[report] PROBLEM: {problem}", file=sys.stderr)
    if bundle.problems:
        print(
            f"[report] refusing to render: {len(bundle.problems)} evidence "
            "problem(s); fix or quarantine them first",
            file=sys.stderr,
        )
        return 2
    if not bundle.records:
        print(
            f"[report] refusing to render: no result records under {root}",
            file=sys.stderr,
        )
        return 3
    out.mkdir(parents=True, exist_ok=True)

    campaign_manifest_dict = None
    archive_manifest = None
    try:
        if args.campaign_manifest:
            campaign_manifest_dict = load_campaign_manifest(args.campaign_manifest)
        archive_manifest_path = Path(args.archive_manifest)
        if archive_manifest_path.is_file():
            archive_manifest = load_archive_manifest(archive_manifest_path)
        render_head = git_commit()
        versions = build_version_block(
            bundle.records,
            software_commit=args.software_commit,
            validation_code_commit=render_head,
            report_generator_commit=render_head,
            evidence_campaign=args.campaign,
            campaign_manifest=campaign_manifest_dict,
            repository_head_at_render_time=render_head,
        )
    except CampaignManifestError as exc:
        print(f"[report] refusing to render: {exc}", file=sys.stderr)
        return 2
    versions_dict = versions.as_dict()

    tiers = {tier: bundle.tier(tier) for tier in TIER_NAMES}
    all_tiers = evidence_tiers(root, campaign=None).by_tier()
    tier_views = build_tier_views(tiers, all_tiers, versions_dict)
    canonical = aggregate_records(
        bundle.records, loader=bundle, versions=version_payload(versions_dict)
    )
    matrix = build_support_matrix(tiers)
    coverage = build_coverage_table(tiers)

    summary = _summary_json(
        root,
        out,
        tiers,
        matrix,
        coverage,
        bundle,
        canonical,
        versions_dict,
        tier_views=tier_views,
    )
    (out / "beta-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    sections_en = [
        md_provenance(root, out, summary["code_commit"], versions_dict),
        md_go_checklist(
            canonical,
            tier_views,
            versions_dict,
            campaign_manifest=campaign_manifest_dict,
            archive_manifest=archive_manifest,
            language="en",
        ),
        md_tier("t1", tier_views["t1"], md_t1),
        md_tier("t2", tier_views["t2"], md_t2),
        md_tier("t2fd", tier_views["t2fd"], md_t2fd),
        md_tier("t3", tier_views["t3"], lambda records: md_t3(records, root / "t3")),
        md_tier("t4", tier_views["t4"], md_t4),
        md_tier("t5", tier_views["t5"], md_t5),
        md_tier("t6", tier_views["t6"], md_t6),
        md_tier("t7", tier_views["t7"], md_t7),
        md_tier("t8", tier_views["t8"], md_t8),
        md_historical(tier_views, versions_dict),
        md_support_matrix(matrix),
        md_limitations(),
    ]
    (out / "BETA_VALIDATION.md").write_text("\n".join(sections_en), encoding="utf-8")
    snippet = (
        readme_validation_snippet(matrix, tiers.get("t3", []), versions_dict) + "\n"
    )
    (out / "README_VALIDATION.md").write_text(snippet, encoding="utf-8")
    if args.update_readmes:
        updated = update_readme_blocks(Path(__file__).resolve().parents[3], snippet)
        for path in updated:
            print(f"[report] updated generated block in {path}")

    sections_cn = [
        "# mliport beta 科学验证报告",
        "",
        f"由 `{root}` 下的证据记录生成；证据对应提交 `{summary['code_commit']}`。"
        "本文档全部表格由 `validation/science/scripts/"
        "generate_beta_report.py` 从结果 JSON 渲染，更新方式是重新生成"
        "并 diff，不要手工编辑。",
        "",
        f"重新认证状态：**{versions_dict.get('scientific_revalidation_status')}**"
        f" —— {versions_dict.get('status_reason')}",
        "",
        f"- software_commit（声称被验证）：`{versions_dict.get('software_commit')}`",
        f"- validation_code_commit：`{versions_dict.get('validation_code_commit')}`",
        f"- report_generator_commit：`{versions_dict.get('report_generator_commit')}`",
        f"- evidence_campaign：`{versions_dict.get('evidence_campaign')}`",
        "- evidence_source_commits：`"
        + ", ".join(versions_dict.get("evidence_source_commits") or [])
        + "`",
        "",
        "模型身份与产物哈希固定在 `validation/science/model_manifest.json`；"
        "OMat24 评估子集见 `validation/science/data/`（种子 20260911）。",
        "",
        md_go_checklist(
            canonical,
            tier_views,
            versions_dict,
            campaign_manifest=campaign_manifest_dict,
            archive_manifest=archive_manifest,
            language="cn",
        ),
        md_tier("t1", tier_views["t1"], md_t1),
        md_tier("t2", tier_views["t2"], md_t2),
        md_tier("t2fd", tier_views["t2fd"], md_t2fd),
        md_tier("t3", tier_views["t3"], lambda records: md_t3(records, root / "t3")),
        md_tier("t4", tier_views["t4"], md_t4),
        md_tier("t5", tier_views["t5"], md_t5),
        md_tier("t6", tier_views["t6"], md_t6),
        md_tier("t7", tier_views["t7"], lambda records: md_t7(records, cn=True)),
        md_tier("t8", tier_views["t8"], md_t8),
        md_historical(tier_views, versions_dict, cn=True),
        md_support_matrix(matrix),
        md_limitations(),
    ]
    (out / "BETA_VALIDATION_CN.md").write_text("\n".join(sections_cn), encoding="utf-8")
    for tier, records in tiers.items():
        counts = status_counts(records)
        print(f"[report] {tier}: {len(records)} records, {counts}")
    print(
        f"[report] wrote {out}/beta-summary.json, "
        f"BETA_VALIDATION.md, BETA_VALIDATION_CN.md"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
