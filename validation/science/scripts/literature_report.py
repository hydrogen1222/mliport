"""Markdown/JSON report for the literature benchmark (tier ``t9``).

Consumes only the canonical evidence layer (``evidence.load_evidence``) plus
the committed literature registry, so a report can be regenerated from saved
records without rerunning a model::

    python validation/science/scripts/literature_report.py \
        --results .validation-work/literature --out .validation-work/literature

Every number in the report is read from a stored record; nothing is
recomputed or rounded differently from the record itself.  Statuses keep their
harness semantics (``pass`` = execution criteria met, not "the model is
accurate"), and each table states whether the reference is
``reference_matched`` or ``cross_method``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_SCIENCE_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(Path(__file__).resolve().parent), str(_SCIENCE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evidence import load_evidence  # noqa: E402
from literature import datasets as literature_datasets  # noqa: E402

REPORT_SCHEMA = "mliport.literature-report/1"

#: ``status`` values that mean "no usable measurement", kept distinct from a
#: scientific difference (see constants.STATUS_SEMANTICS).
NON_MEASUREMENT_STATUSES = ("blocked", "unsupported", "fail")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int | float):
        return f"{value:.{digits}g}" if abs(value) < 1e-3 else f"{value:.4f}"
    return str(value)


def _t9_records(
    results: Path,
) -> tuple[list[dict[str, Any]], Any]:
    bundle = load_evidence(results, tiers=("t9",))
    records = [
        record
        for record in bundle.records
        if str(record.get("test_id", "")).startswith("t9_")
    ]
    return records, bundle


def _latest(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Latest run per profile for a fixed (dataset, split) test id."""
    ordered = sorted(
        records,
        key=lambda r: (str(r.get("created_at") or ""), str(r.get("run_id") or "")),
    )
    return ordered[-1]


def _dataset_split(test_id: str) -> tuple[str, str]:
    parts = test_id.split("_")
    if len(parts) < 3:
        return "unknown", "unknown"
    dataset = parts[1]
    split = "_".join(parts[2:])
    return dataset, split


def _metric(metrics: dict[str, Any], *path: str, default: Any = None) -> Any:
    node: Any = metrics
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return node


def build_summary(
    records: list[dict[str, Any]],
    registry: dict[str, Any],
    bundle: Any,
    tier: str | None = None,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        dataset, split = _dataset_split(str(record.get("test_id", "")))
        grouped.setdefault(
            (dataset, split, str(record.get("profile_id", "unknown"))), []
        ).append(record)
    entries = []
    for (dataset, split, profile_id), group in sorted(grouped.items()):
        record = _latest(group)
        metrics = record.get("metrics") or {}
        forces = metrics.get("forces") or {}
        relative = metrics.get("relative_energy") or {}
        formation = metrics.get("formation_energy") or {}
        stress = metrics.get("stress") or {}
        entries.append(
            {
                "dataset": dataset,
                "split": split,
                "profile_id": profile_id,
                "engine": record.get("engine"),
                "model_identity": record.get("model_identity"),
                "model_sha256": record.get("model_sha256"),
                "dtype": record.get("dtype"),
                "task": record.get("task"),
                "head": record.get("head"),
                "device": (record.get("device") or {}).get("actual"),
                "gpu_name": (record.get("device") or {}).get("gpu_name"),
                "status": record.get("status"),
                "status_semantics": record.get("status_semantics"),
                "comparison_mode": metrics.get("comparison_mode"),
                "reference_level": metrics.get("reference_level"),
                "n_structures": metrics.get("n_structures"),
                "n_evaluated": metrics.get("n_evaluated"),
                "n_runs": len(group),
                "run_ids": [r.get("run_id") for r in group],
                "energy_per_atom": metrics.get("energy_per_atom_eV"),
                "relative_energy": relative.get("relative_energy_eV_per_atom"),
                "formation_energy": {
                    key: formation.get(key)
                    for key in (
                        "available",
                        "rmse_eV_per_atom",
                        "mae_eV_per_atom",
                        "protocol_variant",
                    )
                }
                if formation
                else None,
                "rank_agreement": formation.get("rank_agreement"),
                "hull": formation.get("hull"),
                "forces": {
                    key: forces.get(key)
                    for key in (
                        "component_mae_eV_A",
                        "component_rmse_eV_A",
                        "vector_norm_error_eV_A",
                        "direction",
                        "by_element",
                    )
                    if key in forces
                }
                if forces
                else {"available": False},
                "stress": {
                    "available": stress.get("available"),
                    "reason": stress.get("reason"),
                    "component_mae_GPa": stress.get("component_mae_GPa"),
                },
                "labels_available_in_reference": metrics.get(
                    "labels_available_in_reference"
                ),
                "data_overlap": metrics.get("data_overlap"),
                "limitations": metrics.get("limitations"),
                "software_verification": metrics.get("software_verification"),
                "wall_seconds": record.get("wall_seconds"),
                "git_commit": record.get("git_commit"),
                "mliport_version": record.get("mliport_version"),
                "backend_version": record.get("backend_version"),
                "framework_version": record.get("framework_version"),
                "exception": record.get("exception"),
                "record_path": record.get("_path"),
            }
        )
    return {
        "schema": REPORT_SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tier": tier,
        "evidence": {
            "records": len(records),
            "problems": list(getattr(bundle, "problems", []) or []),
            "counts": getattr(bundle, "counts", {}),
        },
        "entries": entries,
        "datasets": {
            key: {
                "title": entry.get("title"),
                "doi": entry.get("doi"),
                "citation": entry.get("citation"),
                "license": entry.get("license"),
                "reference_level": entry.get("reference_level"),
                "comparison_mode": entry.get("comparison_mode"),
                "dft": entry.get("dft"),
                "units": entry.get("units"),
                "provenance": entry.get("provenance"),
                "data_overlap": entry.get("data_overlap"),
            }
            for key, entry in (registry.get("datasets") or {}).items()
        },
    }


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any], lang: str = "en") -> str:
    cn = lang == "cn"
    t = {
        "title": "文献基准测试报告" if cn else "Literature benchmark report",
        "generated": "生成时间" if cn else "Generated",
        "evidence": "证据记录" if cn else "Evidence records",
        "commit": "软件提交" if cn else "Software commit",
        "model": "模型" if cn else "Model",
        "device": "设备" if cn else "Device",
        "summary": "结果总览" if cn else "Result overview",
        "dataset": "数据集" if cn else "Dataset",
        "profile": "模型 profile" if cn else "Model profile",
        "split": "划分" if cn else "Split",
        "status": "状态" if cn else "Status",
        "n": "结构数" if cn else "n",
        "e_abs": "E/原子 MAE (eV)" if cn else "E/atom MAE (eV)",
        "e_rel": "相对能量 MAE (eV/atom)" if cn else "Relative E MAE (eV/atom)",
        "force": "力分量 MAE (eV/A)" if cn else "Force comp. MAE (eV/A)",
        "force_p95": "力矢量 P95 (eV/A)" if cn else "Force vector P95 (eV/A)",
        "formation": "形成能 RMSE (eV/atom)"
        if cn
        else "Formation energy RMSE (eV/atom)",
        "stress": "应力" if cn else "Stress",
        "mode": "比较模式" if cn else "Mode",
        "datasets": "数据集来源与许可" if cn else "Dataset provenance and licences",
        "paper": "论文" if cn else "Paper",
        "doi": "DOI",
        "license_paper": "论文许可" if cn else "Paper licence",
        "license_data": "数据许可" if cn else "Data licence",
        "redistributable": "可再分发" if cn else "Redistributable",
        "level": "参考理论水平" if cn else "Reference level",
        "units": "单位" if cn else "Units",
        "overlap": "训练集重叠风险" if cn else "Training-overlap risk",
        "dft": "DFT 设置" if cn else "DFT settings",
        "limitations": "限制与说明" if cn else "Limitations",
        "not_run": "未运行 / 阻断项" if cn else "Not run / blocked",
        "reproduce": "复现命令" if cn else "Reproduce",
        "notes": "说明" if cn else "Notes",
        "accuracy_note": (
            "`pass` 只表示执行完整性（结构已评估且数值有限），不是精度合格判定。"
            "`cross_method` 数据集不比较绝对总能量。"
            if cn
            else "`pass` means execution integrity only (every structure was evaluated "
            "with finite values), never an accuracy verdict. Absolute total energies "
            "are not compared for `cross_method` references."
        ),
        "accuracy": "绝对能量是否可比" if cn else "Absolute energy comparable",
        "yes": "是" if cn else "yes",
        "no": "否" if cn else "no",
        "na": "无" if cn else "n/a",
    }
    lines = [f"# {t['title']}", ""]
    lines.append(f"- {t['generated']}: `{summary['generated_utc']}`")
    lines.append(f"- {t['evidence']}: {summary['evidence']['records']}")
    lines.append(f"- {t['accuracy_note']}")
    entries = summary["entries"]
    if entries:
        first = entries[0]
        lines.append(
            f"- {t['model']}: `{first.get('model_identity')}` "
            f"({t['profile']} `{first.get('profile_id')}`, dtype "
            f"`{first.get('dtype')}`, task/head `{first.get('task')}/"
            f"{first.get('head')}`)"
        )
        lines.append(
            f"- {t['commit']}: `{first.get('git_commit')}` "
            f"(mliport {first.get('mliport_version')})"
        )
        lines.append(
            f"- {t['device']}: `{first.get('device')}` "
            f"{first.get('gpu_name') or ''}".rstrip()
        )
    lines.append("")
    lines.append(f"## {t['summary']}")
    lines.append("")
    header = [
        t["dataset"],
        t["split"],
        t["profile"],
        t["status"],
        t["mode"],
        t["n"],
        t["e_abs"],
        t["e_rel"],
        t["force"],
        t["force_p95"],
        t["formation"],
        t["stress"],
    ]
    rows = []
    for entry in entries:
        energy = entry.get("energy_per_atom") or {}
        relative = entry.get("relative_energy") or {}
        forces = entry.get("forces") or {}
        vectors = forces.get("vector_norm_error_eV_A") or {}
        formation = entry.get("formation_energy") or {}
        stress = entry.get("stress") or {}
        rows.append(
            [
                entry["dataset"],
                entry["split"],
                str(entry["profile_id"]),
                str(entry["status"]),
                str(entry["comparison_mode"]),
                str(entry["n_structures"]),
                _fmt(energy.get("mae")),
                _fmt(relative.get("mae")),
                _fmt(forces.get("component_mae_eV_A")),
                _fmt(vectors.get("p95")),
                _fmt(formation.get("rmse_eV_per_atom")),
                "available"
                if stress.get("available")
                else (t["no"] if stress else t["na"]),
            ]
        )
    lines.append(_table(rows, header))
    lines.append("")

    lines.append(f"## {t['datasets']}")
    lines.append("")
    for key, meta in summary["datasets"].items():
        used = sorted({e["split"] for e in entries if e["dataset"] == key})
        lines.append(f"### {meta.get('title') or key} (`{key}`)")
        lines.append("")
        lines.append(f"- {t['paper']}: {meta.get('citation')}")
        lines.append(f"- {t['doi']}: `{meta.get('doi')}`")
        license_block = meta.get("license") or {}
        lines.append(
            f"- {t['license_paper']}: `{(license_block.get('paper') or {}).get('id')}`"
        )
        data_license = license_block.get("data") or {}
        lines.append(
            f"- {t['license_data']}: `{data_license.get('id')}` "
            f"({t['redistributable']}: {_fmt(data_license.get('redistributable'))})"
        )
        if data_license.get("evidence"):
            lines.append(f"  - {data_license['evidence']}")
        lines.append(f"- {t['level']}: {meta.get('reference_level')}")
        lines.append(f"- {t['mode']}: `{meta.get('comparison_mode')}`")
        lines.append(
            f"- {t['overlap']}: `{(meta.get('data_overlap') or {}).get('status')}` "
            f"{(meta.get('data_overlap') or {}).get('note') or ''}"
        )
        units = meta.get("units") or {}
        lines.append(
            f"- {t['units']}: energy `{units.get('energy')}`; forces "
            f"`{units.get('forces')}`; stress `{units.get('stress')}`"
        )
        dft = meta.get("dft") or {}
        dft_bits = ", ".join(
            f"{name}={dft[name]}"
            for name in (
                "software",
                "functional",
                "dispersion",
                "plane_wave_cutoff_eV",
                "plane_wave_cutoff_Ry",
                "pseudopotentials",
            )
            if dft.get(name)
        )
        lines.append(f"- {t['dft']}: {dft_bits or 'see registry'}")
        if dft.get("per_entry_settings"):
            lines.append(f"  - {dft['per_entry_settings']}")
        used_text = ", ".join(used) if used else "-"
        lines.append(f"- splits: {used_text}")
        lines.append("")
        for entry in [e for e in entries if e["dataset"] == key]:
            lines.append(
                f"#### `{entry['split']}` — {entry['status']} "
                f"({entry['n_evaluated']}/{entry['n_structures']} evaluated)"
            )
            lines.append("")
            energy = entry.get("energy_per_atom") or {}
            if energy.get("available") is False:
                lines.append(f"- E/atom: n/a ({energy.get('reason')})")
            elif energy:
                lines.append(
                    f"- E/atom: MAE {_fmt(energy.get('mae'))} eV, RMSE "
                    f"{_fmt(energy.get('rmse'))} eV, P95 {_fmt(energy.get('p95'))} eV "
                    f"— {t['accuracy']}: "
                    f"{t['yes'] if entry['comparison_mode'] == 'reference_matched' else t['no']}"
                )
                if energy.get("interpretation"):
                    lines.append(f"  - {energy['interpretation']}")
            relative = entry.get("relative_energy") or {}
            if relative:
                lines.append(
                    f"- Relative energy: MAE {_fmt(relative.get('mae'))} eV/atom, "
                    f"RMSE {_fmt(relative.get('rmse'))} eV/atom, P95 "
                    f"{_fmt(relative.get('p95'))} eV/atom"
                )
            forces = entry.get("forces") or {}
            if forces.get("component_mae_eV_A") is not None:
                vectors = forces.get("vector_norm_error_eV_A") or {}
                lines.append(
                    f"- Forces: component MAE {_fmt(forces.get('component_mae_eV_A'))} "
                    f"eV/A, RMSE {_fmt(forces.get('component_rmse_eV_A'))} eV/A, "
                    f"vector P95 {_fmt(vectors.get('p95'))} eV/A, P99 "
                    f"{_fmt(vectors.get('p99'))} eV/A"
                )
                direction = forces.get("direction") or {}
                if direction.get("angle_deg_mean") is not None:
                    lines.append(
                        f"- Direction: mean angle {_fmt(direction.get('angle_deg_mean'))} deg, "
                        f"median {_fmt(direction.get('angle_deg_median'))} deg, P95 "
                        f"{_fmt(direction.get('angle_deg_p95'))} deg, "
                        f"fraction > 10 deg {_fmt(direction.get('frac_angle_above_10deg'))}"
                    )
                by_element = forces.get("by_element") or {}
                if by_element:
                    worst = sorted(
                        by_element.items(),
                        key=lambda item: item[1].get("component_mae_eV_A", 0.0),
                        reverse=True,
                    )[:5]
                    lines.append(
                        "- Per element (worst 5): "
                        + ", ".join(
                            f"{element} {_fmt(data.get('component_mae_eV_A'))}"
                            for element, data in worst
                        )
                        + " eV/A"
                    )
            elif forces.get("available") is False:
                lines.append(f"- Forces: n/a ({forces.get('reason')})")
            formation = entry.get("formation_energy") or {}
            if formation.get("available"):
                rank = entry.get("rank_agreement") or {}
                lines.append(
                    f"- Formation energy: RMSE {_fmt(formation.get('rmse_eV_per_atom'))} "
                    f"eV/atom, MAE {_fmt(formation.get('mae_eV_per_atom'))} eV/atom, "
                    f"Spearman rho {_fmt(rank.get('spearman_rho'), 4)} "
                    f"(max rank shift {_fmt(rank.get('max_rank_shift'), 3)})"
                )
                lines.append(f"  - protocol: {formation.get('protocol_variant')}")
                hull = entry.get("hull") or {}
                if hull:
                    lines.append(
                        f"  - hull agreement: {len(hull.get('agreement') or [])} shared, "
                        f"only model {hull.get('only_model')}, "
                        f"only reference {hull.get('only_reference')}"
                    )
            stress = entry.get("stress") or {}
            if stress.get("available"):
                lines.append(
                    f"- Stress: component MAE {_fmt(stress.get('component_mae_GPa'))} GPa"
                )
            elif stress:
                lines.append(f"- Stress: n/a ({stress.get('reason')})")
            for note in entry.get("limitations") or []:
                lines.append(f"- {t['notes']}: {note}")
            lines.append("")

    blocked = [
        entry for entry in entries if entry["status"] in NON_MEASUREMENT_STATUSES
    ]
    lines.append(f"## {t['not_run']}")
    lines.append("")
    if not blocked:
        lines.append("- " + ("没有阻断项。" if cn else "none"))
    else:
        for entry in blocked:
            lines.append(
                f"- `{entry['dataset']}/{entry['split']}`: {entry['status']} — "
                f"{entry.get('exception') or 'see record'}"
            )
    lines.append("")
    lines.append(f"## {t['reproduce']}")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "python validation/science/scripts/benchmark.py --tier core "
        "--datasets lips25,lips20,omat24 --engine <engine> "
        "--profile-id <profile> --model <weights> "
        "--manifest validation/science/model_manifest.json --out <workdir>"
    )
    lines.append("```")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--registry", default=str(literature_datasets.registry_path()))
    parser.add_argument("--out", required=True)
    parser.add_argument("--tier", default=None)
    parser.add_argument(
        "--lang",
        default="both",
        choices=("en", "cn", "both"),
        help="which Markdown languages to write",
    )
    args = parser.parse_args()
    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    registry = literature_datasets.load_registry(args.registry)
    records, bundle = _t9_records(results)
    summary = build_summary(records, registry, bundle, tier=args.tier)
    (out / "literature-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if args.lang in ("en", "both"):
        (out / "literature-report.md").write_text(
            render_markdown(summary, "en"), encoding="utf-8"
        )
    if args.lang in ("cn", "both"):
        (out / "literature-report_CN.md").write_text(
            render_markdown(summary, "cn"), encoding="utf-8"
        )
    print(
        f"[literature-report] {len(records)} record(s) -> "
        f"{out / 'literature-summary.json'}"
    )
    return 0 if not summary["evidence"]["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
