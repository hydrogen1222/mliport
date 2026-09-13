"""Build a reproducible validation-evidence archive and its manifest (section 16).

The archive contains only what is needed to re-render a campaign report: the
campaign-scoped canonical result records, per-structure metric CSVs, the
pinned model/data manifests and the campaign manifest, plus a README with the
scope and the exact re-render command.  Model weights, long trajectories,
secrets and attic records are never copied.

The committed ``archive_manifest.json`` records the archive sha256 so any
clean clone can verify that an externally hosted archive is the one the
report was rendered from.  Hosting (GitHub Release / Zenodo / OSF) is
deliberately external; ``--archive-url`` records it without changing the
archive itself.

Usage::

    python build_evidence_archive.py --root .validation-work \
        --campaign <campaign-id> --out /tmp/mliport-validation-<id>.tar.zst \
        --software-commit <validated-software-sha> \
        --validation-commit <harness-sha> \
        --campaign-manifest validation/science/campaigns/<id>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCIENCE = REPO / "validation" / "science"
if str(SCIENCE) not in sys.path:
    sys.path.insert(0, str(SCIENCE))

from evidence import load_evidence  # noqa: E402

ARCHIVE_MANIFEST_SCHEMA = "mliport.beta-archive-manifest/1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_readme(
    staging: Path,
    *,
    campaign: str,
    software_commit: str,
    validation_commit: str,
    records: int,
    counts: dict,
) -> None:
    lines = [
        f"# MLIPort validation evidence archive -- {campaign}",
        "",
        f"- software_commit (validated product revision): `{software_commit}`",
        f"- validation_commit (harness): `{validation_commit}`",
        f"- records: {records} ({json.dumps(counts, sort_keys=True)})",
        "",
        "Included: canonical campaign records (`records/`), per-structure T3",
        "CSVs, pinned model/data manifests, the campaign manifest and the",
        "runtime environment. Excluded: model weights, long trajectories,",
        "attic/superseded records and secrets.",
        "",
        "Re-render the committed report from the extracted archive:",
        "",
        "```bash",
        "python validation/science/scripts/generate_beta_report.py \\",
        f"  --evidence-root <extracted-archive> --campaign {campaign} \\",
        f"  --campaign-manifest validation/science/campaigns/{campaign}.json \\",
        "  --out validation/science/reports --update-readmes",
        "```",
        "",
    ]
    (staging / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build(
    *,
    root: Path,
    campaign: str,
    out: Path,
    software_commit: str,
    validation_commit: str,
    archive_url: str | None = None,
    campaign_manifest: Path | None = None,
    manifest_out: Path,
) -> dict:
    """Create ``out`` and ``manifest_out``; returns the archive manifest."""
    bundle = load_evidence(root, campaign=campaign)
    if bundle.problems:
        raise SystemExit(
            f"refusing to archive: {len(bundle.problems)} evidence problem(s): "
            + "; ".join(bundle.problems[:3])
        )
    if not bundle.records:
        raise SystemExit(f"refusing to archive: no records for campaign {campaign!r}")

    counts: dict[str, int] = {}
    for record in bundle.records:
        status = str(record.get("status"))
        counts[status] = counts.get(status, 0) + 1

    staging = Path(tempfile.mkdtemp(prefix="mliport-evidence-"))
    try:
        for record in bundle.records:
            source = root / record["_path"]
            target = staging / "records" / record["_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for csv_path in sorted((root / "t3").glob("t3_errors_*.csv")):
            target = staging / "per_structure_metrics" / csv_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_path, target)
        manifests_dir = staging / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        for name in ("model_manifest.json", "data_manifest.json"):
            source = SCIENCE / name
            if source.is_file():
                shutil.copy2(source, manifests_dir / name)
        if campaign_manifest is not None:
            shutil.copy2(campaign_manifest, manifests_dir / "campaign_manifest.json")
        environment_dir = staging / "environment"
        environment_dir.mkdir(parents=True, exist_ok=True)
        (environment_dir / "runtime.json").write_text(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "generated_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _write_readme(
            staging,
            campaign=campaign,
            software_commit=software_commit,
            validation_commit=validation_commit,
            records=len(bundle.records),
            counts=counts,
        )

        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["tar", "-I", "zstd", "-cf", str(out), "-C", str(staging), "."],
            check=True,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    model_manifest = SCIENCE / "model_manifest.json"
    data_manifest = SCIENCE / "data_manifest.json"
    manifest = {
        "schema": ARCHIVE_MANIFEST_SCHEMA,
        "campaign_id": campaign,
        "archive_sha256": sha256_file(out),
        "archive_url": archive_url,
        "archive_bytes": out.stat().st_size,
        "software_commit": software_commit,
        "validation_commit": validation_commit,
        "model_manifest_sha256": (
            sha256_file(model_manifest) if model_manifest.is_file() else None
        ),
        "data_manifest_sha256": (
            sha256_file(data_manifest) if data_manifest.is_file() else None
        ),
        "records": len(bundle.records),
        "by_status": counts,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".validation-work")
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--software-commit", required=True)
    parser.add_argument("--validation-commit", required=True)
    parser.add_argument("--archive-url", default=None)
    parser.add_argument("--campaign-manifest", default=None)
    parser.add_argument(
        "--manifest-out", default="validation/science/archive_manifest.json"
    )
    args = parser.parse_args()
    manifest = build(
        root=Path(args.root),
        campaign=args.campaign,
        out=Path(args.out),
        software_commit=args.software_commit,
        validation_commit=args.validation_commit,
        archive_url=args.archive_url,
        campaign_manifest=(
            Path(args.campaign_manifest) if args.campaign_manifest else None
        ),
        manifest_out=Path(args.manifest_out),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
