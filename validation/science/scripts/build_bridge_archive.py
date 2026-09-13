"""Build the small exact-release bridge archive and its manifest.

The archive contains only the release-bridge campaign artifacts: the
per-engine smoke records, the campaign summary (identity/environment/config
checks) and the deterministic structure.  Model weights, long trajectories
and the full scientific campaign archive are never copied.

Usage::

    python validation/science/scripts/build_bridge_archive.py \
        --campaign-dir validation/science/bridge/<campaign> \
        --out .validation-work/archive/mliport-validation-release-bridge-<sha>.tar.zst \
        --archive-url https://github.com/hydrogen1222/mliport/releases/download/v2.0.0b3/<name>
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from evidence import (  # noqa: E402
    BridgeArtifactError,
    load_bridge_summary,
)

ARCHIVE_MANIFEST_SCHEMA = "mliport.release-bridge-archive-manifest/1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_validated_summary(campaign_dir: Path) -> dict:
    """Load the campaign summary, refusing anything that is not a pass."""
    summary_path = campaign_dir / "summary.json"
    if not summary_path.is_file():
        raise BridgeArtifactError(f"{summary_path}: missing bridge summary")
    summary = load_bridge_summary(summary_path)
    if summary.get("status") != "pass":
        raise BridgeArtifactError(
            f"{campaign_dir}: bridge status is {summary.get('status')!r}, "
            "refusing to archive a non-passing bridge"
        )
    if not summary.get("engines"):
        raise BridgeArtifactError(f"{campaign_dir}: bridge summary has no engines")
    return summary


def build(campaign_dir: Path, out: Path, archive_url: str | None) -> dict:
    campaign_dir = campaign_dir.resolve()
    summary = load_validated_summary(campaign_dir)
    engines = sorted(summary["engines"])
    with tempfile.TemporaryDirectory(prefix="mliport-bridge-") as tmp:
        staging = Path(tmp)
        for engine in engines:
            source = campaign_dir / engine / f"{engine}.json"
            if not source.is_file():
                raise BridgeArtifactError(f"{source}: missing bridge record")
            target = staging / engine
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / source.name)
        shutil.copy2(campaign_dir / "summary.json", staging / "summary.json")
        structure = campaign_dir / "structure.vasp"
        if structure.is_file():
            shutil.copy2(structure, staging / "structure.vasp")
        (staging / "README.md").write_text(
            "# mliport release-bridge archive\n\n"
            f"campaign: `{summary.get('campaign_id')}`\n"
            f"release candidate: `{summary.get('release_candidate_commit')}`\n"
            f"scientific campaign target: `{summary.get('scientific_campaign_target')}`\n\n"
            "This archive contains the four-backend release bridge smoke only "
            "(model load, single point, explicit-backend/no-backend/strict-config "
            "checks and the fairchem alias).  It is not the full scientific "
            "campaign archive and contains no long trajectories or model weights.\n",
            encoding="utf-8",
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["tar", "-I", "zstd", "-cf", str(out), "-C", str(staging), "."],
            check=True,
        )
    manifest = {
        "schema": ARCHIVE_MANIFEST_SCHEMA,
        "campaign_id": summary.get("campaign_id"),
        "release_candidate_commit": summary.get("release_candidate_commit"),
        "scientific_campaign_target": summary.get("scientific_campaign_target"),
        "archive_sha256": sha256_file(out),
        "archive_bytes": out.stat().st_size,
        "archive_url": archive_url or "(pending)",
        "records": len(engines),
        "engines": engines,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path = campaign_dir / "archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--archive-url", default=None)
    args = parser.parse_args()
    try:
        manifest = build(Path(args.campaign_dir), Path(args.out), args.archive_url)
    except (BridgeArtifactError, subprocess.CalledProcessError) as exc:
        print(f"[bridge-archive] refusing to build: {exc}", file=sys.stderr)
        return 1
    print(
        f"[bridge-archive] {args.out} sha256={manifest['archive_sha256']} "
        f"records={manifest['records']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
