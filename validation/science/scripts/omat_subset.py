"""OMat24 validation-subset acquisition and deterministic selection.

Taskbook section 6.1: a pinned ~256-structure subset of the official
OMat24 *validation* split, stratified across the dataset's own source
subdatasets, chosen with an immutable seed.  The corrected (2024-12-20)
validation files are used, avoiding the duplicate-structure defect fixed
in fairchem issue #942.

This script must run inside an environment that has fairchem-core (it
reads the official LMDB files with fairchem's own reader)::

    <venv-with-fairchem> omat_subset.py --data-dir /mnt/hdd500/omat24-val \
        --out-dir .validation-work/omat24

Outputs (selection record is committed; raw caches are not):
  - ``selection.json``   provenance + chosen indices per stratum
  - ``omat24_subset.extxyz``  the selected structures with reference labels
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import common  # noqa: E402

SEED = 20260911
#: stratum name -> (official corrected val tarball, total size)
STRATA = {
    "rattled-1000-subsampled": (
        "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000-subsampled.tar.gz",
        39785,
    ),
    "rattled-300-subsampled": (
        "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-300-subsampled.tar.gz",
        35579,
    ),
    "aimd-from-PBE-3000-npt": (
        "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-3000-npt.tar.gz",
        62130,
    ),
    "rattled-relax": (
        "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-relax.tar.gz",
        95206,
    ),
}
DEFAULT_PER_STRATUM = 64  # 4 strata x 64 = 256 structures


def sha256_ok(path: Path, expected: str | None) -> bool:
    if expected is None:
        return False
    return common.sha256_file(path) == expected


def ensure_tarball(url: str, data_dir: Path, expected_sha256: str | None) -> Path:
    name = url.rsplit("/", 1)[-1]
    path = data_dir / name
    if path.exists() and sha256_ok(path, expected_sha256):
        return path
    if path.exists():
        print(f"[omat] {name}: hash mismatch, re-downloading", file=sys.stderr)
        path.unlink()
    subprocess.run(["curl", "-sL", "--retry", "3", "-o", str(path), url], check=True)
    if expected_sha256 and not sha256_ok(path, expected_sha256):
        msg = f"downloaded {name} does not match the pinned sha256"
        raise RuntimeError(msg)
    return path


def extract(tarball: Path, data_dir: Path) -> Path:
    target = data_dir / tarball.name[: -len(".tar.gz")]
    if target.exists() and any(target.iterdir()):
        return target
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(data_dir)  # noqa: S202 - pinned upstream archive
    return target


def read_lmdb_frames(lmdb_dir: Path):
    from fairchem.core import datasets as _fc_datasets

    lmdb_files = sorted(
        list(lmdb_dir.rglob("*.aselmdb")) + list(lmdb_dir.rglob("*.lmdb"))
    )
    if not lmdb_files:
        msg = f"no .lmdb/.aselmdb files found under {lmdb_dir}"
        raise RuntimeError(msg)
    if hasattr(_fc_datasets, "AseDBDataset"):  # fairchem-core >= 2
        return _fc_datasets.AseDBDataset({"src": str(lmdb_dir)})
    return _fc_datasets.AseLMDBDataset({"src": str(lmdb_dir)})  # fairchem 1.x


def _as_atoms(item):
    """fairchem >= 2 returns AtomicData; older versions return Atoms."""
    if hasattr(item, "to_ase"):
        atoms_list = item.to_ase()
        return atoms_list[0] if isinstance(atoms_list, list) else atoms_list
    return item


def select_stratum(dataset, per_stratum: int) -> list[int]:
    """Deterministic stratified pick: shuffle indices with the immutable
    seed, take the first ``per_stratum`` -- then sort for stable storage."""
    rng = np.random.default_rng(SEED)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    return sorted(int(i) for i in indices[:per_stratum])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-stratum", type=int, default=DEFAULT_PER_STRATUM)
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parents[1] / "data_manifest.json"),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    pinned = {}
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pinned = {
            s["name"]: s.get("tarball_sha256")
            for s in data_manifest.get("sources", [])
        }

    from ase.io import write as ase_write

    selection: dict[str, dict] = {}
    all_atoms = []
    for name, (url, total) in STRATA.items():
        tarball = ensure_tarball(url, data_dir, pinned.get(name))
        lmdb_dir = extract(tarball, data_dir)
        dataset = read_lmdb_frames(lmdb_dir)
        chosen = select_stratum(dataset, args.per_stratum)
        frames = []
        for i in chosen:
            atoms = _as_atoms(dataset[int(i)])
            atoms.info["omat24_stratum"] = name
            atoms.info["omat24_index"] = int(i)
            frames.append(atoms)
        selection[name] = {
            "tarball": tarball.name,
            "tarball_sha256": common.sha256_file(tarball),
            "dataset_size": int(len(dataset)),
            "official_total": total,
            "selected_indices": chosen,
            "selected": len(frames),
        }
        all_atoms.extend(frames)
        print(f"[omat] {name}: selected {len(frames)} of {len(dataset)}")

    subset_path = out_dir / "omat24_subset.extxyz"
    ase_write(subset_path, all_atoms, format="extxyz")

    record = {
        "schema": common.DATA_MANIFEST_SCHEMA,
        "suite_revision": common.BETA_VALIDATION_SUITE_REVISION,
        "seed": SEED,
        "selection_rule": (
            "numpy default_rng(seed) shuffle of stratum indices, first "
            "per_stratum kept, sorted; strata are the official corrected "
            "OMat24 validation subdatasets (241220, fairchem issue #942)"
        ),
        "per_stratum": args.per_stratum,
        "total_selected": len(all_atoms),
        "strata": selection,
        "subset_sha256": common.sha256_file(subset_path),
    }
    (out_dir / "selection.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[omat] wrote {subset_path} ({len(all_atoms)} structures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
