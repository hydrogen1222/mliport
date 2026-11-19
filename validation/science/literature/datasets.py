"""Registry, cache and acquisition layer for literature DFT references.

The literature benchmark deliberately keeps **provenance** in git and
**data** out of git: the registry pins papers, DOIs, licences, DFT settings,
units and SHA-256 checksums of every upstream file, while the files themselves
are downloaded into a local cache (``MLIPORT_LITERATURE_CACHE`` or
``~/.cache/mliport/literature``) and imported into normalised extxyz there.

Design rules (see ``validation/science/literature/README.md``):

* a missing or hash-mismatched file is an explicit ``DatasetAccessError``,
  never a silent empty dataset;
* offline mode never downloads: it either reuses a verified cache entry or
  fails with the exact command that would populate it;
* only datasets whose registry entry says ``redistributable: true`` may be
  committed; everything else stays in the cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_LITERATURE_ROOT = Path(__file__).resolve().parent
_SCIENCE_ROOT = _LITERATURE_ROOT.parent
if str(_SCIENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCIENCE_ROOT))

REGISTRY_SCHEMA = "mliport.literature-registry/1"
REGISTRY_PATH = _LITERATURE_ROOT / "registry.json"
SCHEMA_PATH = _LITERATURE_ROOT / "registry.schema.json"
CACHE_ENV = "MLIPORT_LITERATURE_CACHE"
DEFAULT_CACHE = Path("~/.cache/mliport/literature")

#: Tier names shared with ``validation/science/scripts/benchmark.py``.
TIERS = ("core", "standard", "research")

#: Required keys of every dataset entry.
REQUIRED_DATASET_FIELDS = (
    "id",
    "title",
    "citation",
    "reference_level",
    "comparison_mode",
    "license",
    "provenance",
    "dft",
    "units",
    "splits",
    "data_overlap",
)

#: Allowed comparison modes (see registry ``comparison_taxonomy``).
COMPARISON_MODES = ("reference_matched", "cross_method", "unknown")


class RegistryError(RuntimeError):
    """The registry is structurally invalid or internally inconsistent."""


class DatasetAccessError(RegistryError):
    """An upstream dataset file is missing, corrupt or unverifiable."""


def registry_path() -> Path:
    return REGISTRY_PATH


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the literature registry."""
    target = Path(path) if path else REGISTRY_PATH
    try:
        registry = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        msg = f"literature registry not found: {target}"
        raise RegistryError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"literature registry is not valid JSON ({target}): {exc}"
        raise RegistryError(msg) from exc
    validate_registry(registry)
    return registry


def validate_registry(registry: dict[str, Any]) -> None:
    """Fail closed on a structurally inconsistent registry.

    Checks the schema id, the dataset fields, the comparison taxonomy, the
    ``sha256`` shape of every pinned file, and the unit/metadata requirements
    that the importer and the evaluator rely on.
    """
    problems: list[str] = []
    if registry.get("schema") != REGISTRY_SCHEMA:
        problems.append(
            f"schema must be {REGISTRY_SCHEMA!r}, got {registry.get('schema')!r}"
        )
    datasets = registry.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        problems.append("datasets must be a non-empty object")
        datasets = {}
    for key, entry in datasets.items():
        prefix = f"datasets.{key}"
        if not isinstance(entry, dict):
            problems.append(f"{prefix} must be an object")
            continue
        for name in REQUIRED_DATASET_FIELDS:
            if name not in entry:
                problems.append(f"{prefix}.{name} is required")
        if entry.get("id") != key:
            problems.append(f"{prefix}.id must equal its key")
        mode = entry.get("comparison_mode")
        if mode not in COMPARISON_MODES:
            problems.append(f"{prefix}.comparison_mode must be one of {COMPARISON_MODES}")
        license_block = entry.get("license") or {}
        for stream in ("paper", "data"):
            block = license_block.get(stream)
            if not isinstance(block, dict) or "id" not in block:
                problems.append(f"{prefix}.license.{stream}.id is required")
        data_license = license_block.get("data") or {}
        if "redistributable" not in data_license:
            problems.append(
                f"{prefix}.license.data.redistributable is required: a dataset "
                "must state explicitly whether its raw files may be committed"
            )
        units = entry.get("units") or {}
        if "energy" not in units:
            problems.append(f"{prefix}.units.energy is required")
        if "stress" not in units:
            problems.append(
                f"{prefix}.units.stress is required (use null when the dataset "
                "has no verified stress label)"
            )
        if not isinstance(entry.get("splits"), dict):
            problems.append(f"{prefix}.splits must be an object")
        # every pinned file must carry a real sha256
        for split_id, spec in (entry.get("splits") or {}).items():
            for spec_file in _iter_split_files(spec):
                sha = spec_file.get("sha256")
                if not isinstance(sha, str) or len(sha) != 64:
                    problems.append(
                        f"{prefix}.splits.{split_id}: file "
                        f"{spec_file.get('path', spec_file.get('id'))!r} lacks a 64-hex sha256"
                    )
            compositions = spec.get("compositions")
            if isinstance(compositions, dict):
                for comp, meta in compositions.items():
                    for name, sha in (meta.get("sha256") or {}).items():
                        if not isinstance(sha, str) or len(sha) != 64:
                            problems.append(
                                f"{prefix}.splits.{split_id}.{comp}.{name}: "
                                "invalid sha256"
                            )
                    for name in ("type_map_sha256", "type_sha256"):
                        sha = meta.get(name)
                        if not isinstance(sha, str) or len(sha) != 64:
                            problems.append(
                                f"{prefix}.splits.{split_id}.{comp}.{name}: "
                                "invalid sha256"
                            )
        selection = entry.get("selection") or {}
        for tier in TIERS:
            if tier not in (selection.get("tiers") or {}):
                problems.append(f"{prefix}.selection.tiers.{tier} is required")
    if problems:
        msg = "invalid literature registry:\n  - " + "\n  - ".join(problems)
        raise RegistryError(msg)


def _iter_split_files(spec: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(spec, dict):
        return ()
    files = spec.get("files")
    if isinstance(files, list):
        return [f for f in files if isinstance(f, dict)]
    return ()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_root(override: str | Path | None = None) -> Path:
    """Resolve the local dataset cache directory."""
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get(CACHE_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_CACHE.expanduser()


def dataset_cache_dir(
    registry: dict[str, Any], dataset_id: str, cache: str | Path | None = None
) -> Path:
    _require_dataset(registry, dataset_id)
    return cache_root(cache) / dataset_id


def _require_dataset(registry: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    try:
        return registry["datasets"][dataset_id]
    except KeyError as exc:
        known = ", ".join(sorted(registry.get("datasets", {})))
        msg = f"unknown literature dataset {dataset_id!r}; known: {known}"
        raise RegistryError(msg) from exc


@dataclass(frozen=True)
class FileSpec:
    """One upstream file with its pinned identity."""

    url: str
    relpath: str
    sha256: str
    bytes: int | None = None
    group: str | None = None


@dataclass
class FetchReport:
    dataset: str
    cache_dir: Path
    downloaded: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)
    offline: bool = False

    @property
    def files(self) -> int:
        return len(self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "cache_dir": str(self.cache_dir),
            "offline": self.offline,
            "files": self.files,
            "downloaded": sorted(self.downloaded),
            "reused": sorted(self.reused),
            "bytes_total": sum(int(e.get("bytes") or 0) for e in self.entries),
        }


def file_specs(registry: dict[str, Any], dataset_id: str) -> list[FileSpec]:
    """Enumerate every upstream file the importer needs for one dataset."""
    entry = _require_dataset(registry, dataset_id)
    provenance = entry.get("provenance") or {}
    prefix = provenance.get("raw_url_prefix", "")
    specs: list[FileSpec] = []
    if dataset_id == "omat24":
        return specs  # committed subset, nothing to download
    if dataset_id == "ms25":
        return specs  # local import only
    if dataset_id == "jarvis":
        file_block = provenance.get("file") or {}
        url = file_block.get("download_url")
        if not url:
            msg = "jarvis provenance.file.download_url is missing"
            raise RegistryError(msg)
        specs.append(
            FileSpec(
                url=url,
                relpath=file_block["name"],
                sha256=file_block["sha256"],
                bytes=file_block.get("bytes"),
            )
        )
        return specs
    for split_id, spec in (entry.get("splits") or {}).items():
        for spec_file in _iter_split_files(spec):
            if "path" in spec_file:
                specs.append(
                    FileSpec(
                        url=prefix + spec_file["path"],
                        relpath=spec_file["path"],
                        sha256=spec_file["sha256"],
                        bytes=spec_file.get("bytes"),
                        group=split_id,
                    )
                )
        compositions = spec.get("compositions")
        if isinstance(compositions, dict):
            for comp, meta in compositions.items():
                base = meta["path_prefix"]
                for name, sha in (meta.get("sha256") or {}).items():
                    specs.append(
                        FileSpec(
                            url=prefix + base + "set.000/" + name,
                            relpath=base + "set.000/" + name,
                            sha256=sha,
                            bytes=(meta.get("bytes") or {}).get(name),
                            group=comp,
                        )
                    )
                for name in ("type.raw", "type_map.raw"):
                    key = "type_map_sha256" if name == "type_map.raw" else "type_sha256"
                    specs.append(
                        FileSpec(
                            url=prefix + base + name,
                            relpath=base + name,
                            sha256=meta[key],
                            group=comp,
                        )
                    )
    if not specs:
        msg = f"dataset {dataset_id!r} declares no downloadable files"
        raise RegistryError(msg)
    return specs


def _download(url: str, target: Path, retries: int = 3, timeout: int = 120) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # unique temp name: two concurrent fetch processes must never share a
    # partially written file
    partial = target.with_name(f".{target.name}.{os.getpid()}.part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "mliport-literature-benchmark/1"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            partial.replace(target)
            return
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 5.0))
    msg = f"failed to download {url} after {retries} attempts: {last_error}"
    raise DatasetAccessError(msg)


def fetch_dataset(
    dataset_id: str,
    cache: str | Path | None = None,
    offline: bool = False,
    retries: int = 3,
    registry: dict[str, Any] | None = None,
) -> FetchReport:
    """Ensure every upstream file exists in the cache and matches its hash."""
    registry = registry or load_registry()
    root = dataset_cache_dir(registry, dataset_id, cache)
    specs = file_specs(registry, dataset_id)
    report = FetchReport(dataset=dataset_id, cache_dir=root, offline=offline)
    if not specs:
        return report
    for spec in specs:
        target = root / spec.relpath
        status = "missing"
        if target.exists():
            if target.stat().st_size == spec.bytes or spec.bytes is None:
                if sha256_file(target) == spec.sha256:
                    report.reused.append(spec.relpath)
                    report.entries.append(
                        {
                            "path": spec.relpath,
                            "sha256": spec.sha256,
                            "bytes": target.stat().st_size,
                            "status": "reused",
                            "group": spec.group,
                        }
                    )
                    continue
            status = "hash_mismatch"
            target.unlink()
        if offline:
            msg = (
                f"{dataset_id}: {spec.relpath} is not available in the local "
                f"cache ({target}) and --offline was requested. Run without "
                f"--offline (or fetch_dataset) to download it from "
                f"{spec.url}"
            )
            raise DatasetAccessError(msg)
        _download(spec.url, target, retries=retries)
        actual = sha256_file(target)
        if actual != spec.sha256:
            target.unlink(missing_ok=True)
            msg = (
                f"{dataset_id}: downloaded {spec.url} does not match the pinned "
                f"sha256 (expected {spec.sha256}, got {actual}); the upstream "
                "file changed or the registry is stale"
            )
            raise DatasetAccessError(msg)
        if spec.bytes is not None and target.stat().st_size != spec.bytes:
            target.unlink(missing_ok=True)
            msg = (
                f"{dataset_id}: downloaded {spec.relpath} has "
                f"{target.stat().st_size} bytes, expected {spec.bytes}"
            )
            raise DatasetAccessError(msg)
        report.downloaded.append(spec.relpath)
        report.entries.append(
            {
                "path": spec.relpath,
                "sha256": spec.sha256,
                "bytes": target.stat().st_size,
                "status": "downloaded" if status == "missing" else "redownloaded",
                "group": spec.group,
            }
        )
    return report


def write_fetch_record(report: FetchReport, root: str | Path) -> Path:
    """Persist the acquisition provenance next to the imported data."""
    target = Path(root) / "fetch.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["schema"] = "mliport.literature-fetch/1"
    payload["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def dataset_entry(registry: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    return _require_dataset(registry, dataset_id)


def tier_selection(
    registry: dict[str, Any], dataset_id: str, tier: str
) -> dict[str, Any]:
    """Resolve the deterministic selection rules for one dataset and tier."""
    entry = _require_dataset(registry, dataset_id)
    if tier not in TIERS:
        msg = f"unknown tier {tier!r}; expected one of {TIERS}"
        raise RegistryError(msg)
    selection = (entry.get("selection") or {}).get("tiers", {}).get(tier)
    if selection is None:
        msg = f"dataset {dataset_id!r} has no selection rule for tier {tier!r}"
        raise RegistryError(msg)
    return {
        "dataset": dataset_id,
        "tier": tier,
        "seed": (entry.get("selection") or {}).get("seed"),
        "rules": selection,
    }
