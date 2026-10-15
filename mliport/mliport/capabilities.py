"""Workload capability claims bound to exact, reviewed validation evidence.

Two evidence layers exist:

- The full, human-reviewable validation records stay at the repository root
  (``validation/runtime/...``). They are provenance, never imported at runtime.
- The packaged curated registry (``mliport/data/validation/*.json``,
  schema ``mliport.capability-evidence/1``) carries only the fields the
  capability resolver needs. It ships inside the wheel so an installed mliport
  can resolve capabilities without a repository checkout.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    # ``Traversable`` lives in ``importlib.resources.abc`` only on Python
    # 3.11+; Python 3.10 exposes it from ``importlib.abc``.  The annotation
    # below is never evaluated at runtime (``from __future__ import
    # annotations``), so the ABC is only needed for type checkers.
    try:  # pragma: no cover - static typing only
        from importlib.resources.abc import Traversable
    except ImportError:  # pragma: no cover - Python 3.10 typing fallback
        from importlib.abc import Traversable
from importlib.resources import files as resource_files
from pathlib import Path

IDENTITY_KEYS = (
    "model_type",
    "model_sha256",
    "backend_version",
    "framework_versions",
    "task",
    "head_effective",
    "dtype_effective",
    "inference_mode",
    "compile_enabled",
    "actual_device_type",
)

CURATED_EVIDENCE_SCHEMA = "mliport.capability-evidence/1"

#: Workloads that must all be ``passed`` for a record to certify gradients.
CORE_WORKLOAD_KEYS = (
    "single_point",
    "shared_calculator",
    "energy_force_consistency",
)

#: Fields every curated evidence record must carry.
EVIDENCE_REQUIRED_KEYS = (
    "schema",
    "evidence_id",
    "source_record",
    "model_identity",
    "workloads",
)


@dataclass(frozen=True)
class CalculatorCapabilities:
    energy: bool
    forces: bool
    stress: bool
    energy_gradient_consistency: Literal["validated", "unknown", "not_guaranteed"] = (
        "unknown"
    )
    validation_evidence_id: str | None = None

    def __post_init__(self):
        if (
            self.energy_gradient_consistency == "validated"
            and not self.validation_evidence_id
        ):
            raise ValueError("Validated capabilities require an evidence ID")


def model_identity(model: dict) -> dict:
    return {key: model.get(key) for key in IDENTITY_KEYS}


def _package_evidence_dir() -> Traversable:
    # Anchor on the real ``mliport.data`` package.  ``importlib.resources.files()``
    # requires Package semantics (``__spec__.submodule_search_locations``) on
    # Python 3.10/3.11, which a plain module such as ``mliport.capabilities``
    # does not satisfy (module anchors are a Python 3.12 addition).  A true
    # package works on every supported version, for both editable and wheel
    # installs.
    return resource_files("mliport.data").joinpath("validation")


def _iter_evidence_documents(
    evidence_dir: Path | Traversable | None,
):
    """Yield ``(file_name, json_text)`` for every ``*.json`` evidence document.

    ``evidence_dir=None`` selects the packaged curated registry via
    ``importlib.resources`` (works for editable and wheel installs). An absent
    directory simply yields nothing.
    """
    directory = _package_evidence_dir() if evidence_dir is None else evidence_dir
    try:
        if not directory.is_dir():
            return
        entries = sorted(
            (entry for entry in directory.iterdir() if entry.name.endswith(".json")),
            key=lambda entry: entry.name,
        )
    except (FileNotFoundError, NotADirectoryError, ModuleNotFoundError):
        return
    for entry in entries:
        if not entry.is_file():
            continue
        try:
            yield entry.name, entry.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            warnings.warn(
                f"Ignoring unreadable validation evidence {entry.name}: {exc}",
                stacklevel=3,
            )


def _looks_like_evidence(record: object) -> bool:
    return isinstance(record, dict) and bool(
        {"schema", "evidence_id", "source_record", "model_identity", "workloads"}
        & set(record)
    )


def _warn_broken_evidence(name: str, reason: str) -> None:
    warnings.warn(
        f"Ignoring malformed validation evidence {name}: {reason}",
        stacklevel=4,
    )


def _require_valid_evidence_record(name: str, record: dict) -> None:
    """Hard-fail when a record whose identity matches the query is malformed."""
    if record.get("schema") != CURATED_EVIDENCE_SCHEMA:
        raise ValueError(
            f"Validation evidence {name} matches the requested model identity but "
            f"has schema {record.get('schema')!r}; expected "
            f"{CURATED_EVIDENCE_SCHEMA!r}. Refusing to resolve capabilities "
            "against evidence with the wrong schema (fail closed)."
        )
    missing = [key for key in EVIDENCE_REQUIRED_KEYS if key not in record]
    if missing:
        raise ValueError(
            f"Validation evidence {name} matches the requested model identity "
            f"but is missing required fields {missing} (fail closed)."
        )
    if not isinstance(record.get("evidence_id"), str) or not record["evidence_id"]:
        raise ValueError(
            f"Validation evidence {name} has a non-string or empty evidence_id "
            "(fail closed)."
        )
    workloads = record.get("workloads")
    if not isinstance(workloads, dict) or not all(
        isinstance(entry, dict) and isinstance(entry.get("status"), str)
        for entry in workloads.values()
    ):
        raise ValueError(
            f"Validation evidence {name} has malformed workloads; every entry "
            "needs a string 'status' (fail closed)."
        )


def _record_is_validated(record: dict) -> bool:
    workloads = record["workloads"]
    return all(
        workloads.get(key, {}).get("status") == "passed" for key in CORE_WORKLOAD_KEYS
    )


def _resolve_evidence(model: dict, evidence_dir: Path | Traversable | None):
    identity = model_identity(model)
    matches: list[tuple[str, dict]] = []
    for name, text in _iter_evidence_documents(evidence_dir):
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            _warn_broken_evidence(name, f"invalid JSON ({exc})")
            continue
        if not isinstance(record, dict):
            _warn_broken_evidence(
                name,
                f"top-level JSON value must be an object, got {type(record).__name__}",
            )
            continue
        if record.get("model_identity") != identity:
            # Unrelated document: warn only when it *looks* like evidence but is
            # broken, so silent data loss stays visible without failing runs
            # that merely pass by an unrelated malformed file.
            if _looks_like_evidence(record) and (
                record.get("schema") != CURATED_EVIDENCE_SCHEMA
                or not isinstance(record.get("workloads"), dict)
            ):
                _warn_broken_evidence(
                    name,
                    f"expected schema {CURATED_EVIDENCE_SCHEMA!r} with a dict "
                    "'workloads' mapping",
                )
            continue
        # The identity matches the query: schema/shape errors are hard failures.
        _require_valid_evidence_record(name, record)
        matches.append((name, record))
    if not matches:
        return None
    states = {_record_is_validated(record) for _, record in matches}
    if len(states) > 1:
        names = [name for name, _ in matches]
        raise ValueError(
            "Conflicting validation evidence for the exact model identity: "
            f"{names} disagree on whether the required workloads "
            f"{CORE_WORKLOAD_KEYS} passed (fail closed)."
        )
    if not states.pop():
        return None
    # Agreeing validated records: pick deterministically by file name.
    return "validated", matches[0][1]["evidence_id"]


def resolve_capabilities(model: dict, properties, *, evidence_dir: Path | None = None):
    state = "not_guaranteed" if model.get("direct_forces") is True else "unknown"
    evidence_id = None
    if state != "not_guaranteed":
        resolved = _resolve_evidence(model, evidence_dir)
        if resolved is not None:
            state, evidence_id = resolved
    return CalculatorCapabilities(
        "energy" in properties,
        "forces" in properties,
        "stress" in properties,
        state,
        evidence_id,
    )


def require_neb_capability(capability: CalculatorCapabilities, *, experimental: bool):
    if not capability.energy or not capability.forces:
        raise ValueError("NEB requires energy and forces")
    if capability.energy_gradient_consistency == "not_guaranteed":
        raise ValueError("NEB rejects a model with non-guaranteed energy gradients")
    if capability.energy_gradient_consistency == "unknown" and not experimental:
        raise ValueError(
            "NEB energy-gradient consistency is unvalidated for this exact model/runtime. "
            "Explicitly opt in with --allow-unvalidated-neb for an experimental run."
        )
