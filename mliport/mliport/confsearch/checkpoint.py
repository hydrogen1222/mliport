"""Atomic checkpoints, resume fingerprint and incomplete-attempt archiving.

Reuses the NEB helpers (``atomic_write_json``, ``canonical_fingerprint``) so the
durability and identity contracts are the same ones the rest of mliport already
relies on.  Spec section 11.2: a checkpoint is written after the start-quench
stage and after every (replica, cycle) unit, and resuming re-runs the
interrupted unit from the checkpointed RNG state.  Raw files of an interrupted
attempt are moved to ``incomplete_attempt_N/`` and never deleted
(AGENTS.md rule 11).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mliport.confsearch.errors import ConfsearchPreparationError
from mliport.confsearch.revisions import (
    CS_CHECKPOINT_SCHEMA_REVISION,
    CS_SCIENTIFIC_REVISION,
)
from mliport.neb.io import atomic_write_json, canonical_fingerprint

if TYPE_CHECKING:
    from typing import Any

CHECKPOINT_SCHEMA = "mliport.confsearch-checkpoint/1"
RESUME_FINGERPRINT_SCHEMA = "mliport.confsearch-resume-fingerprint/1"


@dataclass
class CheckpointData:
    """Everything needed to resume a search exactly where it stopped."""

    unit_index: int = 0
    stage: str = "prepared"
    replicas: list[dict] = field(default_factory=list)
    rng_states: dict[str, Any] = field(default_factory=dict)
    registry: dict = field(default_factory=dict)
    discovery: list[dict] = field(default_factory=list)
    baseline: dict = field(default_factory=dict)
    cumulative_md_ps: float = 0.0
    quench_index: int = 0

    def to_dict(self) -> dict:
        return {
            "schema": CHECKPOINT_SCHEMA,
            "checkpoint_schema_revision": CS_CHECKPOINT_SCHEMA_REVISION,
            "scientific_revision": CS_SCIENTIFIC_REVISION,
            "unit_index": int(self.unit_index),
            "stage": self.stage,
            "replicas": self.replicas,
            "rng_states": self.rng_states,
            "registry": self.registry,
            "discovery": self.discovery,
            "baseline": self.baseline,
            "cumulative_md_ps": float(self.cumulative_md_ps),
            "quench_index": int(self.quench_index),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CheckpointData:
        return cls(
            unit_index=int(payload.get("unit_index", 0)),
            stage=str(payload.get("stage", "prepared")),
            replicas=list(payload.get("replicas") or []),
            rng_states=dict(payload.get("rng_states") or {}),
            registry=dict(payload.get("registry") or {}),
            discovery=list(payload.get("discovery") or []),
            baseline=dict(payload.get("baseline") or {}),
            cumulative_md_ps=float(payload.get("cumulative_md_ps", 0.0)),
            quench_index=int(payload.get("quench_index", 0)),
        )


def build_resume_fingerprint(
    *,
    options,
    model: Mapping[str, Any] | None,
    input_hashes: Mapping[str, Any],
    regions: Mapping[str, Any],
    seed: int | None,
    mliport_version: str,
) -> tuple[dict[str, Any], str]:
    """Identity of a search: everything except pure output/verbosity settings.

    ``cs_cycles`` is deliberately included: ``extend`` is the only supported way
    to change the budget, and it records the extension instead of pretending the
    fingerprint is unchanged.
    """
    import dataclasses  # noqa: PLC0415

    options_dict = {
        item.name: _jsonable(getattr(options, item.name))
        for item in dataclasses.fields(options)
        if not item.name.startswith("_")
    }
    value = {
        "schema": RESUME_FINGERPRINT_SCHEMA,
        "software": {
            "mliport_version": mliport_version,
            "cs_scientific_revision": CS_SCIENTIFIC_REVISION,
            "cs_checkpoint_schema_revision": CS_CHECKPOINT_SCHEMA_REVISION,
        },
        "model": dict(model or {}),
        "input_hashes": dict(input_hashes),
        "regions": dict(regions),
        "seed": seed,
        "options": options_dict,
    }
    return canonical_fingerprint(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def compare_fingerprints(
    recorded: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
    """Return the differing key paths between two resume fingerprints."""
    differences: list[str] = []

    def walk(left: Any, right: Any, path: str) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right)):
                walk(left.get(key), right.get(key), f"{path}.{key}")
            return
        if left != right:
            differences.append(f"{path}: recorded {left!r} != current {right!r}")

    walk(recorded, current, "")
    return differences


class ConfsearchCheckpointStore:
    """Write/read the single ``checkpoints/confsearch.json`` of one run."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.directory = self.run_dir / "checkpoints"
        self.path = self.directory / "confsearch.json"

    def write(self, data: CheckpointData, fingerprint: Mapping[str, Any]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = data.to_dict()
        payload["resume_fingerprint"] = dict(fingerprint)
        payload["resume_fingerprint_sha256"] = canonical_fingerprint(dict(fingerprint))[
            1
        ]
        atomic_write_json(self.path, payload)
        return self.path

    def load(self) -> tuple[CheckpointData, dict[str, Any]]:
        if not self.path.is_file():
            raise ConfsearchPreparationError(f"No confsearch checkpoint at {self.path}")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        recorded = payload.get("resume_fingerprint")
        if not isinstance(recorded, Mapping):
            raise ConfsearchPreparationError("Checkpoint has no resume fingerprint")
        expected = payload.get("resume_fingerprint_sha256")
        actual = canonical_fingerprint(dict(recorded))[1]
        if expected is not None and expected != actual:
            raise ConfsearchPreparationError(
                "Checkpoint resume fingerprint is corrupted (sha256 mismatch)"
            )
        return CheckpointData.from_dict(payload), dict(recorded)

    def archive_incomplete(self, *, replica: int, cycle: int) -> Path | None:
        """Move an interrupted cycle's raw directory aside; never delete it."""
        source = self.run_dir / "raw" / f"replica_{replica:03d}" / f"cycle_{cycle:03d}"
        if not source.is_dir():
            return None
        attempt = 1
        while (target := source.parent / f"incomplete_attempt_{attempt}").exists():
            attempt += 1
        source.rename(target)
        return target


__all__ = [
    "CHECKPOINT_SCHEMA",
    "RESUME_FINGERPRINT_SCHEMA",
    "CheckpointData",
    "ConfsearchCheckpointStore",
    "build_resume_fingerprint",
    "compare_fingerprints",
]
