"""On-disk cache for the fit-independent part of an MSD analysis.

Refitting only ``--fit-start-ps``/``--fit-stop-ps`` must not repeat trajectory
loading, drift preparation, the FFT estimator and the local-exponent analysis.
:func:`compute_msd_core` output is stored here under a key that contains the
*content* identity of the source (trajectory SHA-256 plus loader sidecars), the
trajectory-interpretation options, the drift policy, the estimator options and
the package versions, but not the fit window.

The format is plain NumPy ``.npz`` plus JSON (no pickle): a modified
trajectory, changed drift settings or changed estimator options give a
different key, and a missing/corrupt/mismatching entry is simply a cache miss.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np

#: Bump when the core's contents or the algorithms behind it change.
MSD_CORE_REVISION = 1
MSD_CORE_SCHEMA = "mliport.msd-core/1"
_MANIFEST = "core.json"
_ARRAYS = "core.npz"


class _Unsupported(TypeError):
    pass


def _encode(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.generic):
        return _encode(value.item(), arrays)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise _Unsupported(f"unsupported array dtype {value.dtype}")
        key = f"a{len(arrays)}"
        arrays[key] = np.ascontiguousarray(value)
        return {"__ndarray__": key}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(item, arrays) for item in value]}
    if isinstance(value, list):
        return [_encode(item, arrays) for item in value]
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            if any(key.startswith("__") for key in value):
                raise _Unsupported("reserved dict key")
            return {key: _encode(item, arrays) for key, item in value.items()}
        return {
            "__dict_items__": [
                [_encode(key, arrays), _encode(item, arrays)]
                for key, item in value.items()
            ]
        }
    raise _Unsupported(f"unsupported type {type(value).__name__}")


def _decode(value: Any, arrays: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item, arrays) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__ndarray__"}:
            return np.array(arrays[value["__ndarray__"]])
        if set(value) == {"__tuple__"}:
            return tuple(_decode(item, arrays) for item in value["__tuple__"])
        if set(value) == {"__dict_items__"}:
            return {
                _decode(key, arrays): _decode(item, arrays)
                for key, item in value["__dict_items__"]
            }
        return {key: _decode(item, arrays) for key, item in value.items()}
    return value


def msd_core_key(identity: dict[str, Any]) -> str:
    """Stable key for a canonical, JSON-serialisable identity description."""

    canonical = json.dumps(
        {"schema": MSD_CORE_SCHEMA, "revision": MSD_CORE_REVISION, **identity},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MsdCoreCache:
    """Directory-backed store for MSD cores, keyed by :func:`msd_core_key`."""

    def __init__(self, root: Path, key: str) -> None:
        self.root = Path(root)
        self.key = key
        self.directory = self.root / key

    def load(self) -> dict[str, Any] | None:
        """Cached core, or ``None`` on any miss or integrity problem."""

        try:
            manifest = json.loads((self.directory / _MANIFEST).read_text("utf-8"))
            if (
                manifest.get("schema") != MSD_CORE_SCHEMA
                or manifest.get("revision") != MSD_CORE_REVISION
                or manifest.get("key") != self.key
            ):
                return None
            npz = self.directory / _ARRAYS
            if manifest.get("arrays_sha256") != _sha256(npz):
                return None
            with np.load(npz, allow_pickle=False) as arrays:
                return _decode(manifest["core"], arrays)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def store(self, core: dict[str, Any]) -> bool:
        """Persist ``core`` atomically; ``False`` if it cannot be cached."""

        try:
            arrays: dict[str, np.ndarray] = {}
            encoded = _encode(core, arrays)
        except _Unsupported:
            return False
        staging = self.root / f".tmp-{self.key}-{uuid.uuid4().hex[:8]}"
        try:
            staging.mkdir(parents=True, exist_ok=False)
            np.savez(staging / _ARRAYS, **arrays)
            manifest = {
                "schema": MSD_CORE_SCHEMA,
                "revision": MSD_CORE_REVISION,
                "key": self.key,
                "arrays_sha256": _sha256(staging / _ARRAYS),
                "core": encoded,
            }
            (staging / _MANIFEST).write_text(
                json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
            )
            if self.directory.exists():
                shutil.rmtree(self.directory, ignore_errors=True)
            os.replace(staging, self.directory)
            return True
        except OSError:
            return False
        finally:
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "MSD_CORE_REVISION",
    "MSD_CORE_SCHEMA",
    "MsdCoreCache",
    "msd_core_key",
]
