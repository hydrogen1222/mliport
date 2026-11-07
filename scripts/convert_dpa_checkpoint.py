"""Convert a TRUSTED DeepMD PyTorch training checkpoint for safe inference.

DeepMD 3.1.3 uses torch.load(weights_only=True) even for user-supplied .pt files.
Some training checkpoints contain numpy.float64 objects in _extra_state and
optimizer metadata, which PyTorch's restricted unpickler rejects.

This offline, explicit-opt-in utility removes the optimizer and converts only
NumPy scalar values in the model state to equivalent Python scalars. It NEVER
changes tensor values or dtype, and proves that the result loads with
weights_only=True. Untrusted pickle files MUST NOT be passed to this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(value: Any, *, torch: Any, numpy: Any, changes: list[str], path: str):
    """Keep tensor objects; replace numpy scalars without changing numeric values."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, numpy.generic):
        normalized = value.item()
        if not isinstance(normalized, int | float | bool | complex | str | bytes):
            raise TypeError(f"Unsupported NumPy scalar at {path}: {type(value)!r}")
        changes.append(f"{path}: {type(value).__name__} -> {type(normalized).__name__}")
        return normalized
    if value is None or isinstance(value, str | bytes | bool | int | float | complex):
        return value
    if isinstance(value, dict | OrderedDict):
        if not all(isinstance(key, str | int) for key in value):
            raise TypeError(f"Unsupported mapping key at {path}")
        return type(value)(
            (key, _normalize(item, torch=torch, numpy=numpy, changes=changes,
                             path=f"{path}.{key}"))
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        items = [
            _normalize(item, torch=torch, numpy=numpy, changes=changes,
                       path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
        return tuple(items) if isinstance(value, tuple) else items
    raise TypeError(
        f"Unsupported value in model state at {path}: {type(value)!r}; "
        "refusing an implicit conversion"
    )


def _tensors(value: Any, prefix: str = ""):
    """Visit all named tensor leaves, including nested model state."""
    import torch  # noqa: PLC0415

    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for name, child in value.items():
            yield from _tensors(child, f"{prefix}.{name}")
    elif isinstance(value, tuple | list):
        for idx, child in enumerate(value):
            yield from _tensors(child, f"{prefix}[{idx}]")


def convert_checkpoint(source: Path, target: Path, *, trusted: bool) -> dict[str, Any]:
    """Write a weights-only-compatible .pt and its separate provenance JSON.

    Unsafe pickle unpickling is allowed ONLY when the caller passes trusted=True.
    The original file is never modified; existing output files are refused.
    """
    if not trusted:
        raise PermissionError(
            "Refusing unsafe checkpoint deserialization without explicit trust. "
            "Only convert checkpoints from a verified trusted source."
        )
    if source.suffix.lower() != ".pt" or target.suffix.lower() != ".pt":
        raise ValueError("Source and target must both have .pt extensions")
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == target.resolve():
        raise ValueError("Never overwrite the original DPA training checkpoint")
    provenance_file = target.with_suffix(".conversion.json")
    if target.exists() or provenance_file.exists():
        raise FileExistsError("Refusing to overwrite converted model or provenance")
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    # This is intentionally the only use of weights_only=False in the utility.
    # An untrusted pickle can execute arbitrary code during this operation.
    original = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(original, dict) or not isinstance(original.get("model"), dict):
        raise ValueError("Expected a DeepMD training checkpoint with a model mapping")
    model = original["model"]
    extra = model.get("_extra_state")
    if not isinstance(extra, dict) or not isinstance(extra.get("model_params"), dict):
        raise ValueError("Missing DeepMD model._extra_state.model_params")

    changes: list[str] = []
    converted = _normalize(model, torch=torch, numpy=np, changes=changes, path="model")
    before = dict(_tensors(model))
    after = dict(_tensors(converted))
    if not before or before.keys() != after.keys():
        raise ValueError("Model tensor names changed or no model tensors found")
    for name in before:
        a, b = before[name], after[name]
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise ValueError(f"Tensor identity changed during conversion: {name}")

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        torch.save({"model": converted}, temp)
        # The same restricted unpickler that DeePMD uses must accept our output.
        verified = torch.load(temp, map_location="cpu", weights_only=True)
        tensors = dict(_tensors(verified["model"]))
        if tensors.keys() != before.keys() or any(
            before[name].dtype != tensors[name].dtype
            or before[name].shape != tensors[name].shape
            or not torch.equal(before[name], tensors[name])
            for name in before
        ):
            raise ValueError("The saved checkpoint did not preserve model tensors")
        record = {
            "schema": "mliport.dpa-checkpoint-conversion/1",
            "source_path": str(source.resolve()),
            "source_sha256": _sha256(source),
            "output_path": str(target.resolve()),
            "output_sha256": _sha256(temp),
            "torch_version": torch.__version__,
            "tensor_count": len(before),
            "tensor_dtypes": sorted({str(t.dtype) for t in before.values()}),
            "numeric_conversion": changes,
            "dropped_top_level_keys": sorted(str(k) for k in original if k != "model"),
            "verification": "weights_only=True reload + tensor names/dtypes/shapes/values equality",
            "warning": "Only verified the serialization contract, not DPA predictions or energy references",
        }
        # Publish both only after all checks pass. They contain no model tensors.
        # On an error, leave no file that could be mistaken for verified output.
        os.replace(temp, target)
        try:
            provenance_file.write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return record
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original trusted DeepMD training .pt")
    parser.add_argument("target", type=Path, help="New inference-compatible .pt")
    parser.add_argument(
        "--i-trust-this-file", action="store_true",
        help="Required: allow unsafe pickle deserialization of this specific source",
    )
    args = parser.parse_args()
    details = convert_checkpoint(
        args.source, args.target, trusted=args.i_trust_this_file
    )
    print(json.dumps(details, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
