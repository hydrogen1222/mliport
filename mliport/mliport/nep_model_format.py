"""Read NEP model identity without loading Calorine or executing GPUMD.

GPUMD has two distinct on-disk formats: legacy nep4 models and
format-versioned models beginning with nep <n>. The latter name the
actual potential in the separate model field. Reject unsupported potential
families before either execution runtime can treat them as energy NEP.
Only inspect the header; GPUMD remains authoritative for numerical blocks.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

_SYMBOL = re.compile(r"[A-Z][a-z]?")
_LEGACY = re.compile(r"nep([0-9]+)(?:_(zbl))?", re.IGNORECASE)
_VERSIONED_FIELDS = frozenset({
    "format_version", "model", "cutoff", "zbl", "n_max", "basis_size",
    "l_max", "ann", "energy_reference",
})


def _float_fields(values: list[str], *, name: str) -> list[float]:
    try:
        result = [float(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"Invalid {name} numerical values: {values[:4]!r}") from exc
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"Non-finite {name} numerical values")
    return result


def read_nep_model_metadata(path: str | Path) -> dict[str, Any]:
    """Return a validated header and its scientific identity fields.

    For format-version 1, a standard energy model nep is the only
    supported family. Original model bytes are never modified.
    """
    model_path = Path(path)
    try:
        with model_path.open("r", encoding="utf-8") as handle:
            first = handle.readline()
            if not first:
                raise ValueError(f"NEP model file is empty: {model_path}")
            tokens = first.strip().split()
            had_leading_blank = not tokens
            if not tokens:
                if first.startswith("\ufeff"):
                    raise ValueError("NEP model header must not start with a BOM")
                # Historical models sometimes begin with empty lines.
                first = next((s for s in handle if s.strip()), "")
                tokens = first.strip().split()
            if not tokens:
                raise ValueError(f"NEP model file is empty: {model_path}")
            versioned = tokens[0] == "nep"
            match = _LEGACY.fullmatch(tokens[0])
            if not versioned and match is None:
                raise ValueError(
                    f"Malformed NEP model header (unrecognised type) {first.strip()!r}"
                )
            if versioned and (had_leading_blank or not first.startswith("nep ") or first.startswith("\ufeff")):
                raise ValueError("Versioned NEP header must be the physical first line")
            if len(tokens) < 3:
                raise ValueError("NEP model header needs a count and element symbols")
            try:
                count = int(tokens[1])
            except ValueError as exc:
                raise ValueError("NEP model element count is not an integer") from exc
            elements = tokens[2:]
            if count < 1 or len(elements) != count:
                raise ValueError("NEP header element count does not match its symbols")
            if len(set(elements)) != count or any(
                _SYMBOL.fullmatch(x) is None for x in elements
            ):
                raise ValueError("NEP element symbols are invalid or duplicated")
            fields: dict[str, list[str]] = {}
            for raw in handle:
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                words = line.split()
                if words[0] == "@":
                    break
                key = words[0].lower()
                if versioned and key not in _VERSIONED_FIELDS:
                    raise ValueError(f"Unsupported NEP v1 header field: {key}")
                if key in fields:
                    raise ValueError(f"Repeated NEP header field: {key}")
                fields[key] = words[1:]
                if len(fields) > 32:
                    raise ValueError("NEP header contains too many fields")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read NEP model file {model_path}: {exc}") from exc

    model_family = "nep"
    format_version: int | None = None
    legacy_version = int(match.group(1)) if match else None
    if versioned:
        required = {
            "format_version", "model", "cutoff", "n_max", "basis_size",
            "l_max", "ann", "energy_reference",
        }
        if "model" in fields and fields["model"] != ["nep"]:
            raise ValueError(
                f"Unsupported NEP model type {fields['model']!r}; only 'nep' is supported"
            )
        missing = sorted(required - fields.keys())
        if missing:
            raise ValueError(f"NEP v1 model header missing fields: {missing}")
        if fields["format_version"] != ["1"]:
            raise ValueError(
                f"Unsupported NEP file format_version: {fields['format_version']!r}"
            )
        format_version = 1
        if len(fields["model"]) != 1:
            raise ValueError("NEP model field must have exactly one value")
        model_family = fields["model"][0].lower()
        if model_family != "nep":
            raise ValueError(
                f"Unsupported NEP model type {model_family!r}; only energy model 'nep' is supported"
            )

    cutoff_tokens = fields.get("cutoff")
    radial: float | None = None
    angular: float | None = None
    by_element: list[dict[str, Any]] | None = None
    max_neighbours_radial: int | None = None
    max_neighbours_angular: int | None = None
    if cutoff_tokens is not None:
        values = _float_fields(cutoff_tokens, name="cutoff")
        if versioned:
            if len(values) == 4:
                pairs = [(values[0], values[1])] * count
            elif len(values) == 2 * count + 2:
                pairs = [(values[2 * i], values[2 * i + 1]) for i in range(count)]
            else:
                raise ValueError("NEP v1 cutoff needs 4 or 2*T+2 numbers")
            neighbors = values[-2:]
            if any(x < 0 or not x.is_integer() for x in neighbors):
                raise ValueError("NEP cutoff neighbour capacities must be nonnegative integers")
            max_neighbours_radial, max_neighbours_angular = map(int, neighbors)
        else:
            if len(values) not in (2, 4):
                raise ValueError("Legacy NEP cutoff needs 2 or 4 numbers")
            pairs = [(values[0], values[1])] * count
            if len(values) == 4:
                if any(x < 0 or not x.is_integer() for x in values[-2:]):
                    raise ValueError("NEP cutoff neighbour capacities must be integers")
                max_neighbours_radial, max_neighbours_angular = map(int, values[-2:])
        if any(r <= 0 or a <= 0 for r, a in pairs):
            raise ValueError("NEP cutoffs must be positive")
        radial = max(r for r, _ in pairs)
        angular = max(a for _, a in pairs)
        by_element = [
            {"element": element, "radial_A": r, "angular_A": a}
            for element, (r, a) in zip(elements, pairs, strict=True)
        ]

    zbl: list[float] | None = None
    if "zbl" in fields:
        zbl = _float_fields(fields["zbl"], name="zbl")
        if len(zbl) not in (2, 3) or any(x < 0 for x in zbl):
            raise ValueError("Invalid NEP ZBL specification")

    energy_reference: list[float] | None = None
    if "energy_reference" in fields:
        energy_reference = _float_fields(fields["energy_reference"], name="energy_reference")
        if len(energy_reference) != count:
            raise ValueError("NEP energy_reference must have exactly one value per element")

    return {
        "model_token": tokens[0],
        "model_type": model_family,
        "legacy_model_version": legacy_version,
        "format_version": format_version,
        "element_types": elements,
        "declared_element_count": count,
        "cutoff_radial": radial,
        "cutoff_angular": angular,
        "cutoff_by_element_A": by_element,
        "max_neighbours_radial": max_neighbours_radial,
        "max_neighbours_angular": max_neighbours_angular,
        "zbl": zbl,
        "energy_reference_eV": energy_reference,
        "raw_first_line": first.strip(),
    }
