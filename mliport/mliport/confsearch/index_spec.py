"""Adsorbate index specifications for confsearch.

``cs_adsorbate_indices`` is a comma-separated list of 0-based indices and
inclusive ranges, for example ``300-341`` or ``0,5,7-9``.  It is 0-based like
ASE; CP2K/VMD users count from 1, which is called out in the docs and in the
error messages.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mliport.confsearch.errors import ConfsearchError

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Hard cap on the number of indices one range may expand to before the atom
#: count is known; guards against ``0-999999999`` typo bombs.
MAX_RANGE_SPAN = 1_000_000

_SINGLE = re.compile(r"\d+")
_RANGE = re.compile(r"(\d+)\s*-\s*(\d+)")
_NEGATIVE = re.compile(r"-\s*\d+")


def parse_index_spec(
    spec: str | int | Iterable[int],
    *,
    n_atoms: int | None = None,
    label: str = "cs_adsorbate_indices",
) -> tuple[int, ...]:
    """Parse an index specification into a sorted, duplicate-free tuple.

    Args:
        spec: ``"0,5,7-9"`` (string), a single index, or an iterable of indices.
        n_atoms: When given, every index must be a valid 0-based atom index.
        label: Option name used in error messages.

    Raises:
        ConfsearchError: on malformed syntax, negative or reversed ranges,
            duplicates within one specification, or out-of-range indices.
    """
    if isinstance(spec, str):
        indices = _parse_string(spec, label=label)
    elif isinstance(spec, int) and not isinstance(spec, bool):
        indices = (spec,)
    else:
        try:
            raw = list(spec)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ConfsearchError(
                f"{label} must be a string like '0,5,7-9', got {spec!r}"
            ) from exc
        indices = tuple(_require_int(value, label=label) for value in raw)

    if not indices:
        raise ConfsearchError(f"{label} selects no atoms")

    if n_atoms is not None:
        out_of_range = sorted({index for index in indices if index >= n_atoms})
        if out_of_range:
            raise ConfsearchError(
                f"{label} contains indices outside the {n_atoms}-atom structure: "
                f"{out_of_range} (the specification is 0-based, like ASE; "
                "CP2K/VMD indices count from 1)"
            )
        negative = sorted({index for index in indices if index < 0})
        if negative:
            raise ConfsearchError(
                f"{label} contains negative indices {negative}; atom indices are "
                "0-based and must be >= 0"
            )

    unique = sorted(set(indices))
    if len(unique) != len(indices):
        duplicates = sorted({index for index in indices if indices.count(index) > 1})
        raise ConfsearchError(
            f"{label} repeats indices {duplicates}; list each atom once"
        )
    return tuple(unique)


def _require_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfsearchError(f"{label} must contain integers, got {value!r}")
    return value


def _parse_string(spec: str, *, label: str) -> tuple[int, ...]:
    text = spec.strip()
    if not text:
        raise ConfsearchError(f"{label} is empty")
    indices: list[int] = []
    for token in text.split(","):
        item = token.strip()
        if not item:
            raise ConfsearchError(
                f"{label}={spec!r} has an empty entry (check the commas)"
            )
        if _SINGLE.fullmatch(item):
            indices.append(int(item))
            continue
        if _NEGATIVE.fullmatch(item):
            raise ConfsearchError(
                f"{label}={spec!r}: negative index {item!r}; atom indices are "
                "0-based and must be >= 0 (CP2K/VMD count from 1)"
            )
        match = _RANGE.fullmatch(item)
        if match is None:
            raise ConfsearchError(
                f"{label}={spec!r}: cannot parse {item!r}; expected an index or an "
                "inclusive range such as '7-9'"
            )
        start, end = int(match.group(1)), int(match.group(2))
        if end < start:
            raise ConfsearchError(
                f"{label}={spec!r}: reversed range {item!r}; write it as "
                f"'{end}-{start}'"
            )
        if end - start + 1 > MAX_RANGE_SPAN:
            raise ConfsearchError(
                f"{label}={spec!r}: range {item!r} spans more than "
                f"{MAX_RANGE_SPAN} atoms"
            )
        indices.extend(range(start, end + 1))
    return tuple(indices)


def format_index_spec(indices: Iterable[int]) -> str:
    """Render indices as a compressed ``"a,b-c"`` specification."""
    ordered = sorted(set(int(index) for index in indices))
    if not ordered:
        return ""
    parts: list[str] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    parts.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(parts)


__all__ = ["MAX_RANGE_SPAN", "format_index_spec", "parse_index_spec"]
