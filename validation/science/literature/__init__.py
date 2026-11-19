"""Literature-referenced DFT benchmarks for mliport (tier ``t9``).

Public API::

    from literature import datasets, importers, metrics

    registry = datasets.load_registry()
    report = datasets.fetch_dataset("lips25", cache=None)
    selection = importers.import_dataset("lips25", tier="core")

The registry (``registry.json``) is the only source of dataset provenance,
licences, units and checksums; raw data lives in the local cache and is never
committed unless the registry marks it redistributable.
"""

from __future__ import annotations

from . import datasets, importers, metrics  # noqa: F401

__all__ = ["datasets", "importers", "metrics"]
