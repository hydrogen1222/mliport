"""Core library of the MD-based configuration search (confsearch).

Phase P1 of the PR #61 implementation plan: options, regions, covalent graphs
and status rules, the focused fingerprint, the deduplicating registry and
automatic placement.  Nothing here loads a calculator; the search engine
(phase P2) drives these pieces.
"""

from __future__ import annotations

from mliport.confsearch.errors import (
    ConfsearchError,
    ConfsearchGradientConsistencyError,
    ConfsearchInternalError,
    ConfsearchNotImplementedError,
    ConfsearchPreparationError,
)
from mliport.confsearch.fingerprint import (
    FingerprintGrid,
    FocusedFingerprint,
    build_fingerprint,
    fingerprint_distance,
)
from mliport.confsearch.geometry import (
    mic_distance_matrix,
    mic_vectors,
    minimum_distance,
    self_image_distance,
    vacuum_gap,
)
from mliport.confsearch.graphs import (
    ClassificationReference,
    CovalentGraph,
    GraphDifference,
    MinimumClassification,
    broken_reference_bonds,
    classify_minimum,
    covalent_bonds,
)
from mliport.confsearch.index_spec import format_index_spec, parse_index_spec
from mliport.confsearch.options import CS_DEFAULTS, ConfsearchOptions, steps_from_ps
from mliport.confsearch.placement import (
    PlacementResult,
    place_molecule,
    random_rotation_matrix,
)
from mliport.confsearch.regions import (
    Regions,
    SurfaceGeometry,
    mobile_indices,
    resolve_regions,
    validate_all,
    validate_surface,
)
from mliport.confsearch.registry import (
    MinimumOrigin,
    MinimumRecord,
    MinimumRegistry,
    UniqueMinimum,
)
from mliport.confsearch.revisions import (
    CS_CHECKPOINT_SCHEMA_REVISION,
    CS_SCIENTIFIC_REVISION,
    CS_SPEC_REVISION,
)

__all__ = [
    "CS_CHECKPOINT_SCHEMA_REVISION",
    "CS_DEFAULTS",
    "CS_SCIENTIFIC_REVISION",
    "CS_SPEC_REVISION",
    "ClassificationReference",
    "ConfsearchError",
    "ConfsearchGradientConsistencyError",
    "ConfsearchInternalError",
    "ConfsearchNotImplementedError",
    "ConfsearchOptions",
    "ConfsearchPreparationError",
    "CovalentGraph",
    "FingerprintGrid",
    "FocusedFingerprint",
    "GraphDifference",
    "MinimumClassification",
    "MinimumOrigin",
    "MinimumRecord",
    "MinimumRegistry",
    "PlacementResult",
    "Regions",
    "SurfaceGeometry",
    "UniqueMinimum",
    "broken_reference_bonds",
    "build_fingerprint",
    "classify_minimum",
    "covalent_bonds",
    "fingerprint_distance",
    "format_index_spec",
    "mic_distance_matrix",
    "mic_vectors",
    "minimum_distance",
    "mobile_indices",
    "parse_index_spec",
    "place_molecule",
    "random_rotation_matrix",
    "resolve_regions",
    "self_image_distance",
    "steps_from_ps",
    "validate_all",
    "validate_surface",
    "vacuum_gap",
]
