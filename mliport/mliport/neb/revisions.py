"""Explicit compatibility boundaries, independent of release versioning.

Increment the scientific revision for changes that alter path preparation,
forces or convergence semantics; increment the checkpoint revision when the
stored geometry/state contract changes. Neither is a FIRE-state restart.
"""

NEB_SCIENTIFIC_REVISION = 1
NEB_CHECKPOINT_SCHEMA_REVISION = 1
