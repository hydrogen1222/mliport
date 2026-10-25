"""Scientific and checkpoint revisions for the confsearch workflow.

Any change to a scientific default, classification rule, fingerprint definition
or ranking rule must bump :data:`CS_SCIENTIFIC_REVISION`; a checkpoint written
by an older revision is then refused on resume instead of silently mixing two
definitions.  A change to the checkpoint file layout bumps
:data:`CS_CHECKPOINT_SCHEMA_REVISION`.
"""

from __future__ import annotations

#: confsearch spec revision 2 (PR #61) is the specification implemented here.
CS_SPEC_REVISION = 2

#: Bumped whenever a scientific rule or default changes.
CS_SCIENTIFIC_REVISION = 1

#: Bumped whenever the checkpoint JSON layout changes.
CS_CHECKPOINT_SCHEMA_REVISION = 1

__all__ = [
    "CS_CHECKPOINT_SCHEMA_REVISION",
    "CS_SCIENTIFIC_REVISION",
    "CS_SPEC_REVISION",
]
