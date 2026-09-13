"""Schema ids, status vocabulary and immutable evidence constants."""

from __future__ import annotations

BETA_VALIDATION_SUITE_REVISION = 1
RESULT_SCHEMA_V1 = "mliport.beta-validation-result/1"
RESULT_SCHEMA = "mliport.beta-validation-result/2"
RESULT_SCHEMAS = (RESULT_SCHEMA_V1, RESULT_SCHEMA)
SUMMARY_SCHEMA_V1 = "mliport.beta-validation-summary/1"
SUMMARY_SCHEMA = "mliport.beta-validation-summary/2"
MODEL_MANIFEST_SCHEMA = "mliport.beta-model-manifest/1"
DATA_MANIFEST_SCHEMA = "mliport.beta-data-manifest/1"

#: Version of the loader/aggregator interpretation logic.  Bump whenever the
#: meaning of a stored record changes (new validation rule, new roll-up rule,
#: new migration); the version is attached to every normalized record so the
#: report can state which validator interpreted the evidence.
VALIDATION_LOGIC_VERSION = 1

STATUSES = (
    "pass",
    "fail",
    "characterized",
    "insufficient_sampling",
    "unsupported",
    "blocked",
)

#: What each status means, recorded with every normalized record so a
#: downstream reader never has to guess whether e.g. ``characterized`` is
#: a success.
STATUS_SEMANTICS = {
    "pass": "acceptance criteria met",
    "fail": "acceptance criteria violated",
    "characterized": "behaviour measured and recorded; no pass/fail claim",
    "insufficient_sampling": "measurement unresolved with the available data",
    "unsupported": "property is not provided by this backend/profile",
    "blocked": "execution could not start or complete",
}

VOLATILE_RECORD_FIELDS = (
    "run_id",
    "wall_seconds",
    "peak_vram_mib",
    "generated_at",
)

ALLOWED_WRAPPER_MODULES = (
    "mliport.calculator",
    "mliport.calculators.uma",
    "mliport.calculators.mace_calc",
    "mliport.calculators.dpa_calc",
    "mliport.calculators.grace_calc",
    # Historical records produced before the rename carry the old module
    # paths; they are legitimate evidence, not a wrapper violation.
    "mlipx.calculator",
    "mlipx.calculators.mace_calc",
    "mlipx.calculators.dpa_calc",
    "mlipx.calculators.grace_calc",
)

#: Schema id prefix shared by every version of the result document; used to
#: decide "declared result record" vs "ancillary sidecar" without guessing.
RESULT_SCHEMA_PREFIX = RESULT_SCHEMA_V1.split("/")[0]

#: Rename migration (the project was previously published under the colliding
#: name ``mlipx``; see the task book section 14).  Historical archive records
#: keep their original schema-id namespace on disk and are migrated in memory
#: by the canonical loader, never rewritten in place.
LEGACY_PROJECT_NAME = "mlipx"
LEGACY_RESULT_SCHEMA_PREFIXES = ("mlipx.beta-validation-result/",)
LEGACY_SCHEMA_PREFIXES = (
    "mlipx.beta-validation-result/",
    "mlipx.beta-validation-summary/",
    "mlipx.beta-model-manifest/",
    "mlipx.beta-data-manifest/",
    "mlipx.beta-campaign/",
)

#: Fields that may carry the producing software version.  New records write
#: ``mliport_version``; historical records carry ``mlipx_version``.
VERSION_FIELDS = ("mliport_version", "mlipx_version")
