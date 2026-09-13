"""Schema ids, status vocabulary and immutable evidence constants."""

from __future__ import annotations

BETA_VALIDATION_SUITE_REVISION = 1
RESULT_SCHEMA_V1 = "mlipx.beta-validation-result/1"
RESULT_SCHEMA = "mlipx.beta-validation-result/2"
RESULT_SCHEMAS = (RESULT_SCHEMA_V1, RESULT_SCHEMA)
SUMMARY_SCHEMA_V1 = "mlipx.beta-validation-summary/1"
SUMMARY_SCHEMA = "mlipx.beta-validation-summary/2"
MODEL_MANIFEST_SCHEMA = "mlipx.beta-model-manifest/1"
DATA_MANIFEST_SCHEMA = "mlipx.beta-data-manifest/1"

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
    "mlipx.calculator",
    "mlipx.calculators.mace_calc",
    "mlipx.calculators.dpa_calc",
    "mlipx.calculators.grace_calc",
)

#: Schema id prefix shared by every version of the result document; used to
#: decide "declared result record" vs "ancillary sidecar" without guessing.
RESULT_SCHEMA_PREFIX = RESULT_SCHEMA_V1.split("/")[0]
