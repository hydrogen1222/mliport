# Validation status

This page explains **what mliport's validation claims mean** and where the
evidence lives. The authoritative, generated reports are:

- `validation/science/reports/BETA_VALIDATION.md` (English, full campaign),
- `validation/science/reports/BETA_VALIDATION_CN.md` (Chinese),
- `validation/science/README_VALIDATION.md` (short generated summary).

Machine-readable evidence:

```text
validation/science/campaigns/<campaign>.json     campaign manifest (target commit, records)
validation/science/archive_manifest.json         evidence archive + checksums
validation/science/model_manifest.json           pinned model profiles
validation/science/capability_matrix.json        interface expressibility matrix
mliport/mliport/data/validation/*.json           packaged capability evidence
```

## Installer smoke evidence (historical)

This table is the pre-beta installer smoke evidence; the workflow-level beta
campaign above/below is the current evidence base. It is kept because it is
record-backed — a `passed` cell links to the runtime record that supports it.

| Backend / model | SP | Short MD | E–F gradient | NEB smoke | GRACE cache | Records |
|---|---|---|---|---|---|---|
| UMA | not run | not run | not run | not run | n/a | [uma.json](../validation/runtime/v100/uma.json) (no offline wheelhouse on this machine) |
| MACE float64 | passed | passed | passed | passed | n/a | [mace.json](../validation/runtime/v100/mace.json) |
| MACE float32 | passed | passed | passed | passed | n/a | [mace-float32.json](../validation/runtime/v100/mace-float32.json) |
| DPA `Domains_Alloy` | passed | passed | passed | passed | n/a | [dpa.json](../validation/runtime/v100/dpa.json) |
| GRACE float32 | passed | passed | passed | passed | passed (ON and OFF) | [grace.json](../validation/runtime/v100/grace.json), [grace-nocache.json](../validation/runtime/v100/grace-nocache.json) |

The NEB smoke was a short 5-image fixed-cell run on a 4-atom Cu cell with
`saddle_validation: not_performed`; a converged band is not a verified
transition state.

## Status vocabulary

| Status | Meaning |
|---|---|
| `software_validated` | mliport's software paths for the listed workflows passed on the stated runtime/model combination |
| `model_characterized` | the *model* was characterised on a benchmark; this is a property of the model, not of mliport |
| `historical_evidence` | a past campaign record kept for reference; it does not describe the current commit |
| `not_run` | the combination was not executed; no claim is made |
| `unsupported` | the backend/interface cannot express the feature; documented as such |
| `experimental` | runnable but outside the supported envelope; results may change |
| `pending_revalidation` | was covered before, but current-head evidence is not yet regenerated |

Do not translate one status into another: a `software_validated` adapter
smoke is not a model-accuracy claim, and a `model_characterized` benchmark is
not a statement about your chemistry.

## What a campaign record contains

Each record ties together: the software commit, workflow/tier, backend and
framework versions, model identity (path/task/head/dtype/SHA-256), device
(requested vs actual), status, and the artifacts/hashes. A record is only
valid for the commit and model identity it names.

## Tiers

The campaign uses tiered coverage (T1-T8 in the reports): backend/workflow
smoke and consistency checks, produced-value checks, numerical/physical
sanity checks (e.g. benchmarks), and performance/transport analyses. Read
each tier's scope in the report before quoting a number.

## Honest boundaries

- Validation covers the **listed** runtimes, GPUs and model profiles. It does
  not certify arbitrary checkpoints, hardware or chemistries.
- Performance/benchmark numbers are hardware- and version-specific and are
  reported with the commit they were measured on.
- MD-derived quantities depend on sampling; analyses record their windows and
  uncertainties.
- If evidence for a claim is missing, the status is `not_run` — never
  "assumed working".

## LGPS fresh-install acceptance

The 2.0.0b3 cycle adds a user-path acceptance run: the documented install
commands, four GPU backends, the LGPS workflow matrix, the analysis modules,
a CPU install and a final clean-room pass. The report and its machine-readable
evidence live here:

- [LGPS_ACCEPTANCE_REPORT.md](../validation/acceptance/LGPS_ACCEPTANCE_REPORT.md)
- [beta-summary.json](../validation/science/reports/beta-summary.json)
- [BETA_VALIDATION.md](../validation/science/reports/BETA_VALIDATION.md)

Installation-matrix rows for the acceptance host, the exact commands and the
remaining limitations are recorded there rather than inferred from CI.
