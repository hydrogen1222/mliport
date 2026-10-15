# Validation

This page explains what mliport's validation claims mean, so that you can tell a
fact from a hope, and where the evidence lives. It does not repeat the numbers;
those belong to the reports and change with every campaign.

The reports are generated from the campaign records:

- [BETA_VALIDATION.md](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION.md): the full campaign, in English
- [BETA_VALIDATION_CN.md](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION_CN.md): the same in Chinese
- [README_VALIDATION.md](https://github.com/hydrogen1222/mliport/blob/main/validation/science/README_VALIDATION.md): a one-page summary

and the machine-readable evidence behind them:

```text
validation/science/campaigns/<campaign>.json     campaign manifest: target commit, records
validation/science/archive_manifest.json         evidence archive and checksums
validation/science/model_manifest.json           pinned model profiles
validation/science/capability_matrix.json        which interface can express which option
```

## What a status means

Every cell in a report carries one of these words, and they are not
interchangeable.

| Status | Meaning |
|---|---|
| `software_validated` | mliport's code paths for the listed workflow ran correctly on the stated runtime and model |
| `model_characterized` | the *model* was measured on a benchmark; that is a property of the model, not of mliport |
| `historical_evidence` | a record from an earlier campaign, kept for reference; it says nothing about the current commit |
| `not_run` | the combination was not executed, so no claim is made |
| `unsupported` | the backend or interface cannot express the feature, and the docs say so |
| `experimental` | runs, but sits outside the supported envelope and may change |
| `pending_revalidation` | covered before; evidence for the current head has not been regenerated yet |

A `software_validated` smoke test says the adapter works. It is not a statement
about accuracy, and a `model_characterized` benchmark is not a statement about
your chemistry.

## What a record contains

A record ties together the software commit, the workflow and tier, the backend and
framework versions, the model identity (path, task, head, dtype, SHA-256), the
device (requested and actual), the status, and the artifacts with their hashes.
It is valid for that commit and that model, and for nothing else.

The campaign is organised in tiers, T1 to T8 in the reports: backend and workflow
smoke tests, consistency checks, produced-value checks, numerical and physical
sanity checks such as benchmarks, and performance and transport analyses. Read
a tier's scope in the report before you quote one of its numbers.

## Installer smoke evidence

This table predates the workflow-level campaign above. It is kept because every
`passed` cell is backed by a runtime record you can open.

| Backend / model | SP | Short MD | E–F gradient | NEB smoke | GRACE cache | Records |
|---|---|---|---|---|---|---|
| UMA | not run | not run | not run | not run | n/a | [uma.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/uma.json) (no offline wheelhouse on this machine) |
| MACE float64 | passed | passed | passed | passed | n/a | [mace.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/mace.json) |
| MACE float32 | passed | passed | passed | passed | n/a | [mace-float32.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/mace-float32.json) |
| DPA `Domains_Alloy` | passed | passed | passed | passed | n/a | [dpa.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/dpa.json) |
| GRACE float32 | passed | passed | passed | passed | passed (ON and OFF) | [grace.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/grace.json), [grace-nocache.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/grace-nocache.json) |

The NEB smoke was a short 5-image fixed-cell run on a 4-atom Cu cell with
`saddle_validation: not_performed`. A converged band is not a verified
transition state.

## LGPS fresh-install acceptance

The 2.0.0b3 cycle added a run that follows the documented user path: the install
commands, four GPU backends, the LGPS workflow matrix, the analysis modules, a CPU
install and a final clean-room pass. The report and its evidence are here:

- [LGPS_ACCEPTANCE_REPORT.md](https://github.com/hydrogen1222/mliport/blob/main/validation/acceptance/LGPS_ACCEPTANCE_REPORT.md)
- [beta-summary.json](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/beta-summary.json)

The installation-matrix rows for the acceptance host, the exact commands and the
remaining limitations are recorded there rather than inferred from CI.

## Where validation stops

- It covers the runtimes, GPUs and model profiles it lists. It does not certify an
  arbitrary checkpoint, machine or chemistry.
- Performance numbers belong to the hardware and versions they were measured on,
  and are reported with the commit.
- Anything derived from MD depends on sampling. Analyses record their windows and
  uncertainties.
- Where there is no evidence, the status is `not_run`. It is never "assumed to
  work".
