# Scientific Validation Harness (beta)

Reproducible GPU validation of the four production engines that mlipx
wraps (UMA, MACE-OMAT-0, DPA-3.1-3M, GRACE-2L-OMAT).  This directory holds
the harness: manifests, deterministic fixtures, per-tier scripts, JSON
schemas, and the aggregator.  **Run outputs stay out of git** — only the
harness, the pinned manifests, and the OMat24 subset provenance are
committed.

## Layout

| Path | Purpose |
| --- | --- |
| `model_manifest.json` | Pinned model identities (sha256 of the exact artifacts; no local paths) |
| `data_manifest.json` | Pinned OMat24 validation-subset provenance (corrected 241220 files) |
| `scripts/common.py` | Shared schemas, identity/UUID hashing, result records |
| `scripts/engines.py` | Engine construction through the product `CalculatorFactory` (no fallbacks) |
| `scripts/fixtures.py` | Deterministic bulk fixtures (Cu, Si, MgO + triclinic/distorted extras) |
| `scripts/inference.py` | T1 — first-call/warm-call timings, E/F/stress, VRAM |
| `scripts/invariance.py` | T2 — repeatability floor, permutation/PBC/translation/rotation invariance, A→B→A cache-state check |
| `scripts/finite_difference.py` | T2-FD — ASE native FD sweep vs analytical forces/stress |
| `scripts/transforms.py` | Rotation/Voigt algebra with hand-verified known answers |
| `scripts/build_cases.py` | Regenerates/pins `cases/manifests/fixtures.json` |
| `scripts/omat_subset.py` | OMat24 val-subset acquisition + deterministic selection |
| `scripts/run_suite.py` | Sequential per-engine orchestrator (backend venvs) |
| `scripts/aggregate.py` | Validates + aggregates records into a summary |
| `schemas/` | JSON Schemas for result records and summaries |

## Core rules encoded here

- **Calculator identity is proven, not assumed** — every record stores the
  live wrapper module/class; anything outside the mlipx wrappers aborts
  the run (`common.calculator_identity`).
- **Tolerances are never hard-coded by hand** — they are `max(10 × measured
  repeatability floor, absolute floor)`, and the policy is stored in every
  record's `parameters`.
- **A changed artifact is a new identity** — `resolve_profile` refuses to
  run on hash mismatch; the manifest must be updated deliberately.
- **Stress signs are never hand-flipped**; FD stress vs analytical stress
  is compared component-wise (normals and shears separately).
- **GPU UUIDs are hashed** before entering any record.

## Quickstart (single engine, backend venv)

```bash
python validation/science/scripts/run_suite.py \
  --tiers t1,t2,t2fd \
  --engine mace --profile-id mace_omat \
  --model models/mace/mace-omat-0-medium.model \
  --manifest validation/science/model_manifest.json \
  --out .validation-work
```

Aggregate everything:

```bash
python validation/science/scripts/aggregate.py \
  --results .validation-work \
  --manifest validation/science/model_manifest.json \
  --out .validation-work/summary.json
```

CPU-only regression tests (no backend needed):

```bash
pytest tests/mlipx/mlipx/test_science_harness.py -q
```
