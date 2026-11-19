# Literature benchmarks

mliport ships a literature benchmark suite that measures every engine against
published DFT reference data (LiPS-25, LiPS20, OMat24, JARVIS-DFT, and local
MS25 imports). It extends the existing scientific validation harness with one
tier, `t9`, and reuses its result schema, evidence aggregator and backend
isolation.

Two questions stay separate. `status=pass` means the calculation ran through
mliport correctly and produced finite values; it is not an accuracy verdict.
The accuracy is the recorded error metrics, always labelled with the
comparison mode.

## Quick start

```bash
python validation/science/scripts/benchmark.py \
  --tier core --datasets lips25,lips20,omat24 \
  --engine mace --profile-id mace_omat \
  --model /path/to/mace-omat-0-medium.model \
  --manifest validation/science/model_manifest.json \
  --out .validation-work/literature
```

The command imports the pinned reference data into the local cache, evaluates
the model on the core selection, aggregates the evidence and writes
`literature-report.md`, `literature-report_CN.md` and
`literature-summary.json` into `<out>`.

## Datasets and licences

| Dataset | Reference | Reference level | Labels used | Data licence |
| --- | --- | --- | --- | --- |
| LiPS-25 | Fragapane & Deringer, JCTC 2026 | PBEsol / VASP | energy, forces, formation energy | unclear (no upstream LICENSE) |
| LiPS20 | Bihani et al., Digital Discovery 2024 | PBE-D3 / CP2K | energy, forces | unclear (empty LICENCE) |
| OMat24 | Meta FAIR validation subset | PBE(+U) / VASP | energy, forces | CC-BY-4.0 |
| JARVIS-DFT | NIST jdft_3d | OptB88vdW / VASP | energy per atom | CC-BY-4.0 |
| MS25 | Maxson et al., JCIM 2025 | per subset | energy, forces, stress | local import only |

Datasets whose data licence is unclear are downloaded at run time and never
committed to the repository. The authoritative provenance (DOI, upstream
commit, per-file SHA-256, DFT settings, units) lives in
`validation/science/literature/registry.json`.

Only OMat24 is `reference_matched`: the checkpoints were trained on it, so
absolute energy errors are interpretable. LiPS-25, LiPS20 and JARVIS-DFT are
`cross_method`, so the suite reports forces, within-group relative energies,
formation energies and rankings, and marks the absolute energy block as a
diagnostic. A stored virial is never converted into stress unless the dataset
documents a stress label.

## Tiers

| Tier | Literature selection | Additional harness tiers |
| --- | --- | --- |
| `core` | deterministic small selection per dataset | none, CI-friendly |
| `standard` | larger deterministic selection | inference, relax/EOS, short NVE drift |
| `research` | full selection | `t1`, `t2`, `t2fd`, `t4`, `t5`, `t6`, `t8`; T7 transport is a separate two-phase manual task |

Every selection is recorded in `resolved.json` and in each
`data/<dataset>/selection.json` together with the seed, the rule and the
upstream file hashes, so a reported number can be traced back to the exact
structures that produced it.

## Reading the results

- `E/atom` is the direct total-energy difference; compare it only for
  `reference_matched` datasets.
- Relative energy removes the group mean on both sides and measures
  structural or thermal energy differences within one composition.
- Force blocks report component MAE/RMSE, per-atom vector P95/P99, direction
  angles, strata by reference force magnitude and per-element errors.
- Formation energy follows the published LiPS-25 protocol, including the
  phase ordering and the convex-hull agreement.
- Stress is reported as unavailable when no verified DFT stress label exists.
- Blocked datasets and unsupported properties are listed with their reason;
  they are never silently dropped or faked.
- `--max-structures N` caps a smoke run; the record's `n_structures`,
  `sampling.n_structures` and the per-structure CSV all describe the
  structures that were actually evaluated.

## Data, cache and offline use

```bash
python validation/science/scripts/benchmark.py \
  --tier core --datasets lips25 --import-only \
  --engine mace --profile-id mace_omat --model /path/to/weights \
  --manifest validation/science/model_manifest.json \
  --cache /data/mliport-literature --offline \
  --out .validation-work/literature-lips25
```

The cache defaults to `~/.cache/mliport/literature` and can be set with
`--cache` or `MLIPORT_LITERATURE_CACHE`. `--offline` reuses only verified
cache entries and fails with the command that would populate the cache. A
hash mismatch deletes the file and aborts with the expected and actual
digests.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| missing cache entry with `--offline` | run once without `--offline`, or point `--cache` at an existing cache |
| pinned sha256 mismatch | a stale or partial download was deleted; rerun to fetch it again |
| hundreds of eV in `E/atom` | cross-method offset, not an error; read the relative-energy and force blocks |
| `Stress: n/a` | the dataset has no verified stress label; a virial is not converted |
| no backend virtualenv | pass `--venv`, set `MLIPORT_SCIENCE_VENV_<ENGINE>`, or create `.venv-<engine>` |

## See also

- The full guide and registry documentation: [literature suite README](https://github.com/hydrogen1222/mliport/blob/main/validation/science/literature/README.md)
- The existing scientific validation tiers: [validation](validation.md)
- Citation and licence details: [citations](citations.md)
