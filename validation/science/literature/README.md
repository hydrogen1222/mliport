# Literature benchmark suite (tier `t9`)

Reproducible accuracy measurements of the mliport engines against **published
DFT reference data**. This is an extension of the existing scientific
validation harness, not a second harness: it reuses `scripts/common.py`,
`scripts/engines.py`, the `evidence/` layer, the result schema and the
aggregator, and it adds one tier (`t9`), a registry, importers and a report.

The suite answers two different questions and never mixes them:

| Question | Where it is answered |
| --- | --- |
| Does mliport drive the backend correctly? | `status` + `metrics.software_verification` of every t9 record |
| How accurate is the model against DFT? | the recorded error metrics, with an explicit comparison mode |

`status = pass` means *execution integrity*: every requested reference
structure was evaluated through the mliport calculator and produced finite
values. It is **not** an accuracy verdict, and no universal MLIP error
threshold is invented anywhere in this suite.

## Datasets

The registry `registry.json` is the single source of truth for provenance,
licences, DFT settings, units, checksums and selection rules.

| id | Reference | DOI | Reference level | Labels used | Licence of the data |
| --- | --- | --- | --- | --- | --- |
| `lips25` | LiPS-25 (Fragapane & Deringer, JCTC 2026) | 10.1021/acs.jctc.5c02006 | PBEsol / VASP 6.4.3 | energy, forces, tie-line formation energies | no LICENSE file in the upstream repository → **download only, never committed** |
| `lips20` | LiPS20 / EGraFFBench (Bihani et al., Digital Discovery 2024) | 10.1039/D4DD00027G | PBE-D3 / CP2K | energy, forces (virial recorded, not used) | upstream LICENSE file is empty → **download only, never committed** |
| `omat24` | OMat24 validation subset (Meta FAIR) | — | PBE(+U) / VASP | energy, forces | CC-BY-4.0, subset already committed by this harness |
| `jarvis` | JARVIS-DFT 3D (`jdft_3d`) | 10.6084/m9.figshare.6815699 | OptB88vdW / VASP | energy per atom | CC-BY-4.0 |
| `ms25` | MS25 (Maxson et al., JCIM 2025) | 10.1021/acs.jcim.5c01262 | per subset (paper) | energy, forces, stress (if supplied) | download blocked at retrieval time (Materials Data Facility / Globus) → **local import only** |

`omat24` is not re-downloaded: the existing committed
`validation/science/data/omat24_subset.extxyz` (256 structures, seed
20260911) is reused, and the importer refuses to run if its hash changed.

## Scientific rules encoded here

1. **Reference-matched vs cross-method.** `omat24` is `reference_matched`:
   the checkpoints were trained on OMat24, so absolute energy errors are
   interpretable. `lips25` (PBEsol), `lips20` (PBE-D3) and `jarvis`
   (OptB88vdW) are `cross_method`: absolute total energies contain an
   arbitrary offset (pseudopotentials + functional), so the suite reports
   *forces*, *within-group relative energies*, *formation energies* and
   *rankings*, and labels the absolute energy block as a diagnostic.
   Cross-method absolute energies from different datasets are never averaged
   into a single number.
2. **A virial is not a stress.** LiPS-25 and LiPS20 both publish a `virial`
   array whose sign/normalisation convention is not documented in the paper
   or the repository. The importers hash and record it as provenance and set
   `reference_virial_used_as_stress: false`; `metrics.stress` is reported as
   `available: false`. A stress MAE is only produced when a dataset carries a
   documented stress label (MS25 local imports can).
3. **No invented labels.** A missing force/stress label becomes
   `{"available": false, "reason": ...}`. The 10 unlabelled LiPS-25
   `task1_unrelaxed` structures are geometry-only; they are relaxed with ASE
   `ExpCellFilter` + `BFGS` and compared against the DFT-relaxed geometry.
4. **Group means, not hidden offsets.** Relative energies subtract the mean
   per-atom energy of the *same* group (composition/formula/trajectory) on
   both the reference and the model side. No reference value ever enters a
   model prediction.
5. **Formation energies follow the published protocol.** LiPS-25 Eq. 1:
   `Ef/atom = (E_cell/(7-4x) - x·E_Li2S^fu - (1-x)·E_P2S5^fu)/(7-4x)`, with
   both end points taken from the same energy set. A regression test asserts
   the end points are exactly zero and the reconstruction reproduces the
   published hull.
6. **Relaxation uses upstream ASE algorithms** (no custom optimizer), with a
   recorded `fmax`, step budget and convergence flag per structure. A budget
   that runs out is reported as `converged: false`, never hidden.

## Quickstart

```bash
# 1. import the pinned references into the local cache (no model needed)
python validation/science/scripts/benchmark.py \
  --tier core --datasets lips25,lips20,omat24 \
  --engine mace --profile-id mace_omat \
  --model /path/to/mace-omat-0-medium.model \
  --manifest validation/science/model_manifest.json \
  --out .validation-work/literature --import-only

# 2. run the full core tier (import + t9 evaluation + aggregate + report)
python validation/science/scripts/benchmark.py \
  --tier core --datasets lips25,lips20,omat24 \
  --engine mace --profile-id mace_omat \
  --model /path/to/mace-omat-0-medium.model \
  --manifest validation/science/model_manifest.json \
  --out .validation-work/literature

# 3. read the results
sed -n '1,80p' .validation-work/literature/literature-report.md
python -c "import json;d=json.load(open('.validation-work/literature/literature-summary.json'));print(d['entries'][0]['dataset'])"
```

`--max-structures N` caps a smoke run; the record's `n_structures`,
`sampling.n_structures` and the per-structure CSV all describe the structures
that were actually evaluated. Post-processing exit codes are collected in
`<out>/postprocess.json`; a failed aggregate/report makes the command exit
non-zero without touching the stored t9 records.

The engine is always started through its own virtualenv (`--venv`, or
`MLIPORT_SCIENCE_VENV_<ENGINE>`, or `<repo>/.venv-<engine>`), exactly as
`run_suite.py` already does. The orchestrator itself imports no backend.

## Tiers

| Tier | Literature data | Existing harness tiers |
| --- | --- | --- |
| `core` | 10 relaxed crystals, 4 melt–quench snapshots, 24 test structures, 80 LiPS20 frames, 24 JARVIS structures, 32 OMat24 structures | none (CI-friendly) |
| `standard` | larger deterministic selections (96 LiPS-25 test structures, 320 LiPS20 frames, 120 JARVIS structures, all 256 OMat24) | `t1` inference, `t4` relax/cellrelax/EOS, `t6` NVE drift |
| `research` | full selections (384 LiPS-25 test structures, 800 LiPS20 frames, 640 JARVIS structures) | `t1`, `t2`, `t2fd`, `t4`, `t5`, `t6`, `t8` with their own budgets; **T7 transport is a separate manual two-phase task** (see below) |

T7 is not launched by `benchmark.py`. It needs one MD phase in the engine
virtualenv and one analysis phase (kinisi, Nernst-Einstein, GEMDAT) in the
analysis environment, and `run_suite.py` has no analysis-venv channel. The
orchestrator therefore writes both copy-pasteable phases to
`resolved.json → manual_tasks.t7`, prints a notice, and never reports the
research tier as fully green. `MLIPORT_SCIENCE_VENV_ANALYSIS` names the
analysis interpreter; without it the template keeps an explicit placeholder.

The exact selections are resolved from the registry, written to
`<out>/resolved.json`, and repeated in each `<out>/data/<dataset>/selection.json`
with the seed, the rule, the chosen source indices, the upstream commit and
the SHA-256 of every contributing file.

## Engines and capability gating

`t9` loads every model through `engines.build_engine` → the product
`CalculatorFactory`, exactly like the rest of the harness, for the four pinned
engines (`uma`, `mace`, `dpa`, `grace`). Nothing is silently substituted:

* `--dtype` is only interpreted by MACE; the other engines record
  `upstream/model-defined`.
* `--head` selects the DPA branch / UMA-MACE head and is recorded.
* NEP is **not** part of this harness: no NEP profile is pinned in
  `model_manifest.json`, its ASE and native-GPUMD runtimes are separate
  products, and `literature_eval.py` rejects `--engine nep` before loading
  anything. That is an `unsupported`/`blocked` capability, never a failed
  scientific test.
* A missing backend virtualenv produces a `blocked` record from the
  orchestrator; a model that cannot be loaded is a `blocked` execution, not a
  scientific difference.
* Stress, NPT, NEB and dispersion support remain the capability limits the
  existing harness already enforces; `t9` only measures what the calculator
  actually exposes.

## Data management

* Cache: `--cache`, else `$MLIPORT_LITERATURE_CACHE`, else
  `~/.cache/mliport/literature`. Raw upstream files and imported extxyz are
  **never** committed (`.gitignore` does not need to know about them because
  they live outside the repository by default).
* Every file is verified against the registry SHA-256 after download and on
  every reuse. A mismatch deletes the file and aborts with the expected and
  actual hashes.
* `--offline` never downloads: it reuses verified cache entries or fails with
  the command that would populate the cache.
* Downloads retry three times, write to a process-unique `.part` file and only
  then replace the target.
* A dataset that cannot be acquired is recorded as a `blocked` result record
  with the reason, never as an empty dataset.

## Outputs

| Path | Content |
| --- | --- |
| `<out>/resolved.json` | fully resolved configuration: tier plan, selections, engine venv, exact commands |
| `<out>/data/<dataset>/selection.json` | provenance of the imported reference (hashes, rules, seed, commit) |
| `<out>/data/<dataset>/fetch.json` | download/reuse status of every upstream file |
| `<out>/data/<dataset>/*.extxyz` | normalised reference structures with reference labels |
| `<out>/t9/*.json` | one `mliport.beta-validation-result/2` record per dataset×split×profile |
| `<out>/t9/t9_<dataset>_<split>_<engine>.csv` | per-structure errors |
| `<out>/summary.json` | canonical aggregate of every record (existing aggregator) |
| `<out>/literature-summary.json` | machine-readable literature report |
| `<out>/literature-report.md`, `literature-report_CN.md` | human-readable report (EN / 中文) |

## Reading the report

* **E/atom** is the direct difference of total energies per atom. Read it as
  an accuracy number only when the mode is `reference_matched`; otherwise it
  is labelled as a diagnostic containing an arbitrary cross-method offset.
* **Relative E** is the group-mean-removed per-atom energy error; it measures
  whether the model reproduces structural/thermal energy differences inside
  one composition.
* **Force** blocks report component MAE/RMSE, per-atom vector-norm P95/P99,
  direction angles above 0.1 eV/Å, error strata by reference force magnitude
  and per-element breakdowns.
* **Formation energy** blocks report RMSE/MAE against the LiPS-25 protocol,
  Spearman ρ of the phase ordering and which phases lie on the convex hull
  for the reference and for the model.
* **Stress** is `available: false` for every dataset without a verified
  stress label, with the reason recorded.
* **Limitations** lists what the numbers cannot support (cross-method bias,
  missing stress, per-atom migration identity, ...).

## MS25 and other user-supplied references

MS25 could not be downloaded: its DOI resolves to a Materials Data Facility
landing page whose download requires the MDF/Globus client, and the DataCite
metadata carries no rights statement. The registry records this as
`access: blocked`, and the only supported path is an explicit local manifest:

```json
{
  "schema": "mliport.literature-local-manifest/1",
  "dataset": "ms25",
  "subset": "mgo_surfaces",
  "license": {"id": "user-supplied", "redistributable": false},
  "reference_level": "PBE (VASP)",
  "dft": {"software": "VASP 6.4.3", "functional": "PBE", "cutoff_eV": 500},
  "comparison_mode": "cross_method",
  "units": {"energy": "eV", "forces": "eV/Angstrom", "stress": null},
  "files": [{"path": "mgo_surfaces.xyz", "sha256": "<64 hex>"}]
}
```

```bash
python validation/science/scripts/benchmark.py --tier core --datasets ms25 \
  --reference-manifest /data/ms25/manifest.json \
  --engine mace --profile-id mace_omat --model ... \
  --manifest validation/science/model_manifest.json --out .validation-work/ms25
```

The importer verifies every hash before reading a single frame, requires the
DFT metadata (so an unlabelled comparison cannot happen) and refuses files
without a `SinglePointCalculator` energy label.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `DatasetAccessError: ... is not available in the local cache ... --offline` | run once without `--offline`, or point `--cache` at an existing cache |
| `... does not match the pinned sha256` | upstream changed or a partial download was interrupted; the file is deleted automatically, rerun to fetch it again |
| `--tiers t9 requires --dataset and --split` | `run_suite.py` is the low-level entry; use `benchmark.py`, or pass both flags plus `--subset` |
| `E/atom MAE` looks enormous (hundreds of eV) | that is a `cross_method` absolute-energy offset, not an error; read the relative-energy and force blocks |
| `Stress: n/a` | the dataset has no verified stress label; a virial is not converted |
| `converged: false` in a relaxation record | the recorded step budget ran out; the number is reported, not silently extended |
| no backend venv | `--venv`, `MLIPORT_SCIENCE_VENV_<ENGINE>` or `<repo>/.venv-<engine>` |

## Adding a dataset

1. Add one entry to `registry.json`: citation/DOI, licence (paper **and**
   data, with the evidence you checked), provenance commit/version, DFT
   settings, units (including whether stress is really labelled), splits with
   per-file SHA-256, tier selections and the training-overlap assessment.
2. Add an importer function in `importers.py` that writes normalised extxyz
   plus `selection.json`, and register it in `import_dataset`.
3. Add a CPU test with a synthetic fixture (no network) and, when the licence
   allows redistribution, a small committed known-answer file; otherwise keep
   the data in the cache.
4. Run `python validation/science/scripts/benchmark.py --tier core
   --datasets <id> --import-only ...` and `pytest
   tests/mliport/mliport/test_literature_benchmark.py -q`.

## What this suite does not claim

* It does not rank the engines: different reference levels, datasets and
  selections are reported side by side, never merged into one score.
* It does not claim statistical convergence of the MD-derived properties; the
  kinetic tasks of LiPS-25 (conductivity, Arrhenius) remain a research
  extension, and the existing `t7` analysis suite is the place for transport
  measurements.
* It does not redistribute datasets whose licence is unclear.
