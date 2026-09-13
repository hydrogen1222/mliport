# GRACE

GRACE models are exported SavedModels loaded through `tensorpotential`. `MODEL_PATH` points at the SavedModel directory, not a single file.

## Install environment

| Environment | Python | Runtime stack |
|---|---|---|
| `.venv-grace` | 3.10, 3.11, 3.12 | `tensorpotential==0.6.0` (tensorflow 2.20.0) |

```bash
./scripts/install_mliport.sh --engines grace --device auto
```

## Supported profiles

| Profile | Identity | dtype | task | head | model SHA-256 |
|---|---|---|---|---|---|
| `grace_omat` | GRACE-2L-OMAT-medium-base (upstream default precision build) | upstream/model-defined | bulk | None | `d497af4c7d9a...` |
| `grace_omat_fp64` | GRACE-2L-OMAT-medium-base fp64 build | float64 | bulk | None | `53b945efce37...` |

## Minimal command

```bash
.venv-grace/bin/mliport sp structure.cif --model /path/to/saved_model --model-type grace
```

## Precision

The dtype is part of the profile identity. This backend's precision is model-defined; mliport records the effective dtype from the backend, never a value it assumes.

## Stress

mliport reports stress only if the loaded checkpoint exposes it (`implemented_properties`). Never assume stress support from the engine name: the T1 evidence records `stress_supported` per profile, and `mliport doctor` reports what the current model provides.

## Head / task semantics

GRACE models do not expose mliport-level heads; `TASK` is the explicit PBC semantic (`bulk`/`molecule`).

## Model acquisition

Download official GRACE exported SavedModels from the upstream AMS-ICAMS-RUB release channels; the model path is the SavedModel directory.

## V100 capability evidence

| evidence id | dtype | head | backend | workloads |
|---|---|---|---|---|
| `v100-grace-87e72b50a2eb-native-cacheTrue` | ['float32'] | None | 0.6.0 ({'tensorflow': '2.20.0'}) | energy_force_consistency:passed, neb:passed, shared_calculator:passed, short_md:passed, single_point:passed |

## License and citation

mliport is MIT-licensed; the upstream model/software must be cited separately. See [citations](../citations.md) for the authoritative upstream references.

## Known limitations

- `MODEL_PATH` must be the SavedModel directory.
- TensorFlow GPU visibility must be isolated per process (the queue worker and installer do this); do not override it inside a run.
- Stress support depends on the checkpoint.
