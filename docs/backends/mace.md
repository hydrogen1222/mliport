# MACE

MACE checkpoints are loaded through `mace-torch`. MACE profiles are selected by checkpoint plus dtype (`float64` is the primary accuracy-first profile; `float32` is a separate identity and is never mixed with float64 results).

## Install environment

| Environment | Python | Runtime stack |
|---|---|---|
| `.venv-mace` | 3.10, 3.11, 3.12 | `mace-torch==0.3.16` (torch 2.8.0) |

```bash
./scripts/install_mliport.sh --engines mace --device auto
```

## Supported profiles

| Profile | Identity | dtype | task | head | model SHA-256 |
|---|---|---|---|---|---|
| `mace_omat` | MACE-OMAT-0 medium, float64 inference | float64 | bulk | None | `d4b14be9afa2...` |
| `mace_omat_float32` | MACE-OMAT-0 medium, float32 inference | float32 | bulk | None | `d4b14be9afa2...` |

## Minimal command

```bash
.venv-mace/bin/mliport sp structure.cif --model /path/to/mace-omat-0-medium.model --model-type mace
```

## Precision

The dtype is part of the profile identity. MACE profiles declare `float64` (primary) and `float32` (separate identity) explicitly.

## Stress

mliport reports stress only if the loaded checkpoint exposes it (`implemented_properties`). Never assume stress support from the engine name: the T1 evidence records `stress_supported` per profile, and `mliport doctor` reports what the current model provides.

## Head / task semantics

`HEAD` selects a head when the checkpoint provides several; single-head checkpoints do not need it. `TASK` is the explicit PBC semantic (`bulk`/`molecule`).

## Model acquisition

Download MACE checkpoints from the upstream MACE releases. The pinned `mace_omat` profiles use `mace-omat-0-medium.model` and are referenced by SHA-256 in `validation/science/model_manifest.json`.

## V100 capability evidence

| evidence id | dtype | head | backend | workloads |
|---|---|---|---|---|
| `v100-mace-75428afe3a1d-float32-cacheTrue` | float32 | default | 0.3.16 ({'torch': '2.8.0+cu126'}) | energy_force_consistency:passed, neb:passed, shared_calculator:passed, short_md:passed, single_point:passed |
| `v100-mace-75428afe3a1d-float64-cacheTrue` | float64 | default | 0.3.16 ({'torch': '2.8.0+cu126'}) | energy_force_consistency:passed, neb:passed, shared_calculator:passed, short_md:passed, single_point:passed |

## License and citation

mliport is MIT-licensed; the upstream model/software must be cited separately. See [citations](../citations.md) for the authoritative upstream references.

## Known limitations

- The float32 profile is a separate identity; never compare its energies with float64 results without saying so.
- Multi-head checkpoints require an explicit `HEAD`; mliport refuses to guess.
- MACE stress support depends on the checkpoint; check the T1 `stress_supported` record.
