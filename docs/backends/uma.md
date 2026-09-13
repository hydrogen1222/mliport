# UMA (FAIRChem)

UMA checkpoints from Meta FAIR are loaded through the upstream `fairchem-core` stack. `uma` and the upstream package alias `fairchem` are the same runtime; `omol`/`oc20`/`omat` task families are selected with `TASK`/`--task`.

## Install environment

| Environment | Python | Runtime stack |
|---|---|---|
| `.venv` | 3.11, 3.12 | `fairchem-core==2.21.0` (torch 2.8.0) |

```bash
./scripts/install_mliport.sh --engines uma --device auto
```

## Supported profiles

| Profile | Identity | dtype | task | head | model SHA-256 |
|---|---|---|---|---|---|
| `uma_omat` | UMA-s 1.2 (OMat task head) | upstream/model-defined | omat | None | `ba5c0d912efa...` |

## Minimal command

```bash
.venv/bin/mliport sp structure.cif --model /path/to/uma.pt --model-type uma --task omat
```

## Precision

The dtype is part of the profile identity. This backend's precision is model-defined; mliport records the effective dtype from the backend, never a value it assumes.

## Stress

mliport reports stress only if the loaded checkpoint exposes it (`implemented_properties`). Never assume stress support from the engine name: the T1 evidence records `stress_supported` per profile, and `mliport doctor` reports what the current model provides.

## Head / task semantics

`TASK` selects the UMA task family (`omat` for periodic bulk, `omol` for molecules, `oc20` for catalysis). There is no separate `HEAD` field for UMA.

## Model acquisition

Download UMA checkpoints from the upstream fairchem/UMA release channels and verify the SHA-256 before use. mliport never redistributes model weights; see [citations](../citations.md).

## V100 capability evidence

No packaged capability evidence for this backend yet; see the validation report for the current campaign records.

## License and citation

mliport is MIT-licensed; the upstream model/software must be cited separately. See [citations](../citations.md) for the authoritative upstream references.

## Known limitations

- UMA task families are not interchangeable: a molecule run must use an `omol` task and a periodic bulk run an `omat` task.
- Upstream precision is model-defined; mliport reports the framework version rather than pretending to control model weights.
- `fairchem` is an alias, not a second backend.
