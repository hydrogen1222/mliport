# DPA (DeepMD-kit)

DPA checkpoints are loaded through `deepmd-kit`. Multi-task checkpoints require an explicit branch `HEAD`; mliport fails closed instead of guessing a head.

## Install environment

| Environment | Python | Runtime stack |
|---|---|---|
| `.venv-dpa` | 3.10, 3.11, 3.12 | `deepmd-kit==3.1.3` (torch 2.10.0) |

```bash
./scripts/install_mliport.sh --engines dpa --device auto
```

## Supported profiles

| Profile | Identity | dtype | task | head | model SHA-256 |
|---|---|---|---|---|---|
| `dpa_omat` | DPA-3.1-3M, branch Omat24 | upstream/model-defined | bulk | Omat24 | `86dd3a804d78...` |

## Minimal command

```bash
.venv-dpa/bin/mliport sp structure.cif --model /path/to/DPA-3.1-3M.pt --model-type dpa --head Omat24
```

## Precision

The dtype is part of the profile identity. This backend's precision is model-defined; mliport records the effective dtype from the backend, never a value it assumes.

## Stress

mliport reports stress only if the loaded checkpoint exposes it (`implemented_properties`). Never assume stress support from the engine name: the T1 evidence records `stress_supported` per profile, and `mliport doctor` reports what the current model provides.

## Head / task semantics

Multi-task checkpoints expose named branches. Pass the intended branch with `--head`/`HEAD`; omitting it on a multi-task checkpoint is a fatal configuration error, never a silent default branch.

## Model acquisition

Download DPA checkpoints from the upstream DeePMD-kit/DPA release channels. Read the available branches with `dp --pt show <model> model-branch` and pass the intended branch as `HEAD`.

## V100 capability evidence

| evidence id | dtype | head | backend | workloads |
|---|---|---|---|---|
| `v100-dpa-876354744aea-native-cacheTrue` | ['float32'] | Domains_Alloy | 3.1.3 ({'torch': '2.10.0+cu126'}) | energy_force_consistency:passed, neb:passed, shared_calculator:passed, short_md:passed, single_point:passed |

## License and citation

mliport is MIT-licensed; the upstream model/software must be cited separately. See [citations](../citations.md) for the authoritative upstream references.

## Known limitations

- Multi-task checkpoints require an explicit `HEAD`; the product fails closed rather than using an arbitrary branch.
- DPA has no `inference_mode` concept; UMA-only options are rejected in strict config mode.
- Stress support depends on the checkpoint and its branch.
