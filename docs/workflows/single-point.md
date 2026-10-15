# Single point

A single point evaluates the energy, the forces and, when the model provides
it, the stress tensor of one structure. Nothing moves and the cell stays as it
is. It is the cheapest way to check that a model, a structure and a device all
work together before you commit to anything longer.

```bash
.venv-mace/bin/mliport sp structure.cif \
  --model mace-omat-0-medium.model --model-type mace --output runs/sp
```

The structure can be anything ASE reads (CIF, XYZ, POSCAR, …). It has to match
what the model expects: a periodic crystal wants a bulk task (`omat` for UMA,
`bulk` elsewhere), an isolated molecule a molecular one (`omol` or `molecule`).

## Options you will use

| Option | Meaning |
|---|---|
| `--model`, `--model-type` | the checkpoint and the backend; both are required unless an alias or profile supplies them |
| `--task` | UMA task family, or the explicit `bulk` / `molecule` semantic for the other backends |
| `--head` | the DPA branch or MACE head, when the checkpoint has several |
| `--device` | `cpu` (default), `cuda`, `gpu` or `cuda:N`; the requested and the actual device are both recorded |
| `--charge`, `--spin` | electronic-state metadata for molecules (UMA `omol`) |
| `--write-outcar`, `--write-stress` | which outputs to write |
| `--inference-mode`, `--dtype` | backend-specific precision and inference options |
| `--overwrite` | allow replacing results already in the output directory |

## What you get

In the output directory: `OUTCAR`, `CONTCAR` (a copy of the input structure),
`mliport_results.json` with energy, forces and stress, `resolved_config.json`
and `run.log`. [Output files](../outputs.md) describes each one. The stress is
only present when the loaded checkpoint can compute it.

## When it fails

A model that cannot be loaded, a multi-task DPA checkpoint without `--head`, a
task that does not fit the backend, or a runtime error all end the run with a
non-zero exit code and the reason. mliport never substitutes a different model
or a default backend to get past a problem. Energy and forces must be finite;
otherwise the run aborts instead of writing NaN into the outputs.

## Reading the number

A single point can only be compared with another single point that has the same
model identity (model, head, dtype, task) and the same structure and electronic
state. Absolute energies are not physical observables across models; see
[Models](../models.md).

```bash
.venv-dpa/bin/mliport sp structure.cif --model DPA-3.1-3M.pt \
  --model-type dpa --head Omat24 --device cuda:0 \
  --output runs/dpa-sp
```
