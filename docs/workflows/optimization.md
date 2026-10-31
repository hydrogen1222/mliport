# Relaxation

`mliport opt` minimizes the forces on the atoms, and optionally the stress on
the cell, with an ASE optimizer running on whichever backend you chose.

```bash
.venv-mace/bin/mliport opt structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --fmax 0.05 --max-steps 50 --output runs/opt
```

## Options you will use

| Option | Meaning |
|---|---|
| `--fmax` | force convergence threshold in eV/Å (default 0.05) |
| `--max-steps` | hard step limit (default 500) |
| `--optimizer` | `FIRE` (default), `BFGS` or `LBFGS` |
| `--cell-opt` | relax the cell too; needs a model that provides stress |
| `--fix-symmetry` | preserve the crystal symmetry during the relaxation |
| `--model`, `--model-type`, `--task`, `--head`, `--device` | model and backend selection, as for a [single point](single-point.md) |

`--cell-opt` is strict about what it accepts. It needs a model with stress
support, a fully periodic structure, and a bulk task. For an isolated molecule,
partial periodicity or a model without stress it stops with an explanation
instead of quietly relaxing the positions only.

## What you get

`CONTCAR` is the final structure. `OUTCAR` and `OSZICAR` hold the energy and
force trace, and `mliport_results.json` records whether the relaxation
converged, how many steps it took, and the final energy and forces, alongside
`resolved_config.json` and `run.log`.

## Converged or not

If the step limit is reached before `--fmax`, the run exits with status `2`,
keeps the partial trajectory and records the reason. That is a relaxation that
did not converge, not a successful one, and the structure it leaves should be
inspected before you use it. A backend or runtime error exits with `1`, keeps
the partial results and records the error in `mliport_results.json`.

## What a relaxed structure means

The result is a minimum of *this model's* potential energy surface, not a DFT
minimum, and it is only meaningful within one model identity and one reference.
Cell optimization changes the reference volume, so look at the final cell before
comparing energies.

```bash
.venv-grace/bin/mliport opt structure.cif --model saved_model \
  --model-type grace --fmax 0.03 --max-steps 200 --output runs/grace-opt
```
