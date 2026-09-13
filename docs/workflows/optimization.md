# Relaxation / optimization (OPT)

**Purpose.** Minimise forces (and optionally the cell) with an ASE optimizer
through the selected backend.

**Minimal command.**

```bash
.venv-mace/bin/mliport opt structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --fmax 0.05 --max-steps 50 --output runs/opt
```

**Important options.**

| Option | Meaning |
|---|---|
| `--fmax` | force convergence threshold in eV/Å (default 0.05) |
| `--max-steps` | hard step limit (default 500) |
| `--optimizer` | `FIRE` (default), `BFGS`, `LBFGS` |
| `--cell-opt` | also relax the cell — requires stress support from the model |
| `--fix-symmetry` | preserve crystal symmetry during the relaxation |
| `--model`, `--model-type`, `--task`, `--head`, `--device` | model/backend selection as in [single-point.md](single-point.md) |

**Inputs.** A structure, a model, and the same explicit backend selection as
SP. `--cell-opt` requires a model whose `has_stress` is true; otherwise the
run fails closed instead of silently doing a fixed-cell optimisation.

**Outputs.** `CONTCAR` (final structure), `OUTCAR`/`OSZICAR` (energy/force
trace), `XDATCAR` (trajectory), `resolved_config.json`, result JSON with
`converged`, `nsteps`, final energy/forces, and a failure reason when not
converged.

**Failure semantics.**

- `converged: false` (step limit reached) exits with the dedicated
  non-converged status, keeps the partial trajectory and records
  `failure_reason`. It is **not** a successful relaxation.
- A backend/runtime error exits non-zero; partial results are retained.

**Scientific caveats.** Relaxation is only meaningful within one model
identity and one reference. A relaxed structure is a minimum of *that*
potential energy surface, not a DFT minimum. Cell optimisation changes the
reference volume; compare energies only after checking the final cell.

**Example.**

```bash
.venv-grace/bin/mliport opt structure.cif --model saved_model \
  --model-type grace --fmax 0.03 --max-steps 200 --output runs/grace-opt
```
