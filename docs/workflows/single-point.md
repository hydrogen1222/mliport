# Single point (SP)

**Purpose.** Evaluate energy, forces and (when the model provides it) the
stress tensor for one structure without changing positions or the cell.

**Minimal command.**

```bash
.venv-mace/bin/mliport sp structure.cif \
  --model mace-omat-0-medium.model --model-type mace --output runs/sp
```

**Important options.**

| Option | Meaning |
|---|---|
| `--model`, `--model-type` | model path and explicit backend (required unless an alias/profile supplies them) |
| `--task` | UMA task family or the explicit `bulk`/`molecule` PBC semantic |
| `--head` | DPA branch / MACE head when the checkpoint needs one |
| `--device` | `cpu`, `cuda`, `cuda:N`; the requested and actual device are both recorded |
| `--charge`, `--spin` | molecular electronic-state metadata (UMA `omol`) |
| `--write-outcar`, `--write-stress` | output selection |
| `--inference-mode`, `--dtype` | backend-specific precision/inference options |

**Inputs.** One ASE-readable structure (CIF, XYZ, POSCAR, ...), a model
checkpoint and an explicit backend. The structure must match the backend's
task semantics (periodic → `omat`/`bulk`, isolated molecule → `omol`/`molecule`).

**Outputs.** `OUTCAR`, `OSZICAR`, `resolved_config.json`, the result JSON,
`arrays.npz` (energy/forces/stress), and the copied structure. See
[outputs.md](../outputs.md).

**Failure semantics.** Failure to load the model, a multi-task checkpoint
without `--head`, a task/backend mismatch or a runtime error exits non-zero
with the reason; mliport never substitutes a fallback model or a default
backend. `stress` is only present if the loaded checkpoint exposes it.

**Scientific caveats.** A single point is only comparable to another single
point with the same model identity (model, head, dtype, task) and the same
structure/electronic state. The absolute energy is not a physical observable
across models. See [models.md](../models.md).

**Example.**

```bash
.venv-dpa/bin/mliport sp structure.cif --model DPA-3.1-3M.pt \
  --model-type dpa --head Omat24 --device cuda:0 \
  --output runs/dpa-sp
```
