# Geometry relaxation

`mliport opt` minimizes the forces on the atoms — and optionally the stress
on the cell — by iteratively stepping downhill on the potential energy surface
until the maximum force drops below a convergence threshold.

## The physics

A geometry relaxation solves:

$$\min_{\{\mathbf{r}_i\}} E(\{\mathbf{r}_i\})$$

subject to the constraint that $|\mathbf{F}_i| = |\nabla_i E| < f_\text{max}$
for every atom $i$. The optimizer computes forces at each step and proposes a
new set of positions that should lower the energy. This is a **local**
minimization: it finds the nearest minimum from the starting geometry, not the
global one.

With `--cell-opt`, the cell vectors are also relaxed using the stress tensor,
which lets the volume and shape adjust. This needs a model that provides stress,
a fully periodic structure, and a bulk task.

## Command line

```bash
mliport opt structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --fmax 0.05 --max-steps 50 --output runs/opt
```

### Cell optimization

```bash
mliport opt structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --fmax 0.02 --cell-opt --output runs/cell-opt
```

`--cell-opt` is strict: it requires stress support, full periodicity, and a
bulk task. For an isolated molecule, partial periodicity, or a model without
stress, it stops with an explanation instead of quietly relaxing positions
only.

### With different backends

```bash
# GRACE
mliport opt structure.cif --model saved_model \
  --model-type grace --fmax 0.03 --max-steps 200 \
  --output runs/grace-opt

# DPA with cell optimization
mliport opt structure.cif --model DPA-3.1-3M.pt \
  --model-type dpa --head Omat24 --device cuda \
  --fmax 0.02 --cell-opt --output runs/dpa-opt
```

## Python API

```python
from mliport.api import run_optimization

results = run_optimization(
    structure="structure.cif",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    fmax=0.02,
    max_steps=200,
    cell_opt=True,
    output_dir="runs/opt",
)
print(f"Converged: {results['converged']}")
print(f"Steps: {results['nsteps']}")
print(f"Final energy: {results['energy']:.4f} eV")
```

The function accepts an ASE `Atoms` object or a file path. It returns a
dictionary with `energy`, `forces`, `converged`, `nsteps`, and the relaxation
trajectory.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--fmax` | 0.05 eV/Å | Force convergence threshold |
| `--max-steps` | 500 | Hard step limit |
| `--optimizer` | `FIRE` | Optimization algorithm: `FIRE`, `BFGS` or `LBFGS` |
| `--cell-opt` | off | Also relax the cell; needs stress, full PBC, and a bulk task |
| `--fix-symmetry` | off | Preserve crystal symmetry during the relaxation |
| `--model`, `--model-type`, `--task`, `--head`, `--device` | — | Model and backend selection, as for a [single point](single-point.md) |

### Choosing an optimizer

- **FIRE** (default): robust and simple; good for structures far from
  equilibrium. Uses velocity Verlet with adaptive time stepping.
- **BFGS**: a quasi-Newton method that builds an approximate Hessian.
  Converges faster when close to the minimum, but can struggle with noisy
  force fields.
- **LBFGS**: limited-memory BFGS, uses less RAM than BFGS for large systems.

## Output files

| File | Contents |
|---|---|
| `CONTCAR` | The final (relaxed) structure |
| `OUTCAR` | VASP-compatible trace of energy and forces at each step |
| `OSZICAR` | One-line-per-step energy record (like VASP's ionic steps) |
| `mliport_results.json` | Convergence status, step count, final energy and forces |
| `resolved_config.json` | Complete resolved configuration |
| `run.log` | Human-readable log |

## Convergence and exit codes

| Outcome | Exit code | `converged` field |
|---|---|---|
| Forces below `--fmax` within `--max-steps` | `0` | `true` |
| Step limit reached before convergence | `2` | `false` |
| Runtime error or bad configuration | `1` | — |

When the step limit is reached, the partial trajectory is preserved and the
reason is recorded. The structure left behind should be inspected before use —
it is a relaxation that **did not converge**, not a successful one.

## What the result means

The relaxed structure is a minimum of **this model's** potential energy
surface, not a DFT minimum. It is meaningful only within one model identity
and one reference. If you used `--cell-opt`, the reference volume has
changed, so check the final cell before comparing energies with other
calculations.

A common workflow is to relax with MLIP, then run a DFT single point on the
MLIP-relaxed structure to get a DFT energy at the MLIP geometry. This is
cheaper than a full DFT relaxation, but the geometry is only as good as the
potential.
