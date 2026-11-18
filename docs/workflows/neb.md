# NEB and CI-NEB

`mliport neb` finds a **minimum-energy path** between two endpoint structures
using the nudged elastic band (NEB) method, and reports the energy barrier
between them. The optional climbing-image (CI-NEB) variant refines the
saddle-point estimate.

## The physics

The nudged elastic band method (Henkelman et al., J. Chem. Phys. **113**,
9901, 2000) threads a chain of images between two relaxed endpoints and
optimizes them simultaneously. Spring forces keep the images evenly spaced
along the path, while the component of the true force perpendicular to the
path drives each image toward the minimum-energy path.

In **CI-NEB** (Henkelman et al., J. Chem. Phys. **113**, 9978, 2000), the
highest image is freed from the spring forces and instead climbs along the
path direction to approach the saddle point. This gives a better
barrier estimate without needing more images.

The barrier is defined as:

$$E_\text{barrier} = E_\text{saddle} - E_\text{reactant}$$

where $E_\text{saddle}$ is the energy of the highest image on the converged
band. mliport reports the **sampled** barrier (the highest image minus the
end state), not a fitted saddle.

## Command line

```bash
mliport neb --initial initial.vasp --final final.vasp \
  --model mace-omat-0-medium.model --model-type mace \
  --images 7 --climb --output runs/neb
```

You supply two structures with the same atoms in the same order (or an
explicit `--atom-map` when the order differs), a model on an explicit backend,
and enough memory for `images + 2` copies of the calculator.

### With IDPP interpolation

```bash
mliport neb --initial initial.vasp --final final.vasp \
  --model mace-omat-0-medium.model --model-type mace \
  --images 7 --climb --interpolation idpp \
  --output runs/neb-idpp
```

IDPP (image-dependent pair potential) interpolation (Smidstrup et al.,
J. Chem. Phys. **140**, 214106, 2014) builds a better initial path than
linear interpolation by minimizing changes in interatomic distances. It is
recommended for complex rearrangements where linear interpolation would cause
atoms to overlap.

## Python API

```python
from mliport.api import run_neb

results = run_neb(
    initial="initial.vasp",
    final="final.vasp",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    n_intermediate_images=7,
    climb=True,
    output_dir="runs/neb",
)
print(f"Forward barrier: {results['barrier_forward_sampled_eV']:.3f} eV")
print(f"Reverse barrier: {results['barrier_reverse_sampled_eV']:.3f} eV")
print(f"Reaction energy: {results['reaction_energy_eV']:.3f} eV")
```

## Gradient preflight: why it checks before running

A NEB only makes sense if the model's forces are the gradient of its energy.
Otherwise the band is optimized against forces that do not describe the energy
surface, and the barrier is meaningless. So mliport checks first.

On the endpoints and interpolated images, it compares the force projection
$-\mathbf{F} \cdot \hat{\mathbf{u}}$ along several directions with central
finite differences of the model energy. Two step sizes are combined by
Richardson extrapolation, and the tolerance accounts for the rounding noise
of the model's dtype. The run is refused when:

- The forces are not the gradient of the energy
- A value is not finite
- The energy resolution cannot settle the question (e.g. a large float32 cell)
- The model predicts forces directly (`direct_forces = true`; always refused)

The measurement is stored as `gradient_preflight` in `run_context.json` and
`neb_results.json`. It is a spot check, not a proof over the whole surface.

## Options

### Band setup

| Option | Default | Meaning |
|---|---|---|
| `--initial`, `--final` | *(required)* | Endpoint structures |
| `--images` | 7 | Number of intermediate images (endpoints not counted) |
| `--climb` | off | Enable climbing image for saddle refinement |
| `--interpolation` | `linear` | `linear` or `idpp` for the initial path |
| `--path-convention` | `mic` | `mic` (minimum image convention) or `unwrapped` |
| `--spring` | 0.1 eV/Å² | Spring constant between images |
| `--atom-map` | — | 0-based final indices in initial-atom order, comma-separated |
| `--image-shifts` | — | Integer lattice shifts per atom, e.g. `0,0,0;1,0,0` |

### Convergence

| Option | Default | Meaning |
|---|---|---|
| `--fmax` | 0.03 eV/Å | Force convergence threshold |
| `--max-steps` | 1000 | Step limit |
| `--pre-fmax` | 0.1 eV/Å | First-stage threshold (before CI switches on) |
| `--pre-max-steps` | 300 | First-stage step limit |
| `--maxstep` | 0.1 Å | Largest optimizer step |
| `--min-distance` | 0.5 Å | Refuse images with atoms closer than this |

### Endpoints

| Option | Default | Meaning |
|---|---|---|
| `--endpoint-policy` | `validate` | `validate` (check endpoints are relaxed) or `relax` (relax them first) |
| `--endpoint-fmax` | 0.02 eV/Å | Endpoint force threshold |
| `--endpoint-steps` | 500 | Endpoint relaxation budget (with `relax`) |

### IDPP tuning

| Option | Default | Meaning |
|---|---|---|
| `--idpp-fmax` | — | IDPP convergence threshold |
| `--idpp-steps` | — | IDPP step limit |
| `--idpp-mic` | — | Use minimum image convention for IDPP |

## Endpoint policy: validate or relax

With `--endpoint-policy validate` (default), the endpoints must already be
relaxed: if either has a force above `--endpoint-fmax`, mliport stops and says
which one. With `relax`, it relaxes both endpoints first, then builds the band.
Either way, the endpoint treatment changes your reference energy, so always
record which policy you used.

## Output files

| File | Contents |
|---|---|
| `neb_results.json` | Energies, forces, convergence, barrier numbers |
| `run_context.json` | Model identity and gradient preflight |
| `resolved_config.json` | Resolved configuration |
| `artifacts.json` | File listing with hashes |
| `checkpoints/` | Complete-band checkpoints at regular intervals |
| `vasp_path/` | One `POSCAR` per image (VASP NEB layout) |

The barrier fields in `neb_results.json`:

- `barrier_forward_sampled_eV`: highest image minus initial state
- `barrier_reverse_sampled_eV`: highest image minus final state
- `reaction_energy_eV`: final minus initial energy
- `barrier_status`: whether an interior peak was found

## Resuming

```bash
mliport neb --resume runs/neb --output runs/neb
```

A resume continues in the original run directory and checks the recorded
identity: model, task, head and dtype, cell and PBC, constraints, band setup and
options. Step budgets are part of that identity in this beta, so a resume that
changes `--max-steps` is rejected with `Resume fingerprint is incompatible`, and
the message names the fields that differ (for example
`neb_options.max_steps (checkpoint 200 != requested 123)`). Omit an option to
inherit the checkpointed value; passing the same value explicitly is allowed.
The checkpoint is left untouched when a resume is refused.

A resume restores the band geometry but restarts the optimizer; it does not
replay the previous optimizer state. Convergence and the final barrier can
differ from an uninterrupted run, so always verify convergence.

## Interpreting the barrier

- A converged NEB saddle candidate is not a verified transition state. A
  Hessian (frequency) calculation would be needed to confirm it has exactly
  one imaginary mode along the reaction coordinate.
- A barrier is meaningful only within one model identity and one NEB protocol
  (images, interpolation, endpoint policy, spring).
- The atom mapping must be chemically correct — wrong mappings produce
  spurious barriers.

## Capability matrix

Not every NEB option is available from every interface. This table is generated from `validation/science/capability_matrix.json`. 

<!-- BEGIN GENERATED: neb-capability -->
| Feature | API | direct CLI | INCAR | TUI |
|---|---|---|---|---|
| endpoints | yes | yes | yes | yes |
| atom map | yes | yes | no | yes |
| image shifts | yes | yes | no | yes |
| resume | yes | yes | no | yes |
| path convention | yes | yes | yes | yes |
| interpolation | yes | yes | yes | yes |
| method climb | yes | yes | yes | yes |
| pre relaxation | yes | yes | yes | yes |
| min distance guard | yes | yes | yes | yes |
<!-- END GENERATED -->

