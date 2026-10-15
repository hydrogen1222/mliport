# NEB and CI-NEB

`mliport neb` finds a minimum-energy path between two endpoint structures with a
fixed-cell nudged elastic band, optionally refined by a climbing image, and
reports the barrier it found.

```bash
.venv-mace/bin/mliport neb --initial initial.vasp --final final.vasp \
  --model mace-omat-0-medium.model --model-type mace --images 7 \
  --climb --output runs/neb
```

You supply two structures with the same atoms in the same order (or an explicit
`--atom-map` when the order differs), a model on an explicit backend, and enough
memory for `images + 2` copies of the calculator.

## What happens before the band moves

A NEB only makes sense if the model's forces are the gradient of its energy.
Otherwise the band is optimized against forces that do not describe the surface
it is supposed to be walking along, and the barrier is meaningless without any
sign that something is wrong. So mliport checks first.

On the band it is about to optimize, both endpoints and interpolated images, it
compares the force projection `-F·u` along a few directions (the force
direction and random ones) with central finite differences of the model energy.
Two step sizes are combined by Richardson extrapolation, and the tolerance
includes the rounding noise of the model's dtype. The run is refused when:

- the forces are not the gradient of the energy,
- a value is not finite,
- the energy resolution cannot settle the question (a large float32 cell, for
  instance), or
- the model predicts forces directly rather than differentiating its energy
  (UMA does; such models are always refused).

The measurement is stored as `gradient_preflight` in `run_context.json` and
`neb_results.json`. It is a spot check of consistency on the sampled structures
and directions, not a proof over the whole energy surface and not a statement
about how accurate the model is.

## Options you will use

| Option | Meaning |
|---|---|
| `--initial`, `--final` | the endpoint structures (required unless resuming) |
| `--images` | number of intermediate images, endpoints not counted (default 7) |
| `--climb` | refine the saddle with a climbing image (default off) |
| `--fmax`, `--max-steps` | convergence threshold in eV/Å (default 0.03) and step limit (default 1000) |
| `--pre-fmax`, `--pre-max-steps` | with `--climb`, the looser first stage before the climbing image switches on (default 0.1 eV/Å, 300 steps) |
| `--spring`, `--maxstep` | spring constant (default 0.1 eV/Å²) and largest optimizer step (default 0.1 Å) |
| `--interpolation` | `linear` (default) or `idpp` for the initial path; tune IDPP with `--idpp-fmax`, `--idpp-steps`, `--idpp-mic` |
| `--path-convention` | `mic` (default) or `unwrapped` |
| `--endpoint-policy` | `validate` (default) or `relax`, see below |
| `--endpoint-fmax`, `--endpoint-steps` | the endpoint force threshold (default 0.02 eV/Å) and relaxation budget (default 500) |
| `--min-distance` | refuse images with atoms closer than this (default 0.5 Å) |
| `--atom-map` | 0-based final indices in initial-atom order, comma-separated |
| `--image-shifts` | integer lattice shifts per atom, such as `0,0,0;1,0,0` |
| `--checkpoint-interval` | save the whole band every N steps (default 10) |
| `--resume` | continue from a checkpoint or a NEB run directory |

### Endpoints: validate or relax

With the default `--endpoint-policy validate`, the endpoints must already be
relaxed: if either has a force above `--endpoint-fmax`, mliport stops and says
which one. With `relax` it relaxes both endpoints first and builds the band
afterwards. Either way, the endpoint treatment changes your reference, so
record it. `relax` cannot rewrite a band that was supplied explicitly.

## What you get

`neb_results.json` holds the energies, forces, convergence state and the
barrier numbers (`barrier_forward_sampled_eV`, `barrier_reverse_sampled_eV`,
`reaction_energy_eV`, and a `barrier_status` that says whether an interior peak
was found). Next to it are `run_context.json`, `resolved_config.json`,
`artifacts.json`, a `checkpoints/` directory with complete-band checkpoints, and
a `vasp_path/` directory with one `POSCAR` per image for tools that expect
VASP's NEB layout.

The barrier is the *sampled* one: the highest image on the band minus the end
state. mliport does not fit a saddle and leaves the fitted fields empty.

## When it fails or does not converge

A band that does not converge is reported as such, with the best band so far and
a reason. It is not a converged barrier. A bad atom mapping or inconsistent
endpoints fail before the band is built. A resume locks the original run
directory, so the band cannot be restarted with different settings by accident.

## What a barrier is worth

- A converged NEB saddle candidate is not a verified transition state. Checking
  it with a Hessian (frequency) calculation is a separate analysis that a
  converged band does not imply.
- A barrier compares only within one model and head identity and one NEB
  protocol (images, interpolation, endpoint policy, spring).
- It is meaningful only if the endpoints are the states you intended and the
  atom mapping is chemically right.

## Which entry point can express what

Not every NEB option can be set from every interface. This table is generated
from `validation/science/capability_matrix.json`.

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

`yes` means that entry point can express the feature; it is not an equivalence
or validation claim. A `no` or `partial` entry has a note in that file, and
nothing is translated silently: the atom mapping and image shifts, for example,
are available through the API, the direct CLI and the TUI only, not through an
INCAR file.

## Resuming

```bash
.venv-mace/bin/mliport neb --resume runs/neb --output runs/neb
```

A resume continues in the original run directory and checks the recorded
identity: model, task, head and dtype, cell and PBC, constraints, band setup and
options. Step budgets are part of that identity in this beta, so a resume that
changes `--max-steps` is rejected with `Resume fingerprint is incompatible`. The
checkpoint is left untouched when that happens.
