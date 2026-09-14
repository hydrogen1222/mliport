# NEB / CI-NEB

**Purpose.** Find a minimum-energy path between two endpoints using
fixed-cell nudged elastic band (with optional climbing image), and report the
forward barrier.

**Minimal command.**

```bash
.venv-mace/bin/mliport neb --initial initial.vasp --final final.vasp   --model mace-omat-0-medium.model --model-type mace --images 7   --allow-unvalidated-neb --output runs/neb
```

The current checkpoints are outside the validated NEB envelope: mliport
cannot yet certify their energy-gradient consistency for this exact
model/runtime, so it refuses to run without the explicit
`--allow-unvalidated-neb` opt-in. Such runs are recorded as
`workflow_smoke_only` and do not claim a physical barrier. Once a runtime has
a validated capability record the flag is no longer required.

**Important options.**

| Option | Meaning |
|---|---|
| `--initial`, `--final` | endpoint structures (mandatory) |
| `--images` | number of intermediate images (endpoints additional) |
| `--climb` / `NEB_CLIMB` | climbing-image refinement |
| `--atom-map` | 0-based mapping of final indices in initial-atom order |
| `--image-shifts` | integer lattice shifts per atom (`0,0,0;1,0,0`) |
| `--neb-interpolation`, `--neb-path-convention` | IDPP interpolation and path convention |
| `--neb-pre-fmax`, `--neb-pre-max-steps` | endpoint pre-relaxation before interpolation |
| `--endpoint-policy`, `--endpoint-fmax`, `--endpoint-steps` | how endpoint images are constrained |
| `--neb-spring`, `--neb-maxstep`, `--neb-min-distance` | band/spring parameters |
| `--checkpoint-interval` | band checkpoint frequency |
| `--resume` | resume from a checkpoint/directory |
| `--allow-unvalidated-neb` | opt-in for NEB configurations outside the validated envelope (recorded, not silent) |

**Inputs.** Two structures with consistent atom order (or an explicit
`--atom-map`), a model, an explicit backend, and enough memory for
`images + 2` calculators.

**Outputs.** `NEB_BAND`/band trajectory, per-image energies/forces,
`resolved_config.json`, checkpoint files, and a result JSON with `barrier`,
convergence state and the sampled barrier value.

**Failure semantics.** Non-convergence is reported with the best band and a
failure reason; it is not a converged barrier. Bad atom mapping or
inconsistent endpoints fail before the band is built. Checkpoint/resume
locks the original run directory so the band cannot be silently restarted
with different settings.

**Scientific caveats.**

- A NEB saddle candidate is **not** a verified transition state. Hessian
  (frequency) verification is a separate analysis and is not implied by a
  converged band.
- The barrier is only comparable within one model/head identity and one NEB
  protocol (images, interpolation, endpoint policy, spring).
- Endpoint pre-relaxation changes the reference; record it.
- The saddle is meaningful only if the endpoints are the intended states and
  the atom mapping is chemically correct.

**Interface capability matrix**

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

This matrix is generated from `validation/science/capability_matrix.json`. A `yes` means the entry point can express the feature; it is not an equivalence or validation claim. A `no`/`partial` entry carries a note in that file.

**Example (resume).**

```bash
.venv-mace/bin/mliport neb --resume runs/neb --output runs/neb
```

Resume continues in the original run directory and checks the recorded
identity (model, task/head/dtype, cell/PBC, constraints, band setup and
options). Step budgets are part of that identity in this beta: a resume that
changes `--max-steps` is rejected with `Resume fingerprint is incompatible`;
the explicit error leaves the checkpoint untouched.
