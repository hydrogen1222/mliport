# Troubleshooting

Most of mliport's errors are deliberate: it stops where a guess would give you a
wrong answer that looks right. So the fix is nearly always to supply the missing
fact, not to find a switch that turns the check off. Each entry below gives the
symptom, the usual cause, how to confirm it and what to do.

## Installation

**The installer exits before installing anything.** An unknown engine, a
`--python` that one of the requested backends cannot use, or `--device cuda` on
a machine without a supported GPU. Run it again with `--dry-run`; the plan it
prints, and the error, name the constraint. Ask for a Python version that every
requested backend accepts (UMA needs 3.11 or newer) and a device the machine has.
See [Installation](installation.md).

**A launcher is missing or points at the wrong environment.** The installer
did not finish, or mliport was installed by hand. `ls bin/` and
`readlink -f bin/mliport-mace` show where it points. Re-run the installer for
that engine; do not edit a launcher.

## Devices and GPUs

**The backend runs on the CPU although you asked for `--device cuda`, or crashes
on the first kernel.** Either the framework wheel has no kernels for your GPU
architecture or the GPU is not visible. `mliport doctor --engine <engine>
--device cuda` and `nvidia-smi` tell you which. Use the installer's pinned wheel
route for your architecture, and do not override `CUDA_VISIBLE_DEVICES` inside a
queued job.

**A run records `actual_device: unknown`.** The backend does not expose which
device it used, or the run fell back. Treat that record as not attributed to a
GPU; fix the visibility and run again before you write down a methods claim.

## Models and configuration

**Model not found, or the checksum does not match.** A wrong path, an
incomplete download or a different checkpoint. Compare the SHA-256 with
`validation/science/model_manifest.json` (`mliport doctor` reports what it can)
and download again from the upstream source. Never rename a different checkpoint
to make it match.

**`No MLIP backend was selected.`** There is no default backend. Pass
`--model-type mace|dpa|grace|uma`, or declare `engine` in a `[model:...]` alias.

**`Strict config validation failed` or `not applicable to engine`.** A
misspelled key, or an option that belongs to another backend. The message names
the key and where it came from, and `mliport config explain <KEY>` shows the
canonical spelling. Fix the key. Use `--lenient-config` (or `strict_config =
false`) only when you really do want warnings instead of errors.

**A DPA checkpoint wants a `HEAD`.** The checkpoint has several branches and you
did not say which. `dp --pt show <model> model-branch` lists them; pass the one
you mean with `--head`. mliport fails closed here and never silently picks a
"default" head, so a result from the wrong head cannot be produced this way.

**Stress, or `--cell-opt`, is refused.** The loaded checkpoint does not implement
stress. `mliport doctor --model <model> --engine <engine>` reports `has_stress`.
Use a checkpoint that does, or run with a fixed cell. A stress tensor is never
made up from forces.

**A change of model or option seems to have no effect.** An existing run
directory or analysis ID was reused. Compare `resolved_config.json` and the model
identity in the result with what you asked for; use `--force` to recompute an
analysis, or write to a new `--output`. mliport does not reuse a calculator across
a changed model identity.

## Runs

**An MD run aborts: a force above `--fmax-abort`, a NaN energy, a runaway
temperature.** The timestep is too large, the structure was not relaxed, the
charge or spin state is inconsistent, or the model is being used outside its
domain. Look at `raw/md.csv` and `vasp/OUTCAR` around the abort; the abort record
names the threshold. Pre-relax, shorten the timestep, check the task, and run
again with a recorded seed. Raising `--fmax-abort` only hides the problem.

**NEB does not converge.** Typical causes are endpoints that were not relaxed or
do not match, a wrong atom map, too few images or too few steps. Look at the
band energies and `run_context.json`. Fix the endpoints or the mapping, add
images or steps, or tune the spring. A saddle candidate is not a verified
transition state; confirming one with a Hessian is a separate step.

**NEB is refused: "not verifiably the gradient of its energy".** Before the first
step, NEB compares the model's forces with finite differences of its energy on
your band. The refusal means the forces do not match the energy (a model that
predicts forces directly), a value was not finite, or the energy
resolution is too coarse to tell (a large float32 cell: use float64 or a smaller
cell). The message gives the image, the direction and the error, and the whole
record is in `NEBGradientConsistencyError.summary`.

**NEB resume says "Resume fingerprint is incompatible".** A resume continues in the
original run directory with the recorded options, and the step budgets are part of
that identity. Changing `--max-steps` while resuming is rejected, so leave it out
and the recorded budget is used.

## Analysis

**`analyze` asks for options.** `transport` needs `--mobile`, `--charge` and
`--fit-start-ps`, and an explicit `--lag-step-ps` with `--lag-stop-ps` when the
default lag grid would pass the memory guard. `electrolyte` needs either `--sites`
or `--discover-sites-from-density`. `rdf` needs `--center` and `--neighbor`;
`msd` needs `--mobile`.

**Transport refuses to fit, or `D` has a huge uncertainty.** There are too few
saved frames in the diffusive regime, the saved-frame interval is wrong, or the
drift policy is hiding the signal. The result JSON reports the lag grid, the fit
window and the sampling diagnostics. Run longer, state the frame interval, and fit
again over an explicit diffusive window. Do not lower the sampling requirement to
get a number out.

**GEMDAT reports `backend_error`.** The GEMDAT backend failed (site mapping,
density parameters, a missing dependency). That is not "zero jumps". Read the error
in `results.json`, try explicit `--sites`, and fix the definition or the
environment. Keep the `backend_error` record rather than reporting a mechanism.

**GEMDAT takes very long on a long trajectory.** Discovering sites on a 1 ns,
400-atom LGPS run is expensive. Analyze a segment with an explicit
`--frame-interval-fs`, or allow it more time.

**`analyze` rejects the `step` column of `raw/md.csv`.** Step indices are
integers, so a fractional or out-of-range `step` value is refused instead of
being truncated into a step axis that is not in the file. Repair the
trajectory's `md.csv`, or analyze the ASE trajectory or XDATCAR directly.

## Fresh-install notes (2.0.0b3)

These were seen on a clean V100 machine and fixed before 2.0.0b3. They are here so
that you recognize the message if you meet it.

**"no local Python 3.10-3.13 runtime with packaging".** The installer needs a
3.10–3.13 interpreter to plan the install. It now picks the uv-managed one and,
when that has no `packaging`, runs the planner in a throwaway
`uv run --with packaging` environment. If you set `MLIPORT_INSTALL_PYTHON`
yourself, it has to be 3.10–3.13 and import `packaging`; `uv python install 3.12`
gives you one.

**`--engines mace dpa grace uma` is rejected as extra arguments.** Fixed: `--engines`
takes the space-separated and the comma-separated form.

**The doctor passes but `analyze transport` or `electrolyte` is missing.** Fixed:
the installer now installs `mliport[analysis-all]`, which brings scipy, matplotlib,
kinisi and gemdat. Run the installer again instead of adding packages by hand.

**DPA fails with `No package metadata was found for mpich`.** `deepmd-kit` reads
MPICH metadata through its `torch` extra. The DPA environment now installs a
matching `mpich`, so reinstalling DPA is enough:

```bash
./scripts/install_mliport.sh --engines dpa --device auto --clean
```

**DPA or GRACE says "requires process-level device isolation".** The CLI restarts
DPA and GRACE commands in a fresh process with `CUDA_VISIBLE_DEVICES` set when it
is unset, so the documented per-environment command just works. The Python API
cannot restart its caller: export `CUDA_VISIBLE_DEVICES=<uuid>` (or `""` for the
CPU) before importing those frameworks. A GPU's UUID comes from
`nvidia-smi --query-gpu=uuid --format=csv,noheader`.

**`mliport tui` waits when started from a script.** The TUI needs an interactive
terminal and now exits with an error without one. For headless automation use the
Python API.

**Hugging Face downloads fail behind a SOCKS proxy.** Install `httpx[socks]` in the
download environment and keep bracketed entries out of `NO_PROXY`; see
[Models](models.md).
