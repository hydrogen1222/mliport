# Troubleshooting

Each entry follows: **symptom → likely cause → diagnostic → action**. Fix the
first failing cause; do not work around a fail-closed error by disabling a
check.

## Installation

- **Symptom:** `./scripts/install_mliport.sh` exits before installing.
  **Cause:** unknown engine, unsupported `--python`, or `--device cuda`
  without a supported GPU.
  **Diagnostic:** `./scripts/install_mliport.sh --dry-run ...` prints the
  plan; the error names the constraint.
  **Action:** request a Python version compatible with **every** requested
  backend (UMA needs >=3.11) and a supported device; see
  [installation.md](installation.md).

- **Symptom:** launcher missing or points at the wrong environment.
  **Cause:** the installer did not finish, or mliport was installed manually.
  **Diagnostic:** `ls bin/`, `readlink -f bin/mliport-mace`.
  **Action:** re-run the installer for that engine; never hand-edit a
  launcher.

## GPU runtime

- **Symptom:** the backend reports CPU despite `--device cuda`, or crashes on
  the first kernel.
  **Cause:** framework wheel does not contain kernels for the GPU, or the
  device is not visible.
  **Diagnostic:** `mliport doctor --engine <engine> --device cuda` and
  `nvidia-smi`.
  **Action:** use the installer's pinned wheel route for your architecture;
  do not override `CUDA_VISIBLE_DEVICES` inside a queued job.

- **Symptom:** run says `actual_device: unknown`.
  **Cause:** the backend does not expose a device identity (or the run fell
  back). **Action:** treat the record as not GPU-attributed; fix visibility
  and re-run before writing methods claims.

## Model access

- **Symptom:** model not found or checksum mismatch.
  **Cause:** wrong path, incomplete download, or a different checkpoint.
  **Diagnostic:** compare the SHA-256 with
  `validation/science/model_manifest.json` (`mliport doctor` reports what it
  can).
  **Action:** re-download from the upstream source; never rename a different
  checkpoint to match.

## Configuration

- **Symptom:** `No MLIP backend was selected.`
  **Cause:** no `--model-type`/`MODEL_TYPE` and no model alias/profile.
  **Action:** pass an explicit backend (`--model-type mace|dpa|grace|uma`) or
  declare `engine` in a `[model:...]` alias. There is no implicit default.

- **Symptom:** `Strict config validation failed` / `not applicable to engine`.
  **Cause:** misspelled key or a backend-specific option given to another
  backend (CFG-01).
  **Diagnostic:** the error names the key and source;
  `mliport config explain <KEY>` shows the canonical name.
  **Action:** fix the key/backend. Only use `--lenient-config` /
  `strict_config = false` when you deliberately accept warnings.

## DPA head

- **Symptom:** `HEAD` required / branch error on a multi-task DPA checkpoint.
  **Cause:** the checkpoint exposes several branches and no head was given.
  **Diagnostic:** `dp --pt show <model> model-branch`.
  **Action:** pass the intended branch with `--head`/`HEAD`. mliport
  **fails closed**; it never silently picks a "default" head, so a wrong-head
  result cannot be produced this way.

## Stress unsupported

- **Symptom:** `--cell-opt` or a stress request fails, or stress is absent
  from the output.
  **Cause:** the loaded checkpoint does not implement stress.
  **Diagnostic:** `mliport doctor --model <model> --engine <engine>` reports
  `has_stress`; the T1 evidence records `stress_supported` per profile.
  **Action:** use a stress-capable checkpoint or run fixed-cell; do not
  fabricate a stress tensor from forces.

## MD instability

- **Symptom:** run aborts with a force above `--fmax-abort`, NaN energy, or
  exploding temperature.
  **Cause:** timestep too large, bad structure/pre-relaxation, inconsistent
  charge/spin, or a model outside its domain.
  **Diagnostic:** inspect `raw/md.csv` and `vasp/OUTCAR` around the abort;
  the abort record names the force threshold.
  **Action:** pre-relax, reduce the timestep, check the task/electronic
  state, and re-run with a recorded seed. Do not lower `--fmax-abort` to
  "make it pass".

## NEB non-convergence

- **Symptom:** `neb_results.json` reports not converged.
  **Cause:** endpoints not pre-relaxed/mismatched, bad atom map, too few
  images, insufficient steps.
  **Diagnostic:** inspect the band energies/images and
  `run_context.json`.
  **Action:** fix endpoints/mapping, increase images/steps or tune the
  spring; a saddle candidate is not a verified transition state — Hessian
  verification remains separate.

## Transport insufficient sampling

- **Symptom:** transport refuses to fit, or `D` has huge uncertainty.
  **Cause:** too few saved frames in the diffusive regime, wrong saved-frame
  interval, or drift policy masking the signal.
  **Diagnostic:** the analysis JSON reports the lag grid, fit window and
  sampling diagnostics.
  **Action:** run longer/equilibrate more, state the saved-frame interval,
  and re-fit over an explicit diffusive window. Never lower the sampling
  requirement to force a number.

## GEMDAT backend failure

- **Symptom:** `backend_error` in the electrolyte analysis.
  **Cause:** the GEMDAT backend failed (site mapping/density parameters,
  missing dependency), which is **not** "zero jumps".
  **Diagnostic:** read the error in `analysis.json`; re-run with explicit
  `--sites`.
  **Action:** fix the site definition/parameters or the environment; keep the
  backend_error record rather than reporting a mechanism.

## Cache invalidation

- **Symptom:** changing a model/option appears to have no effect, or results
  look stale.
  **Cause:** an existing run directory/analysis ID was reused.
  **Diagnostic:** compare `resolved_config.json` and model identity in the
  result with your request; use `--force` for analyses.
  **Action:** write to a new `--output` directory. mliport does not silently
  reuse a calculator across changed model identity.

## Fresh-install acceptance notes (2.0.0b3)

These failures were observed on a clean V100 installation and fixed before
the 2.0.0b3 release. They are listed with the exact remedy so the message can
be recognised in the field.

**The installer stops with "no local Python 3.10-3.12 runtime with
packaging".** The bootstrap needs a 3.10-3.12 interpreter to plan the
install. The current installer selects the uv-managed interpreter and, when
that interpreter has no `packaging`, runs the planner in an ephemeral
`uv run --with packaging` environment. If you set `MLIPORT_INSTALL_PYTHON`
yourself, the interpreter must be 3.10-3.12 and must import `packaging`. You
can also preselect one with `uv python install 3.12`.

**`--engines mace dpa grace uma` is rejected as extra arguments.** Fixed in
2.0.0b3: `--engines` accepts both the documented space-separated form and the
comma-separated form.

**The doctor passes but `analyze transport`/`electrolyte` are missing.**
Fixed in 2.0.0b3: the installer now installs `mliport[analysis-all]`, which
covers scipy/matplotlib, kinisi and gemdat. Re-run the installer instead of
adding packages by hand.

**DPA fails with `No package metadata was found for mpich`.** `deepmd-kit`
imports MPICH metadata from its `torch` extra. Fixed in 2.0.0b3: the DPA
environment installs a matching `mpich` package. Reinstalling DPA is enough:

```bash
./scripts/install_mliport.sh --engines dpa --device auto --clean
```

**DPA or GRACE says "requires process-level device isolation".** The CLI
restarts DPA/GRACE commands in a fresh process with `CUDA_VISIBLE_DEVICES`
when the variable is unset, so the documented per-environment command works.
The Python API cannot restart the caller: export
`CUDA_VISIBLE_DEVICES=<uuid>` (or `""` for CPU) before importing those
frameworks. One GPU UUID can be read with
`nvidia-smi --query-gpu=uuid --format=csv,noheader`.

**NEB refuses to run without `--allow-unvalidated-neb`.** This is the
capability gate, not a bug: the current checkpoints have no validated
energy-gradient consistency record for this runtime. The run is explicitly
experimental and does not claim a physical barrier.

**NEB resume says "Resume fingerprint is incompatible".** Resume continues in
the original run directory and keeps the recorded options. In this beta the
step budgets are part of the identity, so changing `--max-steps` during a
resume is rejected; drop the override to continue with the recorded budget.
For UMA the checkpoint's implicit charge/spin defaults are now carried into
the comparison, so a plain resume works.

**`analyze` asks for missing options.** `transport` requires
`--charge` and `--fit-start-ps` (and explicit `--lag-step-ps`/`--lag-stop-ps`
when the default lag grid would exceed the memory guard). `electrolyte`
requires either `--sites` or `--discover-sites-from-density`. `rdf` requires
`--center` and `--neighbor`; `density` and `msd` require `--mobile`.

**GEMDAT on a long, large trajectory takes very long.** Site discovery on a
1 ns, 400-atom LGPS run exceeds a short acceptance budget. Analyse a segment
with an explicit `--frame-interval-fs`, or give it a longer window.

**`mliport tui` waits when started from a script.** The TUI requires an
interactive terminal; without one it now exits with an error. Use the TUI
tests or the Python API for headless automation.

**Hugging Face downloads fail behind a SOCKS proxy.** Install
`httpx[socks]` in the download environment and keep `NO_PROXY` free of
bracketed entries; see [models.md](models.md).
