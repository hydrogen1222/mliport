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
