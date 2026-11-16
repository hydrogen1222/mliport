# Batch calculations

`mliport batch` runs one calculation — a single point or a geometry
relaxation — over every structure in a directory. The model is loaded once
and reused for every structure, so you pay the startup cost a single time.

## When to use batch

Batch mode is for **throughput**: you have many structures and one model, and
you want results for all of them. Common use cases include:

- Screening a set of candidate structures from a structure prediction run
- Computing single-point energies for a training or validation set
- Re-relaxing structures from one model with another
- Benchmarking a model against a dataset

## Command line

```bash
# Single-point over all structures in a directory
mliport batch structures/ \
  --model mace-omat-0-medium.model --model-type mace \
  --calc-type sp --pattern "*.cif" \
  --output batch_results

# Relaxation with a pattern for nested directories
mliport batch ML-OPT/ \
  --model my-grace-model --model-type grace --device cuda \
  --calc-type opt --pattern '*/rank_*/1.cif' \
  --fmax 0.05 --max-steps 1000 --optimizer FIRE --no-cell-opt \
  --output runs/ml-opt

Quote the pattern so the shell does not expand it. Each structure's output
directory mirrors its path below the input directory
(`runs/ml-opt/<composition>/rank_0001/1/`), so files with the same name never
overwrite each other. Two inputs that would still map to one output directory
stop the batch before anything runs.

## What you get

One run directory per structure under `--output`, each with the usual single
point or relaxation files, plus progress and failure reporting for the batch as
a whole. Every structure keeps its own model identity and resolved
configuration. The `batch_summary.json` entries expose `status` as
`completed`, `not_converged`, or `failed`. Only completed runs have
`success: true`; non-converged optimizations remain in the `failed`
aggregate count and also increment `not_converged`.

If the full summary cannot be serialised, a degraded document is written
instead: it carries `"summary_status": "degraded"`, the original error in
`metadata.summary_write_error`, and the per-structure status. If even that
cannot be written, the batch fails with an error naming both problems instead
of pretending the summary was saved; the per-structure `batch_status.json`
files are unaffected either way.

## When something fails

A structure that fails is reported as failed; it never turns into a zero-energy
entry, and the batch carries on with the next structure. A missing or empty
input directory fails closed.

The same holds when a backend returns a value that is not a usable number. A
non-finite `energy` or `fmax` (NaN, infinity) or a value that is not a number
at all makes that structure `failed`, with the replaced fields listed in
`non_finite_fields` / `unserialisable_fields` and the reason in `error`; the
value is stored as `null` (or as a placeholder string) and never as `0`, and
the other structures and the batch summary are still written.

## Interrupted batches and `--resume`

Every structure records its outcome the moment it finishes:
`batch_status.json` in its own directory, and one line appended to
`batch_progress.jsonl` in the batch directory. `batch_manifest.json` stores the
settings that define the batch: model (path and, for a local checkpoint, the
sha256 of its weights), task, calculator options and the relaxation settings. `batch_summary.json` is written when the batch ends.

If the batch is interrupted, rerun the same command with `--resume`:

- structures recorded as `completed` or `not_converged` are reused, not
  recomputed (rerunning a deterministic relaxation with the same settings would
  give the same answer), but only if the input file still has the content it
  had when it was computed (sha256) and every output file the run wrote is
  still there with its recorded size; otherwise the structure runs again;
- structures recorded as `failed`, and the one that was running when the batch
  stopped, run again from their input structure;
- the batch refuses to resume if any setting in `batch_manifest.json` differs,
  so results computed with different settings are never mixed. To relax the
  unconverged structures further, use a new `--output` directory.

`--resume` cannot be combined with `--overwrite`; it appends to the existing
`run.log`. The structure that was
running restarts from its input geometry; the optimizer does not continue from
the interrupted step.

## A caution

Batch mode is for throughput, not a consistency guarantee. Check that every
structure fits the backend's task (periodic or molecular), and do not mix
structures that need different heads or tasks in one batch; run them separately
and say so.

```bash
.venv-dpa/bin/mliport batch structures/ \
  --model DPA-3.1-3M.pt --model-type dpa --head Omat24 \
  --calc-type sp --output runs/dpa-batch
```

Quote the `--pattern` so the shell does not expand it. Each structure's output
directory mirrors its path below the input directory, so files with the same
name never overwrite each other.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--calc-type` | `sp` | What to run per structure: `sp` or `opt` |
| `--pattern` | `*.cif`, `*.xyz`, `*.vasp`, `POSCAR*` | File glob; may span subdirectories (`'*/rank_*/1.cif'`, `'**/*.cif'`) |
| `--output` | `batch_results` | Batch output directory |
| `--resume` | — | Continue an interrupted batch |
| `--fmax` | 0.05 eV/Å | Relaxation force threshold (with `--calc-type opt`) |
| `--max-steps` | 500 | Relaxation step limit (with `--calc-type opt`) |
| `--optimizer` | `FIRE` | Relaxation optimizer (with `--calc-type opt`) |
| `--cell-opt` | off | Cell optimization (with `--calc-type opt`) |
| `--fix-symmetry` | off | Preserve symmetry (with `--calc-type opt`) |
| `--model`, `--model-type`, `--task`, `--head`, `--device` | — | Model and backend selection |

Relaxation options (`--fmax`, `--max-steps`, `--optimizer`, `--cell-opt`,
`--fix-symmetry`) are rejected when `--calc-type` is `sp`.

## Output structure

One run directory per structure under `--output`, each containing the usual
single-point or relaxation files. Every structure keeps its own model
identity and resolved configuration. At the batch level:

| File | Contents |
|---|---|
| `batch_summary.json` | Overall status: completed, failed, and not-converged counts |
| `batch_progress.jsonl` | One line per structure as it finishes |
| `batch_manifest.json` | Settings that define the batch (model, options, file hashes) |

Each entry in `batch_summary.json` exposes `status` as `completed`,
`not_converged`, or `failed`. Only completed runs have `success: true`;
non-converged optimizations count as both `failed` and `not_converged`.

## Error handling

A structure that fails is reported as failed — it never becomes a zero-energy
entry. The batch continues with the next structure. A missing or empty input
directory fails closed before anything runs. Two input structures that would
map to the same output directory are detected and rejected before the batch
starts.

## Resuming interrupted batches

Every structure records its outcome immediately: `batch_status.json` in its
own directory and one line appended to `batch_progress.jsonl`. To continue
after an interruption:

```bash
mliport batch structures/ --model ... --calc-type sp \
  --output batch_results --resume
```

Resume rules:

- **Completed** and **not_converged** structures are reused (not recomputed),
  but only if the input file has the same sha256 and every output file is
  intact.
- **Failed** structures and the one that was running when the batch stopped
  are rerun from their input geometry.
- The batch **refuses to resume** if any setting in `batch_manifest.json`
  differs — results from different settings are never mixed.
- `--resume` cannot be combined with `--overwrite`.

## Important caveat

Batch mode is for throughput, not a consistency guarantee. Ensure that every
structure fits the backend's task (periodic vs molecular), and do not mix
structures that need different heads or tasks in one batch — run them
separately.
