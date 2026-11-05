# Batch

`mliport batch` runs one calculation, a single point by default, over every
structure in a directory, with one model, one backend and one configuration.

```bash
.venv-mace/bin/mliport batch structures/ \
  --model mace-omat-0-medium.model --model-type mace \
  --calc-type sp --pattern "*.cif" \
  --output batch_results
```

The model is loaded once and the calculator is reused for every structure, which
is the point of batching: you pay the startup cost a single time.

## Options you will use

| Option | Meaning |
|---|---|
| `--calc-type` | what to run per structure: `sp` (default) or `opt` |
| `--pattern` | file glob relative to the input directory; it may span sub-directories (`'*/rank_*/1.cif'`, `'**/*.cif'`). Without it mliport finds `*.cif`, `*.xyz`, `*.vasp` and `POSCAR*` directly in the input directory |
| `--output` | the batch output directory (default `batch_results`) |
| `--fmax`, `--max-steps`, `--optimizer`, `--cell-opt`, `--fix-symmetry` | relaxation settings for `--calc-type opt`, with the same meaning and defaults as `mliport opt`; they are rejected for `--calc-type sp` |
| `--resume` | continue an interrupted batch in the same `--output` directory |
| `--model`, `--model-type`, `--task`, `--head`, `--device` | model and backend selection |

## Relaxing structures spread over many directories

Generated structures often share a file name and differ only in their
directory, for example `ML-OPT/<composition>/rank_0001/1.cif`. Point batch at
the top directory and select the files with a pattern:

```bash
mliport batch ML-OPT/ \
  --model my-grace-model --model-type grace --device cuda \
  --calc-type opt --pattern '*/rank_*/1.cif' \
  --fmax 0.05 --max-steps 1000 --optimizer FIRE --no-cell-opt \
  --output runs/ml-opt
```

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

## When something fails

A structure that fails is reported as failed; it never turns into a zero-energy
entry, and the batch carries on with the next structure. A missing or empty
input directory fails closed.

## Interrupted batches and `--resume`

Every structure records its outcome the moment it finishes:
`batch_status.json` in its own directory, and one line appended to
`batch_progress.jsonl` in the batch directory. `batch_manifest.json` stores the
settings that define the batch: model, task, calculator options and the
relaxation settings. `batch_summary.json` is written when the batch ends.

If the batch is interrupted, rerun the same command with `--resume`:

- structures recorded as `completed` or `not_converged` are reused, not
  recomputed (rerunning a deterministic relaxation with the same settings would
  give the same answer);
- structures recorded as `failed`, and the one that was running when the batch
  stopped, run again from their input structure;
- the batch refuses to resume if any setting in `batch_manifest.json` differs,
  so results computed with different settings are never mixed. To relax the
  unconverged structures further, use a new `--output` directory.

`--resume` cannot be combined with `--overwrite`. The structure that was
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
