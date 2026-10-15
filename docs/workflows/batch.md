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
| `--pattern` | file glob; without it mliport finds `*.cif`, `*.xyz`, `*.vasp` and `POSCAR*` |
| `--output` | the batch output directory (default `batch_results`) |
| `--model`, `--model-type`, `--task`, `--head`, `--device` | model and backend selection |

## What you get

One run directory per structure under `--output`, each with the usual single
point or relaxation files, plus progress and failure reporting for the batch as
a whole. Every structure keeps its own model identity and resolved
configuration.

## When something fails

A structure that fails is reported as failed; it never turns into a zero-energy
entry. A missing or empty input directory fails closed.

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
