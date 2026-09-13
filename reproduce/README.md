# Reproduction entry points

These commands never train a network and never modify the frozen result CSVs.
Generated files are written below `reproduce/_outputs/`, which is ignored by
Git.

## Source-only implementation check

```bash
bash reproduce/smoke_test.sh --source-only
```

This deterministic check compares the explicit DFA contraction with both the
diagnostic capture and the production `Trainer._local_batch` autograd result.

## Canonical checkpoint smoke replay

Place the preprocessed N-MNIST tree below `data/`, or point `DATA_ROOT` to the
directory which contains `nmnist/frames_number_30_split_by_number/{train,test}`:

```bash
DATA_ROOT=/path/to/data bash reproduce/smoke_test.sh
```

The bundled seed-20260830 path-sanitized checkpoint release copy and fixed probe
are used by default. The command verifies the release hash, its canonical
identity through the transform report, and probe order; it then recomputes H1
delta on all 512 basis-eval samples and checks r95 plus the
within-time/between-sample decomposition.
Exit code 3 means that a required external artifact (usually the dataset) was
missing; the JSON file contains a machine-readable reason.

## Figures and end-to-end verification

```bash
bash reproduce/make_figures.sh
DATA_ROOT=/path/to/data bash reproduce/reproduce_main_diagnostics.sh
```

Set `PYTHON_BIN`, `SMOKE_DEVICE`, `SMOKE_OUTPUT_DIR`, or `FIGURE_OUTPUT_DIR` to
override the corresponding runtime setting. `SMOKE_DEVICE=cuda` is optional;
CPU replay is supported.

`build_appendix_tables.mjs` is retained as release-authoring provenance for the
already published CSV/XLSX bundle. It uses the authoring environment's
`@oai/artifact-tool` and is not a reviewer runtime dependency; all reviewer
checks use the Python/shell entry points above.
