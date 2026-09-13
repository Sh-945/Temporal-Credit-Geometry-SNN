# Temporal Modulation and Local-Error Spectra in SNNs

Reviewer-facing reproducibility repository for the ICASSP 2027 manuscript.
It contains the production implementation, exact per-seed result tables,
fixed train/validation/probe indices, sanitized checkpoint metadata, and one
scientifically identical path-sanitized checkpoint release copy for a real
no-training smoke replay.

No result was reconstructed from manuscript prose. No main experiment was
retrained while assembling this release. Accuracy and geometry retain their
frozen checkpoint rules: validation-selected checkpoints for task accuracy and
completed epoch 100 for spectral diagnostics.

## Quick verification

Create the pinned public dependency environment (or install the pinned
requirements), then:

```bash
conda env create -f environment.yml
conda activate icassp27-credit-geometry
python -m pip install -e .
python verify_artifact.py
bash reproduce/smoke_test.sh --source-only
bash reproduce/make_figures.sh
```

`verify_artifact.py` checks every published byte against
`ARTIFACT_MANIFEST_SHA256.csv`, verifies the bundled release hash and its frozen
canonical identity, checks all nine table row counts, and recomputes frozen
headline means from per-seed rows. The exact archived GPU/CUDA build is recorded
in `metadata/runtime_environment.txt`; `environment.yml` is the portable public
dependency specification.

For the full checkpoint smoke, separately provide the matching pre-framed
N-MNIST NPZ tree described in `data/README.md` and run:

```bash
DATA_ROOT=/path/containing/nmnist bash reproduce/smoke_test.sh
```

The command loads the bundled SNN-DFA seed-20260830 epoch-100 release copy,
verifies the exact 1,024-probe order, replays the 512-sample `basis_eval` half,
and recomputes H1 delta, r95, the explicit local weight-gradient contraction,
and the within-time/between-sample decomposition. It performs no optimizer step.
The release-time run passed; its portable JSON is in
`metadata/smoke_results/checkpoint_smoke_seed_20260830_H1.json`.
`metadata/checkpoints/checkpoint_release_provenance.json` links the release hash
to the original canonical SHA-256 and verifies every non-config tensor/state is
unchanged after serialization. Only location-only config fields were sanitized.

## Repository layout

```text
README.md                     overview and commands
REPRODUCIBILITY_AUDIT.md      complete A–J evidence audit
requirements.txt              exact Python package versions
environment.yml               archived runtime specification
configs/                      canonical configs plus labeled legacy configs
src/                          production model/training/diagnostic packages
diagnostics/                  reviewer checks and archived replay logic
reproduce/                    one-command shell and release-building scripts
metadata/                     indices, hashes, configs and smoke evidence
results/                      nine public tables plus frozen source CSVs
supplement/                   detailed audit material and Excel appendix tables
tests/                        implementation regression tests
```

## Frozen scope

- N-MNIST: five seeds `20260830`, `20260831`, `20260901`, `20260908`,
  `20260909`; SNN-DFA, matched SNN-BPTT, temporal ANN-DFA, MeanGate, and
  ShuffledGate.
- Mechanism controls: progressive temporal homogenization, magnitude mask,
  random mask, marginal shuffle, and a three-seed memoryless surrogate.
- DVS-Gesture: three seeds `20260830`, `20260831`, `20260901`, with raw versus
  projected gate cosine, carrier/post/aggregated ranks, and covariance
  decomposition.
- Robustness: probe sizes 128/256/512 and frozen widths 400/800/1200.

The requested numerical tables are directly under `results/`; their exact
source-row lineage is documented in `results/README.md`. A formatted copy is
provided as `supplement/appendix_tables.xlsx`.

## One-click commands

```bash
bash reproduce/reproduce_main_diagnostics.sh --source-only
bash reproduce/make_figures.sh
bash reproduce/smoke_test.sh --source-only
```

Set `DATA_ROOT` and omit `--source-only` for the real checkpoint replay. Set
`SMOKE_DEVICE=cuda` to use a GPU; CPU is supported.

## Reproducibility boundaries

- Third-party datasets are not redistributed.
- The archive contains the post-NPZ loader and exact frame shapes/splits, but
  not the pinned raw-event-to-NPZ converter; this limitation is explicit in the
  audit.
- The manifest records 28 final checkpoint identities. One scientifically
  identical, path-sanitized release binary is bundled for smoke replay; all 28
  have sanitized config records and canonical SHA-256 identities, and no older
  checkpoint is substituted.
- The dataset archives and a portable NPZ content manifest are not bundled.
  Full replay therefore requires a separately supplied matching pre-framed tree.
- DVS H1/H2 raw-spatial gate cosine is available, but raw-spatial credit spectra
  were not archived and are not claimed.
- The source archive had no usable Git commit, so release provenance is based on
  byte-level source/checkpoint/result manifests.
- The repository does not synthesize the unsupported 32×32 DFA/BPTT basis
  heatmap.

See `REPRODUCIBILITY_AUDIT.md` for exact equations, reductions, shapes, source
functions, numerical values, confirmed facts, and unresolved facts.
