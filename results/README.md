# Numerical tables and provenance

The nine top-level CSVs are reviewer-facing views generated from
`results/frozen_sources/`. Every row carries a `source_file` or
`provenance_file` pointer with its source row. Values are selected and joined
programmatically; they are not copied from the manuscript.

| Public table | Rows | Frozen inputs |
|---|---:|---|
| `nmnist_main_per_seed.csv` | 45 | N-MNIST spectral rows + both accuracy checkpoint rules |
| `cutoff_metrics_per_seed.csv` | 75 | final basis-eval timestep r50/r80/r90/r95/r99 |
| `threshold_free_metrics_per_seed.csv` | 75 | stable and entropy effective ranks |
| `homogenization_per_seed_layer_alpha.csv` | 75 | 5 seeds × 3 layers × 5 alpha values |
| `controls_per_seed.csv` | 945 | magnitude mask, random mask, marginal shuffle, MeanGate, ShuffledGate |
| `memoryless_per_seed.csv` | 9 | 3 seeds × 3 layers |
| `dvs_per_seed.csv` | 9 | DVS accuracy, gate spaces, ranks and covariance rows |
| `probe_size_sensitivity.csv` | 2700 | 3 methods × 5 seeds × 3 layers × 3 N × 20 repeats |
| `width_robustness.csv` | 27 | widths 400/800/1200 × 3 seeds × 3 layers |

`reproduce/build_appendix_tables.mjs` is the release-authoring script used to
construct these CSVs and `supplement/appendix_tables.xlsx`. The frozen inputs
remain byte-preserved so each join/filter can be independently audited. Its
`@oai/artifact-tool` dependency belongs to the authoring environment and is not
needed by the Python reviewer checks.

`frozen_sources/nmnist/fig2_subspace_overlap.csv` is an additional byte-preserved
canonical figure-source table. It is intentionally not converted into a 32 x 32
heatmap: the matching independently trained DFA/BPTT bases were not retained.
See `../supplement/fig2d_data_audit.md`.

Important semantics:

- task accuracy: validation-selected checkpoint unless explicitly labeled
  `epoch100`;
- spectral geometry: completed epoch-100 checkpoint;
- canonical N-MNIST and DVS geometry: fixed `basis_eval` split;
- uncertainty: sample standard deviation over seeds (`ddof=1`);
- `selected_positions` in the probe table are zero-based positions inside the
  fixed 512-example N-MNIST evaluation half;
- raw DVS gate rows carry projected credit ranks as shared metadata; they are
  not raw-spatial credit spectra.

The byte-level release manifest is `../ARTIFACT_MANIFEST_SHA256.csv`.
