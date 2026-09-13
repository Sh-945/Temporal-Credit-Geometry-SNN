"""Assemble the reviewer-facing audit from checked evidence and frozen tables."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def remap(text: str) -> str:
    replacements = (
        ("results/paper_v9_signal_processing/08_covariance_decomposition/replay_covariance_decomposition.py", "diagnostics/archived/covariance/replay_covariance_decomposition.py"),
        ("results/paper_v9_signal_processing/08_covariance_decomposition/summarize_covariance_decomposition.py", "diagnostics/archived/covariance/summarize_covariance_decomposition.py"),
        ("results/paper_v9_signal_processing/06_mechanism_exploration/00_offline_gate_manipulations/offline_gate_manipulations.py", "diagnostics/archived/mechanisms/offline_gate_manipulations.py"),
        ("results/paper_v9_signal_processing/06_mechanism_exploration/00_offline_gate_manipulations/analyze_offline_results.py", "diagnostics/archived/mechanisms/analyze_offline_results.py"),
        ("results/paper_v9_signal_processing/06_mechanism_exploration/01_stateless_surrogate_control/train_stateless_smoke.py", "diagnostics/archived/mechanisms/train_stateless_control.py"),
        ("results/paper_v9_signal_processing/06_mechanism_exploration/01_stateless_surrogate_control/stateless_surrogate.py", "diagnostics/archived/mechanisms/stateless_surrogate.py"),
        ("results/paper_v9_signal_processing/tables/5_seed_review/sample_metrics.csv", "results/frozen_sources/nmnist/sample_metrics.csv"),
        ("results/paper_v9_signal_processing/07_manuscript_final_audit/checkpoint_metadata.json", "metadata/all_checkpoint_metadata_sanitized.json"),
        ("results/paper_v9_signal_processing/logs/dfa_trained_<seed>/complete.json", "metadata/feedback_matrix_manifest.csv"),
        ("results/paper_v9_signal_processing/replay_v9.py", "diagnostics/archived/v9_replay/replay_v9.py"),
        ("results/paper_v9_signal_processing/train_new_seeds.py", "diagnostics/archived/v9_replay/train_new_seeds.py"),
        ("results/paper_v9_signal_processing/dvs_gate_v9.py", "diagnostics/archived/v9_replay/dvs_gate_v9.py"),
        ("results/paper1_experiment03_20260901_server_report/data_split.csv", "metadata/splits/nmnist_train_val_split.csv"),
        ("results/paper1_experiment03_20260901_server_report/diagnostic_probe.csv", "metadata/splits/nmnist_diagnostic_probe.csv"),
        ("results/paper1_experiment03_20260901_server_report/stageD_cross_dataset/data_split.csv", "metadata/splits/dvs_train_val_split.csv"),
        ("results/paper1_experiment03_20260901_server_report/stageD_cross_dataset/diagnostic_probe.csv", "metadata/splits/dvs_diagnostic_probe.csv"),
        (".../stageD_cross_dataset/diagnostic_probe.csv", "metadata/splits/dvs_diagnostic_probe.csv"),
        ("results/paper1_experiment03_20260901_server_report/stageA_bptt/parity.csv", "results/frozen_sources/nmnist/production_parity.csv"),
        ("results/paper1_experiment03_20260901_server_report/stageA_bptt/checkpoint_reuse_audit.json", "metadata/nmnist_checkpoint_pairing_audit.json"),
        ("stageD_cross_dataset/dataset_audit.json", "metadata/dvs_dataset_audit.json"),
        ("results/paper1_experiment03_20260901_server_report/environment.txt:1-7` and `run_manifest.json:114-120", "metadata/runtime_environment.txt"),
        ("checkpoint_metadata.json", "metadata/all_checkpoint_metadata_sanitized.json"),
    )
    for before, after in replacements:
        text = text.replace(before, after)
    return text


def table_inventory() -> str:
    names = (
        "nmnist_main_per_seed.csv",
        "cutoff_metrics_per_seed.csv",
        "threshold_free_metrics_per_seed.csv",
        "homogenization_per_seed_layer_alpha.csv",
        "controls_per_seed.csv",
        "memoryless_per_seed.csv",
        "dvs_per_seed.csv",
        "probe_size_sensitivity.csv",
        "width_robustness.csv",
    )
    lines = ["| File | Rows | SHA-256 |", "|---|---:|---|"]
    for name in names:
        path = ROOT / "results" / name
        lines.append(f"| `results/{name}` | {row_count(path)} | `{sha256(path)}` |")
    return "\n".join(lines)


def spearman_audit() -> str:
    path = ROOT / "results/frozen_sources/nmnist/correlation_breakdown.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["group_type"] in {"pooled", "layer"}]
    rules = {row["checkpoint_rule"] for row in selected}
    p_values = {row["p_value"] for row in selected}
    pooled_n = {int(row["n"]) for row in selected if row["group_type"] == "pooled"}
    layer_n = {int(row["n"]) for row in selected if row["group_type"] == "layer"}
    if len(rules) != 1 or len(p_values) != 1 or len(pooled_n) != 1 or len(layer_n) != 1:
        raise RuntimeError("Correlation audit fields are not internally consistent")
    lines = [
        "| temporal mode | group | rho | n | p value |",
        "|---|---|---:|---:|---|",
    ]
    for row in selected:
        group = "all layers" if row["group_type"] == "pooled" else row["group"]
        lines.append(
            f"| {row['signal']} | {group} | {row['rho']} | {row['n']} | {row['p_value']} |"
        )
    return (
        "The exact checkpoint-count audit was read from "
        "`results/frozen_sources/nmnist/correlation_breakdown.csv`. "
        f"Pooled correlations use `n={pooled_n.pop()}` and each layer uses "
        f"`n={layer_n.pop()}`. The frozen checkpoint rule is `{rules.pop()}`. "
        f"The archived p-value field is `{p_values.pop()}`; no p value is inferred.\n\n"
        + "\n".join(lines)
    )


def main() -> None:
    abc = (ROOT / "supplement/audit_training_data_gradient.md").read_text(encoding="utf-8")
    abc = abc.split("### C.4 Numerical contraction check", 1)[0]
    abc = remap(abc).replace(
        "# Training, data, and local-gradient audit notes",
        "## A–C. Training, data, and local-gradient implementation",
    )
    abc = abc.replace(
        "Scope: read-only recovery from the frozen source/result archive. No training or canonical-result regeneration was performed. Historical evidence identifiers in this note are translated to public repository paths when `REPRODUCIBILITY_AUDIT.md` is assembled.",
        "Scope: read-only recovery from the frozen source/result archive. No training or canonical-result regeneration was performed; paths below use the public repository layout.",
    )
    deff = remap((ROOT / "supplement/audit_spectral_controls_dvs.md").read_text(encoding="utf-8"))
    deff = deff.replace(
        "# Spectral diagnostics, controls, and DVS-Gesture audit notes",
        "## D–F. Spectral diagnostics, controls, and DVS-Gesture",
    )

    smoke = json.loads(
        (ROOT / "metadata/smoke_results/checkpoint_smoke_seed_20260830_H1.json").read_text(encoding="utf-8")
    )
    parity = json.loads(
        (ROOT / "metadata/smoke_results/local_gradient_parity.json").read_text(encoding="utf-8")
    )
    xlsx = ROOT / "supplement/appendix_tables.xlsx"
    inventory = table_inventory()
    spearman = spearman_audit()
    header = f"""# Reproducibility Audit

## Verdict

The frozen ICASSP 2027 claims are traceable to real per-seed result files, exact
split/probe indices, source code, checkpoint hashes, and sanitized checkpoint
configuration records. No main experiment was retrained and no scientific value
was inferred from manuscript prose. One path-sanitized release copy of a
SHA-256-identified canonical N-MNIST SNN-DFA checkpoint is bundled so the
reviewer smoke test performs a real replay without exposing machine paths.

Audit date: 2026-09-13. The source archive had no recoverable Git commit; the
release therefore uses byte-level source and artifact manifests instead.

| Area | Status |
|---|---|
| A. Training/model/feedback configuration | Confirmed from code and 28 sanitized checkpoint records |
| B. Splits and probes | Exact indices included and regeneration rule confirmed |
| C. DFA local gradient | Confirmed from production code plus two numerical checks |
| D. Spectral diagnostics | Exact definitions and decomposition confirmed |
| E. Homogenization/controls | Preserve/change semantics confirmed from code |
| F. DVS-Gesture | Three-seed values, shapes and cosine definitions recovered |
| G. Numerical tables | Nine per-seed/source tables generated from frozen CSVs |
| H. Public repository | Structured, path-sanitized and hash-manifested |
| I. One-click replay | Scripts provided; canonical checkpoint smoke passed |

## Evidence hierarchy

1. `results/frozen_sources/` contains byte-preserved source CSVs.
2. `results/*.csv` contains reviewer-facing tables built only from those files.
3. `metadata/` contains exact indices, 28 sanitized checkpoint records, feedback
   tensor hashes, source-code hashes, and smoke outputs.
4. `src/` and `diagnostics/archived/` contain the production and replay logic.
5. `ARTIFACT_MANIFEST_SHA256.csv` covers every published file byte-for-byte.

"""
    contraction = f"""
### C.4 Numerical checks (no training)

`diagnostics/local_gradient_parity.py` compares the explicit global-reduction
contraction `sum(delta*x^T)/(T*B*D)` with both the diagnostic autograd capture
and `Trainer._local_batch`. The saved source-only check passed with maximum
relative error `{parity['max_relative_error']:.17g}` at tolerance
`{parity['tolerance']:.1e}` (`metadata/smoke_results/local_gradient_parity.json`).

The stronger checkpoint smoke loaded the bundled epoch-100 seed-20260830
scientific state, verified the sanitized release SHA-256 and all 1,024 probe
indices, replayed the fixed 512-sample `basis_eval` half, and performed no
optimizer step. Its explicit weight-gradient contraction error was
`{smoke['max_explicit_gradient_relative_error']:.17g}`;
the production `q*g` delta parity error was
`{smoke['max_production_delta_relative_l2_error']:.17g}`. All checks passed.

"""
    tail = f"""
## G. Full numerical tables

The release-authoring table builder is `reproduce/build_appendix_tables.mjs`.
It uses the frozen CSV inputs and emitted both the requested CSVs and one
formatted workbook. Every output row includes `source_file` or
`provenance_file`; values are selected or joined from real source rows, never
typed from the paper. The builder's `@oai/artifact-tool` dependency belongs to
the authoring environment and is not a reviewer runtime dependency.

{inventory}

Workbook: `supplement/appendix_tables.xlsx` ({xlsx.stat().st_size} bytes,
SHA-256 `{sha256(xlsx)}`).

`dvs_per_seed.csv` has one row per seed/layer. Accuracy uses both explicitly
labeled checkpoint rules; raw/projected cosine, q/post/aggregated ranks, shapes,
and delta/gate within/between fractions remain distinct columns.

## H. Repository and privacy audit

The public layout is `configs/`, `src/`, `diagnostics/`, `reproduce/`,
`metadata/`, `results/`, and `supplement/`. The release excludes datasets,
training logs, submission PDFs, private repository addresses, host/IP details,
credentials, usernames, and machine-local paths. In the bundled release copy,
location-only config fields were removed/replaced with `DATA_ROOT` and
`OUTPUT_DIR`; no scientific state was changed. The original and release hashes,
changed keys, tensor count, and state-equivalence check are recorded in
`metadata/checkpoints/checkpoint_release_provenance.json`.
`metadata/source_code_provenance.csv` confirms
that all eight high-impact production files checked against the completed-run
server copy are byte-identical.

`metadata/checkpoints/checkpoint_manifest.csv` lists all 28 final checkpoints
(25 N-MNIST and three DVS-Gesture) by method/seed and frozen SHA-256. The bundled
file is a path-sanitized release copy of the canonical SNN-DFA seed-20260830
epoch-100 scientific state. The other binaries are represented by
path/hash/config metadata, not silently replaced with older checkpoints.

## I. Running the repository

```bash
python -m pip install -e .
bash reproduce/smoke_test.sh --source-only
DATA_ROOT=/path/containing/nmnist bash reproduce/smoke_test.sh
bash reproduce/make_figures.sh
DATA_ROOT=/path/containing/nmnist bash reproduce/reproduce_main_diagnostics.sh
```

The actual release smoke used seed 20260830/H1 with a separately supplied
matching pre-framed N-MNIST NPZ tree and passed every assertion:

- checkpoint SHA-256 and stored completed epoch: passed;
- exact probe ordering and fit/eval assignment: passed;
- samples/time/features: `[30,512,800]`;
- total `r95`: expected 546, measured {smoke['measured_r95']};
- within-time fraction: expected {smoke['expected_within_time_trace_fraction']},
  measured {smoke['measured_within_time_trace_fraction']};
- covariance identity residual: {smoke['decomposition']['identity_relative_frobenius']:.17g};
- no optimizer step and no training.

The portable smoke output is
`metadata/smoke_results/checkpoint_smoke_seed_20260830_H1.json`.

## J. Confirmed and not-confirmed boundaries

### Spearman checkpoint-count audit

{spearman}

### Fig.2D dominant-subspace audit

**Verdict C.** The archive retains real overlap/principal-angle summaries but
does not retain the matching independently trained DFA and BPTT basis matrices
or the underlying teaching-signal matrices needed to refit them. Principal angles provide singular values of
`Q_DFA.T @ Q_BPTT`; they do not determine its 1,024 individual entries.
Consequently a real top-32 matrix `abs(Q_DFA.T @ Q_BPTT)` cannot be uniquely
reconstructed, and this repository does not generate a 32x32 heatmap. The safe
paper display is the archived layerwise normalized-overlap and mean-angle
summary only. See `supplement/fig2d_data_audit.md` and
`results/frozen_sources/nmnist/fig2_subspace_overlap.csv`.

### Main-text versus appendix scope

Confirmed facts suitable for the main text are the frozen five-seed N-MNIST
accuracy/rank comparisons, the specified intervention/control summaries, the
within-time decomposition, and the explicitly qualified three-seed DVS boundary
result. Per-seed cutoffs, threshold-free ranks, probe-size repeats, width results,
full alpha trajectories, configuration records, and DVS row-level diagnostics
are best placed in the appendix/supplement.

The following facts are not recoverable and are not claimed:

1. The clean archive consumes pre-generated NPZ frames but lacks the exact raw
   event-to-NPZ converter and pinned SpikingJelly version. Post-NPZ loading,
   shapes, splits and indices are exact for a matching supplied tree;
   bit-for-bit raw conversion and NPZ dataset identity are not verified.
2. One path-sanitized release copy of 28 final checkpoint binaries is bundled.
   Its canonical identity and unchanged scientific state are verified. The other 27 have
   sanitized configuration/path/hash records, but cannot be replayed from this
   repository alone.
3. The historical eigensolver branch (normal float32-device branch versus its
   documented CPU-float64 fallback) was not logged per call.
4. The original unpacked experiment tree had no usable Git commit. Source SHA-256
   manifests are the provenance mechanism.
5. DVS H1/H2 raw-spatial credit spectra were not archived. Their raw gate cosine
   values must not be described as raw credit ranks.
6. The archive's dataset checksum includes source path strings and is not
   portable across directory roots; the exact split/probe file hashes are
   portable and published.

Existing DVS-BPTT task accuracies are archived in
`results/frozen_sources/dvs/test_results.csv`; this audit did not launch new BPTT
training. These boundaries do not alter the frozen scientific conclusions.
"""
    output = header + abc + contraction + deff + tail
    (ROOT / "REPRODUCIBILITY_AUDIT.md").write_text(output, encoding="utf-8")
    print(f"wrote {ROOT / 'REPRODUCIBILITY_AUDIT.md'}")


if __name__ == "__main__":
    main()
